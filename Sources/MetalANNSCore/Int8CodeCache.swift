import Accelerate
import Foundation

/// Lazily-built int8 quantized mirror of a Float32 corpus.
///
/// Per row: symmetric quantization with scale `u = maxAbs / 127`,
/// `code[d] = clamp(round(v[d] / u), -127, 127)`.
///
/// Search derives provable dot-product error bounds from precomputed,
/// PLANAR per-row metadata (one contiguous array per field, so the hot scan
/// loop streams each plane linearly):
///
///   u        = scale
///   w        = u · eBase,  eBase = |ĉ|₁/2 + dim/4 (inflated for rounding)
///   normSq   = ‖v‖²
///   invSqrt  = 1/‖v‖      (0 when ‖v‖² < 1e-20)
///
/// Exactness contract: for any query q and row v,
///   dot(q, v) ∈ qs·u·D ± qs·(w + u·|q̂|₁/2)
/// where D is the exact int32 code dot and qs the query scale. See
/// `BoundedExactScan`.
final class Int8CodeBuffer: @unchecked Sendable {
    let count: Int
    let dim: Int

    // One allocation, planar sections (all 64-byte aligned relative to base).
    private let storage: UnsafeMutableRawPointer
    private let totalBytes: Int

    let codesBase: UnsafeMutablePointer<Int8>
    let uBase: UnsafeMutablePointer<Float>
    let wBase: UnsafeMutablePointer<Float>
    let normSqBase: UnsafeMutablePointer<Float>
    let invSqrtBase: UnsafeMutablePointer<Float>

    init?(count: Int, dim: Int) {
        guard count > 0, dim > 0 else { return nil }
        let align = 64
        func pad(_ bytes: Int) -> Int { (bytes + align - 1) / align * align }

        let codesBytes = pad(count * dim)
        let planeBytes = pad(count * MemoryLayout<Float>.size)
        let total = codesBytes + 4 * planeBytes
        guard let raw = malloc(total) else { return nil }

        self.count = count
        self.dim = dim
        self.storage = raw
        self.totalBytes = total
        self.codesBase = (raw + 0).assumingMemoryBound(to: Int8.self)
        self.uBase = (raw + codesBytes).assumingMemoryBound(to: Float.self)
        self.wBase = (raw + codesBytes + planeBytes).assumingMemoryBound(to: Float.self)
        self.normSqBase =
            (raw + codesBytes + 2 * planeBytes).assumingMemoryBound(to: Float.self)
        self.invSqrtBase =
            (raw + codesBytes + 3 * planeBytes).assumingMemoryBound(to: Float.self)
    }

    deinit {
        free(storage)
    }

    /// Quantizes every corpus row into the preallocated planes.
    func fill(corpus: UnsafePointer<Float>) {
        let codesOut = codesBase
        let uOut = uBase
        let wOut = wBase
        let normSqOut = normSqBase
        let invSqrtOut = invSqrtBase

        for row in 0..<count {
            let rowBase = corpus + row * dim
            var maxAbs: Float = 0
            vDSP_maxmgv(rowBase, 1, &maxAbs, vDSP_Length(dim))
            let scale = maxAbs > 0 ? maxAbs / 127.0 : 1.0
            let inverse = 1.0 / scale
            let out = codesOut + row * dim
            var absSum: Int32 = 0
            var normSq: Float = 0
            for d in 0..<dim {
                // Round half away from zero; |code| <= 127 keeps the error
                // model |v_d - s*code| <= s/2 valid.
                let value = rowBase[d]
                normSq += value * value
                let scaled = value * inverse
                var rounded = Int32(scaled >= 0 ? scaled + 0.5 : scaled - 0.5)
                if rounded > 127 { rounded = 127 } else if rounded < -127 { rounded = -127 }
                out[d] = Int8(rounded)
                absSum += rounded < 0 ? -rounded : rounded
            }

            // Inflate generously to absorb all downstream fp rounding.
            let eBase = (Float(absSum) / 2 + Float(dim) / 4) * (1 + 1e-6) + 16
            uOut[row] = scale
            wOut[row] = scale * eBase
            normSqOut[row] = normSq
            invSqrtOut[row] = normSq > 1e-20 ? 1.0 / sqrt(normSq) : 0
        }
    }

    static func build(
        corpus: UnsafePointer<Float>, count: Int, dim: Int
    ) -> Int8CodeBuffer? {
        guard let buffer = Int8CodeBuffer(count: count, dim: dim) else { return nil }
        buffer.fill(corpus: corpus)
        return buffer
    }
}

/// Cache of `Int8CodeBuffer`s keyed by backing MTL buffer identity + shape:
/// explicitly invalidated on in-place storage writes, evicted beyond capacity.
enum Int8CodeCache {
    struct Key: BufferIdentityKeyed {
        let bufferID: ObjectIdentifier
        let bufferLength: Int
        let count: Int
        let dim: Int
    }

    private static let store = IdentityKeyedLRUCache<Key, Int8CodeBuffer>(capacity: 8)

    static func codes(
        for buffer: MTLBuffer,
        vectorCount: Int,
        dim: Int,
        corpus: UnsafePointer<Float>
    ) -> Int8CodeBuffer? {
        guard vectorCount > 0, dim > 0 else { return nil }
        let key = Key(
            bufferID: ObjectIdentifier(buffer),
            bufferLength: buffer.length,
            count: vectorCount,
            dim: dim
        )
        if let cached = store.get(key) { return cached }
        guard
            let built = Int8CodeBuffer.build(
                corpus: corpus, count: vectorCount, dim: dim
            )
        else { return nil }
        store.store(built, for: key)
        return built
    }

    /// Called by mutable storage on in-place writes so stale codes can never
    /// be served after an update.
    static func invalidate(buffer: MTLBuffer) {
        store.invalidate(bufferID: ObjectIdentifier(buffer))
    }

    static func clearAll() {
        store.clear()
    }
}
