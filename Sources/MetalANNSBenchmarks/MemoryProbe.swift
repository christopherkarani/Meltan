import Foundation
#if canImport(Darwin)
import Darwin
#endif

public struct MemorySnapshot: Sendable {
    public let residentBytes: UInt64
    public let peakResidentBytes: UInt64
    public let virtualBytes: UInt64

    public init(residentBytes: UInt64, peakResidentBytes: UInt64, virtualBytes: UInt64) {
        self.residentBytes = residentBytes
        self.peakResidentBytes = peakResidentBytes
        self.virtualBytes = virtualBytes
    }

    public static func zero() -> MemorySnapshot {
        MemorySnapshot(residentBytes: 0, peakResidentBytes: 0, virtualBytes: 0)
    }

    public static func capture() -> MemorySnapshot {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)

        let result = withUnsafeMutablePointer(to: &info) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    reboundPointer,
                    &count
                )
            }
        }

        guard result == KERN_SUCCESS else {
            return .zero()
        }

        // phys_footprint is the modern Darwin RSS analogue used by the OS for
        // memory accounting (matches what Activity Monitor reports). Fall back
        // to resident_size if it is somehow zero on the running platform.
        let resident = info.phys_footprint != 0 ? info.phys_footprint : UInt64(info.resident_size)
        let peak = UInt64(info.resident_size_peak)
        let virtualSize = UInt64(info.virtual_size)

        return MemorySnapshot(
            residentBytes: resident,
            peakResidentBytes: peak,
            virtualBytes: virtualSize
        )
        #else
        return .zero()
        #endif
    }

    public var residentMB: Double { Double(residentBytes) / (1024 * 1024) }
    public var peakResidentMB: Double { Double(peakResidentBytes) / (1024 * 1024) }
}
