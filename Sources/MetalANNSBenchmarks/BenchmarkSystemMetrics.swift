import Darwin
import Foundation

enum BenchmarkSystemMetrics {
    static func residentMemoryBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count
                )
            }
        }

        guard result == KERN_SUCCESS else {
            return 0
        }
        return UInt64(info.resident_size)
    }

    static func memoryDelta(before: UInt64, after: UInt64) -> Int64 {
        if after >= before {
            return Int64(after - before)
        }
        return -Int64(before - after)
    }

    static func fileSizeBytes(at url: URL) -> UInt64 {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return 0
        }

        if !isDirectory.boolValue {
            return singleFileSize(at: url)
        }

        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total: UInt64 = 0
        for case let child as URL in enumerator {
            total += singleFileSize(at: child)
        }
        return total
    }

    private static func singleFileSize(at url: URL) -> UInt64 {
        guard
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
            values.isRegularFile == true,
            let size = values.fileSize
        else {
            return 0
        }
        return UInt64(max(0, size))
    }
}
