import Foundation
#if canImport(Metal)
import Metal
#endif

public struct EnvironmentProbe: Sendable {
    public let osVersion: String
    public let osBuild: String
    public let hostName: String
    public let physicalCoreCount: Int
    public let activeCoreCount: Int
    public let physicalMemoryBytes: UInt64
    public let thermalState: String
    public let lowPowerModeEnabled: Bool
    public let metalDeviceName: String?
    public let metalIsLowPower: Bool?
    public let metalSupportsUnifiedMemory: Bool?
    public let metalRecommendedMaxWorkingSetBytes: UInt64?
    public let processArguments: [String]
    public let buildConfiguration: String
    public let gitSHA: String?

    public init(
        osVersion: String,
        osBuild: String,
        hostName: String,
        physicalCoreCount: Int,
        activeCoreCount: Int,
        physicalMemoryBytes: UInt64,
        thermalState: String,
        lowPowerModeEnabled: Bool,
        metalDeviceName: String?,
        metalIsLowPower: Bool?,
        metalSupportsUnifiedMemory: Bool?,
        metalRecommendedMaxWorkingSetBytes: UInt64?,
        processArguments: [String],
        buildConfiguration: String,
        gitSHA: String?
    ) {
        self.osVersion = osVersion
        self.osBuild = osBuild
        self.hostName = hostName
        self.physicalCoreCount = physicalCoreCount
        self.activeCoreCount = activeCoreCount
        self.physicalMemoryBytes = physicalMemoryBytes
        self.thermalState = thermalState
        self.lowPowerModeEnabled = lowPowerModeEnabled
        self.metalDeviceName = metalDeviceName
        self.metalIsLowPower = metalIsLowPower
        self.metalSupportsUnifiedMemory = metalSupportsUnifiedMemory
        self.metalRecommendedMaxWorkingSetBytes = metalRecommendedMaxWorkingSetBytes
        self.processArguments = processArguments
        self.buildConfiguration = buildConfiguration
        self.gitSHA = gitSHA
    }

    public static func capture() -> EnvironmentProbe {
        let info = ProcessInfo.processInfo

        let metalDeviceName: String?
        let metalIsLowPower: Bool?
        let metalSupportsUnifiedMemory: Bool?
        let metalRecommendedMaxWorkingSetBytes: UInt64?
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            metalDeviceName = device.name
            metalIsLowPower = device.isLowPower
            metalSupportsUnifiedMemory = device.hasUnifiedMemory
            metalRecommendedMaxWorkingSetBytes = UInt64(device.recommendedMaxWorkingSetSize)
        } else {
            metalDeviceName = nil
            metalIsLowPower = nil
            metalSupportsUnifiedMemory = nil
            metalRecommendedMaxWorkingSetBytes = nil
        }
        #else
        metalDeviceName = nil
        metalIsLowPower = nil
        metalSupportsUnifiedMemory = nil
        metalRecommendedMaxWorkingSetBytes = nil
        #endif

        let arguments = CommandLine.arguments.count > 1
            ? Array(CommandLine.arguments.dropFirst())
            : []

        return EnvironmentProbe(
            osVersion: info.operatingSystemVersionString,
            osBuild: readOSBuild(),
            hostName: info.hostName,
            physicalCoreCount: info.processorCount,
            activeCoreCount: info.activeProcessorCount,
            physicalMemoryBytes: info.physicalMemory,
            thermalState: thermalStateString(info.thermalState),
            lowPowerModeEnabled: info.isLowPowerModeEnabled,
            metalDeviceName: metalDeviceName,
            metalIsLowPower: metalIsLowPower,
            metalSupportsUnifiedMemory: metalSupportsUnifiedMemory,
            metalRecommendedMaxWorkingSetBytes: metalRecommendedMaxWorkingSetBytes,
            processArguments: arguments,
            buildConfiguration: currentBuildConfiguration(),
            gitSHA: readGitSHA()
        )
    }

    public func asMetadata() -> [String: String] {
        var metadata: [String: String] = [
            "osVersion": osVersion,
            "osBuild": osBuild,
            "hostName": hostName,
            "physicalCoreCount": String(physicalCoreCount),
            "activeCoreCount": String(activeCoreCount),
            "physicalMemoryBytes": String(physicalMemoryBytes),
            "thermalState": thermalState,
            "lowPowerModeEnabled": String(lowPowerModeEnabled),
            "buildConfiguration": buildConfiguration,
            "processArguments": processArguments.joined(separator: " ")
        ]
        if let metalDeviceName {
            metadata["metalDeviceName"] = metalDeviceName
        }
        if let metalIsLowPower {
            metadata["metalIsLowPower"] = String(metalIsLowPower)
        }
        if let metalSupportsUnifiedMemory {
            metadata["metalSupportsUnifiedMemory"] = String(metalSupportsUnifiedMemory)
        }
        if let metalRecommendedMaxWorkingSetBytes {
            metadata["metalRecommendedMaxWorkingSetBytes"] = String(metalRecommendedMaxWorkingSetBytes)
        }
        if let gitSHA {
            metadata["gitSHA"] = gitSHA
        }
        return metadata
    }

    private static func thermalStateString(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func currentBuildConfiguration() -> String {
        #if DEBUG
        return "debug"
        #else
        return "release"
        #endif
    }

    private static func readOSBuild() -> String {
        #if os(macOS)
        return runProcess(launchPath: "/usr/bin/sw_vers", arguments: ["-buildVersion"]) ?? ""
        #else
        return ""
        #endif
    }

    private static func readGitSHA() -> String? {
        #if os(macOS) || os(Linux)
        if let value = runProcess(launchPath: "/usr/bin/env", arguments: ["git", "rev-parse", "--short", "HEAD"]),
           !value.isEmpty {
            return value
        }
        return nil
        #else
        return nil
        #endif
    }

    private static func runProcess(launchPath: String, arguments: [String]) -> String? {
        #if os(macOS) || os(Linux)
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = Pipe()

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }

        guard process.terminationStatus == 0 else {
            return nil
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let raw = String(data: data, encoding: .utf8) else {
            return nil
        }
        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
        #else
        return nil
        #endif
    }
}
