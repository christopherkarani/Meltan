// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MetalANNS",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "MetalANNS", targets: ["MetalANNS"])
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.0.0"),
    ],
    targets: [
        .target(
            name: "MetalANNSC",
            publicHeadersPath: "include"
        ),
        .target(
            name: "MetalANNSCore",
            dependencies: ["MetalANNSC"],
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "MetalANNS",
            dependencies: [
                "MetalANNSCore",
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MetalANNSTests",
            dependencies: [
                "MetalANNS",
                "MetalANNSCore",
                "MetalANNSC",
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "MetalANNSBenchmarks",
            dependencies: ["MetalANNS", "MetalANNSCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
