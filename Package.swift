// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SchrodingerMetal",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "SchrodingerMetal",
            resources: [
                // Shipped as plain text and compiled at runtime with
                // device.makeLibrary(source:). This avoids needing an
                // Xcode-generated default.metallib.
                .copy("Shaders/Shaders.metal")
            ]
        )
    ]
)
