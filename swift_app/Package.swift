// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "IdealRussianTranscribeApp",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "IdealRussianTranscribeApp", targets: ["IdealRussianTranscribeApp"]),
    ],
    targets: [
        .executableTarget(
            name: "IdealRussianTranscribeApp",
            path: "Sources/IdealRussianTranscribeApp"
        ),
    ]
)
