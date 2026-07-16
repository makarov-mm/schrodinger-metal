import Foundation

// Must match the `Params` struct in Shaders.metal field for field.
// All members are 4-byte scalars in the same order so the byte layout is
// identical on both sides without explicit padding.
struct Params {
    var N: UInt32
    var dx: Float
    var dt: Float
    var hbar: Float
    var mass: Float
    var sigma: Float
    var x0: Float
    var y0: Float
    var kx0: Float
    var ky0: Float
    var potentialStrength: Float
    var brightness: Float
    var absorberStrength: Float
    var time: Float
    var potentialType: UInt32
}

enum PotentialType: UInt32, CaseIterable, Identifiable {
    case free = 0
    case harmonic = 1
    case barrier = 2
    case doubleSlit = 3

    var id: UInt32 { rawValue }

    var label: String {
        switch self {
        case .free: return "Free space"
        case .harmonic: return "Harmonic well"
        case .barrier: return "Barrier"
        case .doubleSlit: return "Double slit"
        }
    }
}
