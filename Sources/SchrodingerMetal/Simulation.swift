import Metal

enum RendererError: Error {
    case deviceUnavailable
    case libraryLoadFailed
    case functionNotFound(String)
    case bufferAllocationFailed
}

// Owns the wavefunction buffer, potential buffer and the compute pipelines,
// and advances the state with the split-step Fourier method (Strang splitting):
//   half potential -> forward FFT -> kinetic -> inverse FFT -> half potential.
final class Simulation {
    let device: MTLDevice
    let n: Int

    private(set) var params: Params

    let psiBuffer: MTLBuffer      // float2 per cell, N*N
    let potentialBuffer: MTLBuffer // float per cell, N*N
    let paramsBuffer: MTLBuffer

    private let fft: FFT2D

    private let initPipeline: MTLComputePipelineState
    private let potentialPipeline: MTLComputePipelineState
    private let kineticPipeline: MTLComputePipelineState
    private let absorbPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary, n: Int, params: Params) throws {
        self.device = device
        self.n = n
        self.params = params

        guard let psi = device.makeBuffer(length: n * n * MemoryLayout<SIMD2<Float>>.stride,
                                          options: .storageModePrivate),
              let pot = device.makeBuffer(length: n * n * MemoryLayout<Float>.stride,
                                          options: .storageModePrivate),
              let pbuf = device.makeBuffer(length: MemoryLayout<Params>.stride,
                                           options: .storageModeShared) else {
            throw RendererError.bufferAllocationFailed
        }
        self.psiBuffer = psi
        self.potentialBuffer = pot
        self.paramsBuffer = pbuf

        self.fft = try FFT2D(device: device, library: library, n: n)

        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw RendererError.functionNotFound(name)
            }
            return try device.makeComputePipelineState(function: fn)
        }
        self.initPipeline = try pipeline("init_state")
        self.potentialPipeline = try pipeline("potential_step")
        self.kineticPipeline = try pipeline("kinetic_step")
        self.absorbPipeline = try pipeline("absorb_step")

        uploadParams()
    }

    func uploadParams() {
        paramsBuffer.contents().copyMemory(from: &params, byteCount: MemoryLayout<Params>.stride)
    }

    func updateParams(_ mutate: (inout Params) -> Void) {
        mutate(&params)
        uploadParams()
    }

    func reset(commandBuffer: MTLCommandBuffer) {
        params.time = 0
        uploadParams()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(initPipeline)
        encoder.setBuffer(psiBuffer, offset: 0, index: 0)
        encoder.setBuffer(potentialBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        dispatch2D(encoder)
        encoder.endEncoding()
    }

    // Advance `substeps` split-step iterations in one command buffer.
    func step(substeps: Int, commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        for _ in 0..<substeps {
            encodePotentialHalfStep(encoder)
            fft.forward(psiBuffer, params: paramsBuffer, in: encoder)
            encodeKineticStep(encoder)
            fft.inverse(psiBuffer, params: paramsBuffer, in: encoder)
            encodePotentialHalfStep(encoder)
            encodeAbsorbStep(encoder)
        }
        encoder.endEncoding()
        params.time += Float(substeps) * params.dt
        uploadParams()
    }

    private func encodePotentialHalfStep(_ encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(potentialPipeline)
        encoder.setBuffer(psiBuffer, offset: 0, index: 0)
        encoder.setBuffer(potentialBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        dispatch2D(encoder)
    }

    private func encodeKineticStep(_ encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(kineticPipeline)
        encoder.setBuffer(psiBuffer, offset: 0, index: 0)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
        dispatch2D(encoder)
    }

    private func encodeAbsorbStep(_ encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(absorbPipeline)
        encoder.setBuffer(psiBuffer, offset: 0, index: 0)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
        dispatch2D(encoder)
    }

    private func dispatch2D(_ encoder: MTLComputeCommandEncoder) {
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: n, height: n, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
    }
}
