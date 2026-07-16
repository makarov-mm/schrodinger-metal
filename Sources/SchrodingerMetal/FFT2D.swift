import Metal

// 2D FFT built from per-row 1D FFTs and a transpose:
//   forward = fftRows -> transpose -> fftRows -> transpose
// The transpose returns the data to its original orientation, so the caller
// always sees the field in row-major (y * N + x) order.
final class FFT2D {
    private let device: MTLDevice
    private let n: Int
    private let logN: UInt32

    private let fftPipeline: MTLComputePipelineState
    private let transposePipeline: MTLComputePipelineState

    // Scratch buffer for the transpose ping-pong. Owned here.
    private let scratch: MTLBuffer

    init(device: MTLDevice, library: MTLLibrary, n: Int) throws {
        precondition(n > 0 && (n & (n - 1)) == 0, "N must be a power of two")
        self.device = device
        self.n = n
        // For a power of two, log2(n) equals the number of trailing zero bits.
        // Integer math avoids pulling in Foundation for log2.
        self.logN = UInt32(n.trailingZeroBitCount)

        guard let fftFn = library.makeFunction(name: "fft_rows"),
              let transposeFn = library.makeFunction(name: "transpose2d") else {
            throw RendererError.functionNotFound("fft_rows / transpose")
        }
        self.fftPipeline = try device.makeComputePipelineState(function: fftFn)
        self.transposePipeline = try device.makeComputePipelineState(function: transposeFn)

        guard let buf = device.makeBuffer(length: n * n * MemoryLayout<SIMD2<Float>>.stride,
                                          options: .storageModePrivate) else {
            throw RendererError.bufferAllocationFailed
        }
        self.scratch = buf
    }

    func forward(_ data: MTLBuffer, params: MTLBuffer, in encoder: MTLComputeCommandEncoder) {
        run(data, params: params, inverse: false, encoder: encoder)
    }

    func inverse(_ data: MTLBuffer, params: MTLBuffer, in encoder: MTLComputeCommandEncoder) {
        run(data, params: params, inverse: true, encoder: encoder)
    }

    private func run(_ data: MTLBuffer, params: MTLBuffer, inverse: Bool, encoder: MTLComputeCommandEncoder) {
        var logNValue = logN
        var inverseFlag: Int32 = inverse ? 1 : 0

        // Pass 1: FFT over rows of `data` (in place).
        encodeRowFFT(data, params: params, logN: &logNValue, inverse: &inverseFlag, encoder: encoder)
        // data -> scratch (transposed)
        encodeTranspose(src: data, dst: scratch, params: params, encoder: encoder)
        // Pass 2: FFT over rows of scratch (originally columns).
        encodeRowFFT(scratch, params: params, logN: &logNValue, inverse: &inverseFlag, encoder: encoder)
        // scratch -> data (transpose back to original orientation).
        encodeTranspose(src: scratch, dst: data, params: params, encoder: encoder)
    }

    private func encodeRowFFT(_ buffer: MTLBuffer,
                              params: MTLBuffer,
                              logN: inout UInt32,
                              inverse: inout Int32,
                              encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(fftPipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(params, offset: 0, index: 1)
        encoder.setBytes(&logN, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&inverse, length: MemoryLayout<Int32>.stride, index: 3)
        // One threadgroup per row, N/2 threads each, N complex numbers in shared memory.
        encoder.setThreadgroupMemoryLength(n * MemoryLayout<SIMD2<Float>>.stride, index: 0)
        let threadsPerGroup = MTLSize(width: n / 2, height: 1, depth: 1)
        let groups = MTLSize(width: n, height: 1, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }

    private func encodeTranspose(src: MTLBuffer,
                                 dst: MTLBuffer,
                                 params: MTLBuffer,
                                 encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(transposePipeline)
        encoder.setBuffer(src, offset: 0, index: 0)
        encoder.setBuffer(dst, offset: 0, index: 1)
        encoder.setBuffer(params, offset: 0, index: 2)
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: n, height: n, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
    }
}
