import Foundation
import MetalKit

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let simulation: Simulation
    private let renderPipeline: MTLComputePipelineState

    // How many split-step iterations to run per displayed frame.
    var substepsPerFrame: Int = 2
    var isPaused: Bool = false
    private var needsReset: Bool = true

    // The render kernel writes into this offscreen texture, which is then
    // blitted to the drawable. Recording reads frames back from here.
    private var offscreenTexture: MTLTexture?

    // Recording state. Three rotating readback buffers let the CPU copy
    // frame i while the GPU is already producing frames i+1 and i+2.
    private static let readbackSlots = 3
    private(set) var isRecording = false
    private var recorder: VideoRecorder?
    private var readbackBuffers: [MTLBuffer] = []
    private var readbackBytesPerRow = 0
    private var nextSlot = 0
    private let captureSemaphore = DispatchSemaphore(value: Renderer.readbackSlots)

    init(device: MTLDevice, n: Int, params: Params) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw RendererError.deviceUnavailable
        }
        self.commandQueue = queue

        // Load and compile the Metal source at runtime. This keeps the package
        // buildable with plain SwiftPM without an Xcode-generated metallib.
        guard let url = Bundle.module.url(forResource: "Shaders", withExtension: "metal"),
              let source = try? String(contentsOf: url, encoding: .utf8) else {
            throw RendererError.libraryLoadFailed
        }
        let library = try device.makeLibrary(source: source, options: nil)

        self.simulation = try Simulation(device: device, library: library, n: n, params: params)

        guard let renderFn = library.makeFunction(name: "render") else {
            throw RendererError.functionNotFound("render")
        }
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)

        super.init()
    }

    // MARK: - Controls forwarded from the UI

    func requestReset() { needsReset = true }

    func updateParams(_ mutate: (inout Params) -> Void) {
        simulation.updateParams(mutate)
    }

    // MARK: - Recording

    func startRecording(to url: URL) throws {
        guard !isRecording else { return }
        guard let offscreen = offscreenTexture else {
            throw RendererError.deviceUnavailable
        }
        recorder = try VideoRecorder(url: url, width: offscreen.width, height: offscreen.height)
        isRecording = true
    }

    func stopRecording(completion: @escaping (URL?) -> Void) {
        guard isRecording, let rec = recorder else {
            completion(nil)
            return
        }
        isRecording = false
        recorder = nil
        // Drain the semaphore to make sure all in-flight GPU readbacks have
        // finished appending before the writer is closed.
        DispatchQueue.global(qos: .userInitiated).async { [captureSemaphore] in
            for _ in 0..<Renderer.readbackSlots { captureSemaphore.wait() }
            for _ in 0..<Renderer.readbackSlots { captureSemaphore.signal() }
            rec.finish { url in
                completion(url)
            }
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        ensureOffscreen(width: drawable.texture.width, height: drawable.texture.height)
        guard let offscreen = offscreenTexture else { return }

        if needsReset {
            simulation.reset(commandBuffer: commandBuffer)
            needsReset = false
        } else if !isPaused {
            simulation.step(substeps: substepsPerFrame, commandBuffer: commandBuffer)
        }

        // Domain-coloring render into the offscreen texture.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(renderPipeline)
            encoder.setTexture(offscreen, index: 0)
            encoder.setBuffer(simulation.psiBuffer, offset: 0, index: 0)
            encoder.setBuffer(simulation.paramsBuffer, offset: 0, index: 1)
            let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
            let grid = MTLSize(width: offscreen.width, height: offscreen.height, depth: 1)
            encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        // Copy offscreen to the drawable, and to a readback buffer if recording.
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            let size = MTLSize(width: offscreen.width, height: offscreen.height, depth: 1)
            blit.copy(from: offscreen,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: size,
                      to: drawable.texture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))

            if isRecording, let rec = recorder,
               captureSemaphore.wait(timeout: .now()) == .success {
                // If all slots are busy the frame is simply skipped instead of
                // stalling the render loop.
                let slot = nextSlot
                nextSlot = (nextSlot + 1) % Renderer.readbackSlots
                let buffer = readbackBuffers[slot]
                let bytesPerRow = readbackBytesPerRow

                blit.copy(from: offscreen,
                          sourceSlice: 0, sourceLevel: 0,
                          sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: size,
                          to: buffer,
                          destinationOffset: 0,
                          destinationBytesPerRow: bytesPerRow,
                          destinationBytesPerImage: bytesPerRow * offscreen.height)

                commandBuffer.addCompletedHandler { [captureSemaphore] _ in
                    rec.appendFrame(from: buffer.contents(), srcBytesPerRow: bytesPerRow)
                    captureSemaphore.signal()
                }
            }
            blit.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Offscreen management

    private func ensureOffscreen(width: Int, height: Int) {
        if let tex = offscreenTexture, tex.width == width, tex.height == height {
            return
        }

        // A size change invalidates the recording surface; stop cleanly.
        if isRecording {
            stopRecording { _ in }
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .private
        offscreenTexture = device.makeTexture(descriptor: descriptor)

        // Blit destination rows must be 256-byte aligned on macOS.
        readbackBytesPerRow = (width * 4 + 255) & ~255
        readbackBuffers = (0..<Renderer.readbackSlots).compactMap { _ in
            device.makeBuffer(length: readbackBytesPerRow * height, options: .storageModeShared)
        }
    }
}
