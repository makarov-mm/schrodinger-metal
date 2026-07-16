import Foundation
import AVFoundation
import CoreVideo

// Wraps AVAssetWriter to append raw BGRA frames at a fixed frame rate.
// All appends are funneled through one serial queue so GPU completion
// handlers from consecutive command buffers cannot interleave writes.
final class VideoRecorder {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let queue = DispatchQueue(label: "schrodinger.video.recorder")

    let width: Int
    let height: Int
    private let fps: Int32
    private var frameIndex: Int64 = 0

    init(url: URL, width: Int, height: Int, fps: Int32 = 60) throws {
        try? FileManager.default.removeItem(at: url)

        self.width = width
        self.height = height
        self.fps = fps

        writer = try AVAssetWriter(outputURL: url, fileType: .mov)

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true

        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
        )

        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
    }

    // Copies one BGRA frame into a pooled CVPixelBuffer and appends it.
    // Called from GPU command buffer completion handlers (background threads).
    // If the writer is briefly not ready, the frame is dropped; the frame
    // index still advances so playback timing stays correct.
    func appendFrame(from src: UnsafeRawPointer, srcBytesPerRow: Int) {
        queue.sync {
            defer { frameIndex += 1 }

            guard input.isReadyForMoreMediaData,
                  let pool = adaptor.pixelBufferPool else { return }

            var pixelBufferOut: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBufferOut)
            guard let pixelBuffer = pixelBufferOut else { return }

            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let dst = CVPixelBufferGetBaseAddress(pixelBuffer) {
                let dstBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
                let rowBytes = width * 4
                for row in 0..<height {
                    memcpy(dst.advanced(by: row * dstBytesPerRow),
                           src.advanced(by: row * srcBytesPerRow),
                           rowBytes)
                }
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

            let time = CMTime(value: frameIndex, timescale: fps)
            adaptor.append(pixelBuffer, withPresentationTime: time)
        }
    }

    func finish(completion: @escaping (URL) -> Void) {
        queue.sync {
            input.markAsFinished()
        }
        writer.finishWriting { [writer] in
            completion(writer.outputURL)
        }
    }
}
