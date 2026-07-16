import AppKit
import MetalKit

// Hosts the Metal view and the control strip. Replaces the former SwiftUI
// ContentView + MetalView pair with plain AppKit so the project builds with
// Command Line Tools only.
final class MainViewController: NSViewController {
    // Grid size, must be a power of two. 256 or 512 are good starting points.
    private let n = 256

    private var renderer: Renderer?
    private var mtkView: MTKView!

    // UI state mirrored into Params.
    private var potential: PotentialType = .doubleSlit
    private var kx0: Float = 6.0
    private var brightness: Float = 3.0
    private var paused = false

    // Controls we need to read back or update.
    private var kxValueLabel: NSTextField!
    private var brightnessValueLabel: NSTextField!
    private var pauseButton: NSButton!
    private var recordButton: NSButton!

    private func makeParams() -> Params {
        Params(
            N: UInt32(n),
            dx: 0.05,
            dt: 0.002,
            hbar: 1.0,
            mass: 1.0,
            sigma: 0.6,
            x0: -3.0,
            y0: 0.0,
            kx0: kx0,
            ky0: 0.0,
            potentialStrength: 400.0,
            brightness: brightness,
            absorberStrength: 150.0,
            time: 0.0,
            potentialType: potential.rawValue
        )
    }

    override func loadView() {
        view = NSView(frame: NSRect(x: 0, y: 0, width: 640, height: 720))
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        mtkView = MTKView(frame: .zero, device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.framebufferOnly = false // the render kernel writes to the drawable
        mtkView.preferredFramesPerSecond = 60
        mtkView.translatesAutoresizingMaskIntoConstraints = false

        do {
            let r = try Renderer(device: device, n: n, params: makeParams())
            mtkView.delegate = r
            renderer = r
        } catch {
            fatalError("Renderer init failed: \(error)")
        }

        let controls = buildControls()
        controls.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(mtkView)
        view.addSubview(controls)

        NSLayoutConstraint.activate([
            mtkView.topAnchor.constraint(equalTo: view.topAnchor),
            mtkView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            mtkView.trailingAnchor.constraint(equalTo: view.trailingAnchor),

            controls.topAnchor.constraint(equalTo: mtkView.bottomAnchor),
            controls.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
            controls.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),
            controls.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -12),
            controls.heightAnchor.constraint(equalToConstant: 120)
        ])
    }

    // MARK: - Controls

    private func buildControls() -> NSView {
        // Row 1: potential selector + pause + reset.
        let segmented = NSSegmentedControl(
            labels: PotentialType.allCases.map { $0.label },
            trackingMode: .selectOne,
            target: self,
            action: #selector(potentialChanged(_:))
        )
        segmented.selectedSegment = PotentialType.allCases.firstIndex(of: potential) ?? 0

        pauseButton = NSButton(title: "Pause", target: self, action: #selector(togglePause(_:)))
        let resetButton = NSButton(title: "Reset", target: self, action: #selector(resetPressed(_:)))
        recordButton = NSButton(title: "Record", target: self, action: #selector(toggleRecord(_:)))

        let row1 = NSStackView(views: [segmented, pauseButton, resetButton, recordButton])
        row1.orientation = .horizontal
        row1.spacing = 8

        // Row 2: momentum slider.
        let kxLabel = NSTextField(labelWithString: "Momentum kx")
        let kxSlider = NSSlider(value: Double(kx0), minValue: 0, maxValue: 20,
                                target: self, action: #selector(kxChanged(_:)))
        kxValueLabel = NSTextField(labelWithString: String(format: "%.1f", kx0))
        kxValueLabel.font = .monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)

        let row2 = NSStackView(views: [kxLabel, kxSlider, kxValueLabel])
        row2.orientation = .horizontal
        row2.spacing = 8

        // Row 3: brightness slider.
        let brightnessLabel = NSTextField(labelWithString: "Brightness")
        let brightnessSlider = NSSlider(value: Double(brightness), minValue: 0.2, maxValue: 10,
                                        target: self, action: #selector(brightnessChanged(_:)))
        brightnessValueLabel = NSTextField(labelWithString: String(format: "%.1f", brightness))
        brightnessValueLabel.font = .monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)

        let row3 = NSStackView(views: [brightnessLabel, brightnessSlider, brightnessValueLabel])
        row3.orientation = .horizontal
        row3.spacing = 8

        // Make sliders take the remaining width.
        kxSlider.setContentHuggingPriority(.defaultLow, for: .horizontal)
        brightnessSlider.setContentHuggingPriority(.defaultLow, for: .horizontal)
        segmented.setContentHuggingPriority(.defaultLow, for: .horizontal)

        let stack = NSStackView(views: [row1, row2, row3])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 10
        stack.distribution = .fillEqually

        // Rows should stretch horizontally inside the vertical stack.
        for row in [row1, row2, row3] {
            row.translatesAutoresizingMaskIntoConstraints = false
            row.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        }
        return stack
    }

    // MARK: - Actions

    @objc private func potentialChanged(_ sender: NSSegmentedControl) {
        let index = sender.selectedSegment
        guard index >= 0 && index < PotentialType.allCases.count else { return }
        potential = PotentialType.allCases[index]
        renderer?.updateParams { $0.potentialType = self.potential.rawValue }
        renderer?.requestReset()
    }

    @objc private func togglePause(_ sender: NSButton) {
        paused.toggle()
        renderer?.isPaused = paused
        pauseButton.title = paused ? "Play" : "Pause"
    }

    @objc private func resetPressed(_ sender: NSButton) {
        renderer?.requestReset()
    }

    @objc private func toggleRecord(_ sender: NSButton) {
        guard let renderer = renderer else { return }

        if renderer.isRecording {
            recordButton.isEnabled = false
            renderer.stopRecording { [weak self] url in
                DispatchQueue.main.async {
                    self?.recordButton.title = "Record"
                    self?.recordButton.isEnabled = true
                    if let url = url {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
            }
        } else {
            do {
                try renderer.startRecording(to: Self.makeOutputURL())
                recordButton.title = "Stop"
            } catch {
                let alert = NSAlert()
                alert.messageText = "Recording failed to start"
                alert.informativeText = String(describing: error)
                alert.runModal()
            }
        }
    }

    private static func makeOutputURL() -> URL {
        let base = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let name = "SchrodingerMetal-\(formatter.string(from: Date())).mov"
        return base.appendingPathComponent(name)
    }

    @objc private func kxChanged(_ sender: NSSlider) {
        kx0 = Float(sender.doubleValue)
        kxValueLabel.stringValue = String(format: "%.1f", kx0)
        renderer?.updateParams { $0.kx0 = self.kx0 }
        renderer?.requestReset()
    }

    @objc private func brightnessChanged(_ sender: NSSlider) {
        brightness = Float(sender.doubleValue)
        brightnessValueLabel.stringValue = String(format: "%.1f", brightness)
        renderer?.updateParams { $0.brightness = self.brightness }
    }
}
