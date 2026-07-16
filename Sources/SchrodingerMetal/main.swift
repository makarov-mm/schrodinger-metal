import AppKit

// Plain AppKit entry point. Top-level code in main.swift avoids the @main
// attribute and any SwiftUI macros, so the project builds with the bare
// Command Line Tools (no full Xcode required).
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
