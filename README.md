# Split-Step Schrodinger (Swift + Metal)

A real-time 2D GPU solver for the time-dependent Schrodinger equation using the
split-step Fourier method, with a live domain-coloring visualization. Written in
Swift and Metal for Apple Silicon.

## Physics

The solver integrates

    i * hbar * d(psi)/dt = [ -hbar^2 / (2m) * laplacian + V(x, y) ] * psi

with Strang splitting over one time step dt:

1. Half potential step in position space: psi *= exp(-i V dt / (2 hbar))
2. Forward FFT into momentum space
3. Full kinetic step: psi_hat *= exp(-i hbar k^2 dt / (2 m))
4. Inverse FFT back to position space
5. Half potential step in position space

This is second-order accurate in dt and unconditionally stable for the free
part, which is why the method stays clean where naive finite differences blow up.

Natural units are used (hbar = 1, m = 1) by default.

## GPU FFT

The FFT is implemented from scratch as a Metal compute shader rather than pulled
from a library. Each row is transformed by one threadgroup using a shared-memory
radix-2 Cooley-Tukey (decimation in time): N/2 threads, one butterfly per thread
per stage, bit-reversed load, log2(N) stages with barriers. The 2D transform is
row FFT, transpose, row FFT, transpose back, so the caller always sees row-major
data. The inverse transform carries the 1/N factor per pass (1/N^2 total).

Grid size N must be a power of two. With one row per threadgroup the shared
memory is N complex numbers (N * 8 bytes), so N up to 1024 fits comfortably.

## Visualization

The render kernel writes straight into the drawable. Phase arg(psi) maps to hue,
and tone-mapped density |psi|^2 maps to brightness, giving the classic complex
domain-coloring look.

## Potentials

Free space, harmonic well, a hard barrier, and a double slit. The double slit
plus a rightward-moving packet reproduces the interference fringes.

## Absorbing boundaries

A sponge layer along the domain borders (8 percent of the grid on each side)
smoothly damps the wavefunction with a quadratic ramp, so outgoing waves are
absorbed instead of wrapping around the periodic FFT domain. The damping rate
is the `absorberStrength` parameter; set it to 0 to restore fully periodic
behavior. Note that with the absorber enabled, total probability is no longer
conserved by design: whatever reaches the border leaves the simulation.

## Recording

The Record button captures the live view into an H.264 .mov on the Desktop at
60 fps and reveals the file in Finder when stopped. Frames are read back
through three rotating buffers so recording does not stall the render loop; if
the writer briefly falls behind, single frames are dropped rather than
blocking. The resulting .mov can be posted directly or converted to mp4/GIF
with ffmpeg.

## Build and run

Requires macOS 14+ and the Xcode Command Line Tools. The UI is plain AppKit
(no SwiftUI), specifically so the project compiles with the bare Command Line
Tools: recent SDKs implement SwiftUI property wrappers as macros whose plugins
ship only with the full Xcode.

    swift run

Or use the helper script: `./build.sh run`. Opening the package in full Xcode
also works.

## Controls

- Potential selector (resets the state)
- Momentum kx slider (initial mean wavenumber of the packet, resets)
- Brightness slider (render exposure only)
- Pause / Play, Reset

Tuning knobs live in `ContentView.makeParams()`: grid size `n`, `dx`, `dt`,
packet width `sigma`, start position, and `potentialStrength`. If a run looks
unstable, reduce `dt` or `substepsPerFrame` (in `Renderer`).

## Status

First pass, not yet run on hardware. The highest-risk areas to verify first:

- FFT butterfly indexing and the bit-reversed load in `fft_rows`
- The `Params` memory layout matching between Swift and Metal
- Inverse FFT normalization (a wrong factor shows up as the field fading or
  exploding over time; total probability should stay roughly constant)

A quick correctness check: set the potential to Free space with kx around 6. The
packet should drift right, spread, and wrap around the periodic boundary without
changing total brightness much.
