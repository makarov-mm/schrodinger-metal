#include <metal_stdlib>
using namespace metal;

// Parameter block shared with the Swift side.
// All fields are 4-byte scalars in the same order on both sides so the
// memory layout matches without padding surprises. Do not add simd vectors here.
struct Params {
    uint  N;                 // grid size (power of two), field is N x N
    float dx;                // spatial step, physical size L = N * dx
    float dt;                // time step
    float hbar;              // reduced Planck constant (natural units: 1)
    float mass;              // particle mass (natural units: 1)
    float sigma;             // initial Gaussian packet width
    float x0;                // initial packet center x (physical)
    float y0;                // initial packet center y (physical)
    float kx0;               // initial mean wavenumber x
    float ky0;               // initial mean wavenumber y
    float potentialStrength; // scales the selected potential
    float brightness;        // render exposure for |psi|^2
    float absorberStrength;  // sponge-layer damping rate at the borders, 0 disables
    float time;              // accumulated simulation time (unused by kernels, kept for parity)
    uint  potentialType;     // 0 free, 1 harmonic, 2 barrier, 3 double slit
};

// ---------------------------------------------------------------------------
// Complex helpers
// ---------------------------------------------------------------------------

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 cexp(float phase) {
    return float2(cos(phase), sin(phase));
}

inline uint bitReverse(uint x, uint bits) {
    uint r = 0;
    for (uint i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    return r;
}

// ---------------------------------------------------------------------------
// 1D FFT over each row, one threadgroup per row.
// Shared-memory radix-2 Cooley-Tukey (decimation in time). N/2 threads per
// group, each thread performs exactly one butterfly per stage.
// The transpose kernel is used between passes to turn "columns" into "rows".
// ---------------------------------------------------------------------------
kernel void fft_rows(device float2*        data       [[buffer(0)]],
                     constant Params&       p          [[buffer(1)]],
                     constant uint&         logN       [[buffer(2)]],
                     constant int&          inverse    [[buffer(3)]],
                     threadgroup float2*    s          [[threadgroup(0)]],
                     uint                   row        [[threadgroup_position_in_grid]],
                     uint                   tid        [[thread_position_in_threadgroup]]) {
    const uint N = p.N;
    device float2* rowData = data + row * N;

    // Load two elements per thread into bit-reversed slots.
    uint i0 = tid;
    uint i1 = tid + N / 2;
    s[bitReverse(i0, logN)] = rowData[i0];
    s[bitReverse(i1, logN)] = rowData[i1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward transform uses -2*pi, inverse uses +2*pi.
    float sign = (inverse != 0) ? 1.0f : -1.0f;

    for (uint len = 2; len <= N; len <<= 1) {
        uint halfLen = len >> 1;
        uint group   = tid / halfLen;
        uint j       = tid % halfLen;
        uint base    = group * len;
        uint a       = base + j;
        uint b       = a + halfLen;

        float angle = sign * 2.0f * M_PI_F * (float)j / (float)len;
        float2 w = cexp(angle);
        float2 u = s[a];
        float2 v = cmul(s[b], w);
        s[a] = u + v;
        s[b] = u - v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Inverse transform carries a 1/N factor; two passes give the full 1/N^2.
    float scale = (inverse != 0) ? (1.0f / (float)N) : 1.0f;
    rowData[i0] = s[i0] * scale;
    rowData[i1] = s[i1] * scale;
}

// Straight transpose, src -> dst. Correctness first; a tiled version can come later.
kernel void transpose2d(device const float2* src [[buffer(0)]],
                      device float2*        dst [[buffer(1)]],
                      constant Params&      p   [[buffer(2)]],
                      uint2                 gid [[thread_position_in_grid]]) {
    const uint N = p.N;
    if (gid.x >= N || gid.y >= N) return;
    dst[gid.x * N + gid.y] = src[gid.y * N + gid.x];
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------

// Half potential step in position space: psi *= exp(-i V dt / (2 hbar)).
kernel void potential_step(device float2*       psi [[buffer(0)]],
                           device const float*   V   [[buffer(1)]],
                           constant Params&      p   [[buffer(2)]],
                           uint2                 gid [[thread_position_in_grid]]) {
    const uint N = p.N;
    if (gid.x >= N || gid.y >= N) return;
    uint idx = gid.y * N + gid.x;
    float phase = -V[idx] * p.dt / (2.0f * p.hbar);
    psi[idx] = cmul(psi[idx], cexp(phase));
}

// Full kinetic step in momentum space: psi_hat *= exp(-i hbar k^2 dt / (2 m)).
// After the forward FFT the data is in standard order (DC at index 0), so the
// fftfreq mapping applies directly.
kernel void kinetic_step(device float2*  psi [[buffer(0)]],
                         constant Params& p   [[buffer(1)]],
                         uint2            gid [[thread_position_in_grid]]) {
    const uint N = p.N;
    if (gid.x >= N || gid.y >= N) return;
    uint idx = gid.y * N + gid.x;

    float L = (float)N * p.dx;
    float twoPiOverL = 2.0f * M_PI_F / L;

    int ix = (gid.x < N / 2) ? (int)gid.x : (int)gid.x - (int)N;
    int iy = (gid.y < N / 2) ? (int)gid.y : (int)gid.y - (int)N;
    float kx = (float)ix * twoPiOverL;
    float ky = (float)iy * twoPiOverL;
    float k2 = kx * kx + ky * ky;

    float phase = -p.hbar * k2 * p.dt / (2.0f * p.mass);
    psi[idx] = cmul(psi[idx], cexp(phase));
}

// Sponge layer: smoothly damps |psi| in a thin frame along the borders so the
// outgoing waves are absorbed instead of wrapping around the periodic domain.
// The quadratic ramp keeps the inner edge of the absorber gentle, which
// minimizes artificial reflections back into the domain.
kernel void absorb_step(device float2*  psi [[buffer(0)]],
                        constant Params& p   [[buffer(1)]],
                        uint2            gid [[thread_position_in_grid]]) {
    const uint N = p.N;
    if (gid.x >= N || gid.y >= N) return;
    if (p.absorberStrength <= 0.0f) return;

    float border = 0.08f * (float)N; // absorber width in cells
    float distX = min((float)gid.x, (float)(N - 1 - gid.x));
    float distY = min((float)gid.y, (float)(N - 1 - gid.y));
    float dist  = min(distX, distY);
    if (dist >= border) return;

    float s = 1.0f - dist / border;  // 0 at the inner edge, 1 at the boundary
    float factor = exp(-p.absorberStrength * s * s * p.dt);
    uint idx = gid.y * N + gid.x;
    psi[idx] *= factor;
}

// ---------------------------------------------------------------------------
// Initial state and potential
// ---------------------------------------------------------------------------

kernel void init_state(device float2*  psi [[buffer(0)]],
                       device float*    V   [[buffer(1)]],
                       constant Params&  p   [[buffer(2)]],
                       uint2             gid [[thread_position_in_grid]]) {
    const uint N = p.N;
    if (gid.x >= N || gid.y >= N) return;
    uint idx = gid.y * N + gid.x;

    // Centered physical coordinates.
    float x = ((float)gid.x - (float)N * 0.5f) * p.dx;
    float y = ((float)gid.y - (float)N * 0.5f) * p.dx;

    // Gaussian wave packet with mean momentum (kx0, ky0). Not globally
    // normalized; the renderer applies its own exposure.
    float r2 = (x - p.x0) * (x - p.x0) + (y - p.y0) * (y - p.y0);
    float envelope = exp(-r2 / (2.0f * p.sigma * p.sigma));
    float phase = p.kx0 * x + p.ky0 * y;
    psi[idx] = envelope * cexp(phase);

    // Potential.
    float value = 0.0f;
    float L = (float)N * p.dx;
    switch (p.potentialType) {
        case 1: { // harmonic well centered at origin
            value = 0.5f * p.potentialStrength * (x * x + y * y);
            break;
        }
        case 2: { // vertical barrier near x = 0
            if (fabs(x) < 0.02f * L) value = p.potentialStrength;
            break;
        }
        case 3: { // double slit: wall near x = 0 with two gaps
            float wallHalf = 0.015f * L;
            float slitHalf = 0.03f * L;   // half-height of each opening
            float slitGap  = 0.10f * L;   // distance of each slit from center
            if (fabs(x) < wallHalf) {
                bool inLower = fabs(y + slitGap) < slitHalf;
                bool inUpper = fabs(y - slitGap) < slitHalf;
                if (!inLower && !inUpper) value = p.potentialStrength;
            }
            break;
        }
        default: // 0 free space
            value = 0.0f;
            break;
    }
    V[idx] = value;
}

// ---------------------------------------------------------------------------
// Domain-coloring render straight into the drawable texture.
// Hue encodes the phase arg(psi), value encodes |psi|^2.
// ---------------------------------------------------------------------------

inline float3 hsv2rgb(float h, float s, float v) {
    float3 k = float3(1.0f, 2.0f / 3.0f, 1.0f / 3.0f);
    float3 p = abs(fract(float3(h) + k) * 6.0f - 3.0f);
    return v * mix(float3(1.0f), clamp(p - 1.0f, 0.0f, 1.0f), s);
}

kernel void render(texture2d<float, access::write> out [[texture(0)]],
                   device const float2*             psi [[buffer(0)]],
                   constant Params&                 p   [[buffer(1)]],
                   uint2                            gid [[thread_position_in_grid]]) {
    uint w = out.get_width();
    uint h = out.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Map drawable pixel to field cell (nearest), resolution independent.
    uint fx = (uint)((float)gid.x / (float)w * (float)p.N);
    uint fy = (uint)((float)gid.y / (float)h * (float)p.N);
    fx = min(fx, p.N - 1);
    fy = min(fy, p.N - 1);
    uint idx = fy * p.N + fx;

    float2 c = psi[idx];
    float mag2 = c.x * c.x + c.y * c.y;
    float phase = atan2(c.y, c.x);               // -pi .. pi
    float hue = (phase + M_PI_F) / (2.0f * M_PI_F);

    // Tone map density so a wide dynamic range stays visible.
    float value = 1.0f - exp(-p.brightness * mag2);

    float3 rgb = hsv2rgb(hue, 1.0f, value);
    out.write(float4(rgb, 1.0f), gid);
}
