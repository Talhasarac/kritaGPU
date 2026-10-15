# Windows GPU brushes

This fork adds a shared OpenGL 4.3 compute service to Krita on Windows. Pixel
Brushes can generate supported tips and composite ordered dabs on the GPU.
Color Smudge has a fused GPU mixing path. The Windows direct canvas uses a
shared D3D11 texture to display eligible RGBA8 strokes while their pixels
remain on the GPU.
These features are experimental and retain Krita's CPU fallback.

## Run

Extract the portable package and start `bin/krita.exe`. GPU brushes are enabled
by default on Windows. The preference is under **Settings > Configure Krita >
Performance > Advanced > GPU accelerated Pixel Brushes (experimental)**.

Start `Krita GPU Direct Canvas.cmd` to select ANGLE/D3D11 presentation for
this process. The older `Krita GPU Native Canvas.cmd` uses the same route.
Direct canvas sharing is also attempted automatically with the ordinary
executable when GPU brushes and a compatible ANGLE canvas are enabled.

The brush compositor still runs native OpenGL compute. It shares a 16 MiB
D3D11 texture through `WGL_NV_DX_interop2` and EGL; canvas updates do not
read brush pixels back to the CPU. A second 16 MiB GPU texture handles the
document's canvas background when needed. Drivers without the required
interop capability keep CPU mirroring. Local hardware: RTX 3050 Ti.

For an eligible stroke, select **F5 > Painting Mode > Build up**, use an RGBA8
image with one unmasked full-opacity Normal paint layer, and keep the image
and display color spaces identical. Wash mode, layers/masks, color-converted
projections, and other pixel formats retain CPU mirroring on ANGLE.

Pure desktop WGL remains available with `KRITA_GPU_NATIVE_CANVAS=wgl`.
Its frame buffer contained correct pixels here, but the native window capture
was white. The D3D11 bridge avoids that presentation/capture problem.

## Features and limits

| Work | GPU support | CPU fallback / remaining CPU work |
| --- | --- | --- |
| Pixel Brush compositing | RGBA U8, U16, F16, F32; native channel order and document profile | Other color models, wraparound, unsupported blend modes and small batches |
| Automatic tips | Default and Gaussian circular/elliptical generators, softness, edge antialiasing, scale, rotation, solid color | Other mask generators, spikes, randomness, density, tiny tips, gradient/lightness coloring |
| Image tips | Static alpha masks and image stamps; cached mip images, GPU affine sampling, mirroring | Animated/pipe tips, lightness maps and gradient maps |
| Texture | Multiply, Subtract and Darken alpha texture, soft/hard strength, canvas alignment | Other texture operations and sharpness postprocessing materialize the CPU tip |
| Smudge | Mask brush per-pixel smear/dulling, color-rate mixing, final mask/selection/compositing | Source snapshots and dulling color sampling remain CPU; stamp/lightness, mirror and two-painter overlay strategies fall back |
| Direct canvas | ANGLE/D3D11 RGBA8 bridge, one unmasked full-opacity Normal paint layer, Build-up painting, LOD 0, matching document/display color space and enabled channels | Other ANGLE pixel formats, indirect/Wash targets, layer stacks/masks and color-converted projections use CPU mirroring per update |

GPU blend operations: Normal, creamy/hard Alpha Darken, Erase, Copy, Multiply,
Screen, Overlay, Darken, Lighten, Add, Subtract, Difference, Exclusion, Color
Dodge, Color Burn, Hard Light, SVG Soft Light and Linear Burn. Channel locks are
supported except for Alpha Darken and Erase, which retain the CPU path when
locks are set. Float RGB values can exceed 1.0; alpha remains normalized.

GPU floating point arithmetic and image sampling can differ slightly from
Krita's CPU implementations. This is not a claim of byte-for-byte parity for
all presets or of a fully GPU-native document model.

## Pipeline

1. Krita still computes input dynamics, spacing, dab positions and scheduler
   jobs on the CPU. Supported tips become small descriptors. Image mip levels
   and texture masks are cached assets rather than regenerated dab images.
   Gaussian tips cache a 4,097-sample CPU radial profile; GPU invocations expand
   that profile into each rotated/scaled dab rather than uploading its pixels.
   Unsupported CPU consumers can materialize an original Krita tip lazily.
2. A sequential painting job builds an ordered dab list and the exact union of
   dirty regions, split at world-aligned 256 x 256 tile boundaries. An update
   fetches at most 256 dabs and targets eight million source words.
3. A dedicated WGL worker owns compute programs, storage buffers and a bounded
   64 MiB resident atlas. Complete cells are initialized from the CPU once.
   Stroke identity, device mutation sequence, animation time and LOD control
   reuse. Unpinned old cells can be evicted; inputs have bounded host caches.
4. A 16 x 16 workgroup assigns a pixel to each invocation. Contributing dabs are
   blended in order, quantized after each dab to the device's storage depth.
   Lists are split into dispatches of at most 256 dabs, with memory barriers.
5. In the normal path, completed dirty output is downloaded and written through
   `KisPaintDevice::writeBytes` inside the existing undo transaction.
6. In the direct canvas path, GPU cells stay pinned. The native worker copies
   changed cells from its storage buffer/PBO into a D3D11 texture on the same
   adapter. ANGLE imports the shared handle with a matching EGL configuration.
   GPU framebuffer blits copy into Krita canvas tiles, including their edge
   padding. A shader composites the configured canvas background when needed.
   WGL interop locking and a completed ANGLE GPU copy protect ownership.
   Updates have zero pixel readback. The stroke
   checkpoints to CPU storage before its transaction closes, when the cache
   fills, or before CPU fallback. Cancellation drops pending GPU work. A retained
   dab journal can replay pending updates if a GPU checkpoint fails.

The CPU remains authoritative for committed documents, filters, undo and saving.
The direct canvas eligibility checks preserve the existing color management
and layer projection behavior.

Small Pixel Brush updates normally need at least four dabs and four million
intersecting dab pixels to use compute. Smudge needs at least 65,536 pixels.
Tip deferral also normally starts at 65,536 pixels to avoid adding profile
setup overhead to small CPU brush updates.
Transfers can outweigh compute savings, particularly on small or lightly
spaced brushes. The direct canvas reduces transfers for eligible strokes.

## Measurements

The earlier resident-compositor build, on an RTX 3050 Ti, reduced median time
for 16/32/128 overlapping 512px dabs from 2.21/2.29/3.58 ms to 1.81/1.83/3.09 ms
per update. Those are prior-build compositor measurements, not a latency claim
for this expanded implementation.

`krita_gpu_brush_compute_test --benchmark-only` runs performance workloads
without executing the unit check suite. It exercises U8, U16, F16, F32 and
GPU-generated soft round tips. The scalar reference is not Krita's optimized
SIMD compositor. See the supplied measurement log for actual local results.
A complete correctness suite has not been run for the new paths.

A live 512px Gaussian dab used a 16,388-byte radial profile in place of a
1,052,676-byte bitmap (about 98% less tip data). With the same nine destination
cells, logged total upload fell from 3,412,140 to 2,375,852 bytes. The generated
dab displayed in the packaged ANGLE build. These are transfer measurements,
not an end-to-end speed or input-latency comparison.

The direct ANGLE canvas was exercised live with a 1000px Build-up Pixel Brush
on a 2048 x 2048 RGBA8 image, without forcing the workload thresholds. Two
eight-dab GPU batches presented 45 canvas cells with zero pixel readback per
update. The stroke-end checkpoint downloaded 16,777,216 bytes for the two
updates. The configured white background displayed correctly, and the result
was saved to `krita-gpu-direct-canvas.kra`. This confirms the presentation
route on the local NVIDIA adapter; it is not an end-to-end speed measurement.

## Build

Shallow clone:

```powershell
git clone --depth 1 --single-branch https://github.com/Talhasarac/kritaGPU.git
```

Use Krita's Windows LLVM MinGW toolchain and matching dependencies:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=path/to/install
cmake --build build --parallel 6
cmake --install build
```

Rebuild all plugins when updating from the earlier GPU version: the lazy dab
metadata changes a shared core class. The new `kritagpubrush` library is shared
by the image library, paintops and native canvas. The backend uses Windows
`opengl32`, `gdi32`, `user32`, `d3d11` and `dxgi`; CUDA and Vulkan SDKs are not required.

The standalone backend can also be built from
`plugins/paintops/defaultpaintops/brush/gpu/tests`. Its existing check cases do
not cover every added format, mode, generated tip or direct canvas path.

## Diagnostics and RenderDoc

- `KRITA_GPU_BRUSH_FORCE=1`: bypass workload thresholds for supported operations.
- `KRITA_GPU_BRUSH_LOG=1`: log dispatches, transfers, resident hits and checkpoints.
- `KRITA_GPU_BRUSH_CPU_TIPS=1`: keep original CPU tip generation.
- `KRITA_GPU_BRUSH_NO_RESIDENCY=1`: disable atlas reuse and direct presentation.
- `KRITA_GPU_NATIVE_CANVAS=1`: select ANGLE/D3D11 for the direct canvas bridge.
- `KRITA_GPU_NATIVE_CANVAS=wgl`: select the legacy desktop OpenGL sharing path.
- `KRITA_GPU_BRUSH_NO_CANVAS_SHARE=1`: keep compute but disable direct presentation.

The program is labeled **Krita Pixel Brush compute**; dispatch debug groups are
**Krita GPU Pixel Brush batch**. Storage buffers are labeled by purpose.
`GPU direct brush batch` lines report zero readback bytes, followed by a
`GPU stroke checkpoint` at synchronization. The initialization log identifies
the actual adapter. The launchers export NVIDIA's Optimus preference hint;
Windows graphics preferences may override it.
