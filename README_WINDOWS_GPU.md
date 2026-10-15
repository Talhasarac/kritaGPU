# Windows GPU brushes

This fork adds a shared OpenGL 4.3 compute service to Krita on Windows. Pixel
Brushes can generate supported tips and composite ordered dabs on the GPU.
Color Smudge has a fused GPU mixing path. An optional native OpenGL canvas
can display eligible strokes before their pixels are downloaded to the CPU.
These features are experimental and retain Krita's CPU fallback.

## Run

Extract the portable package and start `bin/krita.exe`. GPU brushes are enabled
by default on Windows. The preference is under **Settings > Configure Krita >
Performance > Advanced > GPU accelerated Pixel Brushes (experimental)**.

Start `Krita GPU Native Canvas.cmd` to request desktop OpenGL for the direct
canvas path. The ordinary executable also supports compute painting when its
canvas uses ANGLE. The launcher changes the renderer for that process only.

Local limitation: context sharing initialized successfully on the RTX 3050 Ti,
but the desktop OpenGL window appeared blank in the computer capture, including
with canvas sharing disabled. The standard ANGLE window displayed correctly.
Use the ordinary executable on this host. Direct presentation and its complete
stroke/checkpoint lifecycle have not been confirmed through a visible live
stroke here; that path needs further validation before production use.

## Features and limits

| Work | GPU support | CPU fallback / remaining CPU work |
| --- | --- | --- |
| Pixel Brush compositing | RGBA U8, U16, F16, F32; native channel order and document profile | Other color models, wraparound, unsupported blend modes and small batches |
| Automatic tips | Default and Gaussian circular/elliptical generators, softness, edge antialiasing, scale, rotation, solid color | Other mask generators, spikes, randomness, density, tiny tips, gradient/lightness coloring |
| Image tips | Static alpha masks and image stamps; cached mip images, GPU affine sampling, mirroring | Animated/pipe tips, lightness maps and gradient maps |
| Texture | Multiply, Subtract and Darken alpha texture, soft/hard strength, canvas alignment | Other texture operations and sharpness postprocessing materialize the CPU tip |
| Smudge | Mask brush per-pixel smear/dulling, color-rate mixing, final mask/selection/compositing | Source snapshots and dulling color sampling remain CPU; stamp/lightness, mirror and two-painter overlay strategies fall back |
| Direct canvas | Shared native desktop OpenGL context, a single unmasked full-opacity Normal paint layer, direct Build-up painting, LOD 0, matching document/display color space and enabled channels | ANGLE, indirect/Wash targets, layer stacks/masks and color-converted projections use CPU mirroring per update |

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
6. In the direct canvas path, GPU cells stay pinned and are copied from a shared
   storage buffer/PBO into existing canvas textures. The canvas and compute
   contexts exchange fences. Updates have zero pixel readback. The stroke
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
spaced brushes. The native canvas can reduce transfers for eligible strokes.

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
`opengl32`, `gdi32` and `user32`; CUDA and Vulkan SDKs are not required.

The standalone backend can also be built from
`plugins/paintops/defaultpaintops/brush/gpu/tests`. Its existing check cases do
not cover every added format, mode, generated tip or direct canvas path.

## Diagnostics and RenderDoc

- `KRITA_GPU_BRUSH_FORCE=1`: bypass workload thresholds for supported operations.
- `KRITA_GPU_BRUSH_LOG=1`: log dispatches, transfers, resident hits and checkpoints.
- `KRITA_GPU_BRUSH_CPU_TIPS=1`: keep original CPU tip generation.
- `KRITA_GPU_BRUSH_NO_RESIDENCY=1`: disable atlas reuse and direct presentation.
- `KRITA_GPU_NATIVE_CANVAS=1`: request desktop OpenGL at process startup.
- `KRITA_GPU_BRUSH_NO_CANVAS_SHARE=1`: isolate native rendering from sharing.

The program is labeled **Krita Pixel Brush compute**; dispatch debug groups are
**Krita GPU Pixel Brush batch**. Storage buffers are labeled by purpose.
`GPU direct brush batch` lines report zero readback bytes, followed by a
`GPU stroke checkpoint` at synchronization. The initialization log identifies
the actual adapter. The launchers export NVIDIA's Optimus preference hint;
Windows graphics preferences may override it.
