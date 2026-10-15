# Windows GPU Pixel Brushes

This fork adds OpenGL compute compositing to Krita's existing Pixel Brush
engine. The normal brush presets continue to generate their tips, dynamics,
textures, and color on the CPU. Their prepared dabs are then blended on the
GPU in ordered batches. This is an initial GPU painting implementation, not a
port of every Krita brush engine.

## Supported painting

- Windows with a native OpenGL 4.3 or newer driver.
- RGB/Alpha, 8-bit integer paint devices, using the device's existing profile.
- Normal (including alpha and color channel locks), Wash/Alpha Darken
  (creamy and legacy hard flow), and Eraser.
- Selection masks, pressure-dependent opacity/flow, existing brush tip
  textures, and the normal undo/redo and document saving paths.

Unsupported color depths, blend operations, and wraparound painting use the
existing CPU compositor. Small batches also use the CPU because transferring
their pixels to the GPU can cost more than blending them. The initial threshold
is four dabs and four million intersecting dab pixels per update.

The option is enabled by default on Windows. It is exposed in **Settings >
Configure Krita > Performance > Advanced > GPU accelerated Pixel Brushes
(experimental)**. Turn it off to compare with the original compositor. The
native compute context operates separately from Qt's canvas context, including
when the canvas uses ANGLE.

## Pipeline

1. Krita's existing asynchronous dab executor prepares the brush tips. GPU
   updates fetch at most 256 dabs and target eight million source pixels per
   update, including the initial fetch, to keep long strokes inside the buffer
   budget. The existing adaptive update limit can reduce this further.
2. A sequential stroke job merges adjacent CPU scheduler patches, splits their
   exact union at world-aligned 256 by 256 cell boundaries, and creates ordered
   dab index lists. Only dirty intersections are dispatched and read back.
3. A dedicated worker owns a native Windows OpenGL context and a 64 MiB paint
   tile atlas. Each stroke has its own cache identity. Cells are initialized
   from the CPU once, including their clean margins, then retain painted pixels
   on the GPU across updates. Cache hits skip both CPU destination reads and
   destination uploads. A bounded LRU evicts old cells. Device mutation counters,
   animation time and level of detail invalidate stale stroke generations.
   Other callers validate cached regions against their CPU input bytes.
4. Unchanged source pixels, dab parameters, index lists and selections reuse
   their GPU buffers, with a 16 MiB host comparison-cache limit per input buffer.
   Byte comparison detects tip edits and mirroring. No mask
   allocation or bulk upload is needed when there is no selection.
5. A 16 by 16 compute workgroup assigns one invocation to each pixel. It blends
   the contributing dabs in order and rounds to 8-bit storage after each dab.
   Dense lists are split into ordered dispatches of at most 256 dabs.
6. A storage/buffer memory barrier and fence precede dirty-pixel readback. Only a completed
   batch is written through `KisPaintDevice::writeBytes`, inside the stroke's
   existing undo transaction. Failed dispatches leave the CPU pixels untouched
   and fall back to CPU painting.

Dirty output is still synchronized to Krita's CPU storage after each update for
canvas projection, filters, undo and saving. This is GPU-resident stroke
compositing with a CPU mirror; the canvas and document model are not entirely
GPU-native. New cells, changing tips and cache evictions still require uploads.
Performance depends on the GPU, transfer cost, brush size and dab overlap.

Set `KRITA_GPU_BRUSH_NO_RESIDENCY=1` to disable cache reuse for comparison.
The `testLargeBatchTiming` benchmark alternates resident and uncached GPU paths
over five rounds of twelve updates for 16, 32 and 128 overlapping 512px dabs.
It includes packing, synchronization, transfers and CPU writes, and requires
both GPU paths to produce identical output before reporting their timings.
Its CPU figure is a single `KisPainter` call; ordinary strokes can distribute
CPU painting across multiple cores. These are compositing measurements rather
than an end-to-end tablet input and canvas latency benchmark.

The final local run on an RTX 3050 Ti measured the following medians, in
milliseconds per update:

| 512px dabs per update | Upload every update | Resident stroke |
| --- | ---: | ---: |
| 16 | 2.21 | 1.81 |
| 32 | 2.29 | 1.83 |
| 128 | 3.58 | 3.09 |

That run used 14–20% less time with residency. An earlier run showed 10–14%;
GPU clocks and system load affect these timings. Warm repeated updates with
identical tips logged zero destination and total uploaded bytes, while still
reading back 1 MiB of dirty output per update. A live 512px stroke in a 4096px
document used six compute batches and reused resident tiles between them.

## Clone with minimal history

```powershell
git clone --depth 1 --single-branch https://github.com/Talhasarac/kritaGPU.git
```

## Build the standalone GPU check

The compute backend can be compiled and checked without Qt or KDE dependencies:

```powershell
cmake -S plugins/paintops/defaultpaintops/brush/gpu/tests -B build-gpu -G "Visual Studio 17 2022" -A x64
cmake --build build-gpu --config Release
ctest --test-dir build-gpu -C Release --output-on-failure
```

It executes real compute dispatches and checks Normal, Wash, Eraser, selection,
channel locks, clipped tiles, dense batches, concurrent submissions, and
invalid-batch handling against a scalar reference. It requires a working GPU;
it does not substitute CPU rendering when a driver is unavailable.

The optional `krita_gpu_brush_compute_test --benchmark` compares a dense batch
with that scalar reference. It is **not** a comparison with Krita's optimized
SIMD compositor.

## Build Krita

Use Krita's [Windows build instructions](https://docs.krita.org/en/untranslatable_pages/building_krita.html)
and a matching LLVM MinGW dependency/toolchain environment. This backend needs
only the Windows system libraries `opengl32`, `gdi32`, and `user32`; no CUDA SDK,
Vulkan SDK, or proprietary shader is required.

With the dependency environment activated:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=path/to/krita-install
cmake --build build --target install --parallel 6
```

To run the focused integration checks after that build:

```powershell
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target KisGpuBrushPaintTest --parallel 6
ctest --test-dir build -R KisGpuBrushPaintTest --output-on-failure
```

The integration test compares displayed premultiplied color and alpha with
`KisPainter` (within two 8-bit levels), checks
selection and alpha lock, verifies undo/redo, and checks unsupported-mode
fallback. Like other Krita tests it needs Krita's color-space plugins available
in the configured test/runtime environment.

GPU floating point rounding can differ slightly from Krita's CPU SIMD/FMA
rounding. RGB bytes under fully transparent pixels can also differ, including
where the CPU's selection code changes invisible RGB. The integration check
therefore compares visible color and alpha rather than requiring identical
unassociated RGB bytes at zero alpha.

## Inspect dispatches

Set `KRITA_GPU_BRUSH_FORCE=1` before starting the application to bypass the
workload threshold for supported brushes. Set `KRITA_GPU_BRUSH_LOG=1` to log
the dab and dispatch counts, resident tile hits, uploaded bytes and readback
bytes. Remove these variables for normal use.

RenderDoc debug groups are labeled **Krita GPU Pixel Brush batch**, the program
is labeled **Krita Pixel Brush compute**, and the storage buffers have
names describing their contents. A successful initialization log identifies
the GPU selected by Windows for the native OpenGL context. Its adapter can
differ from the canvas adapter on laptops with multiple GPUs.

The Windows executables export NVIDIA's `NvOptimusEnablement` hint to request
the discrete adapter on Optimus laptops. Windows or user graphics preferences
can override this request; the initialization log reports the actual adapter.
