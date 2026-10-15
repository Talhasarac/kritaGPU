/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef KIS_GPU_BRUSH_COMPOSITOR_H
#define KIS_GPU_BRUSH_COMPOSITOR_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32) && !defined(KRITA_GPU_BRUSH_STANDALONE)
#ifdef KRITA_GPU_BRUSH_BUILD
#define KRITA_GPU_BRUSH_EXPORT __declspec(dllexport)
#else
#define KRITA_GPU_BRUSH_EXPORT __declspec(dllimport)
#endif
#else
#define KRITA_GPU_BRUSH_EXPORT
#endif

namespace KisGpuBrush
{
inline constexpr int residentTileSize = 256;
enum class PixelFormat : int { RGBA8, RGBA16, RGBA16F, RGBA32F };
inline unsigned wordsPerPixel(PixelFormat f) { return f == PixelFormat::RGBA8 ? 1 : f == PixelFormat::RGBA32F ? 4 : 2; }
enum class BlendMode : int {
    Over, AlphaDarkenCreamy, AlphaDarkenHard, Erase,
    Multiply, Screen, Overlay, Darken, Lighten, Add, Subtract, Copy,
    Difference, Exclusion, ColorDodge, ColorBurn, HardLight, SoftLight, LinearBurn
};

// This layout is also used by the shader's std430 buffer. Pixels stay in the
// paint device's native C1,C2,C3,A order; there is no profile conversion.
struct alignas(16) Dab
{
    std::int32_t bounds[4] = {}; // x, y, width, height
    std::uint32_t data[4] = {}; // source pixel offset, reserved
    float params[4] = {1.0f, 1.0f, 1.0f, 0.0f}; // opacity, flow, average
    float shape[4] = {};
    float transform[4] = {};
    float colour[4] = {};
    std::uint32_t texture[4] = {}; // pattern offset, width, height, operation
    float textureParams[4] = {}; // canvas offsets, strength, reserved
};
static_assert(sizeof(Dab) == 128, "The GPU dab layout must match std430");

struct Tile
{
    std::int32_t x = 0, y = 0, width = 0, height = 0;
    std::uint32_t pixelOffset = 0;
    std::uint32_t indexOffset = 0, dabCount = 0;
};

struct Batch
{
    // A nonzero paint-device identity enables bounded GPU tile residency.
    // Input pixels are still authoritative: changed CPU pixels invalidate the
    // corresponding cached region, including after undo or CPU fallback.
    std::uint64_t cacheId = 0;
    std::uint64_t cacheGeneration = 0;
    // Stroke sessions can defer CPU reads until a tile is actually missing.
    // A generation must change after any external CPU modification. The
    // callback is synchronous and must not change the paint device.
    std::function<void(const Tile &, std::uint32_t *)> loadPixels;
    PixelFormat format = PixelFormat::RGBA8;
    bool deferReadback = false;
    std::uint64_t presentationKey = 0;
    BlendMode mode = BlendMode::Over;
    std::uint32_t channelFlags = 15;
    std::vector<std::uint32_t> pixels;
    std::vector<std::uint32_t> sourcePixels;
    std::vector<std::uint32_t> selection; // empty = no selection mask (255)
    std::vector<Dab> dabs;
    std::vector<std::uint32_t> dabIndices;
    std::vector<Tile> tiles;
};

struct Result
{
    bool success = false;
    std::string message;
    std::string renderer;
    std::uint64_t dispatches = 0;
    std::uint64_t uploadedBytes = 0;
    std::uint64_t destinationUploadedBytes = 0;
    std::uint64_t readbackBytes = 0;
    std::uint64_t reusedTiles = 0;
    std::uint64_t residentTiles = 0;
};

struct PresentationTile {
    int x=0,y=0,width=0,height=0;
    std::uint32_t buffer=0;
    std::uint64_t byteOffset=0,version=0;
    PixelFormat format=PixelFormat::RGBA8;
};
struct Checkpoint {
    bool success=false;
    std::string message;
    PixelFormat format=PixelFormat::RGBA8;
    std::vector<Tile> tiles;
    std::vector<std::uint32_t> pixels;
};

/**
 * Windows OpenGL 4.3 compute service, independent of Qt's canvas backend.
 * A single worker owns its WGL context for its entire lifetime. Submission is
 * synchronous for the stroke scheduler, and safe from any non-GUI thread.
 * On failure Batch::pixels is unchanged, allowing the caller to use CPU blits.
 */
class KRITA_GPU_BRUSH_EXPORT Compositor
{
public:
    static Compositor &instance();
    ~Compositor();
    bool isAvailable();
    Result render(Batch &batch);
    std::string diagnostic();
    void shutdown();
    // Call with a current native desktop GL canvas. ANGLE keeps CPU mirroring.
    void configureNativeCanvas(void *context, void *dc);
    void registerCanvas(std::uint64_t key, std::uint64_t token, std::function<void()> repaint);
    void unregisterCanvas(std::uint64_t key, std::uint64_t token);
    bool canPresent(std::uint64_t key);
    bool hasPresentation(std::uint64_t key);
    // The callback runs in the canvas GL context. It returns a flushed GLsync
    // protecting its GPU reads; the worker waits before touching shared data.
    bool present(std::uint64_t key, const std::function<void *(const std::vector<PresentationTile> &)> &draw);
    Checkpoint checkpoint(std::uint64_t cacheId, std::uint64_t generation);
    void discard(std::uint64_t cacheId);

    Compositor(const Compositor &) = delete;
    Compositor &operator=(const Compositor &) = delete;

private:
    Compositor();
    struct Private;
    std::unique_ptr<Private> m_d;
};
}
#endif
