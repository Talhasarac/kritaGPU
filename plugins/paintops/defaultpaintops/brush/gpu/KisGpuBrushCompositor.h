/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef KIS_GPU_BRUSH_COMPOSITOR_H
#define KIS_GPU_BRUSH_COMPOSITOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace KisGpuBrush
{
enum class BlendMode : int { Over, AlphaDarkenCreamy, AlphaDarkenHard, Erase };

// This layout is also used by the shader's std430 buffer. Pixels stay in the
// paint device's native C1,C2,C3,A order; there is no profile conversion.
struct alignas(16) Dab
{
    std::int32_t bounds[4] = {}; // x, y, width, height
    std::uint32_t data[4] = {}; // source pixel offset, reserved
    float params[4] = {1.0f, 1.0f, 1.0f, 0.0f}; // opacity, flow, average
};
static_assert(sizeof(Dab) == 48, "The GPU dab layout must match std430");

struct Tile
{
    std::int32_t x = 0, y = 0, width = 0, height = 0;
    std::uint32_t pixelOffset = 0;
    std::uint32_t indexOffset = 0, dabCount = 0;
};

struct Batch
{
    BlendMode mode = BlendMode::Over;
    std::uint32_t channelFlags = 15;
    std::vector<std::uint32_t> pixels;
    std::vector<std::uint32_t> sourcePixels;
    std::vector<std::uint32_t> selection; // one byte-valued uint per pixel
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
};

/**
 * Windows OpenGL 4.3 compute service, independent of Qt's canvas backend.
 * A single worker owns its WGL context for its entire lifetime. Submission is
 * synchronous for the stroke scheduler, and safe from any non-GUI thread.
 * On failure Batch::pixels is unchanged, allowing the caller to use CPU blits.
 */
class Compositor
{
public:
    static Compositor &instance();
    ~Compositor();
    bool isAvailable();
    Result render(Batch &batch);
    std::string diagnostic();
    void shutdown();

    Compositor(const Compositor &) = delete;
    Compositor &operator=(const Compositor &) = delete;

private:
    Compositor();
    struct Private;
    std::unique_ptr<Private> m_d;
};
}
#endif
