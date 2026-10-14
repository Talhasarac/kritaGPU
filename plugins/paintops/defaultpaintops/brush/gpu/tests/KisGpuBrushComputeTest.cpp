/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "../KisGpuBrushCompositor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace KisGpuBrush;
namespace
{
void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}
std::uint32_t pack(const float *c)
{
    std::uint32_t p = 0;
    for (int i = 0; i < 4; ++i) p |= std::uint32_t(std::nearbyint(std::clamp(c[i], 0.0f, 255.0f))) << (8 * i);
    return p;
}
void unpack(std::uint32_t p, float *c)
{
    for (int i = 0; i < 4; ++i) c[i] = float((p >> (8 * i)) & 255);
}
unsigned mul255(unsigned a, unsigned b)
{
    return ((a * b + 128) * 257) >> 16;
}
// Independent scalar reference for the supported Krita compositing equations.
// Walk dabs first (the existing painter's order), rather than mirroring the
// GPU's per-pixel traversal, to also verify ordering and tile index lists.
void reference(Batch &b)
{
    for (const auto &t : b.tiles) {
        for (unsigned i = 0; i < t.dabCount; ++i) {
            const auto &d = b.dabs[b.dabIndices[t.indexOffset + i]];
            const int left = std::max(t.x, d.bounds[0]);
            const int top = std::max(t.y, d.bounds[1]);
            const int right = std::min(t.x + t.width, d.bounds[0] + d.bounds[2]);
            const int bottom = std::min(t.y + t.height, d.bounds[1] + d.bounds[3]);
            for (int y = top; y < bottom; ++y) {
                for (int x = left; x < right; ++x) {
                    const auto offset = t.pixelOffset + (y - t.y) * t.width + x - t.x;
                    const auto srcOffset = d.data[0] + (y - d.bounds[1]) * d.bounds[2] + x - d.bounds[0];
                    if (!b.selection[offset] || !(b.sourcePixels[srcOffset] >> 24)) continue;
                    float src[4], dst[4];
                    unpack(b.sourcePixels[srcOffset], src);
                    unpack(b.pixels[offset], dst);
                    const float mask = b.selection[offset] / 255.0f;
                    if (b.mode == BlendMode::Erase) {
                        const auto a = mul255(mul255(unsigned(src[3]), b.selection[offset]),
                                              unsigned(d.params[0] * 255.0f + 0.5f));
                        dst[3] = float(mul255(255 - a, unsigned(dst[3])));
                    } else if (b.mode == BlendMode::Over) {
                        const float alpha = src[3] * d.params[0] * mask;
                        if (!alpha) continue;
                        const bool locked = !(b.channelFlags & 8);
                        float outputAlpha = dst[3];
                        float weight;
                        if (locked || dst[3] == 255) weight = alpha / 255;
                        else if (dst[3] == 0) {
                            outputAlpha = alpha;
                            weight = 1;
                            if ((b.channelFlags & 7) != 7) dst[0] = dst[1] = dst[2] = 0;
                        } else {
                            outputAlpha = dst[3] + (255 - dst[3]) * alpha / 255;
                            weight = alpha / outputAlpha;
                        }
                        for (int c = 0; c < 3; ++c) {
                            if (b.channelFlags & (1u << c)) dst[c] += (src[c] - dst[c]) * weight;
                        }
                        if (!locked) dst[3] = outputAlpha;
                    } else {
                        const bool hard = b.mode == BlendMode::AlphaDarkenHard;
                        const float flow = d.params[1];
                        const float opacity = d.params[0] * (hard ? flow : 1);
                        const float average = d.params[2] * (hard ? flow : 1);
                        const float coverage = src[3] / 255 * mask;
                        const float alpha = coverage * opacity;
                        if (!alpha) continue;
                        for (int c = 0; c < 3; ++c) {
                            dst[c] = dst[3] ? dst[c] + (src[c] - dst[c]) * alpha : src[c];
                        }
                        const float old = dst[3] / 255;
                        float full;
                        if (average > opacity) {
                            full = average > old ? alpha + (average - alpha) * (old / average) : old;
                        } else full = opacity > old ? old + (opacity - old) * coverage : old;
                        const float zero = hard ? alpha + old - alpha * old : old;
                        dst[3] = (flow == 1 ? full : zero + (full - zero) * flow) * 255;
                    }
                    b.pixels[offset] = pack(dst);
                }
            }
        }
    }
}

Batch makeBatch(BlendMode mode, std::uint32_t flags = 15)
{
    std::mt19937 random(87123);
    Batch b;
    b.mode = mode;
    b.channelFlags = flags;
    b.pixels.resize(17 * 19 + 31 * 21);
    b.selection.resize(b.pixels.size());
    for (std::size_t i = 0; i < b.pixels.size(); ++i) {
        b.pixels[i] = random();
        if (i % 5 == 0) b.pixels[i] &= 0xffffff;
        if (i % 7 == 0) b.pixels[i] |= 0xff000000;
        b.selection[i] = i % 3 == 0 ? 0 : i % 3 == 1 ? 255 : random() % 256;
    }
    b.sourcePixels.resize(35 * 37 * 4);
    for (auto &p : b.sourcePixels) p = random();
    for (int i = 0; i < 40; ++i) {
        Dab d;
        d.bounds[0] = -20 + i % 9;
        d.bounds[1] = -14 + i % 7;
        d.bounds[2] = 35;
        d.bounds[3] = 37;
        d.data[0] = (i % 4) * 35 * 37;
        d.params[0] = i % 11 == 0 ? 0 : float(10 + random() % 90) / 100;
        d.params[1] = i % 4 == 0 ? 1 : float(random() % 100) / 100;
        d.params[2] = float(random() % 101) / 100;
        b.dabs.push_back(d);
    }
    b.tiles.push_back({-13, -7, 17, 19, 0, 0, 40});
    b.tiles.push_back({4, -7, 31, 21, 17 * 19, 40, 40});
    for (int t = 0; t < 2; ++t) for (unsigned i = 0; i < 40; ++i) b.dabIndices.push_back(i);
    return b;
}
void compare(Batch batch, const std::string &name)
{
    auto expected = batch;
    reference(expected);
    const auto result = Compositor::instance().render(batch);
    require(result.success, name + ": " + result.message);
    std::uint64_t dispatches = 0;
    for (const auto &tile : batch.tiles) dispatches += (tile.dabCount + 255) / 256;
    require(result.dispatches == dispatches, "Unexpected GPU dispatch count");
    unsigned maxError = 0;
    for (std::size_t i = 0; i < batch.pixels.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            const int actual = (batch.pixels[i] >> (c * 8)) & 255;
            const int wanted = (expected.pixels[i] >> (c * 8)) & 255;
            maxError = std::max(maxError, unsigned(std::abs(actual - wanted)));
        }
    }
    require(maxError <= 1, name + ": GPU/CPU mismatch, maximum byte error " + std::to_string(maxError));
    std::cout << "PASS " << name << " (max byte difference " << maxError << ")\n";
}

void benchmark()
{
    Batch batch;
    batch.pixels.resize(512 * 512, 0);
    batch.selection.resize(batch.pixels.size(), 255);
    batch.sourcePixels.resize(batch.pixels.size());
    for (int y = 0; y < 512; ++y) for (int x = 0; x < 512; ++x) {
        const float r = std::sqrt(float((x - 256) * (x - 256) + (y - 256) * (y - 256))) / 256;
        const auto alpha = unsigned(std::clamp(1.0f - r, 0.0f, 1.0f) * 255);
        batch.sourcePixels[y * 512 + x] = (alpha << 24) | 0x80a0e0;
    }
    for (unsigned i = 0; i < 128; ++i) {
        Dab d;
        d.bounds[0] = int(i % 7) - 3;
        d.bounds[1] = int(i % 5) - 2;
        d.bounds[2] = d.bounds[3] = 512;
        d.params[0] = 0.1f;
        batch.dabs.push_back(d);
        batch.dabIndices.push_back(i);
    }
    batch.tiles.push_back({0, 0, 512, 512, 0, 0, 128});
    auto cpu = batch;
    const auto start = std::chrono::steady_clock::now();
    reference(cpu);
    const auto middle = std::chrono::steady_clock::now();
    auto result = Compositor::instance().render(batch);
    const auto finish = std::chrono::steady_clock::now();
    require(result.success, result.message);
    std::cout << "128 overlapping 512px dabs: scalar reference "
              << std::chrono::duration<double, std::milli>(middle - start).count()
              << " ms; GPU including transfers "
              << std::chrono::duration<double, std::milli>(finish - middle).count()
              << " ms. This is not a benchmark against Krita's SIMD compositor.\n";
}
}

int main(int argc, char **argv)
{
    try {
        require(Compositor::instance().isAvailable(), Compositor::instance().diagnostic());
        std::cout << Compositor::instance().diagnostic() << '\n';
        compare(makeBatch(BlendMode::Over), "Normal, clipped tiles and selection");
        compare(makeBatch(BlendMode::Over, 7), "Normal with alpha lock");
        compare(makeBatch(BlendMode::Over, 10), "Normal with channel locks");
        compare(makeBatch(BlendMode::AlphaDarkenCreamy), "Creamy wash with flow and opacity ceiling");
        compare(makeBatch(BlendMode::AlphaDarkenHard), "Legacy hard wash");
        compare(makeBatch(BlendMode::Erase), "Eraser with selection");
        auto dense = makeBatch(BlendMode::Over);
        dense.tiles.resize(1);
        dense.tiles[0].dabCount = 600;
        dense.dabIndices.clear();
        for (unsigned i = 0; i < 600; ++i) dense.dabIndices.push_back(i % 40);
        compare(dense, "Dense batch spans multiple ordered dispatches");
        auto invalid = makeBatch(BlendMode::Over);
        invalid.tiles[1].pixelOffset = 0;
        auto original = invalid.pixels;
        require(!Compositor::instance().render(invalid).success && invalid.pixels == original,
                "Overlapping output storage must fail without modifying pixels");
        invalid = makeBatch(BlendMode::Over);
        invalid.dabs[0].data[0] = 0xffffffff;
        original = invalid.pixels;
        require(!Compositor::instance().render(invalid).success && invalid.pixels == original,
                "Invalid source bounds must fail without modifying pixels");
        std::cout << "PASS invalid batch leaves CPU input unchanged\n";
        std::vector<std::future<void>> jobs;
        for (int i = 0; i < 4; ++i) jobs.push_back(std::async(std::launch::async, [i] {
            compare(makeBatch(BlendMode::Over), "Concurrent submission " + std::to_string(i));
        }));
        for (auto &job : jobs) job.get();
        if (argc > 1 && std::strcmp(argv[1], "--benchmark") == 0) benchmark();
        Compositor::instance().shutdown();
        require(!Compositor::instance().isAvailable(), "Worker must stop before DLL teardown");
        auto stopped = makeBatch(BlendMode::Over);
        original = stopped.pixels;
        require(!Compositor::instance().render(stopped).success && stopped.pixels == original,
                "Submission after shutdown must not modify CPU pixels");
        std::cout << "PASS explicit shutdown and stopped-service fallback\n";
        std::cout << "All GPU compute checks passed.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
