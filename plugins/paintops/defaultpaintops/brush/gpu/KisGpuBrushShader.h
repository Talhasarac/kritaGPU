/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef KIS_GPU_BRUSH_SHADER_H
#define KIS_GPU_BRUSH_SHADER_H

namespace KisGpuBrush
{
// The compositing equations follow Krita's KoOptimizedCompositeOpOver32,
// KoAlphaDarkenParamsWrapper and KoCompositeOpErase implementations.
inline constexpr const char *computeShader = R"GLSL(#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
struct Dab { ivec4 bounds; uvec4 data; vec4 params; };
layout(std430, binding = 0) buffer Destination { uint pixels[]; };
layout(std430, binding = 1) readonly buffer Sources { uint sources[]; };
layout(std430, binding = 2) readonly buffer Dabs { Dab dabs[]; };
layout(std430, binding = 3) readonly buffer Indices { uint indices[]; };
layout(std430, binding = 4) readonly buffer Selection { uint mask[]; };
uniform ivec4 tile; // world origin and dimensions
uniform uvec4 ranges; // pixel offset, index offset, count, channel flags
uniform int mode;

vec4 unpackPixel(uint p) {
    return vec4(p & 255u, (p >> 8) & 255u, (p >> 16) & 255u, p >> 24);
}
uint packPixel(vec4 p) {
    uvec4 b = uvec4(roundEven(clamp(p, 0.0, 255.0)));
    return b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
}
uint mul255(uint a, uint b) {
    uint c = a * b + 128u;
    return ((c >> 8) + c) >> 8;
}

vec4 composite(vec4 dst, vec4 src, vec3 params, float selection) {
    if (mode == 3) {
        uint a = mul255(uint(src.a), uint(selection * 255.0 + 0.5));
        a = mul255(a, uint(clamp(params.x * 255.0, 0.0, 255.0) + 0.5));
        dst.a = float(mul255(255u - a, uint(dst.a)));
        return dst;
    }

    precise float opacity = params.x;
    precise float average = params.z;
    if (mode == 2) { opacity *= params.y; average *= params.y; }
    precise float maskAlpha = src.a * (1.0 / 255.0) * selection;
    precise float srcAlpha = src.a * opacity;
    srcAlpha *= selection;
    if (srcAlpha == 0.0) return dst;

    if (mode == 0) {
        bool alphaLocked = (ranges.w & 8u) == 0u;
        precise float newAlpha = dst.a;
        precise float blend;
        if (alphaLocked || dst.a == 255.0) {
            blend = srcAlpha * (1.0 / 255.0);
        } else if (dst.a == 0.0) {
            newAlpha = srcAlpha;
            blend = 1.0;
            if ((ranges.w & 7u) != 7u) dst.rgb = vec3(0.0);
        } else {
            newAlpha = dst.a + (255.0 - dst.a) * srcAlpha * (1.0 / 255.0);
            blend = srcAlpha / newAlpha;
        }
        precise vec3 colour = (src.rgb - dst.rgb) * blend + dst.rgb;
        for (int c = 0; c < 3; ++c) {
            if ((ranges.w & (1u << c)) != 0u) dst[c] = colour[c];
        }
        if (!alphaLocked) dst.a = newAlpha;
        return dst;
    }

    // Wash has a separate opacity ceiling and flow interpolation. Multiplying
    // opacity by flow would change the default creamy wash presets.
    precise float srcNorm = maskAlpha * opacity;
    precise vec3 colour = (src.rgb - dst.rgb) * srcNorm + dst.rgb;
    dst.rgb = dst.a == 0.0 ? src.rgb : colour;
    precise float dstNorm = dst.a * (1.0 / 255.0);
    precise float fullFlow;
    if (average > opacity) {
        fullFlow = average > dstNorm
            ? (average - srcNorm) * (dstNorm / average) + srcNorm : dstNorm;
    } else {
        fullFlow = opacity > dstNorm
            ? (opacity - dstNorm) * maskAlpha + dstNorm : dstNorm;
    }
    precise float zeroFlow = mode == 2
        ? srcNorm + dstNorm - srcNorm * dstNorm : dstNorm;
    precise float resultAlpha = params.y == 1.0
        ? fullFlow : (fullFlow - zeroFlow) * params.y + zeroFlow;
    dst.a = resultAlpha * 255.0;
    return dst;
}

void main() {
    ivec2 local = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(local, tile.zw))) return;
    uint offset = ranges.x + uint(local.y * tile.z + local.x);
    float selection = float(mask[offset]) * (1.0 / 255.0);
    if (selection == 0.0) return;
    ivec2 world = tile.xy + local;
    uint original = pixels[offset];
    vec4 dst = unpackPixel(original);
    for (uint i = 0u; i < ranges.z; ++i) {
        Dab dab = dabs[indices[ranges.y + i]];
        ivec2 p = world - dab.bounds.xy;
        if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, dab.bounds.zw))) continue;
        uint sourceOffset = dab.data.x + uint(p.y * dab.bounds.z + p.x);
        vec4 src = unpackPixel(sources[sourceOffset]);
        if (src.a == 0.0) continue;
        dst = composite(dst, src, dab.params.xyz, selection);
        // Match Krita's byte storage after each dab, rather than accumulating
        // an entire batch in floating point and rounding only at the end.
        dst = unpackPixel(packPixel(dst));
    }
    uint result = packPixel(dst);
    if (result != original) pixels[offset] = result;
}
)GLSL";
}
#endif
