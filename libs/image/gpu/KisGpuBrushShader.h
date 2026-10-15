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
struct Dab { ivec4 bounds; uvec4 data; vec4 params; vec4 shape; vec4 transform; vec4 colour; uvec4 texture; vec4 textureParams; };
layout(std430, binding = 0) buffer Destination { uint pixels[]; };
layout(std430, binding = 1) readonly buffer Sources { uint sources[]; };
layout(std430, binding = 2) readonly buffer Dabs { Dab dabs[]; };
layout(std430, binding = 3) readonly buffer Indices { uint indices[]; };
layout(std430, binding = 4) readonly buffer Selection { uint mask[]; };
layout(std430, binding = 5) buffer Transfer { uint transfer[]; };
uniform ivec4 tile; // world origin and dimensions
uniform uvec4 ranges; // pixel offset, index offset, count, channel flags
uniform uvec4 residency; // atlas pixel offset, stride, enabled, initialize
uniform int mode;
uniform int hasSelection;
uniform int pixelFormat;
uint pixelWords() { return pixelFormat == 0 ? 1u : pixelFormat == 3 ? 4u : 2u; }

vec4 unpackPixel(uint p) {
    return vec4(p & 255u, (p >> 8) & 255u, (p >> 16) & 255u, p >> 24);
}
uint packPixel(vec4 p) {
    uvec4 b = uvec4(roundEven(clamp(p, 0.0, 255.0)));
    return b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
}
vec4 decode(uvec4 p) {
    if (pixelFormat == 0) return unpackPixel(p.x);
    if (pixelFormat == 1) return vec4(p.x & 65535u, p.x >> 16, p.y & 65535u, p.y >> 16) * (255.0 / 65535.0);
    if (pixelFormat == 2) return vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y)) * 255.0;
    return uintBitsToFloat(p) * 255.0;
}
uvec4 encode(vec4 p) {
    if (pixelFormat == 0) return uvec4(packPixel(p), 0u, 0u, 0u);
    p /= 255.0;
    p.a = clamp(p.a, 0.0, 1.0);
    if (pixelFormat == 1) {
        uvec4 v = uvec4(roundEven(clamp(p, 0.0, 1.0) * 65535.0));
        return uvec4(v.x | (v.y << 16), v.z | (v.w << 16), 0u, 0u);
    }
    if (pixelFormat == 2) return uvec4(packHalf2x16(p.xy), packHalf2x16(p.zw), 0u, 0u);
    return floatBitsToUint(p);
}
vec3 blendColour(vec3 s, vec3 d) {
    if (mode == 4) return s * d;
    if (mode == 5) return s + d - s * d;
    if (mode == 6) return mix(2.0*s*d, 1.0-2.0*(1.0-s)*(1.0-d), step(vec3(0.5), d));
    if (mode == 7) return min(s,d);
    if (mode == 8) return max(s,d);
    if (mode == 9) return pixelFormat < 2 ? min(s+d,vec3(1.0)) : s+d;
    if (mode == 10) return max(d-s,vec3(0.0));
    if (mode == 12) return abs(d-s);
    if (mode == 13) return s+d-2.0*s*d;
    if (mode == 14) return min(d / max(vec3(1e-8),1.0-s),vec3(1.0));
    if (mode == 15) return 1.0-min((1.0-d)/max(s,vec3(1e-8)),vec3(1.0));
    if (mode == 16) return mix(2.0*s*d,1.0-2.0*(1.0-s)*(1.0-d),step(vec3(0.5),s));
    if (mode == 17) {
        vec3 g = mix(((16.0*d-12.0)*d+4.0)*d,sqrt(max(d,vec3(0.0))),step(vec3(0.25),d));
        return mix(d-(1.0-2.0*s)*d*(1.0-d),d+(2.0*s-1.0)*(g-d),step(vec3(0.5),s));
    }
    if (mode == 18) return max(s+d-1.0,vec3(0.0));
    return s;
}
vec4 sourcePixel(uint offset);
vec4 tipTexel(Dab dab, ivec2 p) {
    ivec2 size = ivec2(dab.data.z,int(dab.params.w));
    if (any(lessThan(p,ivec2(0))) || any(greaterThanEqual(p,size))) return vec4(0.0);
    uint index=uint(p.y*size.x+p.x);
    vec4 value=dab.data.y==3u ? sourcePixel(dab.data.x+index*pixelWords()) : unpackPixel(sources[dab.data.x+index]);
    value.rgb*=value.a/255.0;
    return value;
}
vec4 smudgeBlend(vec4 dst, vec4 src, float amount, bool copyMode) {
    float sa=src.a/255.0, da=dst.a/255.0;
    float a=copyMode ? mix(da,sa,amount) : sa*amount+da*(1.0-sa*amount);
    vec3 rgb=copyMode ? (dst.rgb*da*(1.0-amount)+src.rgb*sa*amount) :
        (src.rgb*sa*amount+dst.rgb*da*(1.0-sa*amount));
    if (a>0.0) dst.rgb=rgb/a;
    if (copyMode && amount==1.0) dst.rgb=src.rgb;
    dst.a=a*255.0;
    return decode(encode(dst));
}
vec4 sourcePixel(uint offset) {
    uvec4 v=uvec4(0);
    for (uint w=0u;w<pixelWords();++w) v[w]=sources[offset+w];
    return decode(v);
}
vec4 generateTip(Dab dab, ivec2 pixel, ivec2 world) {
    if (dab.data.y==5u) {
        uint p=uint(pixel.y*dab.bounds.z+pixel.x)*pixelWords();
        vec4 background=sourcePixel(dab.texture.x+p);
        vec4 carried=dab.params.w!=0.0 ? dab.transform : sourcePixel(dab.data.x+p);
        if (dab.data.w!=0u) {
            carried=smudgeBlend(carried,dab.colour,dab.shape.z,dab.shape.w!=0.0);
            return smudgeBlend(background,carried,dab.shape.x,dab.shape.y!=0.0);
        }
        return smudgeBlend(smudgeBlend(background,carried,dab.shape.x,dab.shape.y!=0.0),
                            dab.colour,dab.shape.z,dab.shape.w!=0.0);
    }
    if ((dab.data.w & 1u) != 0u) pixel.x = dab.bounds.z-1-pixel.x;
    if ((dab.data.w & 2u) != 0u) pixel.y = dab.bounds.w-1-pixel.y;
    vec2 p = vec2(pixel)-dab.shape.xy;
    vec2 q = vec2(p.x*dab.transform.x-p.y*dab.transform.y,p.x*dab.transform.y+p.y*dab.transform.x);
    vec4 src = dab.colour;
    if (dab.data.y==6u) {
        float radius=length(vec2(q.x,q.y*dab.shape.w))*dab.shape.z;
        float position=clamp(radius,0.0,1.0)*float(dab.data.z-1u);
        uint first=uint(position), last=min(first+1u,dab.data.z-1u);
        float coverage=mix(float(sources[dab.data.x+first]),float(sources[dab.data.x+last]),fract(position))/255.0;
        src.a*=radius>1.0 ? 0.0 : coverage;
    } else if (dab.data.y == 1u) {
        float n = dot(q*dab.shape.zw,q*dab.shape.zw);
        float coverage = 0.0;
        if (n <= 1.0) {
            if (dab.data.z != 0u) q = abs(q)+1.0;
            float nf = dot(q*dab.transform.zw,q*dab.transform.zw);
            float maskValue = nf < 1.0 ? 0.0 : (nf == n ? 1.0 : n*(nf-1.0)/(nf-n));
            coverage = 1.0-clamp(maskValue,0.0,1.0);
        }
        src.a *= coverage;
    } else {
        vec2 centre=vec2(pixel)+0.5;
        vec2 uv=vec2(dot(centre,dab.shape.xy),dot(centre,dab.shape.zw))+dab.transform.xy;
        ivec2 first = ivec2(floor(uv));
        vec2 f = fract(uv);
        vec4 sampled = mix(mix(tipTexel(dab,first),tipTexel(dab,first+ivec2(1,0)),f.x),
                          mix(tipTexel(dab,first+ivec2(0,1)),tipTexel(dab,first+ivec2(1,1)),f.x),f.y);
        if (sampled.a>0.0) sampled.rgb*=255.0/sampled.a;
        if (dab.data.y == 2u) {
            float gray = floor((sampled.r*5.0+sampled.g*16.0+sampled.b*11.0)/32.0);
            src.a *= (1.0-gray/255.0)*sampled.a/255.0;
        } else src = sampled;
    }
    if (dab.texture.y != 0u) {
        ivec2 size = ivec2(dab.texture.yz);
        ivec2 pos = (world-ivec2(dab.textureParams.xy)) % size;
        pos = (pos+size)%size;
        float m = float(sources[dab.texture.x+uint(pos.y*size.x+pos.x)]) / 255.0;
        float a = src.a/255.0;
        float strength = dab.textureParams.z;
        bool soft = dab.textureParams.w != 0.0;
        if (dab.texture.w == 1u) a *= soft ? 1.0-strength+strength*m : strength*m;
        else if (dab.texture.w == 2u) a = max(0.0,a-(soft ? strength*m : m+1.0-strength));
        else if (dab.texture.w == 3u) a = soft ? min(a,1.0-strength+strength*m) : min(a*strength,m);
        src.a = clamp(a,0.0,1.0)*255.0;
    }
    return decode(encode(src));
}
uint mul255(uint a, uint b) {
    uint c = a * b + 128u;
    return ((c >> 8) + c) >> 8;
}

vec4 composite(vec4 dst, vec4 src, vec3 params, float selection) {
    if (mode == 3) {
        if (pixelFormat != 0) { dst.a *= 1.0 - src.a * (1.0 / 255.0) * params.x * selection; return dst; }
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
    if (mode == 11) {
        vec4 result=smudgeBlend(dst,src,opacity*selection,true);
        if ((ranges.w & 8u)==0u && src.a==0.0) result.rgb=dst.rgb;
        for (int c = 0; c < 4; ++c) if ((ranges.w & (1u << c)) == 0u) result[c] = dst[c];
        return result;
    }
    if (srcAlpha == 0.0) return dst;

    if (mode >= 4) {
        float sa = clamp(srcAlpha / 255.0, 0.0, 1.0), da = clamp(dst.a / 255.0, 0.0, 1.0);
        vec3 s = src.rgb / 255.0, d = dst.rgb / 255.0;
        vec3 b = blendColour(s,d);
        bool locked = (ranges.w & 8u) == 0u;
        float a = locked ? da : sa + da * (1.0-sa);
        vec3 result = locked ? mix(d,b,sa) :
            (a > 0.0 ? (s*sa*(1.0-da)+d*da*(1.0-sa)+b*sa*da)/a : d);
        for (int c = 0; c < 3; ++c) if ((ranges.w & (1u << c)) != 0u) dst[c] = result[c] * 255.0;
        if (!locked) dst.a = a * 255.0;
        return dst;
    }

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
    uint words = pixelWords();
    uint offset = ranges.x + uint(local.y * tile.z + local.x) * words;
    uint destination = residency.z != 0u
        ? residency.x + uint(local.y) * residency.y + uint(local.x) * words : offset;
    uvec4 original = uvec4(0);
    for (uint w = 0u; w < words; ++w) original[w] = residency.z != 0u && residency.w != 0u
        ? transfer[offset+w] : pixels[destination+w];
    float selection = hasSelection != 0 ? float(mask[offset/words]) * (1.0 / 255.0) : 1.0;
    ivec2 world = tile.xy + local;
    vec4 dst = decode(original);
    for (uint i = 0u; selection != 0.0 && i < ranges.z; ++i) {
        Dab dab = dabs[indices[ranges.y + i]];
        ivec2 p = world - dab.bounds.xy;
        if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, dab.bounds.zw))) continue;
        vec4 src;
        if (dab.data.y != 0u) src = generateTip(dab,p,world);
        else {
            uint sourceOffset = dab.data.x + uint(p.y * dab.bounds.z + p.x) * words;
            uvec4 source = uvec4(0);
            for (uint w = 0u; w < words; ++w) source[w] = sources[sourceOffset+w];
            src = decode(source);
        }
        if (src.a == 0.0 && mode != 11) continue;
        float coverage=selection;
        if (dab.data.y==5u) coverage*=float(sources[dab.texture.y+uint(p.y*dab.bounds.z+p.x)]) / 255.0;
        dst = composite(dst, src, dab.params.xyz, coverage);
        // Match Krita's byte storage after each dab, rather than accumulating
        // an entire batch in floating point and rounding only at the end.
        dst = decode(encode(dst));
    }
    uvec4 result = selection == 0.0 ? original : encode(dst);
    for (uint w = 0u; w < words; ++w) {
        if (result[w] != original[w] || residency.w != 0u) pixels[destination+w] = result[w];
        if (residency.z != 0u) transfer[offset+w] = result[w];
    }
}
)GLSL";
}
#endif
