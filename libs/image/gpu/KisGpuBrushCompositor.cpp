/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "KisGpuBrushCompositor.h"
#include "KisGpuBrushShader.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#ifndef KRITA_GPU_BRUSH_STANDALONE
#include <QCoreApplication>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/gl.h>
#include <d3d11.h>
#include <dxgi.h>
#endif

namespace KisGpuBrush
{
namespace
{
constexpr std::size_t maxBufferBytes = 128u * 1024u * 1024u;
#ifdef _WIN32
std::mutex nativeConfigMutex;
HGLRC nativeShareContext = nullptr;
int nativePixelFormat = 0;
int nativeProfile=1;
int nativeReset=0x8261;
#endif

bool validBatch(const Batch &b)
{
    const auto words = wordsPerPixel(b.format);
    if (int(b.format) < 0 || int(b.format) > 3 || b.pixels.size() % words || b.sourcePixels.size() % words) return false;
    if (b.tiles.empty() || b.dabs.empty() || b.dabIndices.empty() || b.pixels.empty() ||
        (!b.selection.empty() && b.selection.size() != b.pixels.size() / words) || b.channelFlags > 15 ||
        int(b.mode) < 0 || int(b.mode) > int(BlendMode::LinearBurn)) return false;
    if (b.pixels.size() > maxBufferBytes / 4 ||
        b.sourcePixels.size() > maxBufferBytes / 4 ||
        b.dabs.size() > maxBufferBytes / sizeof(Dab) ||
        b.dabIndices.size() > maxBufferBytes / 4) return false;
    for (auto m : b.selection) if (m > 255) return false;
    for (const auto &d : b.dabs) {
        if (d.bounds[2] <= 0 || d.bounds[3] <= 0) return false;
        const auto area = std::uint64_t(d.bounds[2]) * d.bounds[3];
        if (d.data[1] == 0 && (area * words + d.data[0] > b.sourcePixels.size() || d.data[0] % words)) return false;
        if (d.data[1] > 6 || d.data[1] == 4 || d.data[3] > 3) return false;
        if (d.data[1]==6 && (d.data[2]<2 || std::uint64_t(d.data[0])+d.data[2]>b.sourcePixels.size())) return false;
        if ((d.data[1] == 2 || d.data[1] == 3) && (!d.data[2] || d.data[2]>8192 || !std::isfinite(d.params[3]) || d.params[3] < 1 ||
            d.params[3] > 8192 || std::uint64_t(d.data[2]) * std::uint64_t(d.params[3]) *
                (d.data[1] == 3 ? words : 1) + d.data[0] > b.sourcePixels.size())) return false;
        if (d.data[1] == 5 && (d.data[0] + area*words > b.sourcePixels.size() ||
            d.texture[0] + area*words > b.sourcePixels.size() || d.texture[1]+area > b.sourcePixels.size())) return false;
        if (d.data[1] != 5 && d.texture[1] && (std::uint64_t(d.texture[1]) * d.texture[2] + d.texture[0] > b.sourcePixels.size())) return false;
        if (std::int64_t(d.bounds[0]) + d.bounds[2] > std::numeric_limits<std::int32_t>::max() ||
            std::int64_t(d.bounds[1]) + d.bounds[3] > std::numeric_limits<std::int32_t>::max()) return false;
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(d.params[i]) || d.params[i] < 0 || d.params[i] > 1) return false;
        }
        if (d.data[1]) for (int i=0;i<4;++i) {
            if (!std::isfinite(d.shape[i]) || !std::isfinite(d.transform[i]) ||
                !std::isfinite(d.colour[i]) || !std::isfinite(d.textureParams[i])) return false;
        }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    for (const auto &t : b.tiles) {
        if (t.width <= 0 || t.height <= 0 || t.width > 512 || t.height > 512 ||
            !t.dabCount || t.dabCount > 4096) return false;
        const auto end = std::uint64_t(t.pixelOffset) + std::uint64_t(t.width) * t.height * words;
        if (end > b.pixels.size() || t.pixelOffset % words ||
            std::uint64_t(t.indexOffset) + t.dabCount > b.dabIndices.size()) return false;
        if (std::int64_t(t.x) + t.width > std::numeric_limits<std::int32_t>::max() ||
            std::int64_t(t.y) + t.height > std::numeric_limits<std::int32_t>::max()) return false;
        spans.emplace_back(t.pixelOffset, end);
    }
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first < spans[i - 1].second) return false;
    }
    for (auto index : b.dabIndices) if (index >= b.dabs.size()) return false;
    return true;
}

#ifdef _WIN32
constexpr GLenum shaderStorageBuffer = 0x90D2;
constexpr GLenum dynamicDraw = 0x88E8;
constexpr GLenum computeShaderType = 0x91B9;
constexpr GLenum compileStatus = 0x8B81;
constexpr GLenum linkStatus = 0x8B82;
constexpr GLenum infoLogLength = 0x8B84;
constexpr GLenum maxShaderStorageBlockSize = 0x90DE;
constexpr GLbitfield storageBarrier = 0x2000;
constexpr GLbitfield bufferUpdateBarrier = 0x0200;
constexpr GLbitfield pixelBufferBarrier = 0x0080;
constexpr GLenum syncGpuCommandsComplete = 0x9117;
constexpr GLenum alreadySignaled = 0x911A;
constexpr GLenum conditionSatisfied = 0x911C;
constexpr GLenum waitFailed = 0x911D;

// The Windows SDK only declares OpenGL 1.1. Resolve the small 4.3 API subset
// from our own native context, without depending on Qt/ANGLE or a GL loader.
struct GlFunctions
{
#define GPU_GL_FUNCTION(ret, name, ...) \
    using name##Type = ret (APIENTRY *)(__VA_ARGS__); name##Type name = nullptr;
    GPU_GL_FUNCTION(GLuint, CreateShader, GLenum)
    GPU_GL_FUNCTION(void, ShaderSource, GLuint, GLsizei, const char *const *, const GLint *)
    GPU_GL_FUNCTION(void, CompileShader, GLuint)
    GPU_GL_FUNCTION(void, GetShaderiv, GLuint, GLenum, GLint *)
    GPU_GL_FUNCTION(void, GetShaderInfoLog, GLuint, GLsizei, GLsizei *, char *)
    GPU_GL_FUNCTION(void, DeleteShader, GLuint)
    GPU_GL_FUNCTION(GLuint, CreateProgram, void)
    GPU_GL_FUNCTION(void, AttachShader, GLuint, GLuint)
    GPU_GL_FUNCTION(void, LinkProgram, GLuint)
    GPU_GL_FUNCTION(void, GetProgramiv, GLuint, GLenum, GLint *)
    GPU_GL_FUNCTION(void, GetProgramInfoLog, GLuint, GLsizei, GLsizei *, char *)
    GPU_GL_FUNCTION(void, DeleteProgram, GLuint)
    GPU_GL_FUNCTION(void, UseProgram, GLuint)
    GPU_GL_FUNCTION(GLint, GetUniformLocation, GLuint, const char *)
    GPU_GL_FUNCTION(void, Uniform1i, GLint, GLint)
    GPU_GL_FUNCTION(void, Uniform4i, GLint, GLint, GLint, GLint, GLint)
    GPU_GL_FUNCTION(void, Uniform4ui, GLint, GLuint, GLuint, GLuint, GLuint)
    GPU_GL_FUNCTION(void, GenBuffers, GLsizei, GLuint *)
    GPU_GL_FUNCTION(void, DeleteBuffers, GLsizei, const GLuint *)
    GPU_GL_FUNCTION(void, BindBuffer, GLenum, GLuint)
    GPU_GL_FUNCTION(void, BindBufferBase, GLenum, GLuint, GLuint)
    GPU_GL_FUNCTION(void, BufferData, GLenum, std::ptrdiff_t, const void *, GLenum)
    GPU_GL_FUNCTION(void, BufferSubData, GLenum, std::ptrdiff_t, std::ptrdiff_t, const void *)
    GPU_GL_FUNCTION(void, GetBufferSubData, GLenum, std::ptrdiff_t, std::ptrdiff_t, void *)
    GPU_GL_FUNCTION(void, GetInteger64v, GLenum, std::int64_t *)
    GPU_GL_FUNCTION(void, DispatchCompute, GLuint, GLuint, GLuint)
    GPU_GL_FUNCTION(void, MemoryBarrier, GLbitfield)
    GPU_GL_FUNCTION(void *, FenceSync, GLenum, GLbitfield)
    GPU_GL_FUNCTION(GLenum, ClientWaitSync, void *, GLbitfield, std::uint64_t)
    GPU_GL_FUNCTION(void, DeleteSync, void *)
    GPU_GL_FUNCTION(void, ObjectLabel, GLenum, GLuint, GLsizei, const char *)
    GPU_GL_FUNCTION(void, PushDebugGroup, GLenum, GLuint, GLsizei, const char *)
    GPU_GL_FUNCTION(void, PopDebugGroup, void)
#undef GPU_GL_FUNCTION

    template<class T> static void resolve(T &fn, const char *name)
    {
        PROC p = wglGetProcAddress(name);
        if (!p || reinterpret_cast<std::intptr_t>(p) == -1 ||
            reinterpret_cast<std::uintptr_t>(p) <= 3) {
            p = GetProcAddress(GetModuleHandleW(L"opengl32.dll"), name);
        }
        if (!p) throw std::runtime_error(std::string("Missing OpenGL function: ") + name);
        fn = reinterpret_cast<T>(p);
    }
    void load()
    {
#define LOAD(name) resolve(name, "gl" #name)
        LOAD(CreateShader); LOAD(ShaderSource); LOAD(CompileShader);
        LOAD(GetShaderiv); LOAD(GetShaderInfoLog); LOAD(DeleteShader);
        LOAD(CreateProgram); LOAD(AttachShader); LOAD(LinkProgram);
        LOAD(GetProgramiv); LOAD(GetProgramInfoLog); LOAD(DeleteProgram);
        LOAD(UseProgram); LOAD(GetUniformLocation); LOAD(Uniform1i);
        LOAD(Uniform4i); LOAD(Uniform4ui); LOAD(GenBuffers); LOAD(DeleteBuffers);
        LOAD(BindBuffer); LOAD(BindBufferBase); LOAD(BufferData); LOAD(BufferSubData);
        LOAD(GetBufferSubData); LOAD(GetInteger64v); LOAD(DispatchCompute);
        LOAD(MemoryBarrier); LOAD(FenceSync); LOAD(ClientWaitSync); LOAD(DeleteSync);
        LOAD(ObjectLabel); LOAD(PushDebugGroup); LOAD(PopDebugGroup);
#undef LOAD
    }
};

class GlWorker
{
public:
    bool available = false;
    bool sharedCanvas = false;
    std::vector<void *> consumerFences;
    std::string message, renderer;

    void initialize()
    {
        try {
            WNDCLASSW wc = {};
            wc.style = CS_OWNDC;
            wc.lpfnWndProc = DefWindowProcW;
            wc.hInstance = GetModuleHandleW(nullptr);
            wc.lpszClassName = L"KritaGPUBrushCompute";
            m_class = RegisterClassW(&wc);
            if (!m_class) throw std::runtime_error("Cannot register GPU brush surface");
            m_window = CreateWindowExW(0, wc.lpszClassName, L"", WS_POPUP,
                                       0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr);
            if (!m_window) throw std::runtime_error("Cannot create GPU brush surface");
            m_dc = GetDC(m_window);
            PIXELFORMATDESCRIPTOR pfd = {};
            pfd.nSize = sizeof(pfd);
            pfd.nVersion = 1;
            pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
            pfd.iPixelType = PFD_TYPE_RGBA;
            pfd.cColorBits = 32;
            HGLRC share;
            int format,profile,reset;
            { std::lock_guard<std::mutex> lock(nativeConfigMutex); share=nativeShareContext; format=nativePixelFormat;
                profile=nativeProfile; reset=nativeReset; }
            if (!format) format=ChoosePixelFormat(m_dc,&pfd);
            else DescribePixelFormat(m_dc,format,sizeof(pfd),&pfd);
            if (!format || !SetPixelFormat(m_dc, format, &pfd)) {
                throw std::runtime_error("Cannot set GPU brush pixel format");
            }
            m_context = wglCreateContext(m_dc);
            if (!m_context || !wglMakeCurrent(m_dc, m_context)) {
                throw std::runtime_error("Cannot create native OpenGL context");
            }
            using CreateContext = HGLRC (WINAPI *)(HDC, HGLRC, const int *);
            CreateContext createContext = nullptr;
            GlFunctions::resolve(createContext, "wglCreateContextAttribsARB");
            const int attributes[] = {0x2091, 4, 0x2092, 3, 0x9126, share ? profile : 1,
                0x8256,share ? reset : 0x8261,0};
            HGLRC modern = createContext(m_dc, share, attributes);
            const DWORD shareError=modern || !share ? 0 : GetLastError();
            sharedCanvas=modern && share;
            if (!modern && share) modern=createContext(m_dc,nullptr,attributes);
            if (!modern) throw std::runtime_error("GPU brushes require native OpenGL 4.3");
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(m_context);
            m_context = modern;
            if (!wglMakeCurrent(m_dc, m_context)) throw std::runtime_error("Cannot activate compute context");
            m_gl.load();
            const auto *name = glGetString(GL_RENDERER);
            renderer = name ? reinterpret_cast<const char *>(name) : "Unknown OpenGL GPU";
            std::string lower = renderer;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });
            if (lower.find("llvmpipe") != std::string::npos ||
                lower.find("softpipe") != std::string::npos ||
                lower.find("software") != std::string::npos ||
                lower.find("gdi generic") != std::string::npos) {
                throw std::runtime_error("GPU brushes require a hardware OpenGL driver; found " + renderer);
            }
            m_gl.GetInteger64v(maxShaderStorageBlockSize, &m_maxBufferSize);
            if (m_maxBufferSize < 1024 * 1024) throw std::runtime_error("GPU storage buffers are too small");
            compile();
            m_gl.GenBuffers(6, m_buffers);
            const char *labels[] = {"Krita brush destination", "Krita brush tips",
                                   "Krita brush dabs", "Krita tile dab indices", "Krita brush selection",
                                   "Krita resident paint tile atlas"};
            for (int i = 0; i < 6; ++i) {
                m_gl.BindBuffer(shaderStorageBuffer, m_buffers[i]);
                m_gl.ObjectLabel(0x82E0, m_buffers[i], -1, labels[i]);
            }
            m_gl.ObjectLabel(0x82E2, m_program, -1, "Krita Pixel Brush compute");
            if (glGetError() != GL_NO_ERROR) throw std::runtime_error("Cannot allocate compute buffers");
            available = true;
            message = "OpenGL compute brush compositor ready: " + renderer;
            if (shareError) message += "; canvas sharing failed with WGL error " + std::to_string(shareError);
        } catch (const std::exception &e) {
            message = e.what();
        }
    }

    ~GlWorker()
    {
        if (m_context) {
            releaseDxInterop();
            if (m_gl.DeleteBuffers) m_gl.DeleteBuffers(6, m_buffers);
            if (m_program && m_gl.DeleteProgram) m_gl.DeleteProgram(m_program);
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(m_context);
        }
        if (m_dc) ReleaseDC(m_window, m_dc);
        if (m_window) DestroyWindow(m_window);
        if (m_class) UnregisterClassW(L"KritaGPUBrushCompute", GetModuleHandleW(nullptr));
    }

    Result render(Batch &batch)
    {
        Result result;
        result.renderer = renderer;
        if (!available) { result.message = message; return result; }
        bool debugGroup = false;
        try {
            waitForConsumers();
            while (glGetError() != GL_NO_ERROR) {}
            std::vector<std::uint32_t> output(batch.pixels.size());
            m_gl.PushDebugGroup(0x824A, 0, -1, "Krita GPU Pixel Brush batch");
            debugGroup = true;
            const bool resident = prepareResidentTiles(batch, result);
            if (batch.deferReadback && (!resident || !(sharedCanvas || m_dxInterop))) {
                m_gl.PopDebugGroup();
                result.message="GPU stroke needs a CPU checkpoint before continuing";
                return result;
            }
            if (resident) {
                // This buffer is a compact dirty-pixel transfer surface. It
                // receives CPU data only for cache misses; the compositor
                // reads the previous paint result directly from the atlas.
                reserve(0, batch.pixels.size() * 4);
                for (std::size_t i = 0; i < batch.tiles.size(); ++i) {
                    if (m_prepared[i].reused) ++result.reusedTiles;
                    if (!m_prepared[i].initialize) continue;
                    const Tile &t = batch.tiles[i];
                    if (batch.loadPixels) batch.loadPixels(t, output.data() + t.pixelOffset);
                    const auto bytes = std::size_t(t.width) * t.height * wordsPerPixel(batch.format) * 4;
                    m_gl.BufferSubData(shaderStorageBuffer, std::ptrdiff_t(t.pixelOffset) * 4,
                                       std::ptrdiff_t(bytes),
                                       (batch.loadPixels ? output.data() : batch.pixels.data()) + t.pixelOffset);
                    result.destinationUploadedBytes += bytes;
                }
                result.residentTiles = batch.tiles.size();
                result.uploadedBytes += result.destinationUploadedBytes;
                m_gl.BindBufferBase(shaderStorageBuffer, 0, m_buffers[5]);
                m_gl.BindBufferBase(shaderStorageBuffer, 5, m_buffers[0]);
            } else {
                if (batch.loadPixels) {
                    for (const Tile &t : batch.tiles) batch.loadPixels(t, output.data() + t.pixelOffset);
                }
                upload(0, batch.loadPixels ? output : batch.pixels, false, result);
                result.destinationUploadedBytes = batch.pixels.size() * 4;
                m_gl.BindBufferBase(shaderStorageBuffer, 5, m_buffers[0]);
            }
            upload(1, batch.sourcePixels, resident, result);
            upload(2, batch.dabs, resident, result);
            upload(3, batch.dabIndices, resident, result);
            // An absent selection is an implicit 255 mask, avoiding an entire
            // mask allocation, fill and transfer on ordinary brush strokes.
            static const std::vector<std::uint32_t> noSelection {255};
            upload(4, batch.selection.empty() ? noSelection : batch.selection, resident, result);
            checkError("uploading brush buffers");
            m_gl.UseProgram(m_program);
            m_gl.Uniform1i(m_modeLocation, int(batch.mode));
            m_gl.Uniform1i(m_formatLocation, int(batch.format));
            m_gl.Uniform1i(m_selectionLocation, batch.selection.empty() ? 0 : 1);
            for (std::size_t i = 0; i < batch.tiles.size(); ++i) {
                const Tile &t = batch.tiles[i];
                m_gl.Uniform4i(m_tileLocation, t.x, t.y, t.width, t.height);
                // Keep individual dispatches bounded to avoid a very dense
                // brush batch monopolizing the Windows GPU scheduler.
                for (unsigned first = 0; first < t.dabCount; first += 256) {
                    if (resident) {
                        m_gl.Uniform4ui(m_residencyLocation, m_prepared[i].offset,
                                        residentTileSize * wordsPerPixel(batch.format), 1,
                                        first == 0 && m_prepared[i].initialize ? 1 : 0);
                    } else {
                        m_gl.Uniform4ui(m_residencyLocation, 0, 0, 0, 0);
                    }
                    const unsigned count = std::min(256u, t.dabCount - first);
                    m_gl.Uniform4ui(m_rangesLocation, t.pixelOffset, t.indexOffset + first,
                                    count, batch.channelFlags);
                    m_gl.DispatchCompute((t.width + 15) / 16, (t.height + 15) / 16, 1);
                    ++result.dispatches;
                    if (first + count < t.dabCount) m_gl.MemoryBarrier(storageBarrier);
                }
            }
            m_gl.MemoryBarrier(storageBarrier | bufferUpdateBarrier | pixelBufferBarrier);
            m_gl.PopDebugGroup();
            debugGroup = false;
            checkError("dispatching brush compute");
            void *fence = m_gl.FenceSync(syncGpuCommandsComplete, 0);
            if (!fence) throw std::runtime_error("Cannot fence GPU brush work");
            glFlush();
            bool completed = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            do {
                const GLenum state = m_gl.ClientWaitSync(fence, 0, 20000000);
                completed = state == alreadySignaled || state == conditionSatisfied;
                if (completed || state == waitFailed) break;
            } while (std::chrono::steady_clock::now() < deadline);
            m_gl.DeleteSync(fence);
            if (!completed) throw std::runtime_error("GPU brush fence failed or timed out");
            if (batch.deferReadback) {
                for (auto &tile : m_prepared) {
                    tile.entry->pinned=true;
                    tile.entry->presentationKey=batch.presentationKey;
                }
                result.success=true;
                return result;
            }
            m_gl.BindBuffer(shaderStorageBuffer, m_buffers[0]);
            m_gl.GetBufferSubData(shaderStorageBuffer, 0,
                                 std::ptrdiff_t(output.size() * 4), output.data());
            checkError("reading brush result");
            // Commit only a complete result. A failed dispatch never modifies
            // the CPU device or the caller's input pixels.
            result.readbackBytes = batch.pixels.size() * 4;
            if (resident) rememberResidentPixels(batch, output);
            batch.pixels.swap(output);
            result.success = true;
        } catch (const std::exception &e) {
            if (debugGroup) m_gl.PopDebugGroup();
            available = false;
            message = e.what();
            result.message = message;
        }
        return result;
    }

    std::vector<PresentationTile> presentation(std::uint64_t key) const
    {
        std::vector<PresentationTile> tiles;
        for (const auto &entry:m_resident) {
            if (!entry.second.pinned || entry.second.presentationKey!=key) continue;
            tiles.push_back({std::get<2>(entry.first),std::get<3>(entry.first),residentTileSize,residentTileSize,
                m_buffers[5],entry.second.slot*slotWords*4,entry.second.age,PixelFormat(std::get<4>(entry.first))});
        }
        return tiles;
    }
    Checkpoint checkpoint(std::uint64_t id,std::uint64_t generation)
    {
        Checkpoint result;
        try {
            waitForConsumers();
            m_gl.BindBuffer(shaderStorageBuffer,m_buffers[5]);
            for (auto &entry:m_resident) {
                if (!entry.second.pinned || std::get<0>(entry.first)!=id || std::get<1>(entry.first)!=generation) continue;
                result.format=PixelFormat(std::get<4>(entry.first));
                const auto count=cellPixels*wordsPerPixel(result.format);
                Tile tile;
                tile.x=std::get<2>(entry.first); tile.y=std::get<3>(entry.first);
                tile.width=tile.height=residentTileSize;
                tile.pixelOffset=result.pixels.size();
                result.pixels.resize(result.pixels.size()+count);
                m_gl.GetBufferSubData(shaderStorageBuffer,entry.second.slot*slotWords*4,count*4,
                    result.pixels.data()+tile.pixelOffset);
                result.tiles.push_back(tile);
            }
            checkError("checkpointing GPU stroke");
            for (auto &entry:m_resident) if (std::get<0>(entry.first)==id && std::get<1>(entry.first)==generation) {
                entry.second.pinned=false; entry.second.presentationKey=0;
            }
            result.success=true;
        } catch (const std::exception &error) { result.message=error.what(); }
        return result;
    }
    void discard(std::uint64_t id)
    {
        for (auto &entry:m_resident) if (std::get<0>(entry.first)==id) {
            entry.second.pinned=false; entry.second.presentationKey=0; entry.second.fullyLoaded=false;
            std::fill(entry.second.valid.begin(),entry.second.valid.end(),0);
        }
    }
    void restartSharing()
    {
        if (sharedCanvas) return;
        this->~GlWorker();
        new(this) GlWorker;
        initialize();
    }

    bool configureAngle(void *angleDevice)
    {
        if (m_dxInterop) return true;
        if (!available || !angleDevice) return false;
        IDXGIDevice *dxgiDevice=nullptr;
        IDXGIAdapter *adapter=nullptr;
        IDXGIResource *resource=nullptr;
        try {
            // A separate D3D device avoids using ANGLE's immediate context from
            // the compute thread. The shared resource stays on the same GPU.
            auto *source=static_cast<ID3D11Device *>(angleDevice);
            if (FAILED(source->QueryInterface(__uuidof(IDXGIDevice),reinterpret_cast<void **>(&dxgiDevice))) ||
                FAILED(dxgiDevice->GetAdapter(&adapter))) throw std::runtime_error("Cannot find ANGLE's GPU adapter");
            if (FAILED(D3D11CreateDevice(adapter,D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,nullptr,0,D3D11_SDK_VERSION,
                                        &m_dxDevice,nullptr,nullptr))) throw std::runtime_error("Cannot create canvas transfer device");
            GlFunctions::resolve(m_dxOpen,"wglDXOpenDeviceNV");
            GlFunctions::resolve(m_dxClose,"wglDXCloseDeviceNV");
            GlFunctions::resolve(m_dxRegister,"wglDXRegisterObjectNV");
            GlFunctions::resolve(m_dxUnregister,"wglDXUnregisterObjectNV");
            GlFunctions::resolve(m_dxLock,"wglDXLockObjectsNV");
            GlFunctions::resolve(m_dxUnlock,"wglDXUnlockObjectsNV");
            m_dxInterop=m_dxOpen(m_dxDevice);
            if (!m_dxInterop) throw std::runtime_error("The OpenGL driver cannot share D3D11 textures");
            D3D11_TEXTURE2D_DESC desc={};
            desc.Width=residentTileSize*4; desc.Height=residentTileSize*16;
            desc.MipLevels=desc.ArraySize=1; desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count=1; desc.Usage=D3D11_USAGE_DEFAULT;
            desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_RENDER_TARGET;
            desc.MiscFlags=D3D11_RESOURCE_MISC_SHARED;
            if (FAILED(m_dxDevice->CreateTexture2D(&desc,nullptr,&m_dxTexture)) ||
                FAILED(m_dxTexture->QueryInterface(__uuidof(IDXGIResource),reinterpret_cast<void **>(&resource))) ||
                FAILED(resource->GetSharedHandle(&m_dxShareHandle))) throw std::runtime_error("Cannot allocate shared canvas texture");
            glGenTextures(1,&m_dxGlTexture);
            m_dxObject=m_dxRegister(m_dxInterop,m_dxTexture,m_dxGlTexture,GL_TEXTURE_2D,0x0001 /* READ_WRITE_NV */);
            if (!m_dxObject) throw std::runtime_error("Cannot register shared canvas texture with OpenGL");
            resource->Release(); adapter->Release(); dxgiDevice->Release();
            message+="; direct ANGLE/D3D11 canvas ready";
            return true;
        } catch (const std::exception &error) {
            if (resource) resource->Release();
            if (adapter) adapter->Release();
            if (dxgiDevice) dxgiDevice->Release();
            releaseDxInterop();
            message+="; "+std::string(error.what());
            return false;
        }
    }
    bool canPresent() const { return sharedCanvas || m_dxInterop; }
    AnglePresentation prepareAngle(std::uint64_t key)
    {
        AnglePresentation result;
        if (!m_dxInterop || !m_dxObject) return result;
        result.shareHandle=m_dxShareHandle;
        result.tiles=presentation(key);
        if (result.tiles.empty()) return result;
        bool locked=false;
        try {
            if (!m_dxLock(m_dxInterop,1,&m_dxObject)) throw std::runtime_error("Cannot acquire shared canvas texture");
            locked=true;
            m_gl.BindBuffer(0x88EC /* PIXEL_UNPACK_BUFFER */,m_buffers[5]);
            glBindTexture(GL_TEXTURE_2D,m_dxGlTexture);
            glPixelStorei(0x0CF2 /* UNPACK_ROW_LENGTH */,residentTileSize);
            glPixelStorei(GL_UNPACK_ALIGNMENT,1);
            for (auto &tile:result.tiles) {
                if (tile.format!=PixelFormat::RGBA8) throw std::runtime_error("ANGLE direct canvas requires RGBA8");
                const unsigned slot=tile.byteOffset/(slotWords*4);
                tile.textureX=(slot%4)*residentTileSize;
                tile.textureY=(slot/4)*residentTileSize;
                if (m_dxVersions[slot]==tile.version) continue;
                glTexSubImage2D(GL_TEXTURE_2D,0,tile.textureX,tile.textureY,tile.width,tile.height,
                               0x80E1 /* BGRA */,GL_UNSIGNED_BYTE,reinterpret_cast<void *>(std::uintptr_t(tile.byteOffset)));
                m_dxVersions[slot]=tile.version;
            }
            m_gl.BindBuffer(0x88EC,0);
            glPixelStorei(0x0CF2,0);
            checkError("copying GPU stroke into shared canvas texture");
            if (!m_dxUnlock(m_dxInterop,1,&m_dxObject)) throw std::runtime_error("Cannot release shared canvas texture");
            locked=false;
            result.shareHandle=m_dxShareHandle;
        } catch (const std::exception &error) {
            if (locked) m_dxUnlock(m_dxInterop,1,&m_dxObject);
            releaseDxInterop();
            message+="; "+std::string(error.what());
            return {};
        }
        return result;
    }

private:
    void releaseDxInterop()
    {
        if (m_dxObject && m_dxUnregister) m_dxUnregister(m_dxInterop,m_dxObject);
        m_dxObject=nullptr;
        if (m_dxGlTexture) glDeleteTextures(1,&m_dxGlTexture);
        m_dxGlTexture=0;
        if (m_dxInterop && m_dxClose) m_dxClose(m_dxInterop);
        m_dxInterop=nullptr;
        if (m_dxTexture) m_dxTexture->Release();
        m_dxTexture=nullptr;
        if (m_dxDevice) m_dxDevice->Release();
        m_dxDevice=nullptr; m_dxShareHandle=nullptr;
        std::fill_n(m_dxVersions,64,0);
    }
    ID3D11Device *m_dxDevice=nullptr;
    ID3D11Texture2D *m_dxTexture=nullptr;
    HANDLE m_dxInterop=nullptr,m_dxObject=nullptr,m_dxShareHandle=nullptr;
    GLuint m_dxGlTexture=0;
    std::uint64_t m_dxVersions[64]={};
    HANDLE (WINAPI *m_dxOpen)(void *)=nullptr;
    BOOL (WINAPI *m_dxClose)(HANDLE)=nullptr;
    HANDLE (WINAPI *m_dxRegister)(HANDLE,void *,GLuint,GLenum,GLenum)=nullptr;
    BOOL (WINAPI *m_dxUnregister)(HANDLE,HANDLE)=nullptr;
    BOOL (WINAPI *m_dxLock)(HANDLE,GLint,HANDLE *)=nullptr;
    BOOL (WINAPI *m_dxUnlock)(HANDLE,GLint,HANDLE *)=nullptr;
    static constexpr unsigned maxResidentTiles = 64;
    static constexpr std::size_t cellPixels = residentTileSize * residentTileSize;
    static constexpr std::size_t slotWords = cellPixels * 4;
    using CacheKey = std::tuple<std::uint64_t, std::uint64_t, std::int32_t, std::int32_t, int>;
    struct ResidentTile {
        unsigned slot = 0;
        std::uint64_t age = 0;
        std::uint64_t loadedAge = 0;
        bool fullyLoaded = false;
        bool pinned=false;
        std::uint64_t presentationKey=0;
        std::vector<std::uint32_t> shadow;
        std::vector<unsigned char> valid;
    };
    struct PreparedTile {
        ResidentTile *entry = nullptr;
        std::uint32_t offset = 0;
        bool initialize = true;
        bool reused = false;
    };
    static std::int32_t cellOrigin(std::int32_t v)
    {
        return std::int32_t((std::int64_t(v) - (v < 0 ? residentTileSize - 1 : 0)) /
                            residentTileSize * residentTileSize);
    }
    bool prepareResidentTiles(const Batch &batch, Result &result)
    {
        m_prepared.clear();
        if (!batch.cacheId || batch.tiles.size() > 1024) return false;
        const auto words = wordsPerPixel(batch.format);
        std::set<CacheKey> keys;
        for (const Tile &t : batch.tiles) {
            const auto x = cellOrigin(t.x), y = cellOrigin(t.y);
            if (std::int64_t(t.x) + t.width > std::int64_t(x) + residentTileSize ||
                std::int64_t(t.y) + t.height > std::int64_t(y) + residentTileSize) return false;
            keys.emplace(batch.cacheId, batch.cacheGeneration, x, y, int(batch.format));
        }
        if (keys.size() > maxResidentTiles) return false;
        std::size_t missing=0, reusable=maxResidentTiles-m_resident.size();
        for (const auto &key:keys) if (!m_resident.count(key)) ++missing;
        for (const auto &entry:m_resident) if (!entry.second.pinned && !keys.count(entry.first)) ++reusable;
        if (missing>reusable) return false;
        // Reject intersecting world-space output too. Packed-buffer validation
        // alone cannot protect an atlas from two dispatches sharing pixels.
        for (std::size_t i = 0; i < batch.tiles.size(); ++i) {
            const auto &a = batch.tiles[i];
            for (std::size_t j = i + 1; j < batch.tiles.size(); ++j) {
                const auto &b = batch.tiles[j];
                if (std::int64_t(a.x) < std::int64_t(b.x) + b.width &&
                    std::int64_t(b.x) < std::int64_t(a.x) + a.width &&
                    std::int64_t(a.y) < std::int64_t(b.y) + b.height &&
                    std::int64_t(b.y) < std::int64_t(a.y) + a.height) return false;
            }
        }
        if (!m_capacity[5]) reserve(5, maxResidentTiles * slotWords * 4);
        ++m_cacheAge;
        // Pin all entries needed by this batch before evicting any old entry.
        for (const auto &key : keys) {
            const auto it = m_resident.find(key);
            if (it != m_resident.end()) it->second.age = m_cacheAge;
        }
        for (const auto &key : keys) {
            if (m_resident.count(key)) continue;
            unsigned slot = unsigned(m_resident.size());
            if (m_resident.size() == maxResidentTiles) {
                auto oldest = m_resident.end();
                for (auto it = m_resident.begin(); it != m_resident.end(); ++it) {
                    if (!it->second.pinned && it->second.age != m_cacheAge &&
                        (oldest == m_resident.end() || it->second.age < oldest->second.age)) oldest = it;
                }
                if (oldest == m_resident.end()) return false;
                slot = oldest->second.slot;
                m_resident.erase(oldest);
            }
            ResidentTile entry;
            entry.slot = slot;
            entry.age = m_cacheAge;
            if (!batch.loadPixels) entry.shadow.resize(cellPixels * words);
            entry.valid.resize(cellPixels, 0);
            m_resident.emplace(key, std::move(entry));
        }
        // A stroke owns complete cells. Initialize each cell once, including
        // its currently clean margins, so a moving brush can enter those
        // margins without fetching the destination from the CPU again.
        if (batch.loadPixels) {
            std::vector<std::uint32_t> initial;
            for (const auto &key : keys) {
                auto &entry = m_resident.at(key);
                if (entry.fullyLoaded) continue;
                initial.resize(cellPixels * words);
                Tile cell;
                cell.x = std::get<2>(key); cell.y = std::get<3>(key);
                cell.width = cell.height = residentTileSize;
                batch.loadPixels(cell, initial.data());
                m_gl.BindBuffer(shaderStorageBuffer, m_buffers[5]);
                m_gl.BufferSubData(shaderStorageBuffer, std::ptrdiff_t(entry.slot * slotWords * 4),
                                   std::ptrdiff_t(cellPixels * words * 4), initial.data());
                result.destinationUploadedBytes += cellPixels * words * 4;
                entry.fullyLoaded = true;
                entry.loadedAge = m_cacheAge;
                std::fill(entry.valid.begin(), entry.valid.end(), 1);
            }
        }
        for (const Tile &t : batch.tiles) {
            const auto x = cellOrigin(t.x), y = cellOrigin(t.y);
            auto &entry = m_resident.at(CacheKey(batch.cacheId, batch.cacheGeneration, x, y, int(batch.format)));
            const auto local = std::uint32_t((t.y - y) * residentTileSize + t.x - x);
            bool unchanged = true;
            for (int row = 0; !batch.loadPixels && row < t.height && unchanged; ++row) {
                const auto pos = local + row * residentTileSize;
                unchanged = std::all_of(entry.valid.begin() + pos, entry.valid.begin() + pos + t.width,
                                        [](unsigned char valid) { return valid != 0; });
                unchanged = unchanged && entry.shadow.size() == cellPixels * words &&
                    std::memcmp(entry.shadow.data() + pos * words,
                                batch.pixels.data() + t.pixelOffset + row * t.width * words, t.width * words * 4) == 0;
            }
            m_prepared.push_back({&entry, std::uint32_t(entry.slot * slotWords + local * words), !unchanged,
                                  unchanged && (!batch.loadPixels || entry.loadedAge != m_cacheAge)});
        }
        return true;
    }
    void rememberResidentPixels(const Batch &batch, const std::vector<std::uint32_t> &output)
    {
        if (batch.loadPixels) return;
        const auto words = wordsPerPixel(batch.format);
        for (std::size_t i = 0; i < batch.tiles.size(); ++i) {
            const auto &t = batch.tiles[i];
            auto &entry = *m_prepared[i].entry;
            if (entry.shadow.empty()) entry.shadow.resize(cellPixels * words);
            const auto local = (m_prepared[i].offset - entry.slot * slotWords) / words;
            for (int row = 0; row < t.height; ++row) {
                const auto pos = local + row * residentTileSize;
                std::memcpy(entry.shadow.data() + pos * words,
                            output.data() + t.pixelOffset + row * t.width * words, t.width * words * 4);
                std::fill_n(entry.valid.begin() + pos, t.width, 1);
            }
        }
    }
    void checkError(const char *operation)
    {
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error(std::string("OpenGL error ") + std::to_string(error) + " while " + operation);
        }
    }
    void waitForConsumers()
    {
        bool completed=true;
        for (void *fence:consumerFences) {
            const auto status=m_gl.ClientWaitSync(fence,0,2000000000ull);
            m_gl.DeleteSync(fence);
            if (status!=alreadySignaled && status!=conditionSatisfied) {
                completed=false;
            }
        }
        consumerFences.clear();
        if (!completed) throw std::runtime_error("Canvas GPU synchronization failed");
    }
    void compile()
    {
        GLuint shader = m_gl.CreateShader(computeShaderType);
        m_gl.ShaderSource(shader, 1, &computeShader, nullptr);
        m_gl.CompileShader(shader);
        GLint status = 0;
        m_gl.GetShaderiv(shader, compileStatus, &status);
        if (!status) {
            GLint length = 0;
            m_gl.GetShaderiv(shader, infoLogLength, &length);
            std::string log(std::max(1, length), '\0');
            m_gl.GetShaderInfoLog(shader, GLsizei(log.size()), nullptr, log.data());
            m_gl.DeleteShader(shader);
            throw std::runtime_error("GPU brush shader: " + log);
        }
        m_program = m_gl.CreateProgram();
        m_gl.AttachShader(m_program, shader);
        m_gl.LinkProgram(m_program);
        m_gl.DeleteShader(shader);
        m_gl.GetProgramiv(m_program, linkStatus, &status);
        if (!status) {
            GLint length = 0;
            m_gl.GetProgramiv(m_program, infoLogLength, &length);
            std::string log(std::max(1, length), '\0');
            m_gl.GetProgramInfoLog(m_program, GLsizei(log.size()), nullptr, log.data());
            throw std::runtime_error("GPU brush program: " + log);
        }
        m_modeLocation = m_gl.GetUniformLocation(m_program, "mode");
        m_tileLocation = m_gl.GetUniformLocation(m_program, "tile");
        m_rangesLocation = m_gl.GetUniformLocation(m_program, "ranges");
        m_residencyLocation = m_gl.GetUniformLocation(m_program, "residency");
        m_selectionLocation = m_gl.GetUniformLocation(m_program, "hasSelection");
        m_formatLocation = m_gl.GetUniformLocation(m_program, "pixelFormat");
        if (m_modeLocation < 0 || m_tileLocation < 0 || m_rangesLocation < 0 ||
            m_residencyLocation < 0 || m_selectionLocation < 0 || m_formatLocation < 0) {
            throw std::runtime_error("GPU brush shader uniforms are missing");
        }
    }
    void reserve(unsigned binding, std::size_t bytes)
    {
        if (!bytes || bytes > std::size_t(m_maxBufferSize)) {
            throw std::runtime_error("GPU brush batch exceeds the device buffer limit");
        }
        m_gl.BindBuffer(shaderStorageBuffer, m_buffers[binding]);
        if (bytes > m_capacity[binding]) {
            m_gl.BufferData(shaderStorageBuffer, std::ptrdiff_t(bytes), nullptr, dynamicDraw);
            m_capacity[binding] = bytes;
        }
    }
    template<class T> void upload(unsigned binding, const std::vector<T> &data,
                                  bool cached, Result &result)
    {
        const std::size_t bytes = data.size() * sizeof(T);
        reserve(binding, bytes);
        auto &shadow = m_uploadShadow[binding];
        if (!cached || shadow.size() != bytes || std::memcmp(shadow.data(), data.data(), bytes)) {
            m_gl.BufferSubData(shaderStorageBuffer, 0, std::ptrdiff_t(bytes), data.data());
            result.uploadedBytes += bytes;
            // Bound the additional host cache; large changing brush tips do
            // not need another persistent copy just to discover a cache miss.
            if (cached && bytes <= 16u * 1024u * 1024u) {
                shadow.resize(bytes);
                std::memcpy(shadow.data(), data.data(), bytes);
            } else shadow.clear();
        }
        m_gl.BindBufferBase(shaderStorageBuffer, binding, m_buffers[binding]);
    }
    ATOM m_class = 0;
    HWND m_window = nullptr;
    HDC m_dc = nullptr;
    HGLRC m_context = nullptr;
    GlFunctions m_gl;
    GLuint m_program = 0, m_buffers[6] = {};
    std::size_t m_capacity[6] = {};
    std::vector<unsigned char> m_uploadShadow[5];
    std::map<CacheKey, ResidentTile> m_resident;
    std::vector<PreparedTile> m_prepared;
    std::uint64_t m_cacheAge = 0;
    std::int64_t m_maxBufferSize = 0;
    GLint m_modeLocation = -1, m_tileLocation = -1, m_rangesLocation = -1;
    GLint m_residencyLocation = -1, m_selectionLocation = -1;
    GLint m_formatLocation = -1;
};
#else
class GlWorker
{
public:
    bool available = false;
    std::string message = "GPU brush compositing is currently supported on Windows only";
    std::string renderer;
    bool sharedCanvas=false;
    std::vector<void *> consumerFences;
    void initialize() {}
    Result render(Batch &) { return {false, message, renderer, 0}; }
    std::vector<PresentationTile> presentation(std::uint64_t) const { return {}; }
    Checkpoint checkpoint(std::uint64_t,std::uint64_t) { return {}; }
    void discard(std::uint64_t) {}
    void restartSharing() {}
    bool configureAngle(void *) { return false; }
    bool canPresent() const { return false; }
    AnglePresentation prepareAngle(std::uint64_t) { return {}; }
};
#endif
}

struct Compositor::Private
{
    std::mutex presentationMutex;
    GlWorker *workerPointer=nullptr;
    std::map<std::uint64_t,std::map<std::uint64_t,std::function<void()>>> canvases;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::function<void(GlWorker &)>> work;
    bool stopping = false;
    std::thread thread;

    Private()
        : thread([this] {
            GlWorker worker;
            worker.initialize();
            { std::lock_guard<std::mutex> lock(presentationMutex); workerPointer=&worker; }
            for (;;) {
                std::function<void(GlWorker &)> job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock, [this] { return stopping || !work.empty(); });
                    if (stopping && work.empty()) break;
                    job = std::move(work.front());
                    work.pop_front();
                }
                { std::lock_guard<std::mutex> lock(presentationMutex); job(worker); }
            }
            { std::lock_guard<std::mutex> lock(presentationMutex); workerPointer=nullptr; }
        })
    {
    }
    ~Private() { shutdown(); }
    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_one();
        if (thread.joinable()) thread.join();
    }
    template<class T, class F> T submit(F function)
    {
        auto task = std::make_shared<std::packaged_task<T(GlWorker &)>>(std::move(function));
        auto result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) throw std::runtime_error("GPU brush service has stopped");
            work.emplace_back([task](GlWorker &worker) { (*task)(worker); });
        }
        wake.notify_one();
        return result.get();
    }
};

Compositor &Compositor::instance()
{
    static Compositor compositor;
    return compositor;
}
Compositor::Compositor() : m_d(new Private)
{
#ifndef KRITA_GPU_BRUSH_STANDALONE
    if (auto *app = QCoreApplication::instance()) {
        QObject::connect(app, &QCoreApplication::aboutToQuit, app, [this] { shutdown(); }, Qt::DirectConnection);
    }
#endif
}
Compositor::~Compositor() = default;
bool Compositor::isAvailable()
{
    try {
        return m_d->submit<bool>([](GlWorker &worker) { return worker.available; });
    } catch (const std::exception &) { return false; }
}
std::string Compositor::diagnostic()
{
    try {
        return m_d->submit<std::string>([](GlWorker &worker) { return worker.message; });
    } catch (const std::exception &e) { return e.what(); }
}
void Compositor::shutdown() { m_d->shutdown(); }
Result Compositor::render(Batch &batch)
{
    if (!validBatch(batch)) return {false, "Invalid or oversized GPU brush batch", {}, 0};
    try {
        return m_d->submit<Result>([&batch,this](GlWorker &worker) {
            if (batch.deferReadback && !m_d->canvases.count(batch.presentationKey)) batch.deferReadback=false;
            auto result=worker.render(batch);
            if (result.success && batch.deferReadback) {
                const auto canvas=m_d->canvases.find(batch.presentationKey);
                if (canvas!=m_d->canvases.end()) for (const auto &view:canvas->second) view.second();
            }
            return result;
        });
    } catch (const std::exception &e) { return {false, e.what(), {}, 0}; }
}

void Compositor::configureNativeCanvas(void *context,void *dc)
{
#ifdef _WIN32
    if (!context || !dc) return;
    { std::lock_guard<std::mutex> lock(nativeConfigMutex);
        nativeShareContext=static_cast<HGLRC>(context);
        nativePixelFormat=GetPixelFormat(static_cast<HDC>(dc));
        glGetIntegerv(0x9126,&nativeProfile);
        glGetIntegerv(0x8256,&nativeReset);
        if (nativeProfile!=1 && nativeProfile!=2) nativeProfile=1;
        if (nativeReset!=0x8252) nativeReset=0x8261;
    }
    // Some WGL drivers reject sharing while the source context is current
    // on another thread. Qt is idle during this synchronous handoff.
    const HGLRC previousContext=wglGetCurrentContext();
    const HDC previousDc=wglGetCurrentDC();
    if (previousContext) wglMakeCurrent(nullptr,nullptr);
    try { m_d->submit<bool>([](GlWorker &worker) { worker.restartSharing(); return worker.sharedCanvas; }); }
    catch (const std::exception &) {}
    if (previousContext) wglMakeCurrent(previousDc,previousContext);
#else
    (void)context; (void)dc;
#endif
}
void Compositor::registerCanvas(std::uint64_t key,std::uint64_t token,std::function<void()> repaint)
{
    std::lock_guard<std::mutex> lock(m_d->presentationMutex);
    m_d->canvases[key][token]=std::move(repaint);
}
void Compositor::unregisterCanvas(std::uint64_t key,std::uint64_t token)
{
    std::lock_guard<std::mutex> lock(m_d->presentationMutex);
    const auto it=m_d->canvases.find(key);
    if (it!=m_d->canvases.end()) {
        it->second.erase(token);
        if (it->second.empty()) m_d->canvases.erase(it);
    }
}
bool Compositor::canPresent(std::uint64_t key)
{
    std::lock_guard<std::mutex> lock(m_d->presentationMutex);
    return m_d->workerPointer && m_d->workerPointer->available && m_d->workerPointer->canPresent() && m_d->canvases.count(key);
}
bool Compositor::configureAngleCanvas(void *device)
{
    try { return m_d->submit<bool>([=](GlWorker &worker) { return worker.configureAngle(device); }); }
    catch (const std::exception &) { return false; }
}
AnglePresentation Compositor::prepareAnglePresentation(std::uint64_t key)
{
    try { return m_d->submit<AnglePresentation>([=](GlWorker &worker) { return worker.prepareAngle(key); }); }
    catch (const std::exception &) { return {}; }
}
bool Compositor::hasPresentation(std::uint64_t key)
{
    std::lock_guard<std::mutex> lock(m_d->presentationMutex);
    return m_d->workerPointer && !m_d->workerPointer->presentation(key).empty();
}
bool Compositor::present(std::uint64_t key,const std::function<void *(const std::vector<PresentationTile> &)> &draw)
{
    std::lock_guard<std::mutex> lock(m_d->presentationMutex);
    auto *worker=m_d->workerPointer;
    if (!worker || !worker->sharedCanvas) return false;
    auto tiles=worker->presentation(key);
    if (tiles.empty()) return false;
    if (auto *fence=draw(tiles)) worker->consumerFences.push_back(fence);
    return true;
}
Checkpoint Compositor::checkpoint(std::uint64_t id,std::uint64_t generation)
{
    try { return m_d->submit<Checkpoint>([=](GlWorker &worker) { return worker.checkpoint(id,generation); }); }
    catch (const std::exception &error) { Checkpoint result; result.message=error.what(); return result; }
}
void Compositor::discard(std::uint64_t id)
{
    try { m_d->submit<bool>([=](GlWorker &worker) { worker.discard(id); return true; }); }
    catch (const std::exception &) {}
}
}
