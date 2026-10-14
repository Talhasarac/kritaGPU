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
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/gl.h>
#endif

namespace KisGpuBrush
{
namespace
{
constexpr std::size_t maxBufferBytes = 128u * 1024u * 1024u;

bool validBatch(const Batch &b)
{
    if (b.tiles.empty() || b.dabs.empty() || b.dabIndices.empty() || b.pixels.empty() ||
        b.selection.size() != b.pixels.size() || b.channelFlags > 15 ||
        int(b.mode) < 0 || int(b.mode) > 3) return false;
    if (b.mode != BlendMode::Over && b.channelFlags != 15) return false;
    if (b.pixels.size() > maxBufferBytes / 4 ||
        b.sourcePixels.size() > maxBufferBytes / 4 ||
        b.dabs.size() > maxBufferBytes / sizeof(Dab) ||
        b.dabIndices.size() > maxBufferBytes / 4) return false;
    for (auto m : b.selection) if (m > 255) return false;
    for (const auto &d : b.dabs) {
        if (d.bounds[2] <= 0 || d.bounds[3] <= 0) return false;
        const auto area = std::uint64_t(d.bounds[2]) * d.bounds[3];
        if (area + d.data[0] > b.sourcePixels.size()) return false;
        if (std::int64_t(d.bounds[0]) + d.bounds[2] > std::numeric_limits<std::int32_t>::max() ||
            std::int64_t(d.bounds[1]) + d.bounds[3] > std::numeric_limits<std::int32_t>::max()) return false;
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(d.params[i]) || d.params[i] < 0 || d.params[i] > 1) return false;
        }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    for (const auto &t : b.tiles) {
        if (t.width <= 0 || t.height <= 0 || t.width > 512 || t.height > 512 ||
            !t.dabCount || t.dabCount > 4096) return false;
        const auto end = std::uint64_t(t.pixelOffset) + std::uint64_t(t.width) * t.height;
        if (end > b.pixels.size() ||
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
            const int format = ChoosePixelFormat(m_dc, &pfd);
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
            const int attributes[] = {0x2091, 4, 0x2092, 3, 0x9126, 1, 0};
            HGLRC modern = createContext(m_dc, nullptr, attributes);
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
            m_gl.GenBuffers(5, m_buffers);
            const char *labels[] = {"Krita brush destination", "Krita brush tips",
                                   "Krita brush dabs", "Krita tile dab indices", "Krita brush selection"};
            for (int i = 0; i < 5; ++i) {
                m_gl.BindBuffer(shaderStorageBuffer, m_buffers[i]);
                m_gl.ObjectLabel(0x82E0, m_buffers[i], -1, labels[i]);
            }
            m_gl.ObjectLabel(0x82E2, m_program, -1, "Krita Pixel Brush compute");
            if (glGetError() != GL_NO_ERROR) throw std::runtime_error("Cannot allocate compute buffers");
            available = true;
            message = "OpenGL compute brush compositor ready: " + renderer;
        } catch (const std::exception &e) {
            message = e.what();
        }
    }

    ~GlWorker()
    {
        if (m_context) {
            if (m_gl.DeleteBuffers) m_gl.DeleteBuffers(5, m_buffers);
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
            while (glGetError() != GL_NO_ERROR) {}
            m_gl.PushDebugGroup(0x824A, 0, -1, "Krita GPU Pixel Brush batch");
            debugGroup = true;
            upload(0, batch.pixels);
            upload(1, batch.sourcePixels);
            upload(2, batch.dabs);
            upload(3, batch.dabIndices);
            upload(4, batch.selection);
            checkError("uploading brush buffers");
            m_gl.UseProgram(m_program);
            m_gl.Uniform1i(m_modeLocation, int(batch.mode));
            for (const Tile &t : batch.tiles) {
                m_gl.Uniform4i(m_tileLocation, t.x, t.y, t.width, t.height);
                // Keep individual dispatches bounded to avoid a very dense
                // brush batch monopolizing the Windows GPU scheduler.
                for (unsigned first = 0; first < t.dabCount; first += 256) {
                    const unsigned count = std::min(256u, t.dabCount - first);
                    m_gl.Uniform4ui(m_rangesLocation, t.pixelOffset, t.indexOffset + first,
                                    count, batch.channelFlags);
                    m_gl.DispatchCompute((t.width + 15) / 16, (t.height + 15) / 16, 1);
                    ++result.dispatches;
                    if (first + count < t.dabCount) m_gl.MemoryBarrier(storageBarrier);
                }
            }
            m_gl.MemoryBarrier(storageBarrier | bufferUpdateBarrier);
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
            std::vector<std::uint32_t> output(batch.pixels.size());
            m_gl.BindBuffer(shaderStorageBuffer, m_buffers[0]);
            m_gl.GetBufferSubData(shaderStorageBuffer, 0,
                                 std::ptrdiff_t(output.size() * 4), output.data());
            checkError("reading brush result");
            // Commit only a complete result. A failed dispatch never modifies
            // the CPU device or the caller's input pixels.
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

private:
    void checkError(const char *operation)
    {
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error(std::string("OpenGL error ") + std::to_string(error) + " while " + operation);
        }
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
        if (m_modeLocation < 0 || m_tileLocation < 0 || m_rangesLocation < 0) {
            throw std::runtime_error("GPU brush shader uniforms are missing");
        }
    }
    template<class T> void upload(unsigned binding, const std::vector<T> &data)
    {
        const std::size_t bytes = data.size() * sizeof(T);
        if (!bytes || bytes > std::size_t(m_maxBufferSize)) {
            throw std::runtime_error("GPU brush batch exceeds the device buffer limit");
        }
        m_gl.BindBuffer(shaderStorageBuffer, m_buffers[binding]);
        if (bytes > m_capacity[binding]) {
            m_gl.BufferData(shaderStorageBuffer, std::ptrdiff_t(bytes), data.data(), dynamicDraw);
            m_capacity[binding] = bytes;
        } else {
            m_gl.BufferSubData(shaderStorageBuffer, 0, std::ptrdiff_t(bytes), data.data());
        }
        m_gl.BindBufferBase(shaderStorageBuffer, binding, m_buffers[binding]);
    }
    ATOM m_class = 0;
    HWND m_window = nullptr;
    HDC m_dc = nullptr;
    HGLRC m_context = nullptr;
    GlFunctions m_gl;
    GLuint m_program = 0, m_buffers[5] = {};
    std::size_t m_capacity[5] = {};
    std::int64_t m_maxBufferSize = 0;
    GLint m_modeLocation = -1, m_tileLocation = -1, m_rangesLocation = -1;
};
#else
class GlWorker
{
public:
    bool available = false;
    std::string message = "GPU brush compositing is currently supported on Windows only";
    std::string renderer;
    void initialize() {}
    Result render(Batch &) { return {false, message, renderer, 0}; }
};
#endif
}

struct Compositor::Private
{
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::function<void(GlWorker &)>> work;
    bool stopping = false;
    std::thread thread;

    Private()
        : thread([this] {
            GlWorker worker;
            worker.initialize();
            for (;;) {
                std::function<void(GlWorker &)> job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock, [this] { return stopping || !work.empty(); });
                    if (stopping && work.empty()) break;
                    job = std::move(work.front());
                    work.pop_front();
                }
                job(worker);
            }
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
Compositor::Compositor() : m_d(new Private) {}
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
        return m_d->submit<Result>([&batch](GlWorker &worker) { return worker.render(batch); });
    } catch (const std::exception &e) { return {false, e.what(), {}, 0}; }
}
}
