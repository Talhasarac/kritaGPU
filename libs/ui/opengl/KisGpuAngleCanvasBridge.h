/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef KIS_GPU_ANGLE_CANVAS_BRIDGE_H
#define KIS_GPU_ANGLE_CANVAS_BRIDGE_H

#ifdef Q_OS_WIN
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <QLibrary>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QVector4D>
#include <kis_image.h>
#include "kis_opengl_image_textures.h"

// Keep Qt's ANGLE/D3D11 window presentation while compute runs in native GL.
// EGL imports a shared GPU texture; no brush pixels pass through the CPU.
class KisGpuAngleCanvasBridge
{
public:
    bool initialize()
    {
        if (!m_egl.load()) return false;
#define EGL_LOAD(name) name = reinterpret_cast<decltype(name)>(m_egl.resolve(#name)); if (!name) return false
        EGL_LOAD(eglGetCurrentDisplay);
        EGL_LOAD(eglGetError);
        EGL_LOAD(eglGetProcAddress);
        EGL_LOAD(eglChooseConfig);
        EGL_LOAD(eglCreatePbufferFromClientBuffer);
        EGL_LOAD(eglDestroySurface);
        EGL_LOAD(eglBindTexImage);
        EGL_LOAD(eglReleaseTexImage);
#undef EGL_LOAD
        m_display=eglGetCurrentDisplay();
        if (m_display==EGL_NO_DISPLAY) return false;
        const auto queryDisplay=reinterpret_cast<PFNEGLQUERYDISPLAYATTRIBEXTPROC>(eglGetProcAddress("eglQueryDisplayAttribEXT"));
        const auto queryDevice=reinterpret_cast<PFNEGLQUERYDEVICEATTRIBEXTPROC>(eglGetProcAddress("eglQueryDeviceAttribEXT"));
        EGLAttrib device=0,d3d=0;
        if (!queryDisplay || !queryDevice || !queryDisplay(m_display,EGL_DEVICE_EXT,&device) ||
            !queryDevice(reinterpret_cast<EGLDeviceEXT>(device),EGL_D3D11_DEVICE_ANGLE,&d3d)) return false;
        const EGLint attributes[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,
            EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_BIND_TO_TEXTURE_RGBA,EGL_TRUE,EGL_NONE};
        EGLint count=0;
        if (!eglChooseConfig(m_display,attributes,nullptr,0,&count) || !count) return false;
        m_configs.resize(count);
        if (!eglChooseConfig(m_display,attributes,m_configs.data(),count,&count)) return false;
        m_configs.resize(count);
        return KisGpuBrush::Compositor::instance().configureAngleCanvas(reinterpret_cast<void *>(d3d));
    }

    ~KisGpuAngleCanvasBridge()
    {
        if (auto *ctx=QOpenGLContext::currentContext()) {
            auto *f=ctx->extraFunctions();
            if (m_fbo) f->glDeleteFramebuffers(1,&m_fbo);
            if (m_writeFbo) f->glDeleteFramebuffers(1,&m_writeFbo);
            if (m_texture) f->glDeleteTextures(1,&m_texture);
            if (m_projectionTexture) f->glDeleteTextures(1,&m_projectionTexture);
            if (m_projectionFbo) f->glDeleteFramebuffers(1,&m_projectionFbo);
        }
        if (m_surface!=EGL_NO_SURFACE && eglDestroySurface) eglDestroySurface(m_display,m_surface);
    }

    bool update(KisOpenGLImageTextures *textures,const KisGpuBrush::AnglePresentation &frame)
    {
        if (!frame.shareHandle) return true;
        auto *f=QOpenGLContext::currentContext()->extraFunctions();
        GLint oldFbo=0,oldDrawFbo=0,oldTexture=0,oldActive=0;
        f->glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&oldFbo);
        f->glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&oldDrawFbo);
        f->glGetIntegerv(GL_ACTIVE_TEXTURE,&oldActive);
        f->glActiveTexture(GL_TEXTURE0);
        f->glGetIntegerv(GL_TEXTURE_BINDING_2D,&oldTexture);
        if (m_handle!=frame.shareHandle) {
            if (m_surface!=EGL_NO_SURFACE) eglDestroySurface(m_display,m_surface);
            const EGLint attributes[]={EGL_WIDTH,1024,EGL_HEIGHT,4096,EGL_TEXTURE_FORMAT,EGL_TEXTURE_RGBA,
                EGL_TEXTURE_TARGET,EGL_TEXTURE_2D,EGL_NONE};
            // RGBA8 and BGRA8 configs have identical component sizes. ANGLE
            // requires the config's actual DXGI format to match the resource,
            // so accepting the first eglChooseConfig result is insufficient.
            for (const auto config:m_configs) {
                m_surface=eglCreatePbufferFromClientBuffer(m_display,EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE,
                    frame.shareHandle,config,attributes);
                if (m_surface!=EGL_NO_SURFACE) break;
            }
            if (m_surface==EGL_NO_SURFACE) {
                qWarning()<<"GPU canvas: EGL shared surface creation failed"<<Qt::hex<<eglGetError();
                f->glActiveTexture(oldActive);
                return false;
            }
            m_handle=frame.shareHandle;
        }
        if (!m_texture) {
            f->glGenTextures(1,&m_texture);
            f->glGenFramebuffers(1,&m_fbo);
            f->glGenFramebuffers(1,&m_writeFbo);
        }
        f->glBindTexture(GL_TEXTURE_2D,m_texture);
        if (!eglBindTexImage(m_display,m_surface,EGL_BACK_BUFFER)) {
            qWarning()<<"GPU canvas: EGL shared texture binding failed"<<Qt::hex<<eglGetError();
            f->glBindTexture(GL_TEXTURE_2D,oldTexture);
            f->glActiveTexture(oldActive);
            return false;
        }
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER,m_fbo);
        f->glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,m_texture,0);
        const auto status=f->glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
        const bool complete=status==GL_FRAMEBUFFER_COMPLETE;
        if (!complete) qWarning()<<"GPU canvas: incomplete imported framebuffer"<<Qt::hex<<status<<f->glGetError();
        const bool scissor=f->glIsEnabled(GL_SCISSOR_TEST);
        f->glDisable(GL_SCISSOR_TEST);
        bool projected=true;
        if (complete && !frame.tiles.empty()) {
            KoColor bg=textures->image()->defaultProjectionColor();
            bg.convertTo(textures->image()->colorSpace());
            const auto *pixel=bg.data();
            const QVector4D background(pixel[2]/255.0f,pixel[1]/255.0f,pixel[0]/255.0f,pixel[3]/255.0f);
            if (background.w()>0.0f) projected=projectBackground(f,frame.tiles,background);
        }
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER,m_writeFbo);
        if (complete && projected && !frame.tiles.empty()) textures->updateFromGpuBrushTexture(frame.tiles);
        const auto copyError=f->glGetError();
        if (copyError!=GL_NO_ERROR) qWarning()<<"GPU canvas: shared texture copy failed"<<Qt::hex<<copyError;
        // Complete the GPU copy before releasing D3D ownership. This waits for
        // GPU work, but does not download any image data to host memory.
        f->glFinish();
        if (scissor) f->glEnable(GL_SCISSOR_TEST);
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER,oldFbo);
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER,oldDrawFbo);
        f->glBindTexture(GL_TEXTURE_2D,m_texture);
        const bool released=eglReleaseTexImage(m_display,m_surface,EGL_BACK_BUFFER);
        if (!released) qWarning()<<"GPU canvas: EGL shared texture release failed"<<Qt::hex<<eglGetError();
        f->glBindTexture(GL_TEXTURE_2D,oldTexture);
        f->glActiveTexture(oldActive);
        return complete && projected && released && copyError==GL_NO_ERROR;
    }
private:
    bool projectBackground(QOpenGLExtraFunctions *f,const std::vector<KisGpuBrush::PresentationTile> &tiles,const QVector4D &background)
    {
        if (!m_projection.isLinked()) {
            const char *vertex=R"(#version 300 es
void main() {
    vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
    gl_Position=vec4(p*2.0-1.0,0.0,1.0);
})";
            const char *fragment=R"(#version 300 es
precision highp float;
uniform sampler2D layerPixels;
uniform vec4 background;
out vec4 colour;
void main() {
    vec4 s=texelFetch(layerPixels,ivec2(gl_FragCoord.xy),0);
    float a=s.a+background.a*(1.0-s.a);
    vec3 rgb=s.rgb*s.a+background.rgb*background.a*(1.0-s.a);
    colour=vec4(a>0.0 ? rgb/a : vec3(0.0),a);
})";
            if (!m_projection.addShaderFromSourceCode(QOpenGLShader::Vertex,vertex) ||
                !m_projection.addShaderFromSourceCode(QOpenGLShader::Fragment,fragment) || !m_projection.link()) {
                qWarning()<<"GPU canvas background shader:"<<m_projection.log();
                return false;
            }
            f->glGenTextures(1,&m_projectionTexture);
            f->glBindTexture(GL_TEXTURE_2D,m_projectionTexture);
            f->glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,1024,4096);
            f->glGenFramebuffers(1,&m_projectionFbo);
            f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER,m_projectionFbo);
            f->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,m_projectionTexture,0);
        }
        GLint viewport[4]={},scissorBox[4]={},program=0;
        GLboolean colourMask[4]={};
        f->glGetIntegerv(GL_VIEWPORT,viewport);
        f->glGetIntegerv(GL_SCISSOR_BOX,scissorBox);
        f->glGetIntegerv(GL_CURRENT_PROGRAM,&program);
        f->glGetBooleanv(GL_COLOR_WRITEMASK,colourMask);
        const GLenum states[]={GL_BLEND,GL_DEPTH_TEST,GL_STENCIL_TEST,GL_CULL_FACE};
        GLboolean enabled[4]={};
        for (int i=0;i<4;++i) { enabled[i]=f->glIsEnabled(states[i]); f->glDisable(states[i]); }
        f->glColorMask(true,true,true,true);
        f->glBindTexture(GL_TEXTURE_2D,m_texture);
        f->glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        f->glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER,m_projectionFbo);
        f->glViewport(0,0,1024,4096);
        f->glEnable(GL_SCISSOR_TEST);
        m_projection.bind();
        m_projection.setUniformValue("layerPixels",0);
        m_projection.setUniformValue("background",background);
        for (const auto &tile:tiles) {
            f->glScissor(tile.textureX,tile.textureY,tile.width,tile.height);
            f->glDrawArrays(GL_TRIANGLES,0,3);
        }
        f->glUseProgram(program);
        f->glDisable(GL_SCISSOR_TEST);
        f->glScissor(scissorBox[0],scissorBox[1],scissorBox[2],scissorBox[3]);
        f->glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
        f->glColorMask(colourMask[0],colourMask[1],colourMask[2],colourMask[3]);
        for (int i=0;i<4;++i) if (enabled[i]) f->glEnable(states[i]);
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER,m_projectionFbo);
        return true;
    }
    QLibrary m_egl{QStringLiteral("libEGL")};
    EGLDisplay m_display=EGL_NO_DISPLAY;
    std::vector<EGLConfig> m_configs;
    EGLSurface m_surface=EGL_NO_SURFACE;
    void *m_handle=nullptr;
    GLuint m_texture=0,m_fbo=0,m_writeFbo=0;
    GLuint m_projectionTexture=0,m_projectionFbo=0;
    QOpenGLShaderProgram m_projection;
    decltype(&::eglGetCurrentDisplay) eglGetCurrentDisplay=nullptr;
    decltype(&::eglGetError) eglGetError=nullptr;
    decltype(&::eglGetProcAddress) eglGetProcAddress=nullptr;
    decltype(&::eglChooseConfig) eglChooseConfig=nullptr;
    decltype(&::eglCreatePbufferFromClientBuffer) eglCreatePbufferFromClientBuffer=nullptr;
    decltype(&::eglDestroySurface) eglDestroySurface=nullptr;
    decltype(&::eglBindTexImage) eglBindTexImage=nullptr;
    decltype(&::eglReleaseTexImage) eglReleaseTexImage=nullptr;
};
#endif
#endif
