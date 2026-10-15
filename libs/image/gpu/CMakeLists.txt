# Shared by brush engines and the canvas. No dependency on Qt's GL backend.
kis_add_library(kritagpubrush SHARED KisGpuBrushCompositor.cpp)
target_compile_definitions(kritagpubrush PRIVATE KRITA_GPU_BRUSH_BUILD)
target_link_libraries(kritagpubrush PRIVATE Qt${QT_MAJOR_VERSION}::Core)
if(WIN32)
    target_link_libraries(kritagpubrush PRIVATE opengl32 gdi32 user32)
endif()
install(TARGETS kritagpubrush ${INSTALL_TARGETS_DEFAULT_ARGS})
