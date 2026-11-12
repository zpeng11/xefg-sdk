cmake_minimum_required(VERSION 3.24)
project(XefgRenderValidation LANGUAGES CXX)

# Standalone raster-scene validation harness for the public XeSS-FG swap-chain
# API.  It shares only the header fixtures and the small runtime loader with
# tools/capture; decoder relay, trace and model dump are not built. The shared
# loader's family inspection routine is never called by this executable.
if(NOT WIN32)
    message(FATAL_ERROR "XefgRenderValidation is a Windows-only project")
endif()
if(NOT MSVC)
    message(FATAL_ERROR "XefgRenderValidation requires MSVC (Visual Studio 2022)")
endif()
if(CMAKE_SIZEOF_VOID_P AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "XefgRenderValidation must be configured for x64")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(XESS_SDK_ROOT "" CACHE PATH "XeSS SDK 3.0.2 root (contains inc/xess_fg and inc/xell)")
if(NOT XESS_SDK_ROOT AND DEFINED ENV{XESS_SDK_ROOT})
    set(XESS_SDK_ROOT "$ENV{XESS_SDK_ROOT}")
endif()
if(NOT EXISTS "${XESS_SDK_ROOT}/inc/xess_fg/xefg_swapchain_d3d12.h")
    message(FATAL_ERROR "Set XESS_SDK_ROOT to the XeSS SDK 3.0.2 root")
endif()

# Optional diagnostic path.  Off by default: the normal renderer keeps the
# public-API-only source list above.  When enabled, the existing capture probe
# modules are linked in and --tensor-frame becomes available for one selected
# sequential source frame of the profiled fg78 provider.
option(XEFG_RENDER_VALIDATION_TENSOR_PROBE
    "Build the optional fg78 tensor-capture diagnostic path" OFF)

# Standalone native DX12 sampler probe.  Off by default; it has its own
# executable, source, shader and metadata contract and never links the XeSS-FG
# runtime or provider DLL.
option(XEFG_BUILD_SAMPLER_PROBE
    "Build the standalone native DX12 sampler probe" OFF)

set(CAPTURE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../capture")
set(NATIVE_INCLUDES
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CAPTURE_ROOT}/src"
    "${CAPTURE_ROOT}/include"
    "${CAPTURE_ROOT}/include/xess"
    "${XESS_SDK_ROOT}/inc"
    "${XESS_SDK_ROOT}/inc/xess_fg"
    "${XESS_SDK_ROOT}/inc/xell")

set(RENDER_VALIDATION_SOURCES
    main.cpp
    renderer.cpp
    "${CAPTURE_ROOT}/src/xefg_runtime.cpp")

set(RENDER_VALIDATION_DEFINITIONS
    UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)

if(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
    list(APPEND NATIVE_INCLUDES "${CAPTURE_ROOT}/src/probe")
    list(APPEND RENDER_VALIDATION_SOURCES
        "${CAPTURE_ROOT}/src/probe/trace_writer.cpp"
        "${CAPTURE_ROOT}/src/probe/root_signature.cpp"
        "${CAPTURE_ROOT}/src/probe/resource_map.cpp"
        "${CAPTURE_ROOT}/src/probe/dispatch_trace.cpp"
        "${CAPTURE_ROOT}/src/probe/decode_relay.cpp"
        "${CAPTURE_ROOT}/src/probe/layer_params.cpp"
        "${CAPTURE_ROOT}/src/probe/tensor_dump.cpp")
    list(APPEND RENDER_VALIDATION_DEFINITIONS XEFG_RENDER_VALIDATION_TENSOR_PROBE=1)
    message(STATUS "XefgRenderValidation: tensor-capture diagnostic path enabled")
endif()

add_executable(XefgRenderValidation ${RENDER_VALIDATION_SOURCES})
target_compile_features(XefgRenderValidation PRIVATE cxx_std_20)
target_compile_definitions(XefgRenderValidation PRIVATE ${RENDER_VALIDATION_DEFINITIONS})
target_compile_options(XefgRenderValidation PRIVATE /utf-8)
target_include_directories(XefgRenderValidation PRIVATE ${NATIVE_INCLUDES})
target_link_libraries(XefgRenderValidation PRIVATE d3d12 dxgi d3dcompiler user32 bcrypt)
set_target_properties(XefgRenderValidation PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Runtime shaders are compiled with D3DCompileFromFile.  Keep the source tree
# authoritative and copy it beside the executable after every build.
set(RENDER_SHADER_DIR "${CMAKE_CURRENT_LIST_DIR}/shaders")
add_custom_target(RenderValidationShaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${RENDER_SHADER_DIR}" "$<TARGET_FILE_DIR:XefgRenderValidation>/shaders"
    COMMENT "Copying render-validation shaders")
add_dependencies(XefgRenderValidation RenderValidationShaders)

include(CTest)
if(BUILD_TESTING)
        add_executable(XefgRenderValidationTests
            tests/renderer_tests.cpp
            renderer.cpp)
        target_compile_features(XefgRenderValidationTests PRIVATE cxx_std_20)
        target_compile_definitions(XefgRenderValidationTests PRIVATE
            UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
        target_compile_options(XefgRenderValidationTests PRIVATE /utf-8)
        target_include_directories(XefgRenderValidationTests PRIVATE ${NATIVE_INCLUDES})
        target_link_libraries(XefgRenderValidationTests PRIVATE d3d12 dxgi d3dcompiler user32 bcrypt)
        set_target_properties(XefgRenderValidationTests PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        add_test(NAME xefg_renderer COMMAND XefgRenderValidationTests)
        if(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
            add_executable(XefgDispatchTraceTests tests/dispatch_trace_tests.cpp)
            target_compile_features(XefgDispatchTraceTests PRIVATE cxx_std_20)
            target_compile_definitions(XefgDispatchTraceTests PRIVATE
                UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
            target_compile_options(XefgDispatchTraceTests PRIVATE /utf-8)
            target_include_directories(XefgDispatchTraceTests PRIVATE ${NATIVE_INCLUDES})
            target_link_libraries(XefgDispatchTraceTests PRIVATE d3d12 dxgi user32 bcrypt)
            set_target_properties(XefgDispatchTraceTests PROPERTIES
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
            add_test(NAME xefg_dispatch_trace COMMAND XefgDispatchTraceTests)
        endif()
endif()

if(XEFG_BUILD_SAMPLER_PROBE)
    add_executable(XefgSamplerProbe sampler_probe.cpp)
    target_compile_features(XefgSamplerProbe PRIVATE cxx_std_20)
    target_compile_definitions(XefgSamplerProbe PRIVATE
        UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(XefgSamplerProbe PRIVATE /utf-8)
    target_include_directories(XefgSamplerProbe PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CAPTURE_ROOT}/include")
    target_link_libraries(XefgSamplerProbe PRIVATE
        d3d12 dxgi d3dcompiler user32 bcrypt version)
    set_target_properties(XefgSamplerProbe PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_custom_target(SamplerProbeShaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_LIST_DIR}/shaders" "$<TARGET_FILE_DIR:XefgSamplerProbe>/shaders"
        COMMENT "Copying sampler-probe shaders")
    add_dependencies(XefgSamplerProbe SamplerProbeShaders)
endif()
