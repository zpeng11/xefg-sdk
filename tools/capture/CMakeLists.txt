cmake_minimum_required(VERSION 3.24)
project(XefgCapture LANGUAGES CXX)

# Standalone native capture tools for Intel's public XeSS-FG swap-chain API.
#
# XefgProbe drives a deterministic synthetic scene through the provider, relays
# its program decoder (read-only), records every dispatch with root signature
# and bound resources, optionally snapshots per-stage tensors, and captures each
# presented frame. Nothing is replaced and the model catalog is never modified.
#
# This tree intentionally contains no MFGUnlocker sources, no raw application
# export features and no XeFG replacement logic. It needs only VS 2022, the
# Windows SDK, and the XeSS SDK 3.0.2 headers/runtime supplied by the caller.

if(NOT WIN32)
    message(FATAL_ERROR "XefgCapture is a Windows-only project")
endif()
if(NOT MSVC)
    message(FATAL_ERROR "XefgCapture requires MSVC (Visual Studio 2022)")
endif()
if(CMAKE_SIZEOF_VOID_P AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "XefgCapture must be configured for x64")
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

set(CAPTURE_INCLUDES
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${CMAKE_CURRENT_LIST_DIR}/include/xess"
    "${XESS_SDK_ROOT}/inc"
    "${XESS_SDK_ROOT}/inc/xess_fg"
    "${XESS_SDK_ROOT}/inc/xell")

add_library(capture_common STATIC
    src/xefg_runtime.cpp
    src/scene.cpp
    src/model_dump.cpp)
target_compile_features(capture_common PUBLIC cxx_std_20)
target_compile_definitions(capture_common PUBLIC
    UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(capture_common PRIVATE /utf-8)
target_include_directories(capture_common PUBLIC ${CAPTURE_INCLUDES})
target_link_libraries(capture_common PUBLIC d3d12 dxgi user32 bcrypt)
set_target_properties(capture_common PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ---- probe ------------------------------------------------------------------
add_executable(XefgProbe
    src/probe_main.cpp
    src/probe/trace_writer.cpp
    src/probe/root_signature.cpp
    src/probe/resource_map.cpp
    src/probe/dispatch_trace.cpp
    src/probe/decode_relay.cpp
    src/probe/layer_params.cpp
    src/probe/tensor_dump.cpp)
target_compile_options(XefgProbe PRIVATE /utf-8)
target_link_libraries(XefgProbe PRIVATE capture_common)
set_target_properties(XefgProbe PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ---- tests ------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
    add_executable(XefgSceneTests tests/scene_tests.cpp)
    target_compile_options(XefgSceneTests PRIVATE /utf-8)
    target_link_libraries(XefgSceneTests PRIVATE capture_common)
    set_target_properties(XefgSceneTests PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_test(NAME xefg_scene COMMAND XefgSceneTests)
endif()
