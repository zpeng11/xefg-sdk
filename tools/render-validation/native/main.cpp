// XefgRenderValidation: a standalone raster-scene capture tool.
//
// This executable uses only the public XeSS-FG swap-chain API.  Unlike the
// older capture probe it has no decoder relay, dispatch/trace instrumentation,
// provider-family detection or model-catalog dump.  It renders a deterministic
// DX12 scene through renderer.h, submits the same GPU resources that were
// rendered to the provider, captures every native present through the
// provider's own present queue, and writes evidence manifests consumed by
// tools/render-validation/validate.py, rendered_pairing.py and replay.py.
#include "common.h"

#include "manifest.h"
#include "renderer.h"
#include "swapchain_capture.h"
#include "xefg_runtime.h"

#include <Windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace harness;
using namespace render_validation;

namespace
{
constexpr uint32_t kWidth = kColorWidth;
constexpr uint32_t kHeight = kColorHeight;
constexpr uint32_t kMotionWidth = kGeometryWidth;
constexpr uint32_t kMotionHeight = kGeometryHeight;
constexpr const char* kRendererVersion = "xefg.render_validation.renderer/1";

struct Options
{
    std::string xessDirectory;
    std::string outputDirectory;
    std::string adapterFilter;
    std::string scene = "translation";
    std::string shaderPath;
    uint32_t frames = kMeasuredFrames;
    uint32_t warmup = kWarmupFrames;
    uint32_t pacingMilliseconds = 16;
    bool renderOnly = false;
    bool warp = false;
    bool debugLayer = false;
    bool tagInterpolated = true;
};

std::wstring Widen(const std::string& text)
{
    if (text.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), count);
    return result;
}

std::string Narrow(const std::wstring& text)
{
    if (text.empty()) return {};
    const int count = WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0,
        nullptr, nullptr);
    std::string result(static_cast<size_t>(count), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), count, nullptr, nullptr);
    return result;
}

bool EqualsIgnoreCase(const std::string& text, const std::string& lower)
{
    if (text.size() != lower.size()) return false;
    for (size_t i = 0; i < text.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(text[i])) != lower[i]) return false;
    return true;
}

bool ContainsIgnoreCase(const std::string& text, const std::string& needle)
{
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return lower.find(needle) != std::string::npos;
}

std::filesystem::path ExecutableDirectory()
{
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
}

void Usage()
{
    std::puts(
        "XefgRenderValidation --out <new-dir> [options]\n"
        "  --xess-dir <dir>        folder containing libxess_fg.dll/libxell.dll\n"
        "  --scene NAME            translation|occlusion|camera|cut (default translation)\n"
        "  --frames N              measured frames after warmup (default 8)\n"
        "  --warmup N              warmup frames (default 4)\n"
        "  --pacing-ms N           application frame pacing (default 16)\n"
        "  --render-only           render and read back without loading XeFG\n"
        "  --warp                  use the WARP adapter (render-only only)\n"
        "  --debug                 enable the D3D12 debug layer\n"
        "  --tag-interpolated      enable official generated-frame tagging (default)\n"
        "  --no-tag                disable official generated-frame tagging\n"
        "  --adapter <substring>   pick a DXGI adapter by name\n"
        "  --shader <file|dir>     override shaders/scene.hlsl next to the executable\n");
}

bool ParseOptions(int argc, char** argv, Options& options, std::string& error)
{
    const auto value = [&](int& index, const std::string& name) -> std::string {
        if (index + 1 >= argc)
        {
            error = "missing value for " + name;
            return {};
        }
        return argv[++index];
    };
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") { Usage(); return false; }
        if (argument == "--xess-dir") options.xessDirectory = value(i, argument);
        else if (argument == "--out") options.outputDirectory = value(i, argument);
        else if (argument == "--adapter") options.adapterFilter = value(i, argument);
        else if (argument == "--scene") options.scene = value(i, argument);
        else if (argument == "--shader") options.shaderPath = value(i, argument);
        else if (argument == "--frames") options.frames = std::stoul(value(i, argument));
        else if (argument == "--warmup") options.warmup = std::stoul(value(i, argument));
        else if (argument == "--pacing-ms") options.pacingMilliseconds = std::stoul(value(i, argument));
        else if (argument == "--render-only") options.renderOnly = true;
        else if (argument == "--warp") options.warp = true;
        else if (argument == "--debug") options.debugLayer = true;
        else if (argument == "--tag-interpolated") options.tagInterpolated = true;
        else if (argument == "--no-tag") options.tagInterpolated = false;
        else
        {
            error = "unknown argument: " + argument;
            return false;
        }
        if (!error.empty()) return false;
    }
    if (options.outputDirectory.empty())
    {
        error = "--out is required";
        return false;
    }
    if (!IsKnownScene(options.scene))
    {
        error = "--scene expects translation, occlusion, camera, or cut";
        return false;
    }
    if (options.frames < 1 || options.frames > 65532)
    {
        error = "--frames must be in 1..65532";
        return false;
    }
    if (options.warmup + options.frames > 65536)
    {
        error = "warmup + frames must not exceed 65536";
        return false;
    }
    if (options.renderOnly && options.warp == false && options.adapterFilter.empty())
    {
        // A hardware adapter is the normal render-only source too; no error.
    }
    if (!options.renderOnly && options.warp)
    {
        error = "XeFG frame generation requires a hardware adapter; use --render-only with --warp";
        return false;
    }
    if (options.xessDirectory.empty()) options.xessDirectory = Narrow(ExecutableDirectory().wstring());
    return true;
}

std::filesystem::path ResolveShaderPath(const Options& options)
{
    std::filesystem::path path;
    if (!options.shaderPath.empty())
        path = Widen(options.shaderPath);
    else
        path = ExecutableDirectory() / L"shaders" / L"scene.hlsl";
    if (std::filesystem::is_directory(path))
        path /= L"scene.hlsl";
    if (!std::filesystem::exists(path))
    {
        // Also accept a shader copied directly beside the executable.
        const std::filesystem::path beside = ExecutableDirectory() / L"scene.hlsl";
        if (std::filesystem::exists(beside)) path = beside;
    }
    return path;
}

bool PrepareOutput(const std::filesystem::path& directory, std::string& error)
{
    std::error_code code;
    if (std::filesystem::exists(directory, code))
    {
        if (!std::filesystem::is_directory(directory, code))
        {
            error = "--out exists and is not a directory: " + directory.string();
            return false;
        }
        if (!std::filesystem::is_empty(directory, code))
        {
            error = "--out must be a new or empty directory: " + directory.string();
            return false;
        }
    }
    std::filesystem::create_directories(directory, code);
    if (code)
    {
        error = "could not create --out: " + code.message();
        return false;
    }
    return true;
}

bool ParseSceneKind(const std::string& name, SceneKind& scene)
{
    return SceneFromName(name, scene);
}

std::string FrameFile(const char* prefix, uint32_t frame, const char* extension)
{
    char text[64]{};
    std::snprintf(text, sizeof(text), "%s_%05u.%s", prefix, frame, extension);
    return text;
}

bool ReadbackChecked(ID3D12Resource* resource, size_t expectedBytes, std::vector<uint8_t>& bytes,
    const char* what, std::string& error)
{
    if (!resource)
    {
        error = std::string(what) + " resource is null";
        return false;
    }
    std::string readbackError;
    if (!Readback(resource, bytes, readbackError))
    {
        error = std::string(what) + " readback failed: " + readbackError;
        return false;
    }
    if (bytes.size() != expectedBytes)
    {
        std::ostringstream text;
        text << what << " readback returned " << bytes.size() << " bytes, expected " << expectedBytes;
        error = text.str();
        return false;
    }
    return true;
}

bool WriteFrameInputs(const std::filesystem::path& directory, uint32_t frame, Renderer& renderer,
    std::string& error)
{
    struct Item
    {
        const char* prefix;
        const char* extension;
        ID3D12Resource* resource;
        size_t bytes;
        const char* label;
    };
    const size_t colorBytes = static_cast<size_t>(kWidth) * kHeight * 4;
    const size_t geometryBytes = static_cast<size_t>(kMotionWidth) * kMotionHeight * 4;
    const Item items[] = {
        {"color", "bgra", renderer.Color(), colorBytes, "color"},
        {"ui", "bgra", renderer.Ui(), colorBytes, "ui"},
        {"backbuffer", "bgra", renderer.Backbuffer(), colorBytes, "backbuffer"},
        {"depth", "r32f", renderer.Depth(), geometryBytes, "depth"},
        {"mv", "rg16f", renderer.Motion(), geometryBytes, "motion"},
    };
    for (const auto& item : items)
    {
        std::vector<uint8_t> bytes;
        if (!ReadbackChecked(item.resource, item.bytes, bytes, item.label, error)) return false;
        if (!WriteRawFile(directory / FrameFile(item.prefix, frame, item.extension), bytes, error))
            return false;
    }
    return true;
}

void Barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
    D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    list->ResourceBarrier(1, &barrier);
}

bool SubmitRecorded(WarpFixture& fixture, std::string& error)
{
    if (FAILED(fixture.commands->Close()))
    {
        error = "command list close failed";
        return false;
    }
    ID3D12CommandList* lists[]{fixture.commands.Get()};
    fixture.queue->ExecuteCommandLists(1, lists);
    if (!fixture.SignalAndWait())
    {
        error = "GPU wait timed out";
        return false;
    }
    return true;
}

int32_t TagResource(oracle::Runtime& runtime, xefg_swapchain_handle_t fg,
    ID3D12GraphicsCommandList* list, uint32_t frame, xefg_swapchain_resource_type_t type,
    ID3D12Resource* resource, uint32_t width, uint32_t height)
{
    xefg_swapchain_d3d12_resource_data_t data{};
    data.type = type;
    data.validity = XEFG_SWAPCHAIN_RV_ONLY_NOW;
    data.resourceBase = {0, 0};
    data.resourceSize = {width, height};
    data.pResource = resource;
    data.incomingState = D3D12_RESOURCE_STATE_COPY_SOURCE;
    return static_cast<int32_t>(runtime.xefgSwapChainD3D12TagFrameResource(fg, list, frame, &data));
}

// Copies the renderer's current backbuffer into the provider proxy swap chain,
// then records the four public input tags on the same command list.  The caller
// has already rendered and read back the same resources.
bool CopyBackbufferAndTag(WarpFixture& fixture, IDXGISwapChain3* proxy, Renderer& renderer,
    oracle::Runtime& runtime, xefg_swapchain_handle_t fg, uint32_t frame,
    std::array<int32_t, 6>& submissionResults, std::string& error)
{
    ComPtr<ID3D12Resource> target;
    if (FAILED(proxy->GetBuffer(proxy->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&target))) || !target)
    {
        error = "proxy back buffer unavailable";
        return false;
    }
    const D3D12_RESOURCE_DESC targetDesc = target->GetDesc();
    if (targetDesc.Format != kColorFormat || targetDesc.Width != kWidth || targetDesc.Height != kHeight
        || targetDesc.DepthOrArraySize != 1)
    {
        error = "proxy back buffer is not a 640x360 BGRA8 texture";
        return false;
    }
    if (!fixture.SubmitAndWait())
    {
        error = "command list reset failed";
        return false;
    }
    Barrier(fixture.commands.Get(), target.Get(), D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_COPY_DEST);
    fixture.commands->CopyResource(target.Get(), renderer.Backbuffer());
    Barrier(fixture.commands.Get(), target.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PRESENT);

    submissionResults[0] = TagResource(runtime, fg, fixture.commands.Get(), frame,
        XEFG_SWAPCHAIN_RES_DEPTH, renderer.Depth(), kMotionWidth, kMotionHeight);
    submissionResults[1] = TagResource(runtime, fg, fixture.commands.Get(), frame,
        XEFG_SWAPCHAIN_RES_MOTION_VECTOR, renderer.Motion(), kMotionWidth, kMotionHeight);
    submissionResults[2] = TagResource(runtime, fg, fixture.commands.Get(), frame,
        XEFG_SWAPCHAIN_RES_HUDLESS_COLOR, renderer.Color(), kWidth, kHeight);
    submissionResults[3] = TagResource(runtime, fg, fixture.commands.Get(), frame,
        XEFG_SWAPCHAIN_RES_UI, renderer.Ui(), kWidth, kHeight);
    return SubmitRecorded(fixture, error);
}

struct SetupResults
{
    int32_t createContext = -1;
    int32_t createXell = -1;
    int32_t init = -1;
    int32_t getSwapChain = -1;
    int32_t setNumInterpolatedFrames = -1;
    int32_t setEnabled = -1;
    int32_t setUiCompositionState = -1;
    int32_t enableOnlyInterpolation = -1;
    int32_t enableTagInterpolated = -1;
    int32_t getProperties = -1;
    int32_t destroy = -1;
};

void CALLBACK LogCallback(const char* message, xefg_swapchain_logging_level_t level, void* userData)
{
    static_cast<oracle::Runtime*>(userData)->AppendLog(message, static_cast<int>(level));
}

std::string LuidString(const LUID& luid)
{
    char text[40]{};
    std::snprintf(text, sizeof(text), "%08lx:%08lx",
        static_cast<unsigned long>(luid.HighPart), static_cast<unsigned long>(luid.LowPart));
    return text;
}

const char* MessageSeverityName(D3D12_MESSAGE_SEVERITY severity)
{
    switch (severity)
    {
    case D3D12_MESSAGE_SEVERITY_CORRUPTION: return "CORRUPTION";
    case D3D12_MESSAGE_SEVERITY_ERROR: return "ERROR";
    case D3D12_MESSAGE_SEVERITY_WARNING: return "WARNING";
    case D3D12_MESSAGE_SEVERITY_INFO: return "INFO";
    default: return "MESSAGE";
    }
}

bool WriteDebugMessages(const std::filesystem::path& path, ID3D12Device* device, bool debugLayer,
    std::string& error)
{
    struct Message
    {
        std::string severity;
        int32_t id = 0;
        std::string description;
    };
    std::vector<Message> errors;
    uint64_t errorCount = 0;
    uint64_t corruptionCount = 0;
    uint64_t warningCount = 0;
    uint64_t totalCount = 0;
    ComPtr<ID3D12InfoQueue> info;
    if (device && SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&info))) && info)
    {
        totalCount = info->GetNumStoredMessages();
        for (UINT64 index = 0; index < totalCount; ++index)
        {
            SIZE_T bytes = 0;
            if (FAILED(info->GetMessage(index, nullptr, &bytes)) || !bytes) continue;
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(info->GetMessage(index, message, &bytes))) continue;
            if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR) ++errorCount;
            else if (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) ++corruptionCount;
            else if (message->Severity == D3D12_MESSAGE_SEVERITY_WARNING) ++warningCount;
            if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR
                || message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                || message->Severity == D3D12_MESSAGE_SEVERITY_WARNING)
            {
                Message record;
                record.severity = MessageSeverityName(message->Severity);
                record.id = static_cast<int32_t>(message->ID);
                record.description = message->pDescription ? message->pDescription : "";
                errors.push_back(std::move(record));
            }
        }
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    file << "{\n";
    file << "  \"schema\": \"xefg.debug_messages/1\",\n";
    file << "  \"debug_layer\": " << (debugLayer ? "true" : "false") << ",\n";
    file << "  \"total_count\": " << totalCount << ",\n";
    file << "  \"error_count\": " << errorCount << ",\n";
    file << "  \"corruption_count\": " << corruptionCount << ",\n";
    file << "  \"warning_count\": " << warningCount << ",\n";
    file << "  \"messages\": [\n";
    for (size_t index = 0; index < errors.size(); ++index)
    {
        file << "    {\"severity\": \"" << errors[index].severity << "\", \"id\": "
             << errors[index].id << ", \"description\": \"" << JsonEscape(errors[index].description)
             << "\"}" << (index + 1 < errors.size() ? "," : "") << '\n';
    }
    file << "  ]\n}\n";
    file.close();
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    if (errorCount || corruptionCount)
    {
        error = "D3D12 validation reported errors; see " + path.string();
        return false;
    }
    return true;
}

std::string ResetFramesJson(SceneKind scene, uint32_t warmup, uint32_t frames)
{
    std::ostringstream text;
    text << "[0";
    if (scene == SceneKind::Cut && kCutFrame < warmup + frames)
        text << ", " << kCutFrame;
    text << "]";
    return text.str();
}

bool WriteMetaJson(const std::filesystem::path& path, const Options& options, SceneKind scene,
    bool warp, const std::string& adapterName, const DXGI_ADAPTER_DESC1& adapterDescription,
    const std::string& luid, const std::string& moduleHash, const std::string& shaderSource,
    const std::string& shaderHash, const SetupResults& setup, uint32_t maxInterpolatedFrames,
    bool initializationFallback, int64_t qpcFrequency, std::string& error)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    file << "{\n";
    file << "  \"schema\": \"xefg.render_validation_meta/1\",\n";
    file << "  \"tool\": \"XefgRenderValidation\",\n";
    file << "  \"adapter\": \"" << JsonEscape(adapterName) << "\",\n";
    file << "  \"vendor_id\": " << adapterDescription.VendorId << ",\n";
    file << "  \"device_id\": " << adapterDescription.DeviceId << ",\n";
    file << "  \"sub_sys_id\": " << adapterDescription.SubSysId << ",\n";
    file << "  \"revision\": " << adapterDescription.Revision << ",\n";
    file << "  \"adapter_luid\": \"" << luid << "\",\n";
    file << "  \"warp\": " << (warp ? "true" : "false") << ",\n";
    file << "  \"xefg_family\": \"public-api\",\n";
    file << "  \"xefg_dll_sha256\": \"" << moduleHash << "\",\n";
    file << "  \"renderer\": {\"version\": \"" << kRendererVersion
         << "\", \"shader_source\": \"" << JsonEscape(shaderSource)
         << "\", \"shader_sha256\": \"" << shaderHash << "\"},\n";
    file << "  \"shader_source\": \"" << JsonEscape(shaderSource) << "\",\n";
    file << "  \"shader_sha256\": \"" << shaderHash << "\",\n";
    file << "  \"qpc_frequency\": " << qpcFrequency << ",\n";
    file << "  \"configuration\": {"
         << "\"width\": " << kWidth << ", \"height\": " << kHeight
         << ", \"motion_width\": " << kMotionWidth << ", \"motion_height\": " << kMotionHeight
         << ", \"scene\": \"" << JsonEscape(options.scene) << "\", \"scene_revision\": 1"
         << ", \"ui\": true, \"ui_texture_tagged\": true, \"ui_mode\": 3"
         << ", \"ui_mode_requested\": \"hudless-uitexture\", \"ui_mode_explicit\": true"
         << ", \"max_interpolated_frames\": " << maxInterpolatedFrames
         << ", \"initialization_fallback\": " << (initializationFallback ? "true" : "false")
         << ", \"frames\": " << options.frames << ", \"warmup\": " << options.warmup
         << ", \"pacing_ms\": " << options.pacingMilliseconds
         << ", \"only_interpolation\": " << (options.renderOnly ? "false" : "true")
         << ", \"tag_interpolated\": "
         << ((!options.renderOnly && options.tagInterpolated) ? "true" : "false")
         << ", \"render_only\": " << (options.renderOnly ? "true" : "false")
         << ", \"reset_frames\": " << ResetFramesJson(scene, options.warmup, options.frames)
         << ", \"debug_layer\": " << (options.debugLayer ? "true" : "false") << "},\n";
    file << "  \"api_results\": {"
         << "\"create_context\": " << setup.createContext
         << ", \"create_xell\": " << setup.createXell
         << ", \"init\": " << setup.init
         << ", \"get_swap_chain\": " << setup.getSwapChain
         << ", \"set_num_interpolated_frames\": " << setup.setNumInterpolatedFrames
         << ", \"set_enabled\": " << setup.setEnabled
         << ", \"set_ui_composition_state\": " << setup.setUiCompositionState
         << ", \"enable_only_interpolation\": " << setup.enableOnlyInterpolation
         << ", \"enable_tag_interpolated\": " << setup.enableTagInterpolated
         << ", \"get_properties\": " << setup.getProperties
         << ", \"destroy\": " << setup.destroy << "}\n";
    file << "}\n";
    file.flush();
    if (!file.good())
    {
        error = "could not write " + path.string();
        return false;
    }
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    LARGE_INTEGER qpcFrequency{};
    QueryPerformanceFrequency(&qpcFrequency);
    AddVectoredExceptionHandler(1, [](EXCEPTION_POINTERS* info) -> LONG {
        if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION
            && info->ExceptionRecord->ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION
            && info->ExceptionRecord->ExceptionCode != EXCEPTION_STACK_OVERFLOW)
            return EXCEPTION_CONTINUE_SEARCH;
        const auto address = reinterpret_cast<uintptr_t>(info->ExceptionRecord->ExceptionAddress);
        HMODULE module = nullptr;
        char path[MAX_PATH]{};
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(address), &module);
        GetModuleFileNameA(module, path, MAX_PATH);
        std::fprintf(stderr, "FATAL 0x%08lx at %p (%s+0x%llx)\n", info->ExceptionRecord->ExceptionCode,
            reinterpret_cast<void*>(address), path,
            static_cast<unsigned long long>(address - reinterpret_cast<uintptr_t>(module)));
        return EXCEPTION_CONTINUE_SEARCH;
    });

    Options options;
    std::string error;
    if (!ParseOptions(argc, argv, options, error))
    {
        if (!error.empty()) std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    SceneKind sceneKind = SceneKind::Translation;
    if (!ParseSceneKind(options.scene, sceneKind))
    {
        std::fprintf(stderr, "unsupported scene: %s\n", options.scene.c_str());
        return 2;
    }

    const std::filesystem::path outputDirectory = Widen(options.outputDirectory);
    if (!PrepareOutput(outputDirectory, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const std::filesystem::path applicationInputsDirectory = outputDirectory / L"application_inputs";
    const std::filesystem::path nativeMidpointDirectory = outputDirectory / L"native_mid";
    std::error_code filesystemError;
    std::filesystem::create_directories(applicationInputsDirectory, filesystemError);
    std::filesystem::create_directories(nativeMidpointDirectory, filesystemError);
    if (!std::filesystem::is_directory(applicationInputsDirectory)
        || !std::filesystem::is_directory(nativeMidpointDirectory))
    {
        std::fprintf(stderr, "could not create output subdirectories\n");
        return 1;
    }

    const std::filesystem::path shaderPath = ResolveShaderPath(options);
    const std::string shaderSource = shaderPath.generic_string();
    const std::string shaderHash = Sha256File(shaderPath);
    if (shaderHash.empty())
    {
        std::fprintf(stderr, "shader source is missing or unreadable: %s\n", shaderSource.c_str());
        return 1;
    }

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
    {
        std::fprintf(stderr, "CreateDXGIFactory1 failed\n");
        return 1;
    }
    ComPtr<IDXGIAdapter1> selected;
    DXGI_ADAPTER_DESC1 adapterDescription{};
    if (!options.warp)
    {
        for (UINT index = 0;; ++index)
        {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 description{};
            if (FAILED(adapter->GetDesc1(&description))) continue;
            if (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (!options.adapterFilter.empty()
                && !ContainsIgnoreCase(Narrow(description.Description), options.adapterFilter))
                continue;
            selected = adapter;
            adapterDescription = description;
            break;
        }
        if (!selected)
        {
            std::fprintf(stderr, "no hardware adapter matched\n");
            return 1;
        }
        std::printf("using adapter: %s\n", Narrow(adapterDescription.Description).c_str());
    }
    else
    {
        ComPtr<IDXGIFactory4> factory4;
        ComPtr<IDXGIAdapter> warpAdapter;
        if (SUCCEEDED(factory.As(&factory4))) factory4->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter));
        ComPtr<IDXGIAdapter1> warpAdapter1;
        if (warpAdapter) warpAdapter.As(&warpAdapter1);
        if (warpAdapter1) warpAdapter1->GetDesc1(&adapterDescription);
        std::printf("using WARP adapter\n");
    }

    WarpFixture fixture;
    if (!fixture.Initialize(options.debugLayer, selected.Get()) || !fixture.factory)
    {
        std::fprintf(stderr, "D3D12 device/queue initialization failed\n");
        return 1;
    }
    TestWindow window(L"XefgRenderValidationWindow", static_cast<LONG>(kWidth), static_cast<LONG>(kHeight));
    if (!window.window)
    {
        std::fprintf(stderr, "window creation failed\n");
        return 1;
    }
    window.Show();
    window.Pump(2);

    Renderer renderer;
    if (!renderer.Initialize(fixture.device.Get(), fixture.queue.Get(), shaderPath, error))
    {
        std::fprintf(stderr, "renderer initialization failed: %s\n", error.c_str());
        return 1;
    }
    if (!renderer.Ready())
    {
        std::fprintf(stderr, "renderer reported not ready after initialization\n");
        return 1;
    }

    SetupResults setup;
    std::string moduleHash;
    xefg_swapchain_handle_t fg = nullptr;
    xell_context_handle_t xell = nullptr;
    ComPtr<IDXGISwapChain3> proxy;
    std::shared_ptr<oracle::CaptureContext> capture;
    ComPtr<IDXGIFactory2> captureFactory;
    oracle::Runtime runtime;
    bool initializationFallback = false;
    uint32_t maxInterpolatedFrames = 0;
    int32_t onlyInterpolationResult = 0;
    int32_t tagInterpolatedResult = 0;
    const bool tagRequested = !options.renderOnly && options.tagInterpolated;

    if (!options.renderOnly)
    {
        if (!runtime.Load(Widen(options.xessDirectory), error))
        {
            std::fprintf(stderr, "XeFG runtime load failed: %s\n", error.c_str());
            return 1;
        }
        moduleHash = oracle::ModuleFileSha256(runtime.fg);
        std::printf("provider sha256: %s\n", moduleHash.c_str());

        capture = std::make_shared<oracle::CaptureContext>();
        if (!capture->Initialize(fixture.device.Get(), fixture.queue.Get(),
            (outputDirectory / L"captures").wstring(), error))
        {
            std::fprintf(stderr, "capture initialization failed: %s\n", error.c_str());
            return 1;
        }

        DXGI_SWAP_CHAIN_DESC1 chainDescription{};
        chainDescription.Width = kWidth;
        chainDescription.Height = kHeight;
        chainDescription.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        chainDescription.SampleDesc.Count = 1;
        chainDescription.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        chainDescription.BufferCount = 3;
        chainDescription.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        chainDescription.Scaling = DXGI_SCALING_STRETCH;
        chainDescription.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;

        captureFactory = oracle::WrapFactory(fixture.factory.Get(), capture);
        if (!captureFactory)
        {
            std::fprintf(stderr, "capture factory wrapping failed\n");
            return 1;
        }

        setup.createContext = static_cast<int32_t>(
            runtime.xefgSwapChainD3D12CreateContext(fixture.device.Get(), &fg));
        if (setup.createContext != int32_t(XEFG_SWAPCHAIN_RESULT_SUCCESS) || !fg)
        {
            std::fprintf(stderr, "xefgSwapChainD3D12CreateContext failed: %d\n", setup.createContext);
            return 1;
        }
        runtime.xefgSwapChainSetLoggingCallback(fg, XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING, LogCallback, &runtime);

        setup.createXell = static_cast<int32_t>(runtime.xellD3D12CreateContext(fixture.device.Get(), &xell));
        if (setup.createXell != 0)
        {
            std::fprintf(stderr, "XeLL creation failed: %d\n", setup.createXell);
            return 1;
        }
        xell_sleep_params_t sleep{};
        sleep.bLowLatencyMode = 1;
        runtime.xellSetSleepMode(xell, &sleep);
        const auto latencyResult = runtime.xefgSwapChainSetLatencyReduction(fg, xell);
        if (latencyResult != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        {
            std::fprintf(stderr, "XeFG latency context failed: %d\n", int(latencyResult));
            return 1;
        }

        xefg_swapchain_d3d12_init_params_t initParameters{};
        initParameters.pApplicationSwapChain = nullptr;
        initParameters.initFlags = 0;
        initParameters.maxInterpolatedFrames = 1;
        initParameters.uiMode = XEFG_SWAPCHAIN_UI_MODE_HUDLESS_UITEXTURE;
        setup.init = static_cast<int32_t>(runtime.xefgSwapChainD3D12InitFromSwapChainDesc(fg, window.window,
            &chainDescription, nullptr, fixture.queue.Get(), captureFactory.Get(), &initParameters));
        if (setup.init != int32_t(XEFG_SWAPCHAIN_RESULT_SUCCESS))
        {
            initializationFallback = true;
            initParameters.maxInterpolatedFrames =
                XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES;
            setup.init = static_cast<int32_t>(runtime.xefgSwapChainD3D12InitFromSwapChainDesc(fg, window.window,
                &chainDescription, nullptr, fixture.queue.Get(), captureFactory.Get(), &initParameters));
        }
        if (setup.init != int32_t(XEFG_SWAPCHAIN_RESULT_SUCCESS))
        {
            std::fprintf(stderr, "XeFG initialization failed: %d\n", setup.init);
            return 1;
        }
        maxInterpolatedFrames = initParameters.maxInterpolatedFrames;

        setup.getSwapChain = static_cast<int32_t>(
            runtime.xefgSwapChainD3D12GetSwapChainPtr(fg, IID_PPV_ARGS(&proxy)));
        if (setup.getSwapChain != int32_t(XEFG_SWAPCHAIN_RESULT_SUCCESS) || !proxy)
        {
            std::fprintf(stderr, "xefgSwapChainD3D12GetSwapChainPtr failed: %d\n", setup.getSwapChain);
            return 1;
        }
        setup.setNumInterpolatedFrames =
            static_cast<int32_t>(runtime.xefgSwapChainSetNumInterpolatedFrames(fg, 1));
        setup.setEnabled = static_cast<int32_t>(runtime.xefgSwapChainSetEnabled(fg, 1));
        setup.setUiCompositionState = static_cast<int32_t>(runtime.xefgSwapChainSetUiCompositionState(
            fg, XEFG_SWAPCHAIN_UI_COMPOSITION_STATE_ENABLED));
        onlyInterpolationResult = static_cast<int32_t>(runtime.xefgSwapChainEnableDebugFeature(
            fg, XEFG_SWAPCHAIN_DEBUG_FEATURE_SHOW_ONLY_INTERPOLATION, 1, nullptr));
        setup.enableOnlyInterpolation = onlyInterpolationResult;
        if (tagRequested)
        {
            tagInterpolatedResult = static_cast<int32_t>(runtime.xefgSwapChainEnableDebugFeature(
                fg, XEFG_SWAPCHAIN_DEBUG_FEATURE_TAG_INTERPOLATED_FRAMES, 1, nullptr));
        }
        setup.enableTagInterpolated = tagInterpolatedResult;

        xefg_swapchain_properties_t properties{};
        setup.getProperties = static_cast<int32_t>(runtime.xefgSwapChainD3D12GetProperties(
            fg, &initParameters, 0, 0, DXGI_FORMAT_UNKNOWN, &properties));
        std::printf("XeFG ready: uiMode=%d maxInterpolations=%u descriptors=%u fallback=%s\n",
            int(XEFG_SWAPCHAIN_UI_MODE_HUDLESS_UITEXTURE), properties.maxSupportedInterpolations,
            properties.requiredDescriptorCount, initializationFallback ? "yes" : "no");
    }

    const uint32_t totalFrames = options.warmup + options.frames;
    std::vector<SceneFrameRecord> sceneFrames;
    sceneFrames.reserve(totalFrames);
    std::vector<PresentStatusRecord> statuses;
    statuses.reserve(options.renderOnly ? 0 : totalFrames);
    bool apiFailure = false;
    if (!options.renderOnly
        && (setup.createContext != 0 || setup.init != 0 || setup.getSwapChain != 0
            || setup.setNumInterpolatedFrames != 0 || setup.setEnabled != 0
            || setup.setUiCompositionState != 0 || onlyInterpolationResult != 0
            || (tagRequested && tagInterpolatedResult != 0) || setup.getProperties != 0))
        apiFailure = true;

    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        const bool resetHistory = (frame == 0) || IsCutResetFrame(frame, sceneKind);
        FrameState state = MakeFrameState(sceneKind, frame, false);
        state.historyReset = resetHistory;
        FrameState midpointState = MakeFrameState(sceneKind, frame, true);
        midpointState.historyReset = resetHistory;

        if (!renderer.Render(midpointState, sceneKind, true, error))
        {
            std::fprintf(stderr, "midpoint render failed at frame %u: %s\n", frame, error.c_str());
            return 1;
        }
        const std::string midpointName = "native_mid/" + FrameFile("frame", frame, "bgra");
        std::vector<uint8_t> midpointBytes;
        if (!ReadbackChecked(renderer.Backbuffer(), static_cast<size_t>(kWidth) * kHeight * 4,
            midpointBytes, "midpoint backbuffer", error))
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        if (!WriteRawFile(nativeMidpointDirectory / FrameFile("frame", frame, "bgra"), midpointBytes, error))
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }

        if (!renderer.Render(state, sceneKind, false, error))
        {
            std::fprintf(stderr, "current render failed at frame %u: %s\n", frame, error.c_str());
            return 1;
        }
        if (!WriteFrameInputs(applicationInputsDirectory, frame, renderer, error))
        {
            std::fprintf(stderr, "input readback failed at frame %u: %s\n", frame, error.c_str());
            return 1;
        }

        SceneFrameRecord record;
        record.index = frame;
        record.hasPrevious = frame > 0;
        record.previousFrame = frame > 0 ? frame - 1 : 0;
        record.resetHistory = resetHistory;
        record.midpoint = midpointName;
        std::memcpy(record.view.data(), state.view.m, sizeof(state.view.m));
        std::memcpy(record.projection.data(), state.projection.m, sizeof(state.projection.m));
        record.time = state.time;
        record.previousTime = state.previousTime;
        for (uint32_t objectIndex = 0;
             objectIndex < std::min<uint32_t>(state.objectCount, kMaxObjects); ++objectIndex)
        {
            SceneObjectRecord object;
            object.name = state.objects[objectIndex].name;
            std::memcpy(object.world.data(), state.objects[objectIndex].world.m, sizeof(object.world));
            std::memcpy(object.previousWorld.data(), state.objects[objectIndex].previousWorld.m,
                sizeof(object.previousWorld));
            record.objects.push_back(std::move(object));
        }

        if (options.renderOnly)
        {
            sceneFrames.push_back(std::move(record));
            continue;
        }

        if (!CopyBackbufferAndTag(fixture, proxy.Get(), renderer, runtime, fg, frame,
            record.submissionResults, error))
        {
            std::fprintf(stderr, "backbuffer/tag submission failed at frame %u: %s\n", frame, error.c_str());
            return 1;
        }

        xefg_swapchain_frame_constant_data_t constants{};
        std::memcpy(constants.viewMatrix, state.view.m, sizeof(state.view.m));
        std::memcpy(constants.projectionMatrix, state.projection.m, sizeof(state.projection.m));
        constants.jitterOffsetX = 0.0f;
        constants.jitterOffsetY = 0.0f;
        constants.motionVectorScaleX = 1.0f;
        constants.motionVectorScaleY = 1.0f;
        constants.resetHistory = resetHistory ? 1u : 0u;
        constants.frameRenderTime = static_cast<float>(options.pacingMilliseconds);
        record.submissionResults[4] = static_cast<int32_t>(
            runtime.xefgSwapChainTagFrameConstants(fg, frame, &constants));
        record.submissionResults[5] = static_cast<int32_t>(
            runtime.xefgSwapChainSetPresentId(fg, frame));

        capture->SetFrameIndex(frame);
        capture->SetEnabled(frame >= options.warmup);
        LARGE_INTEGER presentStart{}, presentEnd{};
        QueryPerformanceCounter(&presentStart);
        const HRESULT presentResult = proxy->Present(0, 0);
        QueryPerformanceCounter(&presentEnd);
        xefg_swapchain_present_status_t status{};
        const auto statusResult = runtime.xefgSwapChainGetLastPresentStatus(fg, &status);
        PresentStatusRecord statusRecord;
        statusRecord.applicationFrame = frame;
        statusRecord.presentResult = static_cast<int32_t>(presentResult);
        statusRecord.statusResult = static_cast<int32_t>(statusResult);
        statusRecord.framesPresented = status.framesPresented;
        statusRecord.frameGenerationResult = static_cast<int32_t>(status.frameGenResult);
        statusRecord.isFrameGenEnabled = status.isFrameGenEnabled;
        statusRecord.presentStartQpc = static_cast<uint64_t>(presentStart.QuadPart);
        statusRecord.presentEndQpc = static_cast<uint64_t>(presentEnd.QuadPart);
        statuses.push_back(statusRecord);
        if (presentResult != S_OK || statusResult != XEFG_SWAPCHAIN_RESULT_SUCCESS
            || record.submissionResults[4] != 0 || record.submissionResults[5] != 0
            || record.submissionResults[0] != 0 || record.submissionResults[1] != 0
            || record.submissionResults[2] != 0 || record.submissionResults[3] != 0)
            apiFailure = true;
        std::printf("frame %u: present=0x%08lx framesPresented=%u result=%d\n", frame,
            static_cast<unsigned long>(presentResult), status.framesPresented, int(status.frameGenResult));
        if (options.pacingMilliseconds) Sleep(options.pacingMilliseconds);
        sceneFrames.push_back(std::move(record));
    }

    if (sceneFrames.size() != totalFrames)
    {
        std::fprintf(stderr, "internal error: incomplete frame records\n");
        return 1;
    }

    if (!WriteApplicationInputsManifest(applicationInputsDirectory, kWidth, kHeight, kMotionWidth,
        kMotionHeight, options.warmup, options.frames, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    SceneManifest sceneManifest;
    sceneManifest.name = kSceneNames[static_cast<size_t>(sceneKind)];
    sceneManifest.revision = 1;
    sceneManifest.width = kWidth;
    sceneManifest.height = kHeight;
    sceneManifest.motionWidth = kMotionWidth;
    sceneManifest.motionHeight = kMotionHeight;
    sceneManifest.warmup = options.warmup;
    sceneManifest.measuredFrames = options.frames;
    sceneManifest.shaderSource = shaderSource;
    sceneManifest.shaderSha256 = shaderHash;
    sceneManifest.rendererVersion = kRendererVersion;
    sceneManifest.frames = std::move(sceneFrames);
    if (!WriteSceneManifest(outputDirectory / L"scene.json", sceneManifest, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    bool presentsWritten = true;
    if (!options.renderOnly)
    {
        fixture.SignalAndWait();
        proxy.Reset();
        setup.destroy = static_cast<int32_t>(runtime.xefgSwapChainDestroy(fg));
        fg = nullptr;
        capture->SetEnabled(false);

        PresentsManifest presents;
        presents.onlyInterpolationRequested = true;
        presents.onlyInterpolationResult = onlyInterpolationResult;
        presents.tagInterpolatedRequested = tagRequested;
        presents.tagInterpolatedResult = tagInterpolatedResult;
        presents.qpcFrequency = qpcFrequency.QuadPart;
        presents.captureError = capture->Error();
        presents.statuses = std::move(statuses);
        for (const auto& native : capture->Captures())
        {
            const auto relative = std::filesystem::path(native.file).lexically_relative(outputDirectory);
            PresentCaptureRecord record;
            record.sequence = native.sequence;
            record.observedApplicationFrame = native.frameIndex;
            record.ordinal = native.ordinal;
            record.width = native.width;
            record.height = native.height;
            record.format = static_cast<int32_t>(native.format);
            record.file = relative.generic_string();
            record.presentStartQpc = native.presentStartQpc;
            record.presentEndQpc = native.presentEndQpc;
            record.dxgiPresentCount = native.dxgiPresentCount;
            record.dxgiPresentCountResult = native.dxgiPresentCountResult;
            record.nativePresentResult = native.nativePresentResult;
            presents.presents.push_back(std::move(record));
        }
        presentsWritten = WritePresentsManifest(outputDirectory / L"presents.json", presents, error);
        if (!presentsWritten) std::fprintf(stderr, "%s\n", error.c_str());

        {
            std::ofstream logFile(outputDirectory / L"xefg_log.txt", std::ios::binary | std::ios::trunc);
            if (logFile)
                for (const auto& line : runtime.TakeLog()) logFile << line << '\n';
        }
        if (capture) capture->SetEnabled(false);
    }

    const bool metaWritten = WriteMetaJson(outputDirectory / L"meta.json", options, sceneKind, options.warp,
        options.warp ? "Microsoft Basic Render Driver" : Narrow(adapterDescription.Description),
        adapterDescription, LuidString(fixture.device->GetAdapterLuid()), moduleHash, shaderSource,
        shaderHash, setup, maxInterpolatedFrames, initializationFallback, qpcFrequency.QuadPart, error);
    if (!metaWritten) std::fprintf(stderr, "%s\n", error.c_str());

    if (captureFactory) captureFactory.Reset();
    if (xell) runtime.xellDestroyContext(xell);
    if (!options.renderOnly) runtime.Unload();
    if (options.debugLayer)
    {
        std::string debugError;
        if (!WriteDebugMessages(outputDirectory / L"debug_messages.json", fixture.device.Get(),
            options.debugLayer, debugError))
            std::fprintf(stderr, "%s\n", debugError.c_str());
    }
    std::printf("render validation output written to %s\n", outputDirectory.string().c_str());
    if (!options.renderOnly && (!presentsWritten || !capture->Error().empty())) return 1;
    if (apiFailure)
    {
        std::fprintf(stderr, "one or more public API calls reported failure; manifests retain the results\n");
        return 1;
    }
    return 0;
}
