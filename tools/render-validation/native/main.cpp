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

#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
#include "decode_relay.h"
#include "dispatch_trace.h"
#include "trace_writer.h"
#endif

#include <Windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace harness;
using namespace render_validation;

namespace
{
constexpr const char* kRendererVersion = "xefg.render_validation.renderer/2";

constexpr RenderSize kSupportedRenderSizes[] = {
    {640, 360},
    {1280, 720},
    {1920, 1080},
    {2560, 1440},
    {3840, 2160},
};

struct Options
{
    std::string xessDirectory;
    std::string outputDirectory;
    std::string adapterFilter;
    std::string scene = "translation";
    std::string shaderPath;
    RenderSize size;
    uint32_t mode = 3;
    std::optional<uint32_t> pairFrame;
    uint32_t frames = kMeasuredFrames;
    uint32_t warmup = kWarmupFrames;
    uint32_t pacingMilliseconds = 16;
    bool renderOnly = false;
    bool warp = false;
    bool debugLayer = false;
    bool tagInterpolated = true;
    bool identityFgMatrices = false; // Diagnostic control; rendering/input textures stay unchanged.
    // Diagnostic-build only: snapshot one sequential source frame.  The normal
    // build rejects these instead of silently ignoring them.
    std::optional<uint32_t> tensorFrame;
    uint32_t tensorBudgetMB = 1024;
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

bool ParseUint(const std::string& text, uint32_t& value)
{
    if (text.empty()) return false;
    uint32_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return false;
    value = parsed;
    return true;
}

bool ParseRenderSize(const std::string& text, RenderSize& size)
{
    const size_t separator = text.find_first_of("xX");
    if (separator == std::string::npos) return false;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!ParseUint(text.substr(0, separator), width)
        || !ParseUint(text.substr(separator + 1), height))
        return false;
    for (const RenderSize& supported : kSupportedRenderSizes)
    {
        if (width == supported.width && height == supported.height)
        {
            size = supported;
            return true;
        }
    }
    return false;
}

void Usage()
{
    std::puts(
        "XefgRenderValidation --out <new-dir> [options]\n"
        "  --xess-dir <dir>        folder containing libxess_fg.dll/libxell.dll\n"
        "  --scene NAME            translation|occlusion|camera|cut|depth_layers|"
        "screen_cross|camera_turn|nonrigid|transparency (default translation)\n"
        "  --mode N                UI mode 1|3|4 (default 3)\n"
        "  --size WxH              output size 640x360|1280x720|1920x1080|2560x1440|"
        "3840x2160 (default 640x360)\n"
        "  --pair-frame N          mode 1 only: one isolated source pair [N-1,N]\n"
        "  --frames N              measured frames after warmup (default 8)\n"
        "  --warmup N              warmup frames (default 4)\n"
        "  --pacing-ms N           application frame pacing (default 16)\n"
        "  --render-only           render and read back without loading XeFG\n"
        "  --warp                  use the WARP adapter (render-only only)\n"
        "  --debug                 enable the D3D12 debug layer\n"
        "  --tag-interpolated      enable official generated-frame tagging (default)\n"
        "  --no-tag                disable official generated-frame tagging\n"
        "  --identity-fg-matrices  diagnostic: submit identity matrices only to FG\n"
        "  --adapter <substring>   pick a DXGI adapter by name\n"
#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
        "  --tensor-frame N        diagnostic: snapshot one sequential source frame\n"
        "  --tensor-budget-mb N    diagnostic tensor byte budget (default 1024)\n"
#endif
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
        else if (argument == "--size")
        {
            if (!ParseRenderSize(value(i, argument), options.size))
            {
                error = "--size must be one of 640x360, 1280x720, 1920x1080, "
                    "2560x1440, 3840x2160";
                return false;
            }
        }
        else if (argument == "--mode") options.mode = std::stoul(value(i, argument));
        else if (argument == "--pair-frame") options.pairFrame = std::stoul(value(i, argument));
        else if (argument == "--frames") options.frames = std::stoul(value(i, argument));
        else if (argument == "--warmup") options.warmup = std::stoul(value(i, argument));
        else if (argument == "--pacing-ms") options.pacingMilliseconds = std::stoul(value(i, argument));
        else if (argument == "--render-only") options.renderOnly = true;
        else if (argument == "--warp") options.warp = true;
        else if (argument == "--debug") options.debugLayer = true;
        else if (argument == "--tag-interpolated") options.tagInterpolated = true;
        else if (argument == "--no-tag") options.tagInterpolated = false;
        else if (argument == "--identity-fg-matrices") options.identityFgMatrices = true;
        else if (argument == "--tensor-frame") options.tensorFrame = std::stoul(value(i, argument));
        else if (argument == "--tensor-budget-mb") options.tensorBudgetMB = std::stoul(value(i, argument));
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
        error = "--scene expects translation, occlusion, camera, cut, depth_layers, "
            "screen_cross, camera_turn, nonrigid, or transparency";
        return false;
    }
    if (options.mode != 1 && options.mode != 3 && options.mode != 4)
    {
        error = "--mode must be 1, 3, or 4";
        return false;
    }
    if (options.pairFrame)
    {
        if (options.mode != 1)
        {
            error = "--pair-frame is only valid with --mode 1";
            return false;
        }
        SceneKind pairScene = SceneKind::Translation;
        SceneFromName(options.scene, pairScene);
        if (*options.pairFrame > 65535
            || !IsValidPairFrame(pairScene, *options.pairFrame))
        {
            error = "--pair-frame must be in 1..65535 and must not cross the camera-cut "
                "discontinuity (frame 7)";
            return false;
        }
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
#if !defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
    if (options.tensorFrame)
    {
        error = "--tensor-frame requires a renderer built with "
            "XEFG_RENDER_VALIDATION_TENSOR_PROBE=ON";
        return false;
    }
#else
    if (options.tensorFrame)
    {
        // Same contract as tools/render-validation/validate.py: one sequential
        // source frame, inside the run's frames, with a positive budget.
        if (options.renderOnly)
        {
            error = "--tensor-frame is not available with --render-only";
            return false;
        }
        if (options.pairFrame)
        {
            error = "--tensor-frame requires a sequential run, not --pair-frame";
            return false;
        }
        if (*options.tensorFrame >= options.warmup + options.frames)
        {
            error = "--tensor-frame must be inside the sequential frame range";
            return false;
        }
        if (options.tensorBudgetMB == 0)
        {
            error = "--tensor-budget-mb must be positive";
            return false;
        }
    }
#endif
    if (options.renderOnly && options.warp == false && options.adapterFilter.empty())
    {
        // A hardware adapter is the normal render-only source too; no error.
    }
    if (!options.renderOnly && options.warp)
    {
        error = "XeFG frame generation requires a hardware adapter; use --render-only with --warp";
        return false;
    }
    if (options.pairFrame)
    {
        // An isolated pair is a fresh two-submission process; the wrapper may
        // still pass its usual --frames/--warmup values, which are overridden.
        options.warmup = 0;
        options.frames = 2;
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

struct ModeInfo
{
    xefg_swapchain_ui_mode_t uiMode;
    const char* requested;
    bool ui;
    bool uiTextureTagged;
    bool uiComposition;
};

ModeInfo ModeFor(uint32_t mode)
{
    switch (mode)
    {
    case 1: return {XEFG_SWAPCHAIN_UI_MODE_NONE, "none", false, false, false};
    case 4: return {XEFG_SWAPCHAIN_UI_MODE_BACKBUFFER_HUDLESS, "backbuffer-hudless",
        true, false, true};
    case 3:
    default: return {XEFG_SWAPCHAIN_UI_MODE_HUDLESS_UITEXTURE, "hudless-uitexture",
        true, true, true};
    }
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
    const RenderSize& size, std::string& error)
{
    struct Item
    {
        const char* prefix;
        const char* extension;
        ID3D12Resource* resource;
        size_t bytes;
        const char* label;
    };
    const size_t colorBytes = static_cast<size_t>(size.width) * size.height * 4;
    const size_t geometryBytes = static_cast<size_t>(size.GeometryWidth())
        * size.GeometryHeight() * 4;
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
// then records the applicable public input tags on the same command list.  The caller
// has already rendered and read back the same resources.
bool CopyBackbufferAndTag(WarpFixture& fixture, IDXGISwapChain3* proxy, Renderer& renderer,
    const RenderSize& size, oracle::Runtime& runtime, xefg_swapchain_handle_t fg, uint32_t frame,
    bool tagColor, bool tagUi, std::array<std::optional<int32_t>, 6>& submissionResults,
    std::string& error)
{
    ComPtr<ID3D12Resource> target;
    if (FAILED(proxy->GetBuffer(proxy->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&target))) || !target)
    {
        error = "proxy back buffer unavailable";
        return false;
    }
    const D3D12_RESOURCE_DESC targetDesc = target->GetDesc();
    if (targetDesc.Format != kColorFormat || targetDesc.Width != size.width
        || targetDesc.Height != size.height || targetDesc.DepthOrArraySize != 1)
    {
        std::ostringstream text;
        text << "proxy back buffer is not a " << size.width << "x" << size.height
             << " BGRA8 texture";
        error = text.str();
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
        XEFG_SWAPCHAIN_RES_DEPTH, renderer.Depth(), size.GeometryWidth(),
        size.GeometryHeight());
    submissionResults[1] = TagResource(runtime, fg, fixture.commands.Get(), frame,
        XEFG_SWAPCHAIN_RES_MOTION_VECTOR, renderer.Motion(), size.GeometryWidth(),
        size.GeometryHeight());
    // UI mode 1 interpolates the backbuffer and has no hudless-color input, so
    // its color submission is skipped and recorded as null.
    if (tagColor)
    {
        submissionResults[2] = TagResource(runtime, fg, fixture.commands.Get(), frame,
            XEFG_SWAPCHAIN_RES_HUDLESS_COLOR, renderer.Color(), size.width, size.height);
    }
    if (tagUi)
    {
        submissionResults[3] = TagResource(runtime, fg, fixture.commands.Get(), frame,
            XEFG_SWAPCHAIN_RES_UI, renderer.Ui(), size.width, size.height);
    }
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

std::string ResetFramesJson(SceneKind scene, uint32_t warmup, uint32_t frames, bool isolatedPair)
{
    std::ostringstream text;
    text << "[0";
    if (!isolatedPair && scene == SceneKind::Cut && kCutFrame < warmup + frames)
        text << ", " << kCutFrame;
    text << "]";
    return text.str();
}

const char* kSceneCameraJson =
    "{\"eye\":[0,0,0],\"target\":[0,0,-1],\"up\":[0,1,0],\"fov_y_degrees\":50,"
    "\"near\":0.1,\"far\":100,\"aspect\":1.7777778}";

// Optional extension for the two new scenes.  It carries the analytic mesh,
// deformation/material contract and ordering; per-frame transforms stay in
// the regular object records.
std::string BuildSceneExtensionJson(SceneKind scene)
{
    std::ostringstream text;
    if (scene == SceneKind::Nonrigid)
    {
        text << "{\"schema\":\"xefg.render_scene_extension/1\",\"version\":1,"
                "\"scene\":\"nonrigid\",\"camera\":" << kSceneCameraJson << ","
             << "\"grid\":{\"cells_x\":" << kNonrigidGridCells
             << ",\"cells_y\":" << kNonrigidGridCells
             << ",\"vertices_x\":" << kNonrigidGridVertices
             << ",\"vertices_y\":" << kNonrigidGridVertices
             << ",\"vertex_count\":" << (kNonrigidGridVertices * kNonrigidGridVertices)
             << ",\"triangle_count\":" << (kNonrigidGridCells * kNonrigidGridCells * 2)
             << ",\"local_x_range\":[-1,1],\"local_y_range\":[-1,1],"
                "\"vertex_order\":\"row-major iy*(cells_x+1)+ix\","
                "\"cell_triangles\":\"(a,b,d),(a,d,c) with a=iy*(cells_x+1)+ix,b=a+1,"
                "c=a+cells_x+1,d=c+1\",\"winding\":\"ccw_from_plus_z\"},"
             << "\"deformation\":{\"formula\":\"z=A1*sin(kx*x+w1*t)*cos(ky*y+w2*t)+"
                "A2*sin(k2*(x+y)+w3*t+phi2)\",\"A1\":" << kNonrigidPrimaryAmplitude
             << ",\"kx\":" << kNonrigidPrimaryKx << ",\"ky\":" << kNonrigidPrimaryKy
             << ",\"w1\":" << kNonrigidPrimaryOmegaX << ",\"w2\":" << kNonrigidPrimaryOmegaY
             << ",\"A2\":" << kNonrigidSecondaryAmplitude << ",\"k2\":" << kNonrigidSecondaryK
             << ",\"w3\":" << kNonrigidSecondaryOmega << ",\"phi2\":" << kNonrigidSecondaryPhase
             << ",\"dzdx_formula\":\"A1*kx*cos(kx*x+w1*t)*cos(ky*y+w2*t)+"
                "A2*k2*cos(k2*(x+y)+w3*t+phi2)\","
                "\"dzdy_formula\":\"-A1*ky*sin(kx*x+w1*t)*sin(ky*y+w2*t)+"
                "A2*k2*cos(k2*(x+y)+w3*t+phi2)\","
                "\"normal_local\":\"normalize((-dzdx,-dzdy,1))\","
                "\"normal_world\":\"normalize((n.x/2.7,n.y/1.5,n.z/1.0))\","
                "\"world\":\"p=(2.7x,1.5y,z_local-4.2)\","
                "\"time\":\"state.time\",\"previous_time\":\"state.previousTime\","
                "\"previous_time_note\":\"MakeFrameState: t-1, or source_frame-1 for midpoint; "
                "history reset sets previous_time=time\","
                "\"reset\":\"previous_time=time, previous_position=current\","
                "\"midpoint\":\"time=source_frame-0.5, previous_time=source_frame-1\"},"
             << "\"shading\":{\"diffuse\":\"saturate(dot(n_world,normalize((0.35,0.80,0.45))))\","
                "\"rgb\":\"ObjectColor.rgb*(0.22+0.78*diffuse)+"
                "pow(saturate(1-abs(n_world.z)),2)*0.10\",\"alpha\":1},"
                "\"objects\":[{\"name\":\"deform_panel\",\"role\":\"deforming_grid\","
                "\"mesh\":\"grid\"},{\"name\":\"static_backdrop\",\"role\":\"static_background\","
                "\"mesh\":\"cube\"}]}";
        return text.str();
    }
    if (scene == SceneKind::Transparency)
    {
        text << "{\"schema\":\"xefg.render_scene_extension/1\",\"version\":1,"
                "\"scene\":\"transparency\",\"camera\":" << kSceneCameraJson << ","
             << "\"policy\":{\"color_writes\":\"opaque_then_transparent\","
                "\"depth_motion\":\"opaque_only\",\"transparent_depth_write\":false,"
                "\"transparent_writes_mv\":false,\"ui\":\"untouched\","
                "\"opaque_scene_static\":true},"
                "\"blend\":{\"mode\":\"premultiplied_alpha\",\"rgb_src\":\"ONE\","
                "\"rgb_dst\":\"INV_SRC_ALPHA\",\"alpha_src\":\"ONE\","
                "\"alpha_dst\":\"INV_SRC_ALPHA\",\"equation\":\"L=S+D*(1-alpha_s)\"},"
                "\"draw_order\":[\"backdrop\",\"blocker\",\"plane_far\",\"plane_near\"],"
                "\"regions\":{\"single_far\":\"A minus B minus blocker\","
                "\"single_near\":\"B minus A minus blocker\","
                "\"double\":\"A intersect B minus blocker\","
                "\"occluded\":\"blocker area covered by A or B\"},"
                "\"motion\":{\"plane_far\":\"xA=-0.15+0.35*sin(0.50*t),"
                "yA=-0.30+0.10*cos(0.37*t)\",\"plane_near\":"
                "\"xB=0.10+0.30*sin(0.40*t+1.30),yB=0.35+0.08*sin(0.31*t+0.60)\","
                "\"time\":\"state.time\",\"previous_time\":\"state.previousTime\"},"
                "\"objects\":["
                "{\"name\":\"backdrop\",\"role\":\"opaque_backdrop\",\"mesh\":\"panel\","
                "\"rgba\":[0.10,0.15,0.24,1.0],"
                "\"premultiplied_rgba\":[0.10,0.15,0.24,1.0],\"depth_test\":true,"
                "\"depth_write\":true,\"writes_depth_motion\":true,\"sort_index\":0},"
                "{\"name\":\"blocker\",\"role\":\"opaque_blocker\",\"mesh\":\"panel\","
                "\"rgba\":[0.20,0.24,0.26,1.0],"
                "\"premultiplied_rgba\":[0.20,0.24,0.26,1.0],\"depth_test\":true,"
                "\"depth_write\":true,\"writes_depth_motion\":true,\"sort_index\":1},"
                "{\"name\":\"plane_far\",\"role\":\"transparent_plane\",\"mesh\":\"panel\","
                "\"rgba\":[0.80,0.22,0.18,0.55],"
                "\"premultiplied_rgba\":[0.44,0.121,0.099,0.55],\"depth_test\":true,"
                "\"depth_write\":false,\"writes_depth_motion\":false,\"sort_index\":2},"
                "{\"name\":\"plane_near\",\"role\":\"transparent_plane\",\"mesh\":\"panel\","
                "\"rgba\":[0.16,0.60,0.78,0.45],"
                "\"premultiplied_rgba\":[0.072,0.27,0.351,0.45],\"depth_test\":true,"
                "\"depth_write\":false,\"writes_depth_motion\":false,\"sort_index\":3}]}";
        return text.str();
    }
    return {};
}

void PopulateObjectMaterial(SceneObjectRecord& record, const ObjectState& object)
{
    record.hasMaterial = true;
    record.role = object.role;
    record.mesh = MeshName(object.mesh);
    for (int i = 0; i < 4; ++i) record.flatColor[static_cast<size_t>(i)] = object.color[i];
    const float alpha = object.color[3];
    record.premultipliedColor = {object.color[0] * alpha, object.color[1] * alpha,
        object.color[2] * alpha, alpha};
    record.transparent = object.transparent;
    record.depthWrite = object.depthWrite;
    record.writesDepthMotion = object.writesDepthMotion;
    record.sortIndex = object.sortIndex;
}

bool WriteMetaJson(const std::filesystem::path& path, const Options& options, SceneKind scene,
    const RenderSize& size, bool warp, const std::string& adapterName,
    const DXGI_ADAPTER_DESC1& adapterDescription,
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
    const ModeInfo mode = ModeFor(options.mode);
    file << "  \"configuration\": {"
         << "\"width\": " << size.width << ", \"height\": " << size.height
         << ", \"motion_width\": " << size.GeometryWidth()
         << ", \"motion_height\": " << size.GeometryHeight()
         << ", \"mode\": " << options.mode
         << ", \"scene\": \"" << JsonEscape(options.scene) << "\", \"scene_revision\": 1"
         << ", \"ui\": " << (mode.ui ? "true" : "false")
         << ", \"ui_texture_tagged\": " << (mode.uiTextureTagged ? "true" : "false")
         << ", \"ui_mode\": " << int(mode.uiMode)
         << ", \"ui_mode_requested\": \"" << mode.requested << "\", \"ui_mode_explicit\": true"
         << ", \"ui_composition_enabled\": " << (mode.uiComposition ? "true" : "false")
         << ", \"pairing_method\": \"" << (options.pairFrame ? "isolated_pair" : "sequential") << "\""
         << ", \"pair_frame\": ";
    if (options.pairFrame) file << *options.pairFrame;
    else file << "null";
    file << ", \"max_interpolated_frames\": " << maxInterpolatedFrames
         << ", \"initialization_fallback\": " << (initializationFallback ? "true" : "false")
         << ", \"frames\": " << options.frames << ", \"warmup\": " << options.warmup
         << ", \"pacing_ms\": " << options.pacingMilliseconds
         << ", \"only_interpolation\": " << (options.renderOnly ? "false" : "true")
         << ", \"tag_interpolated\": "
         << ((!options.renderOnly && options.tagInterpolated) ? "true" : "false")
         << ", \"render_only\": " << (options.renderOnly ? "true" : "false")
         << ", \"fg_matrix_control\": \"" << (options.identityFgMatrices ? "identity" : "rendered") << "\""
         << ", \"reset_frames\": "
         << ResetFramesJson(scene, options.warmup, options.frames, options.pairFrame.has_value())
         << ", \"debug_layer\": " << (options.debugLayer ? "true" : "false") << "},\n";
    file << "  \"api_results\": {"
         << "\"create_context\": " << setup.createContext
         << ", \"create_xell\": " << setup.createXell
         << ", \"init\": " << setup.init
         << ", \"get_swap_chain\": " << setup.getSwapChain
         << ", \"set_num_interpolated_frames\": " << setup.setNumInterpolatedFrames
         << ", \"set_enabled\": " << setup.setEnabled
         << ", \"set_ui_composition_state\": ";
    if (mode.uiComposition) file << setup.setUiCompositionState;
    else file << "null"; // Mode 1 never calls the composition API.
    file << ", \"enable_only_interpolation\": " << setup.enableOnlyInterpolation
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
    const RenderSize renderSize = options.size;

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
    TestWindow window(L"XefgRenderValidationWindow", static_cast<LONG>(renderSize.width),
        static_cast<LONG>(renderSize.height));
    if (!window.window)
    {
        std::fprintf(stderr, "window creation failed\n");
        return 1;
    }
    window.Show();
    window.Pump(2);

    Renderer renderer;
    if (!renderer.Initialize(fixture.device.Get(), fixture.queue.Get(), shaderPath, error,
        renderSize))
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
    const ModeInfo modeInfo = ModeFor(options.mode);

#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
    std::unique_ptr<probe::TraceWriter> tensorTrace;
    std::unique_ptr<probe::DecodeRelay> tensorRelay;
    std::unique_ptr<probe::DispatchTrace> tensorDispatcher;
    bool tensorCaptureOk = true;
    std::string tensorWaitError;
#endif

    if (!options.renderOnly)
    {
        if (!runtime.Load(Widen(options.xessDirectory), error))
        {
            std::fprintf(stderr, "XeFG runtime load failed: %s\n", error.c_str());
            return 1;
        }
        moduleHash = oracle::ModuleFileSha256(runtime.fg);
        std::printf("provider sha256: %s\n", moduleHash.c_str());

#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
        if (options.tensorFrame
            && _stricmp(moduleHash.c_str(), oracle::kProfiledFg78Sha256) != 0)
        {
            std::fprintf(stderr,
                "tensor capture is calibrated for fg78 %s; refusing %s\n",
                oracle::kProfiledFg78Sha256, moduleHash.c_str());
            return 1;
        }
#endif

        capture = std::make_shared<oracle::CaptureContext>();
        if (!capture->Initialize(fixture.device.Get(), fixture.queue.Get(),
            (outputDirectory / L"captures").wstring(), error))
        {
            std::fprintf(stderr, "capture initialization failed: %s\n", error.c_str());
            return 1;
        }

#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
        // Install after the capture context so its private queue is not mistaken
        // for a provider queue, but before CreateContext so every provider queue,
        // command list and pipeline is observed.  The present capture wrappers are
        // COM delegations, not vtable patches, so they do not conflict.
        if (options.tensorFrame)
        {
            error.clear();
            tensorTrace = std::make_unique<probe::TraceWriter>();
            if (!tensorTrace->Open(outputDirectory / L"trace", error))
            {
                std::fprintf(stderr, "tensor trace open failed: %s\n", error.c_str());
                return 1;
            }
            tensorRelay = std::make_unique<probe::DecodeRelay>();
            if (!tensorRelay->Install(runtime.fg, *tensorTrace,
                outputDirectory / L"trace" / L"shaders",
                outputDirectory / L"trace" / L"params", error))
            {
                std::fprintf(stderr, "decode relay install failed: %s\n", error.c_str());
                return 1;
            }
            tensorDispatcher = std::make_unique<probe::DispatchTrace>();
            probe::DispatchTraceOptions tensorOptions;
            tensorOptions.tensors = true;
            tensorOptions.tensorBudgetMB = options.tensorBudgetMB;
            tensorOptions.tensorDispatches = 0;
            tensorOptions.tensorFirstFrame = *options.tensorFrame;
            tensorOptions.tensorLastFrame = *options.tensorFrame;
            tensorOptions.tensorDirectory = outputDirectory / L"trace" / L"tensors";
            tensorDispatcher->SetOptions(tensorOptions);
            if (!tensorDispatcher->Install(fixture.device.Get(), fixture.commands.Get(),
                *tensorTrace, *tensorRelay, error))
            {
                std::fprintf(stderr, "dispatch trace install failed: %s\n", error.c_str());
                return 1;
            }
            if (!tensorDispatcher->TensorEnabled())
            {
                std::fprintf(stderr, "tensor dump could not be initialized\n");
                return 1;
            }
            std::printf("tensor capture: frame %u budget %u MB\n",
                *options.tensorFrame, options.tensorBudgetMB);
        }
#endif

        DXGI_SWAP_CHAIN_DESC1 chainDescription{};
        chainDescription.Width = renderSize.width;
        chainDescription.Height = renderSize.height;
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
        initParameters.uiMode = modeInfo.uiMode;
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
        if (modeInfo.uiComposition)
        {
            setup.setUiCompositionState = static_cast<int32_t>(
                runtime.xefgSwapChainSetUiCompositionState(
                    fg, XEFG_SWAPCHAIN_UI_COMPOSITION_STATE_ENABLED));
        }
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
        std::printf("XeFG ready: mode=%u uiMode=%d maxInterpolations=%u descriptors=%u fallback=%s\n",
            options.mode, int(modeInfo.uiMode), properties.maxSupportedInterpolations,
            properties.requiredDescriptorCount, initializationFallback ? "yes" : "no");
    }

    const bool isolatedPair = options.pairFrame.has_value();
    const uint32_t totalFrames = options.warmup + options.frames;
    std::vector<SceneFrameRecord> sceneFrames;
    sceneFrames.reserve(totalFrames);
    std::vector<PresentStatusRecord> statuses;
    statuses.reserve(options.renderOnly ? 0 : totalFrames);
    bool apiFailure = false;
    if (!options.renderOnly
        && (setup.createContext != 0 || setup.init != 0 || setup.getSwapChain != 0
            || setup.setNumInterpolatedFrames != 0 || setup.setEnabled != 0
            || (modeInfo.uiComposition && setup.setUiCompositionState != 0)
            || onlyInterpolationResult != 0
            || (tagRequested && tagInterpolatedResult != 0) || setup.getProperties != 0))
        apiFailure = true;

    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        // Local submission index (tags, status, capture labels) versus the
        // rendering source frame/time.  They are identical sequentially and
        // differ in isolated-pair mode.
        const uint32_t localFrame = frame;
        uint32_t sourceFrame = localFrame;
        bool resetHistory = false;
        FrameState state;
        FrameState midpointState;
        if (isolatedPair)
        {
            sourceFrame = *options.pairFrame - 1u + localFrame;
            const bool firstOfPair = localFrame == 0;
            state = MakeIsolatedPairFrameState(sceneKind, sourceFrame, firstOfPair, false);
            midpointState = MakeIsolatedPairFrameState(sceneKind, sourceFrame, firstOfPair, true);
            resetHistory = state.historyReset;
        }
        else
        {
            resetHistory = (localFrame == 0) || IsCutResetFrame(localFrame, sceneKind);
            state = MakeFrameState(sceneKind, sourceFrame, false);
            state.historyReset = resetHistory;
            midpointState = MakeFrameState(sceneKind, sourceFrame, true);
            midpointState.historyReset = resetHistory;
        }
        state.uiEnabled = modeInfo.ui;
        midpointState.uiEnabled = modeInfo.ui;

#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
        // The provider dispatches for this submission happen during Present.
        // The trace stamps every dispatch with this source frame and only the
        // selected frame is allowed to retain tensor snapshots.
        if (tensorDispatcher) tensorDispatcher->SetFrame(sourceFrame);
#endif

        if (!renderer.Render(midpointState, sceneKind, true, error))
        {
            std::fprintf(stderr, "midpoint render failed at frame %u: %s\n", localFrame, error.c_str());
            return 1;
        }
        const std::string midpointName = "native_mid/" + FrameFile("frame", localFrame, "bgra");
        std::vector<uint8_t> midpointBytes;
        if (!ReadbackChecked(renderer.Backbuffer(),
            static_cast<size_t>(renderSize.width) * renderSize.height * 4,
            midpointBytes, "midpoint backbuffer", error))
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        if (!WriteRawFile(nativeMidpointDirectory / FrameFile("frame", localFrame, "bgra"),
            midpointBytes, error))
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }

        if (!renderer.Render(state, sceneKind, false, error))
        {
            std::fprintf(stderr, "current render failed at frame %u: %s\n", localFrame, error.c_str());
            return 1;
        }
        if (!WriteFrameInputs(applicationInputsDirectory, localFrame, renderer, renderSize, error))
        {
            std::fprintf(stderr, "input readback failed at frame %u: %s\n", localFrame, error.c_str());
            return 1;
        }

        SceneFrameRecord record;
        record.index = localFrame;
        record.sceneFrame = sourceFrame;
        record.hasPrevious = localFrame > 0;
        record.previousFrame = localFrame > 0 ? localFrame - 1 : 0;
        record.resetHistory = resetHistory;
        record.midpoint = midpointName;
        std::memcpy(record.view.data(), state.view.m, sizeof(state.view.m));
        std::memcpy(record.projection.data(), state.projection.m, sizeof(state.projection.m));
        record.time = state.time;
        record.previousTime = state.previousTime;
        const bool materialScene = sceneKind == SceneKind::Nonrigid
            || sceneKind == SceneKind::Transparency;
        if (sceneKind == SceneKind::Nonrigid)
        {
            record.hasDeformation = true;
            record.previousIsCurrent = resetHistory || state.previousTime == state.time;
        }
        for (uint32_t objectIndex = 0;
             objectIndex < std::min<uint32_t>(state.objectCount, kMaxObjects); ++objectIndex)
        {
            SceneObjectRecord object;
            object.name = state.objects[objectIndex].name;
            std::memcpy(object.world.data(), state.objects[objectIndex].world.m, sizeof(object.world));
            std::memcpy(object.previousWorld.data(), state.objects[objectIndex].previousWorld.m,
                sizeof(object.previousWorld));
            if (materialScene) PopulateObjectMaterial(object, state.objects[objectIndex]);
            record.objects.push_back(std::move(object));
        }

        if (options.renderOnly)
        {
            sceneFrames.push_back(std::move(record));
            continue;
        }

        if (!CopyBackbufferAndTag(fixture, proxy.Get(), renderer, renderSize, runtime, fg, localFrame,
            modeInfo.ui, modeInfo.uiTextureTagged, record.submissionResults, error))
        {
            std::fprintf(stderr, "backbuffer/tag submission failed at frame %u: %s\n",
                localFrame, error.c_str());
            return 1;
        }

        xefg_swapchain_frame_constant_data_t constants{};
        std::memcpy(constants.viewMatrix, state.view.m, sizeof(state.view.m));
        std::memcpy(constants.projectionMatrix, state.projection.m, sizeof(state.projection.m));
        if (options.identityFgMatrices)
        {
            const Matrix4 identity = IdentityMatrix();
            std::memcpy(constants.viewMatrix, identity.m, sizeof(identity.m));
            std::memcpy(constants.projectionMatrix, identity.m, sizeof(identity.m));
        }
        constants.jitterOffsetX = 0.0f;
        constants.jitterOffsetY = 0.0f;
        constants.motionVectorScaleX = 1.0f;
        constants.motionVectorScaleY = 1.0f;
        constants.resetHistory = resetHistory ? 1u : 0u;
        constants.frameRenderTime = static_cast<float>(options.pacingMilliseconds);
        record.submissionResults[4] = static_cast<int32_t>(
            runtime.xefgSwapChainTagFrameConstants(fg, localFrame, &constants));
        record.submissionResults[5] = static_cast<int32_t>(
            runtime.xefgSwapChainSetPresentId(fg, localFrame));

        capture->SetFrameIndex(localFrame);
        // Isolated-pair runs capture both submissions so neither the reset base
        // nor the single eligible generated pair can be hidden.
        capture->SetEnabled(isolatedPair || localFrame >= options.warmup);
        LARGE_INTEGER presentStart{}, presentEnd{};
        QueryPerformanceCounter(&presentStart);
        const HRESULT presentResult = proxy->Present(0, 0);
        QueryPerformanceCounter(&presentEnd);
        xefg_swapchain_present_status_t status{};
        const auto statusResult = runtime.xefgSwapChainGetLastPresentStatus(fg, &status);
        PresentStatusRecord statusRecord;
        statusRecord.applicationFrame = localFrame;
        statusRecord.presentResult = static_cast<int32_t>(presentResult);
        statusRecord.statusResult = static_cast<int32_t>(statusResult);
        statusRecord.framesPresented = status.framesPresented;
        statusRecord.frameGenerationResult = static_cast<int32_t>(status.frameGenResult);
        statusRecord.isFrameGenEnabled = status.isFrameGenEnabled;
        statusRecord.presentStartQpc = static_cast<uint64_t>(presentStart.QuadPart);
        statusRecord.presentEndQpc = static_cast<uint64_t>(presentEnd.QuadPart);
        statuses.push_back(statusRecord);
        const bool submissionSucceeded = std::all_of(record.submissionResults.begin(),
            record.submissionResults.end(), [](const std::optional<int32_t>& result) {
                return !result || *result == 0;
            });
        if (presentResult != S_OK || statusResult != XEFG_SWAPCHAIN_RESULT_SUCCESS
            || !submissionSucceeded)
            apiFailure = true;
        std::printf("frame %u (scene %u): present=0x%08lx framesPresented=%u result=%d\n",
            localFrame, sourceFrame, static_cast<unsigned long>(presentResult),
            status.framesPresented, int(status.frameGenResult));
        if (options.pacingMilliseconds) Sleep(options.pacingMilliseconds);
        sceneFrames.push_back(std::move(record));
    }

    if (sceneFrames.size() != totalFrames)
    {
        std::fprintf(stderr, "internal error: incomplete frame records\n");
        return 1;
    }

    if (!WriteApplicationInputsManifest(applicationInputsDirectory, renderSize.width,
        renderSize.height, renderSize.GeometryWidth(), renderSize.GeometryHeight(),
        options.warmup, options.frames, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    SceneManifest sceneManifest;
    sceneManifest.name = kSceneNames[static_cast<size_t>(sceneKind)];
    sceneManifest.revision = 1;
    sceneManifest.mode = options.mode;
    sceneManifest.pairingMethod = isolatedPair ? "isolated_pair" : "sequential";
    sceneManifest.hasSourcePair = isolatedPair;
    if (isolatedPair)
    {
        sceneManifest.sourcePair = {
            *options.pairFrame - 1u, *options.pairFrame};
    }
    sceneManifest.width = renderSize.width;
    sceneManifest.height = renderSize.height;
    sceneManifest.motionWidth = renderSize.GeometryWidth();
    sceneManifest.motionHeight = renderSize.GeometryHeight();
    sceneManifest.warmup = options.warmup;
    sceneManifest.measuredFrames = options.frames;
    sceneManifest.shaderSource = shaderSource;
    sceneManifest.shaderSha256 = shaderHash;
    sceneManifest.rendererVersion = kRendererVersion;
    sceneManifest.sceneExtensionJson = BuildSceneExtensionJson(sceneKind);
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
#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
        // The provider's own present queue carries the tensor copies.  Drain it
        // and write the readbacks before destroying the context that owns it.
        if (tensorDispatcher)
        {
            tensorCaptureOk = tensorDispatcher->FinishTensors(tensorWaitError);
            const size_t tensorFiles = tensorDispatcher->TensorFiles();
            const uint64_t tensorBytes = tensorDispatcher->TensorBytes();
            const uint64_t tensorBudget = tensorDispatcher->TensorBudgetBytes();
            const bool tensorEnabled = tensorDispatcher->TensorEnabled();
            const size_t providerQueues = tensorDispatcher->ProviderQueues();
            const std::string traceError = tensorDispatcher->Error();

            if (tensorFiles == 0 && tensorWaitError.empty())
            {
                tensorCaptureOk = false;
                tensorWaitError = "no tensor snapshots were written for the selected frame";
            }
            else if (!tensorEnabled && tensorWaitError.empty())
            {
                tensorCaptureOk = false;
                tensorWaitError =
                    "tensor capture stopped before the selected frame completed (budget or allocation)";
            }
            if (!traceError.empty())
            {
                tensorCaptureOk = false;
                if (tensorWaitError.empty()) tensorWaitError = traceError;
            }
            if (!tensorCaptureOk)
                std::fprintf(stderr, "tensor capture incomplete: %s\n", tensorWaitError.c_str());
            else
                std::printf("tensor capture complete: %zu files, %llu bytes\n",
                    tensorFiles, static_cast<unsigned long long>(tensorBytes));

            if (tensorTrace)
            {
                std::ostringstream meta;
                meta << "{\n";
                meta << "  \"schema\": \"xefg.tensor_trace/1\",\n";
                meta << "  \"complete\": " << (tensorCaptureOk ? "true" : "false") << ",\n";
                meta << "  \"error\": \"" << probe::JsonEscape(tensorCaptureOk ? "" : tensorWaitError)
                     << "\",\n";
                meta << "  \"wait_error\": \"" << probe::JsonEscape(tensorWaitError) << "\",\n";
                meta << "  \"trace_error\": \"" << probe::JsonEscape(traceError) << "\",\n";
                meta << "  \"tensor_enabled\": " << (tensorEnabled ? "true" : "false") << ",\n";
                meta << "  \"tensor_files\": " << tensorFiles << ",\n";
                meta << "  \"tensor_bytes\": " << tensorBytes << ",\n";
                meta << "  \"tensor_budget_bytes\": " << tensorBudget << ",\n";
                meta << "  \"tensor_frame\": " << *options.tensorFrame << ",\n";
                meta << "  \"tensor_first_frame\": " << *options.tensorFrame << ",\n";
                meta << "  \"tensor_last_frame\": " << *options.tensorFrame << ",\n";
                meta << "  \"dispatches\": " << tensorDispatcher->DispatchCount() << ",\n";
                meta << "  \"distinct_programs\": " << tensorDispatcher->DistinctPrograms() << ",\n";
                meta << "  \"patched_tables\": " << tensorDispatcher->PatchedListTables() << ",\n";
                meta << "  \"provider_queues\": " << providerQueues << ",\n";
                meta << "  \"programs\": " << tensorRelay->ProgramCount() << ",\n";
                meta << "  \"parameter_captures\": " << tensorRelay->ParameterCaptureCount() << ",\n";
                meta << "  \"xefg_family\": \"fg78\",\n";
                meta << "  \"xefg_dll_sha256\": \"" << probe::JsonEscape(moduleHash) << "\",\n";
                meta << "  \"configuration\": {\"scene\": \"" << probe::JsonEscape(options.scene)
                     << "\", \"mode\": " << options.mode
                     << ", \"width\": " << renderSize.width
                     << ", \"height\": " << renderSize.height
                     << ", \"warmup\": " << options.warmup
                     << ", \"frames\": " << options.frames << "}\n";
                meta << "}\n";
                tensorTrace->SetMeta(meta.str());
                tensorTrace->Flush();
            }
            tensorDispatcher->Detach();
        }
#endif
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

    const bool metaWritten = WriteMetaJson(outputDirectory / L"meta.json", options, sceneKind,
        renderSize, options.warp, options.warp ? "Microsoft Basic Render Driver"
        : Narrow(adapterDescription.Description),
        adapterDescription, LuidString(fixture.device->GetAdapterLuid()), moduleHash, shaderSource,
        shaderHash, setup, maxInterpolatedFrames, initializationFallback, qpcFrequency.QuadPart, error);
    if (!metaWritten) std::fprintf(stderr, "%s\n", error.c_str());

    if (captureFactory) captureFactory.Reset();
    if (xell) runtime.xellDestroyContext(xell);
#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
    // Restore the provider's decoder and close the trace before the module is
    // unmapped; the exhausted trace also has no more live objects to reference.
    if (tensorRelay) tensorRelay->Restore();
    if (tensorTrace) tensorTrace->Close();
#endif
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
#if defined(XEFG_RENDER_VALIDATION_TENSOR_PROBE)
    if (!tensorCaptureOk)
    {
        std::fprintf(stderr, "tensor capture incomplete; see trace/meta.json: %s\n",
            tensorWaitError.c_str());
        return 1;
    }
#endif
    if (apiFailure)
    {
        std::fprintf(stderr, "one or more public API calls reported failure; manifests retain the results\n");
        return 1;
    }
    return 0;
}
