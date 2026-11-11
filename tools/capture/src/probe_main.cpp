// XefgProbe: research-mode trace of the provider's frame generation.
//
// Runs the same deterministic scene as XefgOracle, but additionally relays the
// provider's program decoder (read-only), records every dispatch with its root
// signature and bound resources, and optionally snapshots per-stage tensors.
// Nothing is replaced and the model catalog is never modified: on a static
// scene the presented frames are bit-identical to an oracle run. With motion
// the provider's output depends on its own timing, so equality is not asserted.
#include "common.h"

#include "model_dump.h"
#include "probe/decode_relay.h"
#include "probe/dispatch_trace.h"
#include "probe/trace_writer.h"
#include "scene.h"
#include "swapchain_capture.h"
#include "xefg_runtime.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace harness;

namespace
{
struct Options
{
    std::string xessDirectory;
    std::string outputDirectory;
    std::string adapterFilter;
    uint32_t width = 640;
    uint32_t height = 360;
    uint32_t motionWidth = 320;
    uint32_t motionHeight = 180;
    bool highResMotionVectors = false;
    bool staticScene = false;
    std::string scene = "standard";
    bool ui = true;
    bool animateUi = false;
    std::string uiMode = "auto";
    float velocityX = 12.0f;
    float velocityY = 4.0f;
    bool flipMotionVectors = false;
    uint32_t frames = 4;
    uint32_t warmup = 6;
    uint32_t pacingMilliseconds = 16;
    bool debugLayer = false;
    bool modelDump = true;
    bool showInterpolationOnly = true;
    bool tensors = false;
    bool dumpInputs = false;
    uint32_t tensorBudgetMB = 512;
    uint32_t tensorDispatches = 0;
    bool calibration = false;
    bool tagInterpolated = false;
    uint32_t resetFrame = 0;
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
        "xefg_probe --xess-dir <dir> --out <dir> [options]\n"
        "  --adapter <substring>   pick a DXGI adapter by name (default: first hardware)\n"
        "  --width/--height N      output size (default 640x360)\n"
        "  --mv-width/--mv-height  low-resolution motion vector size (default 320x180)\n"
        "  --high-res-mv           motion vectors and depth at output size\n"
        "  --static                zero motion\n"
        "  --frames N              traced frames after warmup (default 4)\n"
        "  --warmup N              frames before tracing (default 6)\n"
        "  --pacing-ms N           simulated application frame time (default 16)\n"
        "  --velocity X,Y          object displacement per frame in output pixels\n"
        "  --scene standard|smooth  grid/checker or gradients (default standard)\n"
        "  --ui on|off             tag a UI texture (default on)\n"
        "  --ui-mode auto|none|hudless-uitexture|backbuffer-hudless (default auto)\n"
        "  --animate-ui            move the synthetic UI crosshair per frame\n"
        "  --flip-mv               opposite motion-vector sign convention\n"
        "  --tensors               snapshot bound tensors around each dispatch\n"
        "  --dump-inputs           write raw application inputs to application_inputs/\n"
        "  --tensor-budget-mb N    tensor snapshot budget (default 512)\n"
        "  --tensor-dispatches N   snapshot only the first N traced dispatches\n"
        "  --all-presents          do not enable SHOW_ONLY_INTERPOLATION\n"
        "  --calibration           frame marker + calibration manifest (640x360, mode 3, --dump-inputs)\n"
        "  --tag-interpolated      enable TAG_INTERPOLATED_FRAMES debug feature\n"
        "  --reset-frame N         set constants.resetHistory on frame N (default 0)\n"
        "  --no-model-dump         skip the model catalog dump\n"
        "  --debug                 enable the D3D12 debug layer\n");
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
        else if (argument == "--width") options.width = std::stoul(value(i, argument));
        else if (argument == "--height") options.height = std::stoul(value(i, argument));
        else if (argument == "--mv-width") options.motionWidth = std::stoul(value(i, argument));
        else if (argument == "--mv-height") options.motionHeight = std::stoul(value(i, argument));
        else if (argument == "--high-res-mv") options.highResMotionVectors = true;
        else if (argument == "--static") options.staticScene = true;
        else if (argument == "--scene")
        {
            const std::string scene = value(i, argument);
            if (EqualsIgnoreCase(scene, "standard")) options.scene = "standard";
            else if (EqualsIgnoreCase(scene, "smooth")) options.scene = "smooth";
            else { error = "--scene expects standard or smooth"; return false; }
        }
        else if (argument == "--frames") options.frames = std::stoul(value(i, argument));
        else if (argument == "--warmup") options.warmup = std::stoul(value(i, argument));
        else if (argument == "--pacing-ms") options.pacingMilliseconds = std::stoul(value(i, argument));
        else if (argument == "--ui") options.ui = EqualsIgnoreCase(value(i, argument), "on");
        else if (argument == "--ui-mode")
        {
            const std::string mode = value(i, argument);
            if (!EqualsIgnoreCase(mode, "auto") && !EqualsIgnoreCase(mode, "none")
                && !EqualsIgnoreCase(mode, "hudless-uitexture")
                && !EqualsIgnoreCase(mode, "backbuffer-hudless"))
            {
                error = "--ui-mode expects auto, none, hudless-uitexture, or backbuffer-hudless";
                return false;
            }
            options.uiMode = mode;
        }
        else if (argument == "--animate-ui") options.animateUi = true;
        else if (argument == "--flip-mv") options.flipMotionVectors = true;
        else if (argument == "--tensors") options.tensors = true;
        else if (argument == "--dump-inputs") options.dumpInputs = true;
        else if (argument == "--tensor-budget-mb") options.tensorBudgetMB = std::stoul(value(i, argument));
        else if (argument == "--tensor-dispatches") options.tensorDispatches = std::stoul(value(i, argument));
        else if (argument == "--all-presents") options.showInterpolationOnly = false;
        else if (argument == "--calibration") options.calibration = true;
        else if (argument == "--tag-interpolated") options.tagInterpolated = true;
        else if (argument == "--reset-frame") options.resetFrame = std::stoul(value(i, argument));
        else if (argument == "--no-model-dump") options.modelDump = false;
        else if (argument == "--debug") options.debugLayer = true;
        else if (argument == "--velocity")
        {
            const std::string text = value(i, argument);
            const auto comma = text.find(',');
            if (comma == std::string::npos)
            {
                error = "--velocity expects X,Y";
                return false;
            }
            options.velocityX = std::stof(text.substr(0, comma));
            options.velocityY = std::stof(text.substr(comma + 1));
        }
        else
        {
            error = "unknown argument: " + argument;
            return false;
        }
        if (!error.empty()) return false;
    }
    if (options.xessDirectory.empty()) options.xessDirectory = Narrow(ExecutableDirectory().wstring());
    if (options.outputDirectory.empty())
    {
        error = "--out is required";
        return false;
    }
    return true;
}

ComPtr<ID3D12Resource> CreateTexture(ID3D12Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format,
    D3D12_RESOURCE_STATES state)
{
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = width;
    description.Height = height;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = format;
    description.SampleDesc.Count = 1;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> texture;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description, state, nullptr,
        IID_PPV_ARGS(&texture))))
        return nullptr;
    return texture;
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

bool RecordUpload(WarpFixture& fixture, ID3D12Resource* target, const void* texels,
    uint32_t rowBytes, uint32_t rows, std::vector<ComPtr<ID3D12Resource>>& keep)
{
    const D3D12_RESOURCE_DESC description = target->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    fixture.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    if (!total) return false;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upload;
    if (FAILED(fixture.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))))
        return false;
    void* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, &mapped))) return false;
    for (uint32_t y = 0; y < rows; ++y)
        std::memcpy(static_cast<uint8_t*>(mapped) + footprint.Offset
                + static_cast<size_t>(y) * footprint.Footprint.RowPitch,
            static_cast<const uint8_t*>(texels) + static_cast<size_t>(y) * rowBytes, rowBytes);
    upload->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION destination{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    destination.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION source{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    source.PlacedFootprint = footprint;
    fixture.commands->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    keep.push_back(std::move(upload));
    return true;
}

bool Submit(WarpFixture& fixture)
{
    if (FAILED(fixture.commands->Close())) return false;
    ID3D12CommandList* lists[]{fixture.commands.Get()};
    fixture.queue->ExecuteCommandLists(1, lists);
    return fixture.SignalAndWait();
}

bool WriteApplicationInputs(const std::filesystem::path& directory, uint32_t frame,
    const oracle::SceneFrame& scene, std::string& error)
{
    const auto write = [&](const wchar_t* prefix, const wchar_t* extension,
                           const void* data, size_t bytes) -> bool {
        wchar_t name[64]{};
        swprintf_s(name, L"%s_%05u.%s", prefix, frame, extension);
        const std::filesystem::path path = directory / name;
        if (!oracle::WriteRawFile(path.wstring(), data, bytes))
        {
            error = "could not write application input " + path.string();
            return false;
        }
        return true;
    };
    return write(L"color", L"bgra", scene.hudlessColor.data(), scene.hudlessColor.size())
        && write(L"ui", L"bgra", scene.uiColor.data(), scene.uiColor.size())
        && write(L"backbuffer", L"bgra", scene.backbuffer.data(), scene.backbuffer.size())
        && write(L"depth", L"r32f", scene.depth.data(), scene.depth.size() * sizeof(float))
        && write(L"mv", L"rg16f", scene.motion.data(), scene.motion.size() * sizeof(uint16_t));
}

bool WriteApplicationInputsManifest(const std::filesystem::path& directory, const Options& options,
    uint32_t motionWidth, uint32_t motionHeight, uint32_t totalFrames, std::string& error)
{
    const std::filesystem::path path = directory / L"manifest.json";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write application input manifest " + path.string();
        return false;
    }
    file << "{\n";
    file << "  \"schema\": \"xefg.application_inputs/1\",\n";
    file << "  \"width\": " << options.width << ",\n";
    file << "  \"height\": " << options.height << ",\n";
    file << "  \"motion_width\": " << motionWidth << ",\n";
    file << "  \"motion_height\": " << motionHeight << ",\n";
    file << "  \"frame_count\": " << totalFrames << ",\n";
    file << "  \"first_frame\": 0,\n";
    file << "  \"last_frame\": " << (totalFrames ? totalFrames - 1u : 0u) << ",\n";
    file << "  \"includes_warmup\": true,\n";
    file << "  \"warmup\": " << options.warmup << ",\n";
    file << "  \"traced_frames\": " << options.frames << ",\n";
    file << "  \"frame_semantics\": \"index is the application frame including warmup; "
            "color is the hudless color, backbuffer is the premultiplied UI over it\",\n";
    file << "  \"formats\": {\"color\": \"BGRA8\", \"ui\": \"BGRA8\", \"backbuffer\": \"BGRA8\", "
            "\"depth\": \"R32_FLOAT\", \"motion\": \"R16G16_FLOAT\"},\n";
    file << "  \"frames\": [\n";
    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        const auto name = [frame](const char* prefix, const char* extension) {
            char text[64]{};
            std::snprintf(text, sizeof(text), "%s_%05u.%s", prefix, frame, extension);
            return std::string(text);
        };
        file << "    {\"index\": " << frame << ", \"inputs\": {"
             << "\"color\": \"" << name("color", "bgra") << "\", "
             << "\"ui\": \"" << name("ui", "bgra") << "\", "
             << "\"backbuffer\": \"" << name("backbuffer", "bgra") << "\", "
             << "\"depth\": \"" << name("depth", "r32f") << "\", "
             << "\"motion\": \"" << name("mv", "rg16f") << "\"}}"
             << (frame + 1u < totalFrames ? "," : "") << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    file.flush();
    if (!file.good())
    {
        error = "could not write application input manifest " + path.string();
        return false;
    }
    return true;
}

struct CalibrationFrame
{
    uint32_t index = 0;
    bool resetHistory = false;
};

bool WriteCalibrationManifest(const std::filesystem::path& directory,
    const std::vector<CalibrationFrame>& frames, bool staticScene, std::string& error)
{
    const std::filesystem::path path = directory / L"manifest.json";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write calibration manifest " + path.string();
        return false;
    }
    file << "{\n";
    file << "  \"schema\": \"xefg.calibration/1\",\n";
    file << "  \"static\": " << (staticScene ? "true" : "false") << ",\n";
    file << "  \"marker\": {\"x\": 96, \"y\": 32, \"cell_width\": 8, \"cell_height\": 8, "
            "\"bits\": 32, \"encoding\": \"u16-complement-lsb\", \"threshold\": 128},\n";
    file << "  \"motion_roi\": [80, 80, 560, 300],\n";
    file << "  \"frames\": [\n";
    for (size_t i = 0; i < frames.size(); ++i)
    {
        char name[64]{};
        std::snprintf(name, sizeof(name), "mid_%05u.bgra", frames[i].index);
        file << "    {\"index\": " << frames[i].index
             << ", \"reset_history\": " << (frames[i].resetHistory ? "true" : "false")
             << ", \"midpoint\": \"" << name << "\""
             << ", \"time\": " << std::fixed << std::setprecision(1)
             << (static_cast<double>(frames[i].index) - 0.5)
             << std::defaultfloat << std::setprecision(6) << "}"
             << (i + 1 < frames.size() ? "," : "") << '\n';
    }
    file << "  ]\n}\n";
    file.flush();
    if (!file.good())
    {
        error = "could not write calibration manifest " + path.string();
        return false;
    }
    return true;
}

bool WritePresentsManifest(const std::filesystem::path& directory, oracle::CaptureContext& capture,
    bool interpolationOnly, int interpolationResult, bool tagInterpolated, int tagInterpolatedResult,
    int64_t qpcFrequency, const std::vector<std::string>& statuses)
{
    const auto records = capture.Captures();
    std::ofstream file(directory / L"presents.json", std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file << "{\n  \"schema\": \"xefg.present_capture/2\",\n"
         << "  \"only_interpolation_requested\": " << (interpolationOnly ? "true" : "false") << ",\n"
         << "  \"only_interpolation_result\": " << interpolationResult << ",\n"
         << "  \"tag_interpolated_requested\": " << (tagInterpolated ? "true" : "false") << ",\n"
         << "  \"tag_interpolated_result\": " << tagInterpolatedResult << ",\n"
         << "  \"qpc_frequency\": " << qpcFrequency << ",\n"
         << "  \"capture_error\": \"" << probe::JsonEscape(capture.Error()) << "\",\n"
         << "  \"frame_status\": [\n";
    for (size_t i = 0; i < statuses.size(); ++i)
        file << "    " << statuses[i] << (i + 1 < statuses.size() ? "," : "") << '\n';
    file << "  ],\n  \"presents\": [\n";
    for (size_t i = 0; i < records.size(); ++i)
    {
        const auto& record = records[i];
        const auto relative = std::filesystem::path(record.file).lexically_relative(directory);
        file << "    {\"sequence\": " << record.sequence
             << ", \"observed_application_frame\": " << record.frameIndex
             << ", \"ordinal\": " << record.ordinal
             << ", \"width\": " << record.width << ", \"height\": " << record.height
             << ", \"format\": " << int(record.format)
             << ", \"file\": \"" << probe::JsonEscape(Narrow(relative.generic_wstring())) << "\""
             << ", \"present_start_qpc\": " << record.presentStartQpc
             << ", \"present_end_qpc\": " << record.presentEndQpc
             << ", \"dxgi_present_count\": " << record.dxgiPresentCount
             << ", \"dxgi_present_count_result\": " << record.dxgiPresentCountResult
             << ", \"native_present_result\": " << record.nativePresentResult << "}"
             << (i + 1 < records.size() ? "," : "") << '\n';
    }
    file << "  ]\n}\n";
    file.close();
    return bool(file);
}

void CALLBACK LogCallback(const char* message, xefg_swapchain_logging_level_t level, void* userData)
{
    static_cast<oracle::Runtime*>(userData)->AppendLog(message, static_cast<int>(level));
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
        if (info->ExceptionRecord->NumberParameters >= 2)
            std::fprintf(stderr, "  access=%s address=%p\n",
                info->ExceptionRecord->ExceptionInformation[0] ? "write" : "read",
                reinterpret_cast<void*>(info->ExceptionRecord->ExceptionInformation[1]));
        return EXCEPTION_CONTINUE_SEARCH;
    });
    Options options;
    std::string error;
    if (!ParseOptions(argc, argv, options, error))
    {
        if (!error.empty()) std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }

    const std::filesystem::path outputDirectory = Widen(options.outputDirectory);
    std::error_code filesystemError;
    std::filesystem::create_directories(outputDirectory / L"trace", filesystemError);

    const std::filesystem::path applicationInputsDirectory = outputDirectory / L"application_inputs";
    if (options.dumpInputs)
    {
        std::filesystem::create_directories(applicationInputsDirectory, filesystemError);
        if (!std::filesystem::is_directory(applicationInputsDirectory))
        {
            std::fprintf(stderr, "application input directory unavailable: %s\n",
                applicationInputsDirectory.string().c_str());
            return 1;
        }
    }

    const std::filesystem::path calibrationDirectory = outputDirectory / L"calibration";
    if (options.calibration)
    {
        std::filesystem::create_directories(calibrationDirectory, filesystemError);
        if (!std::filesystem::is_directory(calibrationDirectory))
        {
            std::fprintf(stderr, "calibration directory unavailable: %s\n",
                calibrationDirectory.string().c_str());
            return 1;
        }
    }

    probe::TraceWriter trace;
    if (!trace.Open(outputDirectory / L"trace", error))
    {
        std::fprintf(stderr, "trace open failed: %s\n", error.c_str());
        return 1;
    }

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
    {
        std::fprintf(stderr, "CreateDXGIFactory1 failed\n");
        return 1;
    }
    ComPtr<IDXGIAdapter1> selected;
    for (UINT index = 0;; ++index)
    {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(adapter->GetDesc1(&description))) continue;
        if (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        const std::string name = Narrow(description.Description);
        if (!selected)
            if (options.adapterFilter.empty() || ContainsIgnoreCase(name, options.adapterFilter)) selected = adapter;
    }
    if (!selected)
    {
        std::fprintf(stderr, "no hardware adapter matched\n");
        return 1;
    }
    DXGI_ADAPTER_DESC1 selectedDescription{};
    selected->GetDesc1(&selectedDescription);
    const std::string adapterName = Narrow(selectedDescription.Description);
    std::printf("using adapter: %s\n", adapterName.c_str());

    WarpFixture fixture;
    if (!fixture.Initialize(options.debugLayer, selected.Get()) || !fixture.factory)
    {
        std::fprintf(stderr, "D3D12 device/queue initialization failed\n");
        return 1;
    }

    TestWindow window(L"XefgProbeWindow", 320, 180);
    if (!window.window)
    {
        std::fprintf(stderr, "window creation failed\n");
        return 1;
    }
    window.Show();
    window.Pump(2);

    oracle::Runtime runtime;
    if (!runtime.Load(Widen(options.xessDirectory), error))
    {
        std::fprintf(stderr, "XeFG runtime load failed: %s\n", error.c_str());
        return 1;
    }
    std::string familyError;
    const std::string family = oracle::DetectFamily(runtime.fg, familyError);
    const std::string moduleHash = oracle::ModuleFileSha256(runtime.fg);
    std::printf("libxess_fg family: %s (%s)\n", family.c_str(), moduleHash.c_str());
    // The relay offsets, the parameter frame and the dispatch slots are only
    // valid for this exact build. Family alone is not enough: pass an unrelated
    // fg78 DLL and every captured byte would still look plausible.
    if (family != "fg78" || _stricmp(moduleHash.c_str(), oracle::kProfiledFg78Sha256) != 0)
    {
        std::fprintf(stderr,
            "this probe is calibrated for the fg78 build %s; refusing %s (%s)\n",
            oracle::kProfiledFg78Sha256,
            family.empty() ? familyError.c_str() : family.c_str(), moduleHash.c_str());
        return 1;
    }

    // The relay must be in place before the provider builds its first pipeline.
    probe::DecodeRelay relay;
    if (!relay.Install(runtime.fg, trace, outputDirectory / L"trace" / L"shaders",
        outputDirectory / L"trace" / L"params", error))
    {
        std::fprintf(stderr, "decode relay install failed: %s\n", error.c_str());
        return 1;
    }

    probe::DispatchTrace dispatcher;
    probe::DispatchTraceOptions traceOptions;
    traceOptions.tensors = options.tensors;
    traceOptions.tensorBudgetMB = options.tensorBudgetMB;
    traceOptions.tensorDispatches = options.tensorDispatches;
    traceOptions.tensorDirectory = outputDirectory / L"trace" / L"tensors";
    dispatcher.SetOptions(traceOptions);
    if (!dispatcher.Install(fixture.device.Get(), fixture.commands.Get(), trace, relay, error))
    {
        std::fprintf(stderr, "dispatch trace install failed: %s\n", error.c_str());
        return 1;
    }

    if (options.modelDump)
    {
        std::string dumpError;
        if (!oracle::DumpModelMetadata(runtime.fg, outputDirectory.wstring(), false, 0, dumpError))
            std::fprintf(stderr, "model dump skipped: %s\n", dumpError.c_str());
        else std::printf("model metadata dumped\n");
    }

    auto capture = std::make_shared<oracle::CaptureContext>();
    std::string captureError;
    if (!capture->Initialize(fixture.device.Get(), fixture.queue.Get(),
        (outputDirectory / L"captures").wstring(), captureError))
    {
        std::fprintf(stderr, "capture init failed: %s\n", captureError.c_str());
        return 1;
    }

    DXGI_SWAP_CHAIN_DESC1 chainDescription{};
    chainDescription.Width = options.width;
    chainDescription.Height = options.height;
    chainDescription.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    chainDescription.SampleDesc.Count = 1;
    chainDescription.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    chainDescription.BufferCount = 3;
    chainDescription.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    chainDescription.Scaling = DXGI_SCALING_STRETCH;
    chainDescription.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;

    ComPtr<IDXGIFactory2> captureFactory = oracle::WrapFactory(fixture.factory.Get(), capture);
    if (!captureFactory)
    {
        std::fprintf(stderr, "factory wrapping failed\n");
        return 1;
    }

    xefg_swapchain_handle_t fg = nullptr;
    auto result = runtime.xefgSwapChainD3D12CreateContext(fixture.device.Get(), &fg);
    if (result != XEFG_SWAPCHAIN_RESULT_SUCCESS || !fg)
    {
        std::fprintf(stderr, "xefgSwapChainD3D12CreateContext failed: %d\n", int(result));
        return 1;
    }
    runtime.xefgSwapChainSetLoggingCallback(fg, XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING, LogCallback, &runtime);

    xell_context_handle_t xell = nullptr;
    const auto xellResult = runtime.xellD3D12CreateContext(fixture.device.Get(), &xell);
    if (xellResult == XELL_RESULT_SUCCESS && xell)
    {
        xell_sleep_params_t sleep{};
        sleep.bLowLatencyMode = 1;
        runtime.xellSetSleepMode(xell, &sleep);
        runtime.xefgSwapChainSetLatencyReduction(fg, xell);
    }

    const bool uiModeExplicit = !EqualsIgnoreCase(options.uiMode, "auto");
    xefg_swapchain_ui_mode_t uiMode = XEFG_SWAPCHAIN_UI_MODE_AUTO;
    if (EqualsIgnoreCase(options.uiMode, "none")) uiMode = XEFG_SWAPCHAIN_UI_MODE_NONE;
    else if (EqualsIgnoreCase(options.uiMode, "hudless-uitexture")) uiMode = XEFG_SWAPCHAIN_UI_MODE_HUDLESS_UITEXTURE;
    else if (EqualsIgnoreCase(options.uiMode, "backbuffer-hudless"))
        uiMode = XEFG_SWAPCHAIN_UI_MODE_BACKBUFFER_HUDLESS;
    if (!uiModeExplicit && !options.ui) uiMode = XEFG_SWAPCHAIN_UI_MODE_NONE;

    // Mode 4 interpolates the hudless color and extracts UI from the swap-chain
    // backbuffer, so it tags hudless color but never a UI texture.
    const bool sceneUi = options.ui && uiMode != XEFG_SWAPCHAIN_UI_MODE_NONE;
    const bool tagUi = sceneUi && uiMode != XEFG_SWAPCHAIN_UI_MODE_BACKBUFFER_HUDLESS;
    options.ui = sceneUi;

    if (options.calibration && (options.width != 640 || options.height != 360
        || uiMode != XEFG_SWAPCHAIN_UI_MODE_HUDLESS_UITEXTURE || !options.ui || !options.dumpInputs))
    {
        std::fprintf(stderr,
            "--calibration requires 640x360, --ui-mode hudless-uitexture (mode 3), and --dump-inputs\n");
        return 2;
    }

    xefg_swapchain_d3d12_init_params_t initParameters{};
    initParameters.pApplicationSwapChain = nullptr;
    initParameters.initFlags = 0;
    initParameters.maxInterpolatedFrames = 1;
    initParameters.uiMode = uiMode;
    bool initializationFallback = false;
    result = runtime.xefgSwapChainD3D12InitFromSwapChainDesc(fg, window.window, &chainDescription, nullptr,
        fixture.queue.Get(), captureFactory.Get(), &initParameters);
    if (result != XEFG_SWAPCHAIN_RESULT_SUCCESS)
    {
        // Retry with the maximum interpolation count but never silently switch
        // an explicit UI mode to AUTO: the requested mode is part of the run.
        initializationFallback = true;
        initParameters.maxInterpolatedFrames = XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES;
        result = runtime.xefgSwapChainD3D12InitFromSwapChainDesc(fg, window.window, &chainDescription, nullptr,
            fixture.queue.Get(), captureFactory.Get(), &initParameters);
    }
    if (result != XEFG_SWAPCHAIN_RESULT_SUCCESS)
    {
        std::fprintf(stderr, "XeFG initialization failed for UI mode %d: %d\n", int(uiMode), int(result));
        return 1;
    }

    ComPtr<IDXGISwapChain3> proxy;
    if (runtime.xefgSwapChainD3D12GetSwapChainPtr(fg, IID_PPV_ARGS(&proxy)) != XEFG_SWAPCHAIN_RESULT_SUCCESS || !proxy)
    {
        std::fprintf(stderr, "xefgSwapChainD3D12GetSwapChainPtr failed\n");
        return 1;
    }
    runtime.xefgSwapChainSetNumInterpolatedFrames(fg, 1);
    runtime.xefgSwapChainSetEnabled(fg, 1);
    if (uiMode != XEFG_SWAPCHAIN_UI_MODE_NONE)
        runtime.xefgSwapChainSetUiCompositionState(fg, XEFG_SWAPCHAIN_UI_COMPOSITION_STATE_ENABLED);
    int interpolationOnlyResult = 0;
    if (options.showInterpolationOnly)
        interpolationOnlyResult = int(runtime.xefgSwapChainEnableDebugFeature(
            fg, XEFG_SWAPCHAIN_DEBUG_FEATURE_SHOW_ONLY_INTERPOLATION, 1, nullptr));
    int tagInterpolatedResult = 0;
    if (options.tagInterpolated)
        tagInterpolatedResult = int(runtime.xefgSwapChainEnableDebugFeature(
            fg, XEFG_SWAPCHAIN_DEBUG_FEATURE_TAG_INTERPOLATED_FRAMES, 1, nullptr));
    std::vector<std::string> frameStatuses;

    xefg_swapchain_properties_t properties{};
    runtime.xefgSwapChainD3D12GetProperties(fg, &initParameters, 0, 0, DXGI_FORMAT_UNKNOWN, &properties);
    std::printf("XeFG ready: uiMode=%d maxInterpolations=%u descriptors=%u fallback=%s\n",
        int(uiMode), properties.maxSupportedInterpolations, properties.requiredDescriptorCount,
        initializationFallback ? "yes" : "no");

    const uint32_t motionWidth = options.highResMotionVectors ? options.width
        : std::min(options.motionWidth, options.width);
    const uint32_t motionHeight = options.highResMotionVectors ? options.height
        : std::min(options.motionHeight, options.height);
    auto colorTexture = CreateTexture(fixture.device.Get(), options.width, options.height,
        DXGI_FORMAT_B8G8R8A8_UNORM, D3D12_RESOURCE_STATE_COPY_DEST);
    auto uiTexture = CreateTexture(fixture.device.Get(), options.width, options.height,
        DXGI_FORMAT_B8G8R8A8_UNORM, D3D12_RESOURCE_STATE_COPY_DEST);
    auto depthTexture = CreateTexture(fixture.device.Get(), motionWidth, motionHeight,
        DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto motionTexture = CreateTexture(fixture.device.Get(), motionWidth, motionHeight,
        DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!colorTexture || !uiTexture || !depthTexture || !motionTexture)
    {
        std::fprintf(stderr, "input texture creation failed\n");
        return 1;
    }

    const uint32_t totalFrames = options.warmup + options.frames;
    std::vector<CalibrationFrame> calibrationFrames;
    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        const bool recorded = frame >= options.warmup;
        dispatcher.SetFrame(frame);

        oracle::SceneOptions sceneOptions;
        sceneOptions.width = options.width;
        sceneOptions.height = options.height;
        sceneOptions.highResMotionVectors = options.highResMotionVectors;
        sceneOptions.motionWidth = motionWidth;
        sceneOptions.motionHeight = motionHeight;
        sceneOptions.staticScene = options.staticScene;
        sceneOptions.smoothContent = options.scene == "smooth";
        sceneOptions.velocityX = options.velocityX;
        sceneOptions.velocityY = options.velocityY;
        sceneOptions.ui = options.ui;
        sceneOptions.animateUi = options.animateUi;
        sceneOptions.flipMotionVectors = options.flipMotionVectors;
        sceneOptions.calibration = options.calibration;
        sceneOptions.frameIndex = frame;

        oracle::SceneFrame scene;
        oracle::GenerateScene(sceneOptions, scene);

        if (options.calibration)
        {
            wchar_t name[64]{};
            swprintf_s(name, L"mid_%05u.bgra", frame);
            const std::filesystem::path path = calibrationDirectory / name;
            if (!oracle::WriteRawFile(path.wstring(), scene.expectedMid.data(), scene.expectedMid.size()))
            {
                std::fprintf(stderr, "calibration midpoint dump failed at frame %u\n", frame);
                return 1;
            }
            calibrationFrames.push_back({frame, frame == 0 || frame == options.resetFrame});
        }

        if (options.dumpInputs)
        {
            std::string dumpError;
            if (!WriteApplicationInputs(applicationInputsDirectory, frame, scene, dumpError))
            {
                std::fprintf(stderr, "application input dump failed: %s\n", dumpError.c_str());
                return 1;
            }
        }

        if (!fixture.SubmitAndWait())
        {
            std::fprintf(stderr, "command list reset failed\n");
            return 1;
        }
        std::vector<ComPtr<ID3D12Resource>> keepAlive;
        const bool uploaded =
            RecordUpload(fixture, colorTexture.Get(), scene.hudlessColor.data(), options.width * 4, options.height, keepAlive)
            && RecordUpload(fixture, uiTexture.Get(), scene.uiColor.data(), options.width * 4, options.height, keepAlive)
            && RecordUpload(fixture, depthTexture.Get(), scene.depth.data(), motionWidth * 4, motionHeight, keepAlive)
            && RecordUpload(fixture, motionTexture.Get(), scene.motion.data(), motionWidth * 4, motionHeight, keepAlive);
        if (!uploaded)
        {
            std::fprintf(stderr, "input upload failed\n");
            return 1;
        }
        Barrier(fixture.commands.Get(), colorTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(fixture.commands.Get(), uiTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(fixture.commands.Get(), depthTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(fixture.commands.Get(), motionTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE);

        ComPtr<ID3D12Resource> backbuffer;
        if (FAILED(proxy->GetBuffer(proxy->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&backbuffer))))
        {
            std::fprintf(stderr, "proxy back buffer unavailable\n");
            return 1;
        }
        Barrier(fixture.commands.Get(), backbuffer.Get(), D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_COPY_DEST);
        if (!RecordUpload(fixture, backbuffer.Get(), scene.backbuffer.data(), options.width * 4, options.height, keepAlive))
        {
            std::fprintf(stderr, "back buffer upload failed\n");
            return 1;
        }
        Barrier(fixture.commands.Get(), backbuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PRESENT);

        const auto tag = [&](xefg_swapchain_resource_type_t type, ID3D12Resource* resource, uint32_t width, uint32_t height) {
            xefg_swapchain_d3d12_resource_data_t data{};
            data.type = type;
            data.validity = XEFG_SWAPCHAIN_RV_ONLY_NOW;
            data.resourceBase = {0, 0};
            data.resourceSize = {width, height};
            data.pResource = resource;
            data.incomingState = D3D12_RESOURCE_STATE_COPY_SOURCE;
            return runtime.xefgSwapChainD3D12TagFrameResource(fg, fixture.commands.Get(), frame, &data);
        };
        if (tag(XEFG_SWAPCHAIN_RES_DEPTH, depthTexture.Get(), motionWidth, motionHeight) != XEFG_SWAPCHAIN_RESULT_SUCCESS
            || tag(XEFG_SWAPCHAIN_RES_MOTION_VECTOR, motionTexture.Get(), motionWidth, motionHeight)
                != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        {
            std::fprintf(stderr, "depth/motion tagging failed at frame %u\n", frame);
            return 1;
        }
        if (options.ui)
        {
            if (tag(XEFG_SWAPCHAIN_RES_HUDLESS_COLOR, colorTexture.Get(), options.width, options.height)
                != XEFG_SWAPCHAIN_RESULT_SUCCESS)
            {
                std::fprintf(stderr, "hudless color tagging failed at frame %u\n", frame);
                return 1;
            }
        }
        if (tagUi)
        {
            if (tag(XEFG_SWAPCHAIN_RES_UI, uiTexture.Get(), options.width, options.height)
                != XEFG_SWAPCHAIN_RESULT_SUCCESS)
            {
                std::fprintf(stderr, "UI tagging failed at frame %u\n", frame);
                return 1;
            }
        }
        if (!Submit(fixture))
        {
            std::fprintf(stderr, "input submission failed\n");
            return 1;
        }

        xefg_swapchain_frame_constant_data_t constants{};
        for (int i = 0; i < 16; ++i)
        {
            constants.viewMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            constants.projectionMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        }
        constants.jitterOffsetX = 0.0f;
        constants.jitterOffsetY = 0.0f;
        constants.motionVectorScaleX = 1.0f;
        constants.motionVectorScaleY = 1.0f;
        constants.resetHistory = (frame == 0 || frame == options.resetFrame) ? 1u : 0u;
        constants.frameRenderTime = static_cast<float>(options.pacingMilliseconds);
        runtime.xefgSwapChainTagFrameConstants(fg, frame, &constants);
        runtime.xefgSwapChainSetPresentId(fg, frame);

        capture->SetFrameIndex(frame);
        capture->SetEnabled(recorded);
        LARGE_INTEGER presentStart{}, presentEnd{};
        QueryPerformanceCounter(&presentStart);
        const HRESULT presented = proxy->Present(0, 0);
        QueryPerformanceCounter(&presentEnd);

        xefg_swapchain_present_status_t status{};
        const auto statusResult = runtime.xefgSwapChainGetLastPresentStatus(fg, &status);
        std::ostringstream statusJson;
        statusJson << "{\"application_frame\": " << frame
                   << ", \"present_result\": " << static_cast<int32_t>(presented)
                   << ", \"status_result\": " << int(statusResult)
                   << ", \"frames_presented\": " << status.framesPresented
                   << ", \"frame_generation_result\": " << int(status.frameGenResult)
                   << ", \"is_frame_gen_enabled\": " << status.isFrameGenEnabled
                   << ", \"present_start_qpc\": " << presentStart.QuadPart
                   << ", \"present_end_qpc\": " << presentEnd.QuadPart << "}";
        frameStatuses.push_back(statusJson.str());
        std::printf("frame %u: present=0x%08lx framesPresented=%u result=%d\n", frame,
            static_cast<unsigned long>(presented), status.framesPresented, int(status.frameGenResult));
        if (options.pacingMilliseconds) Sleep(options.pacingMilliseconds);
    }

    if (options.dumpInputs)
    {
        std::string manifestError;
        if (!WriteApplicationInputsManifest(applicationInputsDirectory, options, motionWidth, motionHeight,
            totalFrames, manifestError))
        {
            std::fprintf(stderr, "application input manifest failed: %s\n", manifestError.c_str());
            return 1;
        }
    }

    if (options.calibration)
    {
        std::string manifestError;
        if (!WriteCalibrationManifest(calibrationDirectory, calibrationFrames, options.staticScene,
            manifestError))
        {
            std::fprintf(stderr, "calibration manifest failed: %s\n", manifestError.c_str());
            return 1;
        }
    }

    fixture.SignalAndWait();
    proxy.Reset();
    std::string tensorError;
    const bool tensorOk = dispatcher.FinishTensors(tensorError);
    if (!tensorOk)
        std::fprintf(stderr, "tensor dumps unverified: %s\n", tensorError.c_str());

    std::ostringstream meta;
    meta << "{\n";
    meta << "  \"adapter\": \"" << probe::JsonEscape(adapterName) << "\",\n";
    meta << "  \"vendor_id\": " << selectedDescription.VendorId << ",\n";
    meta << "  \"device_id\": " << selectedDescription.DeviceId << ",\n";
    meta << "  \"xefg_family\": \"" << probe::JsonEscape(family) << "\",\n";
    meta << "  \"xefg_dll_sha256\": \"" << moduleHash << "\",\n";
    meta << "  \"application_inputs\": " << (options.dumpInputs ? "true" : "false") << ",\n";
    meta << "  \"configuration\": {\"width\": " << options.width << ", \"height\": " << options.height
         << ", \"motion_width\": " << motionWidth << ", \"motion_height\": " << motionHeight
         << ", \"scene\": \"" << options.scene << "\", \"scene_revision\": 1"
         << ", \"static\": " << (options.staticScene ? "true" : "false")
         << ", \"ui\": " << (options.ui ? "true" : "false")
         << ", \"ui_texture_tagged\": " << (tagUi ? "true" : "false")
         << ", \"ui_mode\": " << int(uiMode)
         << ", \"ui_mode_requested\": \"" << probe::JsonEscape(options.uiMode) << "\""
         << ", \"ui_mode_explicit\": " << (uiModeExplicit ? "true" : "false")
         << ", \"max_interpolated_frames\": " << initParameters.maxInterpolatedFrames
         << ", \"initialization_fallback\": " << (initializationFallback ? "true" : "false")
         << ", \"animate_ui\": " << (options.animateUi ? "true" : "false")
         << ", \"dump_inputs\": " << (options.dumpInputs ? "true" : "false")
         << ", \"calibration\": " << (options.calibration ? "true" : "false")
         << ", \"tag_interpolated\": " << (options.tagInterpolated ? "true" : "false")
         << ", \"reset_frame\": " << options.resetFrame
         << ", \"frames\": " << options.frames << ", \"warmup\": " << options.warmup
         << ", \"pacing_ms\": " << options.pacingMilliseconds
         << ", \"only_interpolation\": " << (options.showInterpolationOnly ? "true" : "false")
         << ", \"velocity\": [" << options.velocityX << ", " << options.velocityY << "]"
         << ", \"flip_mv\": " << (options.flipMotionVectors ? "true" : "false")
         << ", \"tensors\": " << (options.tensors ? "true" : "false") << "},\n";
    meta << "  \"programs\": " << relay.ProgramCount() << ",\n";
    meta << "  \"parameter_captures\": " << relay.ParameterCaptureCount() << ",\n";
    meta << "  \"dispatches\": " << dispatcher.DispatchCount() << ",\n";
    meta << "  \"distinct_programs\": " << dispatcher.DistinctPrograms() << ",\n";
    meta << "  \"patched_tables\": " << dispatcher.PatchedListTables() << ",\n";
    meta << "  \"tensor_files\": " << dispatcher.TensorFiles() << ",\n";
    meta << "  \"tensor_bytes\": " << dispatcher.TensorBytes() << ",\n";
    meta << "  \"provider_queues\": " << dispatcher.ProviderQueues() << ",\n";
    meta << "  \"tensor_wait\": " << (tensorOk ? "true" : "false") << ",\n";
    meta << "  \"tensor_wait_error\": \"" << probe::JsonEscape(tensorError) << "\",\n";
    meta << "  \"trace_error\": \"" << probe::JsonEscape(dispatcher.Error()) << "\"\n";
    meta << "}\n";
    trace.SetMeta(meta.str());

    {
        std::ofstream logFile(outputDirectory / L"xefg_log.txt", std::ios::binary | std::ios::trunc);
        if (logFile)
            for (const auto& line : runtime.TakeLog()) logFile << line << '\n';
    }

    std::printf("programs=%zu parameterCaptures=%zu dispatches=%llu distinct=%zu\n",
        relay.ProgramCount(), relay.ParameterCaptureCount(),
        static_cast<unsigned long long>(dispatcher.DispatchCount()), dispatcher.DistinctPrograms());

    dispatcher.Detach();
    if (fg) runtime.xefgSwapChainDestroy(fg);
    capture->SetEnabled(false);
    const bool presentsOk = WritePresentsManifest(outputDirectory, *capture,
        options.showInterpolationOnly, interpolationOnlyResult, options.tagInterpolated,
        tagInterpolatedResult, qpcFrequency.QuadPart, frameStatuses);
    if (!presentsOk) std::fprintf(stderr, "could not write presents.json\n");
    if (xell) runtime.xellDestroyContext(xell);
    captureFactory.Reset();
    relay.Restore();
    trace.Close();
    runtime.Unload();
    std::printf("trace written to %s\n", (outputDirectory / L"trace").string().c_str());
    return presentsOk && capture->Error().empty() ? 0 : 1;
}
