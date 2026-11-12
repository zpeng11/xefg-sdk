// XefgSamplerProbe: standalone, one-shot DX12 texture sampler evidence tool.
//
// It has no dependency on the XeSS-FG runtime, the provider DLL or the capture
// probe modules.  One invocation uploads exactly one BGRA8_UNORM texture,
// evaluates a fixed-format request stream (SampleLevel(uv, 0) or Load(x, y))
// through one compute dispatch, and writes the fixed-format samples.bin plus
// metadata.  The fixed sampler configuration is documented in
// shaders/sampler_probe.hlsl and mirrored in the static sampler below.
#include "common.h"
#include "manifest.h"

#include <d3dcompiler.h>
#include <d3d12sdklayers.h>
#include <winver.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace
{
constexpr uint32_t kRequestMagic = 0x58534647u;
constexpr uint32_t kVersion = 1;
constexpr uint32_t kRequestStride = 32;
constexpr uint32_t kRecordStride = 80;
constexpr uint32_t kThreadsPerGroup = 64;
constexpr uint32_t kMaxDimension = 16384;
constexpr UINT kTextureSrvSlot = 0;
constexpr UINT kRequestSrvSlot = 1;
constexpr UINT kOutputUavSlot = 2;
constexpr UINT kDescriptorCount = 3;

constexpr UINT kShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;

const char* kUsage =
    "usage:\n"
    "  XefgSamplerProbe --texture PATH --width W --height H --requests PATH\n"
    "                   --out NEW_DIR [--debug] [--adapter TEXT]\n"
    "  XefgSamplerProbe --self-test --out NEW_DIR [--debug] [--warp] [--adapter TEXT]\n"
    "\n"
    "Texture bytes are raw BGRA8 (B,G,R,A), width*height*4 bytes exactly.\n"
    "Requests: header {0x58534647, 1, count, 32}; 32-byte records\n"
    "  {uint32 id, uint32 op(0=sample,1=load), float u, float v,\n"
    "   int32 x, int32 y, uint32 reserved0, uint32 reserved1}.\n"
    "Output: samples.bin header {0x58534647, 1, count, 80}; 80-byte records\n"
    "  {id, op, u, v, float4 normalized, float4 units=saturate(normalized)*255,\n"
    "   int4 truncated, uint4 padding}.\n";

#pragma pack(push, 1)
struct Request
{
    uint32_t id;
    uint32_t op;
    float u;
    float v;
    int32_t x;
    int32_t y;
    uint32_t reserved0;
    uint32_t reserved1;
};

struct Record
{
    uint32_t id;
    uint32_t op;
    float u;
    float v;
    float normalized[4];
    float units[4];
    int32_t truncated[4];
    uint32_t padding[4];
};
#pragma pack(pop)

static_assert(sizeof(Request) == kRequestStride, "request record must be 32 bytes");
static_assert(sizeof(Record) == kRecordStride, "output record must be 80 bytes");

struct Options
{
    std::string texturePath;
    std::string requestsPath;
    std::string outputDirectory;
    std::string adapterFilter;
    uint32_t width = 0;
    uint32_t height = 0;
    bool debug = false;
    bool warp = false;
    bool selfTest = false;
    bool help = false;
};

struct DeviceContext
{
    std::unique_ptr<harness::WarpFixture> fixture;
    DXGI_ADAPTER_DESC1 description{};
    LUID luid{};
    std::string adapterName;
    bool warp = false;
};

struct ShaderInfo
{
    std::filesystem::path path;
    std::string sourceHash;
    ComPtr<ID3DBlob> blob;
    std::string warnings;
    UINT flags = kShaderFlags;
    std::string dllVersion;
    bool dxbc = false;
    bool dxil = false;
};

struct DebugReport
{
    bool layer = false;
    uint32_t errors = 0;
    uint32_t warnings = 0;
    std::vector<std::string> messages;
};

struct SelfTestCase
{
    std::string name;
    bool passed = false;
    std::string detail;
};

std::wstring Widen(const std::string& text)
{
    if (text.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, 0, text.data(),
        static_cast<int>(text.size()), nullptr, 0);
    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
        result.data(), count);
    return result;
}

std::string Narrow(const std::wstring& text)
{
    if (text.empty()) return {};
    const int count = WideCharToMultiByte(CP_UTF8, 0, text.data(),
        static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<size_t>(count), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
        result.data(), count, nullptr, nullptr);
    return result;
}

std::filesystem::path Utf8Path(const std::string& text)
{
    return std::filesystem::path(Widen(text));
}

bool EqualsIgnoreCase(const std::string& text, const std::string& other)
{
    if (text.size() != other.size()) return false;
    for (size_t i = 0; i < text.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(text[i]))
            != std::tolower(static_cast<unsigned char>(other[i]))) return false;
    return true;
}

bool ContainsIgnoreCase(const std::string& text, const std::string& needle)
{
    std::string lower;
    lower.reserve(text.size());
    for (const char value : text)
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(value))));
    return lower.find(needle) != std::string::npos;
}

std::filesystem::path ExecutableDirectory()
{
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
}

std::string HrText(const char* what, HRESULT hr)
{
    char text[160]{};
    std::snprintf(text, sizeof(text), "%s failed (hr=0x%08lx)", what,
        static_cast<unsigned long>(hr));
    return text;
}

std::string BlobText(ID3DBlob* blob)
{
    if (!blob || !blob->GetBufferPointer()) return {};
    return std::string(static_cast<const char*>(blob->GetBufferPointer()),
        blob->GetBufferSize());
}

std::string LuidText(const LUID& luid)
{
    char text[48]{};
    std::snprintf(text, sizeof(text), "%08lx:%08lx",
        static_cast<unsigned long>(luid.HighPart),
        static_cast<unsigned long>(luid.LowPart));
    return text;
}

bool ParseU32(const std::string& text, uint32_t& value)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

bool ParseArgs(int argc, char** argv, Options& options, std::string& error)
{
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        auto value = [&](std::string& target) {
            if (index + 1 >= argc)
            {
                error = "missing value for " + argument;
                return false;
            }
            target = argv[++index];
            return true;
        };
        if (argument == "--texture")
        {
            if (!value(options.texturePath)) return false;
        }
        else if (argument == "--requests")
        {
            if (!value(options.requestsPath)) return false;
        }
        else if (argument == "--out")
        {
            if (!value(options.outputDirectory)) return false;
        }
        else if (argument == "--adapter")
        {
            if (!value(options.adapterFilter)) return false;
        }
        else if (argument == "--width" || argument == "--height")
        {
            std::string text;
            if (!value(text)) return false;
            uint32_t parsed = 0;
            if (!ParseU32(text, parsed) || parsed == 0 || parsed > kMaxDimension)
            {
                error = argument + " must be an integer in [1, 16384]";
                return false;
            }
            (argument == "--width" ? options.width : options.height) = parsed;
        }
        else if (argument == "--debug") options.debug = true;
        else if (argument == "--warp") options.warp = true;
        else if (argument == "--self-test") options.selfTest = true;
        else if (argument == "--help" || argument == "-h") options.help = true;
        else
        {
            error = "unknown argument: " + argument;
            return false;
        }
    }
    if (options.help) return true;
    if (options.warp && !options.adapterFilter.empty())
    {
        error = "--warp cannot be combined with --adapter";
        return false;
    }
    if (options.outputDirectory.empty())
    {
        error = "--out is required";
        return false;
    }
    if (options.selfTest)
    {
        if (!options.texturePath.empty() || !options.requestsPath.empty()
            || options.width || options.height)
        {
            error = "--self-test does not take capture inputs";
            return false;
        }
        return true;
    }
    if (options.texturePath.empty() || options.requestsPath.empty())
    {
        error = "--texture and --requests are required";
        return false;
    }
    if (!options.width || !options.height)
    {
        error = "--width and --height are required and must be positive";
        return false;
    }
    return true;
}

bool PrepareNewDirectory(const std::filesystem::path& directory, std::string& error)
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

bool ReadFileBytes(const std::filesystem::path& path, std::vector<uint8_t>& bytes,
    std::string& error)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        error = "could not open " + path.string();
        return false;
    }
    const std::streampos size = file.tellg();
    if (size < 0)
    {
        error = "could not size " + path.string();
        return false;
    }
    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    if (!bytes.empty())
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file)
    {
        error = "could not read " + path.string();
        return false;
    }
    return true;
}

std::string D3DCompilerVersion()
{
    wchar_t systemDirectory[MAX_PATH]{};
    GetSystemDirectoryW(systemDirectory, MAX_PATH);
    const std::wstring path = std::wstring(systemDirectory) + L"\\d3dcompiler_47.dll";
    DWORD handle = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &handle);
    if (!size) return "unknown";
    std::vector<uint8_t> data(size);
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data())) return "unknown";
    VS_FIXEDFILEINFO* info = nullptr;
    UINT infoSize = 0;
    if (!VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &infoSize)
        || !info || infoSize < sizeof(VS_FIXEDFILEINFO)) return "unknown";
    char text[64]{};
    std::snprintf(text, sizeof(text), "%u.%u.%u.%u",
        static_cast<unsigned>(HIWORD(info->dwFileVersionMS)),
        static_cast<unsigned>(LOWORD(info->dwFileVersionMS)),
        static_cast<unsigned>(HIWORD(info->dwFileVersionLS)),
        static_cast<unsigned>(LOWORD(info->dwFileVersionLS)));
    return text;
}

bool CompileShader(ShaderInfo& shader, std::string& error)
{
    shader.path = ExecutableDirectory() / L"shaders" / L"sampler_probe.hlsl";
    if (!std::filesystem::exists(shader.path))
        shader.path = ExecutableDirectory() / L"sampler_probe.hlsl";
    if (!std::filesystem::exists(shader.path))
    {
        error = "shader source not found: " + shader.path.string();
        return false;
    }
    shader.sourceHash = render_validation::Sha256File(shader.path);
    if (shader.sourceHash.empty())
    {
        error = "could not hash shader source: " + shader.path.string();
        return false;
    }
    ComPtr<ID3DBlob> errors;
    const HRESULT hr = D3DCompileFromFile(shader.path.c_str(), nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE, "CSMain", "cs_5_1", shader.flags, 0,
        &shader.blob, &errors);
    if (FAILED(hr))
    {
        error = "shader compile failed: " + BlobText(errors.Get());
        return false;
    }
    shader.warnings = BlobText(errors.Get());
    shader.dllVersion = D3DCompilerVersion();
    const auto* bytes = static_cast<const uint8_t*>(shader.blob->GetBufferPointer());
    const size_t size = shader.blob->GetBufferSize();
    shader.dxbc = size >= 4 && std::memcmp(bytes, "DXBC", 4) == 0;
    const uint8_t needle[] = {'D', 'X', 'I', 'L'};
    shader.dxil = std::search(bytes, bytes + size, needle, needle + 4) != bytes + size;
    return true;
}

bool CreateDeviceContext(const Options& options, DeviceContext& context, std::string& error)
{
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr))
    {
        error = HrText("CreateDXGIFactory1", hr);
        return false;
    }

    DXGI_ADAPTER_DESC1 description{};
    context.fixture = std::make_unique<harness::WarpFixture>();
    if (options.warp)
    {
        ComPtr<IDXGIFactory4> factory4;
        ComPtr<IDXGIAdapter> warpAdapter;
        if (FAILED(factory.As(&factory4))
            || FAILED(factory4->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter))))
        {
            error = "WARP adapter is not available";
            return false;
        }
        ComPtr<IDXGIAdapter1> warpAdapter1;
        if (SUCCEEDED(warpAdapter.As(&warpAdapter1)))
            warpAdapter1->GetDesc1(&description);
        if (!context.fixture->Initialize(options.debug, warpAdapter.Get()))
        {
            error = "WARP D3D12 device/queue initialization failed";
            return false;
        }
    }
    else
    {
        struct Candidate
        {
            ComPtr<IDXGIAdapter1> adapter;
            DXGI_ADAPTER_DESC1 description{};
        };
        std::vector<Candidate> candidates;
        for (UINT index = 0;; ++index)
        {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 adapterDescription{};
            if (FAILED(adapter->GetDesc1(&adapterDescription))) continue;
            if (adapterDescription.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            candidates.push_back({adapter, adapterDescription});
        }
        if (candidates.empty())
        {
            error = "no hardware adapter is available (use --warp explicitly for selftest)";
            return false;
        }

        size_t selected = candidates.size();
        if (!options.adapterFilter.empty())
        {
            // An explicit adapter is an exact, case-insensitive match.  There
            // is no fallback to another adapter and never a fallback to WARP.
            for (size_t index = 0; index < candidates.size(); ++index)
            {
                if (EqualsIgnoreCase(Narrow(candidates[index].description.Description),
                    options.adapterFilter))
                {
                    selected = index;
                    break;
                }
            }
            if (selected == candidates.size())
            {
                error = "no hardware adapter exactly matches '" + options.adapterFilter
                    + "'; available:";
                for (const Candidate& candidate : candidates)
                    error += " '" + Narrow(candidate.description.Description) + "'";
                return false;
            }
        }
        else
        {
            // The plan's reference GPU is the RX 6800 XT.  Prefer it when it
            // is present; otherwise take the first hardware adapter.  WARP is
            // only reached through the explicit --warp switch.
            for (size_t index = 0; index < candidates.size(); ++index)
            {
                if (ContainsIgnoreCase(Narrow(candidates[index].description.Description),
                    "6800 xt"))
                {
                    selected = index;
                    break;
                }
            }
            if (selected == candidates.size()) selected = 0;
        }

        ComPtr<IDXGIAdapter1> adapter = candidates[selected].adapter;
        description = candidates[selected].description;
        if (!context.fixture->Initialize(options.debug, adapter.Get()))
        {
            error = "D3D12 device/queue initialization failed on adapter '"
                + Narrow(description.Description) + "'";
            return false;
        }
    }

    if (!context.fixture->device || !context.fixture->queue
        || !context.fixture->allocator || !context.fixture->commands
        || !context.fixture->fence)
    {
        error = "D3D12 fixture is incomplete";
        return false;
    }
    context.description = description;
    context.luid = context.fixture->device->GetAdapterLuid();
    context.adapterName = Narrow(description.Description);
    context.warp = options.warp;
    return true;
}

bool CreateBuffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heapType,
    D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& resource, std::string& error,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heapType;
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = bytes;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &description, state, nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(hr))
    {
        error = HrText("CreateCommittedResource(buffer)", hr);
        return false;
    }
    return true;
}

bool CreateTextureWithUpload(ID3D12Device* device, const uint8_t* bytes, UINT width,
    UINT height, ComPtr<ID3D12Resource>& texture, ComPtr<ID3D12Resource>& upload,
    std::string& error)
{
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = width;
    description.Height = height;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    description.Flags = D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &description, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture));
    if (FAILED(hr))
    {
        error = HrText("CreateCommittedResource(texture)", hr);
        return false;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 totalBytes = 0;
    UINT rowCount = 0;
    UINT64 rowBytes = 0;
    device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rowCount,
        &rowBytes, &totalBytes);
    if (rowCount != height || rowBytes < static_cast<UINT64>(width) * 4)
    {
        error = "unexpected copyable-footprint layout";
        return false;
    }
    if (!CreateBuffer(device, totalBytes, D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ, upload, error)) return false;
    void* mapped = nullptr;
    hr = upload->Map(0, nullptr, &mapped);
    if (FAILED(hr))
    {
        error = HrText("Map(texture upload)", hr);
        return false;
    }
    auto* destination = static_cast<uint8_t*>(mapped);
    for (UINT row = 0; row < height; ++row)
    {
        std::memcpy(destination + footprint.Offset
                + static_cast<UINT64>(row) * footprint.Footprint.RowPitch,
            bytes + static_cast<uint64_t>(row) * width * 4,
            static_cast<size_t>(width) * 4);
    }
    upload->Unmap(0, nullptr);
    return true;
}

void CollectDebugMessages(ID3D12Device* device, bool enabled, DebugReport& report)
{
    report.layer = enabled;
    if (!enabled) return;
    ComPtr<ID3D12InfoQueue> queue;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&queue))) || !queue) return;
    const UINT64 count = queue->GetNumStoredMessages();
    for (UINT64 index = 0; index < count; ++index)
    {
        SIZE_T size = 0;
        if (FAILED(queue->GetMessage(index, nullptr, &size)) || !size) continue;
        std::vector<uint8_t> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(queue->GetMessage(index, message, &size))) continue;
        const D3D12_MESSAGE_SEVERITY severity = message->Severity;
        if (severity <= D3D12_MESSAGE_SEVERITY_ERROR) ++report.errors;
        else if (severity == D3D12_MESSAGE_SEVERITY_WARNING) ++report.warnings;
        if (severity <= D3D12_MESSAGE_SEVERITY_WARNING
            && report.messages.size() < 64 && message->pDescription)
            report.messages.emplace_back(message->pDescription);
    }
}

class Probe
{
public:
    bool Initialize(ID3D12Device* device, ID3DBlob* shader, std::string& error)
    {
        device_ = device;
        if (!CreateRootSignature(error)) return false;
        if (!CreatePipeline(shader, error)) return false;
        D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
        heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDescription.NumDescriptors = kDescriptorCount;
        heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        HRESULT hr = device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&heap_));
        if (FAILED(hr))
        {
            error = HrText("CreateDescriptorHeap", hr);
            return false;
        }
        increment_ = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return true;
    }

    bool Run(const uint8_t* textureBytes, UINT width, UINT height,
        const Request* requests, uint32_t count, std::vector<uint8_t>& records,
        std::string& error)
    {
        harness::WarpFixture& fixture = *fixture_;
        ComPtr<ID3D12Resource> texture;
        ComPtr<ID3D12Resource> textureUpload;
        if (!CreateTextureWithUpload(device_, textureBytes, width, height, texture,
            textureUpload, error)) return false;

        const UINT64 requestBytes = static_cast<UINT64>(std::max(count, 1u)) * kRequestStride;
        ComPtr<ID3D12Resource> requestBuffer;
        if (!CreateBuffer(device_, requestBytes, D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ, requestBuffer, error)) return false;
        void* mappedRequests = nullptr;
        HRESULT hr = requestBuffer->Map(0, nullptr, &mappedRequests);
        if (FAILED(hr))
        {
            error = HrText("Map(request upload)", hr);
            return false;
        }
        if (count)
            std::memcpy(mappedRequests, requests,
                static_cast<size_t>(count) * kRequestStride);
        requestBuffer->Unmap(0, nullptr);

        const UINT64 outputBytes = static_cast<UINT64>(std::max(count, 1u)) * kRecordStride;
        ComPtr<ID3D12Resource> output;
        if (!CreateBuffer(device_, outputBytes, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON, output, error,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return false;
        ComPtr<ID3D12Resource> readback;
        if (!CreateBuffer(device_, outputBytes, D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST, readback, error)) return false;

        D3D12_SHADER_RESOURCE_VIEW_DESC textureView{};
        textureView.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        textureView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        textureView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        textureView.Texture2D.MostDetailedMip = 0;
        textureView.Texture2D.MipLevels = 1;
        textureView.Texture2D.PlaneSlice = 0;
        textureView.Texture2D.ResourceMinLODClamp = 0.0f;
        device_->CreateShaderResourceView(texture.Get(), &textureView, CpuAt(kTextureSrvSlot));

        D3D12_SHADER_RESOURCE_VIEW_DESC requestView{};
        requestView.Format = DXGI_FORMAT_UNKNOWN;
        requestView.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        requestView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        requestView.Buffer.FirstElement = 0;
        requestView.Buffer.NumElements = std::max(count, 1u);
        requestView.Buffer.StructureByteStride = kRequestStride;
        requestView.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        device_->CreateShaderResourceView(requestBuffer.Get(), &requestView,
            CpuAt(kRequestSrvSlot));

        D3D12_UNORDERED_ACCESS_VIEW_DESC outputView{};
        outputView.Format = DXGI_FORMAT_UNKNOWN;
        outputView.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        outputView.Buffer.FirstElement = 0;
        outputView.Buffer.NumElements = std::max(count, 1u);
        outputView.Buffer.StructureByteStride = kRecordStride;
        outputView.Buffer.CounterOffsetInBytes = 0;
        outputView.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        device_->CreateUnorderedAccessView(output.Get(), nullptr, &outputView,
            CpuAt(kOutputUavSlot));

        ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
        hr = fixture.allocator->Reset();
        if (FAILED(hr)) { error = HrText("CommandAllocator::Reset", hr); return false; }
        hr = fixture.commands->Reset(fixture.allocator.Get(), nullptr);
        if (FAILED(hr)) { error = HrText("CommandList::Reset", hr); return false; }
        ID3D12GraphicsCommandList* list = fixture.commands.Get();
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(root_.Get());
        list->SetComputeRoot32BitConstant(0, count, 0);
        list->SetComputeRootDescriptorTable(1, GpuAt(kTextureSrvSlot));
        list->SetComputeRootDescriptorTable(2, GpuAt(kOutputUavSlot));
        list->SetPipelineState(pipeline_.Get());

        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = texture.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = textureUpload.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = textureFootprint(texture.Get());
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);

        D3D12_RESOURCE_BARRIER barriers[2]{};
        barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[0].Transition.pResource = texture.Get();
        barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ResourceBarrier(1, &barriers[0]);

        if (count)
        {
            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Transition.pResource = output.Get();
            barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list->ResourceBarrier(1, &barriers[1]);
            list->Dispatch((count + kThreadsPerGroup - 1) / kThreadsPerGroup, 1, 1);
            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[1].UAV.pResource = output.Get();
            list->ResourceBarrier(1, &barriers[1]);
        }
        barriers[0].Transition.pResource = output.Get();
        barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[0].Transition.StateBefore = count
            ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &barriers[0]);
        list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, outputBytes);
        hr = list->Close();
        if (FAILED(hr)) { error = HrText("CommandList::Close", hr); return false; }

        ID3D12CommandList* lists[] = {list};
        fixture.queue->ExecuteCommandLists(1, lists);
        if (!fixture.SignalAndWait())
        {
            error = "GPU submission did not complete";
            return false;
        }

        D3D12_RANGE readRange{0, static_cast<SIZE_T>(outputBytes)};
        void* mapped = nullptr;
        hr = readback->Map(0, &readRange, &mapped);
        if (FAILED(hr))
        {
            error = HrText("Map(readback)", hr);
            return false;
        }
        records.resize(static_cast<size_t>(count) * kRecordStride);
        if (!records.empty())
            std::memcpy(records.data(), mapped, records.size());
        const D3D12_RANGE writeRange{0, 0};
        readback->Unmap(0, &writeRange);
        return true;
    }

    void SetFixture(harness::WarpFixture* fixture) { fixture_ = fixture; }

private:
    D3D12_CPU_DESCRIPTOR_HANDLE CpuAt(UINT index) const
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = heap_->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(index) * increment_;
        return handle;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE GpuAt(UINT index) const
    {
        D3D12_GPU_DESCRIPTOR_HANDLE handle = heap_->GetGPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<UINT64>(index) * increment_;
        return handle;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT textureFootprint(ID3D12Resource* texture) const
    {
        D3D12_RESOURCE_DESC description = texture->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows = 0;
        UINT64 rowBytes = 0;
        UINT64 total = 0;
        device_->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
            &rowBytes, &total);
        return footprint;
    }

    bool CreateRootSignature(std::string& error)
    {
        D3D12_ROOT_PARAMETER parameters[3]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.ShaderRegister = 0;
        parameters[0].Constants.RegisterSpace = 0;
        parameters[0].Constants.Num32BitValues = 1;
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_DESCRIPTOR_RANGE srvRange{};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = 2;
        srvRange.BaseShaderRegister = 0;
        srvRange.RegisterSpace = 0;
        srvRange.OffsetInDescriptorsFromTableStart = 0;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[1].DescriptorTable.NumDescriptorRanges = 1;
        parameters[1].DescriptorTable.pDescriptorRanges = &srvRange;
        parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_DESCRIPTOR_RANGE uavRange{};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = 1;
        uavRange.BaseShaderRegister = 0;
        uavRange.RegisterSpace = 0;
        uavRange.OffsetInDescriptorsFromTableStart = 0;
        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[2].DescriptorTable.NumDescriptorRanges = 1;
        parameters[2].DescriptorTable.pDescriptorRanges = &uavRange;
        parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        // Fixed contract.  Filter 0x15 is D3D12_FILTER_MIN_MAG_MIP_LINEAR, a
        // non-comparison filter; ComparisonFunc is inert but fixed at
        // LESS_EQUAL.  W uses WRAP even though a 2D texture fetch uses w=0.
        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.MipLODBias = 0.0f;
        sampler.MaxAnisotropy = 16;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        sampler.MinLOD = 0.0f;
        sampler.MaxLOD = FLT_MAX;
        sampler.ShaderRegister = 0;
        sampler.RegisterSpace = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC description{};
        description.NumParameters = 3;
        description.pParameters = parameters;
        description.NumStaticSamplers = 1;
        description.pStaticSamplers = &sampler;
        description.Flags = D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> errors;
        HRESULT hr = D3D12SerializeRootSignature(&description, D3D_ROOT_SIGNATURE_VERSION_1_0,
            &signature, &errors);
        if (FAILED(hr))
        {
            error = "D3D12SerializeRootSignature: " + BlobText(errors.Get());
            return false;
        }
        hr = device_->CreateRootSignature(0, signature->GetBufferPointer(),
            signature->GetBufferSize(), IID_PPV_ARGS(&root_));
        if (FAILED(hr))
        {
            error = HrText("CreateRootSignature", hr);
            return false;
        }
        return true;
    }

    bool CreatePipeline(ID3DBlob* shader, std::string& error)
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
        description.pRootSignature = root_.Get();
        description.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
        const HRESULT hr = device_->CreateComputePipelineState(&description,
            IID_PPV_ARGS(&pipeline_));
        if (FAILED(hr))
        {
            error = HrText("CreateComputePipelineState", hr);
            return false;
        }
        return true;
    }

    ID3D12Device* device_ = nullptr;
    harness::WarpFixture* fixture_ = nullptr;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    ComPtr<ID3D12DescriptorHeap> heap_;
    UINT increment_ = 0;
};

float ByteToFloat(uint8_t value)
{
    return static_cast<float>(value) / 255.0f;
}

std::array<float, 4> NormalizedBgra(uint8_t b, uint8_t g, uint8_t r, uint8_t a)
{
    // BGRA8 memory-byte order maps to logical float4 channels (R, G, B, A).
    return {ByteToFloat(r), ByteToFloat(g), ByteToFloat(b), ByteToFloat(a)};
}

Request MakeSample(uint32_t id, float u, float v)
{
    Request request{};
    request.id = id;
    request.op = 0;
    request.u = u;
    request.v = v;
    return request;
}

Request MakeLoad(uint32_t id, int32_t x, int32_t y)
{
    Request request{};
    request.id = id;
    request.op = 1;
    request.x = x;
    request.y = y;
    return request;
}

bool Nearly(float actual, float expected, std::string& detail, const char* label,
    float tolerance = 2e-6f)
{
    if (std::fabs(actual - expected) <= tolerance) return true;
    char text[256]{};
    std::snprintf(text, sizeof(text), "%s: got %.9g expected %.9g", label,
        static_cast<double>(actual), static_cast<double>(expected));
    detail = text;
    return false;
}

bool CheckRecord(const std::vector<uint8_t>& records, size_t index,
    const Request& request, const std::array<float, 4>& expected, std::string& detail)
{
    if ((index + 1) * kRecordStride > records.size())
    {
        detail = "record list is too short";
        return false;
    }
    Record record{};
    std::memcpy(&record, records.data() + index * kRecordStride, kRecordStride);
    if (record.id != request.id)
    {
        detail = "id mismatch";
        return false;
    }
    if (record.op != request.op)
    {
        detail = "op mismatch";
        return false;
    }
    if (std::memcmp(&record.u, &request.u, sizeof(float)) != 0
        || std::memcmp(&record.v, &request.v, sizeof(float)) != 0)
    {
        detail = "u/v bits changed";
        return false;
    }
    const char names[4] = {'r', 'g', 'b', 'a'};
    for (int channel = 0; channel < 4; ++channel)
    {
        char label[32]{};
        std::snprintf(label, sizeof(label), "normalized.%c", names[channel]);
        if (!Nearly(record.normalized[channel], expected[channel], detail, label))
            return false;
        const float clamped = std::min(std::max(record.normalized[channel], 0.0f), 1.0f);
        std::snprintf(label, sizeof(label), "units.%c", names[channel]);
        if (!Nearly(record.units[channel], clamped * 255.0f, detail, label, 1e-3f))
            return false;
        if (record.truncated[channel] != static_cast<int32_t>(record.units[channel]))
        {
            detail = std::string("truncated.") + names[channel] + " is not (int)units";
            return false;
        }
        std::snprintf(label, sizeof(label), "units-vs-expected.%c", names[channel]);
        if (!Nearly(record.units[channel], expected[channel] * 255.0f, detail, label,
            1e-2f)) return false;
    }
    for (int indexPad = 0; indexPad < 4; ++indexPad)
    {
        if (record.padding[indexPad] != 0)
        {
            detail = "padding is not zero";
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> UniformTexture(UINT width, UINT height, uint8_t b, uint8_t g,
    uint8_t r, uint8_t a)
{
    std::vector<uint8_t> texture(static_cast<size_t>(width) * height * 4);
    for (size_t pixel = 0; pixel < static_cast<size_t>(width) * height; ++pixel)
    {
        texture[pixel * 4 + 0] = b;
        texture[pixel * 4 + 1] = g;
        texture[pixel * 4 + 2] = r;
        texture[pixel * 4 + 3] = a;
    }
    return texture;
}

void SetPixel(std::vector<uint8_t>& texture, UINT width, UINT x, UINT y, uint8_t b,
    uint8_t g, uint8_t r, uint8_t a)
{
    const size_t at = (static_cast<size_t>(y) * width + x) * 4;
    texture[at + 0] = b;
    texture[at + 1] = g;
    texture[at + 2] = r;
    texture[at + 3] = a;
}

void Pixel4x4(UINT x, UINT y, uint8_t& b, uint8_t& g, uint8_t& r, uint8_t& a)
{
    b = static_cast<uint8_t>(16 + (y * 4 + x) * 15);
    g = static_cast<uint8_t>(20 + x * 60);
    r = static_cast<uint8_t>(30 + y * 50);
    a = static_cast<uint8_t>(180 + x * 10 + y);
}

bool RunSelfTest(Probe& probe, std::vector<SelfTestCase>& cases, std::string& error)
{
    // 1. Center sample and Load agree, BGRA8 memory order maps to logical
    //    RGBA float4 channels, and a repeated request reproduces the full
    //    80-byte record.
    {
        SelfTestCase test{"center_bgra_load_repeat", false, {}};
        const UINT width = 2;
        const UINT height = 2;
        const std::vector<uint8_t> texture = UniformTexture(width, height, 10, 20, 30, 40);
        const std::vector<Request> requests = {
            MakeSample(1, 0.5f, 0.5f),
            MakeLoad(2, 1, 1),
            MakeSample(1, 0.5f, 0.5f),
        };
        std::vector<uint8_t> records;
        if (!probe.Run(texture.data(), width, height, requests.data(),
            static_cast<uint32_t>(requests.size()), records, error))
        {
            test.detail = error;
            cases.push_back(test);
            return false;
        }
        const std::array<float, 4> expected = NormalizedBgra(10, 20, 30, 40);
        bool okay = records.size() == requests.size() * kRecordStride;
        if (!okay) test.detail = "unexpected record byte count";
        for (size_t index = 0; okay && index < requests.size(); ++index)
        {
            okay = CheckRecord(records, index, requests[index], expected, test.detail);
            if (!okay) test.detail = "record " + std::to_string(index) + ": " + test.detail;
        }
        if (okay && std::memcmp(records.data(), records.data() + 2 * kRecordStride,
                kRecordStride) != 0)
        {
            okay = false;
            test.detail = "repeated request record is not bit-identical";
        }
        test.passed = okay;
        if (okay) test.detail = "sample(0.5,0.5), load(1,1), duplicate exact";
        cases.push_back(test);
    }

    // 2. 2x2 bilinear half-footprint: each corner contributes 0.25.
    {
        SelfTestCase test{"half_bilinear_2x2", false, {}};
        const UINT width = 2;
        const UINT height = 2;
        std::vector<uint8_t> texture(static_cast<size_t>(width) * height * 4, 0);
        SetPixel(texture, width, 0, 0, 0, 0, 0, 255);
        SetPixel(texture, width, 1, 0, 255, 0, 0, 255);
        SetPixel(texture, width, 0, 1, 0, 255, 0, 255);
        SetPixel(texture, width, 1, 1, 0, 0, 255, 255);
        const std::vector<Request> requests = {MakeSample(10, 0.5f, 0.5f)};
        std::vector<uint8_t> records;
        if (!probe.Run(texture.data(), width, height, requests.data(), 1, records, error))
        {
            test.detail = error;
            cases.push_back(test);
            return false;
        }
        const std::array<float, 4> expected = {0.25f, 0.25f, 0.25f, 1.0f};
        bool okay = records.size() == kRecordStride
            && CheckRecord(records, 0, requests[0], expected, test.detail);
        test.passed = okay;
        if (okay) test.detail = "2x2 center = average of four corners";
        cases.push_back(test);
    }

    // 3. CLAMP U/V at all four outside corners, Load exactness, repeated run
    //    determinism, including a repeated identical request.
    {
        SelfTestCase test{"clamp_load_deterministic", false, {}};
        const UINT width = 4;
        const UINT height = 4;
        std::vector<uint8_t> texture(static_cast<size_t>(width) * height * 4);
        for (UINT y = 0; y < height; ++y)
            for (UINT x = 0; x < width; ++x)
            {
                uint8_t b = 0, g = 0, r = 0, a = 0;
                Pixel4x4(x, y, b, g, r, a);
                SetPixel(texture, width, x, y, b, g, r, a);
            }
        const std::vector<Request> requests = {
            MakeSample(20, -0.5f, -0.5f),
            MakeSample(21, 1.5f, -0.5f),
            MakeSample(22, -0.5f, 1.5f),
            MakeSample(23, 1.5f, 1.5f),
            MakeLoad(24, 0, 0),
            MakeLoad(25, 3, 0),
            MakeLoad(26, 2, 1),
            MakeLoad(26, 2, 1),
        };
        const int32_t expectedX[] = {0, 3, 0, 3, 0, 3, 2, 2};
        const int32_t expectedY[] = {0, 0, 3, 3, 0, 0, 1, 1};
        std::vector<uint8_t> first;
        if (!probe.Run(texture.data(), width, height, requests.data(),
            static_cast<uint32_t>(requests.size()), first, error))
        {
            test.detail = error;
            cases.push_back(test);
            return false;
        }
        bool okay = first.size() == requests.size() * kRecordStride;
        if (!okay) test.detail = "unexpected record byte count";
        for (size_t index = 0; okay && index < requests.size(); ++index)
        {
            uint8_t b = 0, g = 0, r = 0, a = 0;
            Pixel4x4(expectedX[index], expectedY[index], b, g, r, a);
            okay = CheckRecord(first, index, requests[index],
                NormalizedBgra(b, g, r, a), test.detail);
            if (!okay) test.detail = "record " + std::to_string(index) + ": " + test.detail;
        }
        if (okay && std::memcmp(first.data()
                + (requests.size() - 2) * kRecordStride,
                first.data() + (requests.size() - 1) * kRecordStride,
                kRecordStride) != 0)
        {
            okay = false;
            test.detail = "duplicate load record is not bit-identical";
        }
        if (okay)
        {
            std::vector<uint8_t> second;
            if (!probe.Run(texture.data(), width, height, requests.data(),
                static_cast<uint32_t>(requests.size()), second, error))
            {
                okay = false;
                test.detail = error;
            }
            else if (first != second)
            {
                okay = false;
                test.detail = "repeated dispatch is not byte-identical";
            }
        }
        test.passed = okay;
        if (okay) test.detail = "corner clamp, exact loads, repeatable dispatch";
        cases.push_back(test);
    }

    return std::all_of(cases.begin(), cases.end(),
        [](const SelfTestCase& test) { return test.passed; });
}

std::string FlagsHex(UINT flags)
{
    char text[16]{};
    std::snprintf(text, sizeof(text), "0x%08x", flags);
    return text;
}

void WriteDebugJson(std::ofstream& file, const DebugReport& debug)
{
    file << "{\"layer\":" << (debug.layer ? "true" : "false")
         << ",\"errors\":" << debug.errors
         << ",\"warnings\":" << debug.warnings << ",\"messages\":[";
    for (size_t index = 0; index < debug.messages.size(); ++index)
    {
        if (index) file << ',';
        file << '"' << render_validation::JsonEscape(debug.messages[index]) << '"';
    }
    file << "]}";
}

void WriteCompilerJson(std::ofstream& file, const ShaderInfo& shader)
{
    file << "{\"api\":\"D3DCompileFromFile\","
         << "\"dll\":\"d3dcompiler_47.dll\","
         << "\"dll_version\":\"" << render_validation::JsonEscape(shader.dllVersion) << "\","
         << "\"target\":\"cs_5_1\",\"entry\":\"CSMain\","
         << "\"container\":\"" << (shader.dxbc ? "DXBC" : "unknown") << "\","
         << "\"dxil_present\":" << (shader.dxil ? "true" : "false") << ","
         << "\"reference_container\":\"DXIL\","
         << "\"flags\":[\"D3DCOMPILE_ENABLE_STRICTNESS\","
            "\"D3DCOMPILE_OPTIMIZATION_LEVEL3\"],"
         << "\"flags_hex\":\"" << FlagsHex(shader.flags) << "\","
         << "\"precise_math\":false,"
         << "\"compile_warnings\":\""
         << render_validation::JsonEscape(shader.warnings) << "\"}";
}

void WriteSamplerJson(std::ofstream& file)
{
    file << "{\"filter\":\"MIN_MAG_MIP_LINEAR\",\"filter_value\":21,"
         << "\"address_u\":\"CLAMP\",\"address_v\":\"CLAMP\",\"address_w\":\"WRAP\","
         << "\"mip_lod_bias\":0.0,\"max_anisotropy\":16,"
         << "\"comparison_func\":\"LESS_EQUAL\","
         << "\"border_color\":\"OPAQUE_WHITE\","
         << "\"min_lod\":0.0,\"max_lod\":\"FLT_MAX\","
         << "\"comparison_filter\":false}";
}

bool WriteSelfTestJson(const std::filesystem::path& directory, bool passed,
    const std::vector<SelfTestCase>& cases, const DeviceContext& context,
    const ShaderInfo& shader, const DebugReport& debug, std::string& error)
{
    const std::filesystem::path path = directory / L"selftest.json";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    file << "{\n  \"schema\": \"xefg.sampler_probe.selftest/1\",\n";
    file << "  \"passed\": " << (passed ? "true" : "false") << ",\n";
    file << "  \"adapter\": \"" << render_validation::JsonEscape(context.adapterName) << "\",\n";
    file << "  \"adapter_luid\": \"" << LuidText(context.luid) << "\",\n";
    file << "  \"vendor_id\": " << context.description.VendorId << ",\n";
    file << "  \"device_id\": " << context.description.DeviceId << ",\n";
    file << "  \"warp\": " << (context.warp ? "true" : "false") << ",\n";
    file << "  \"shader\": {\"path\": \"" << render_validation::JsonEscape(
        shader.path.string()) << "\", \"sha256\": \"" << shader.sourceHash
        << "\", \"compiler\": ";
    WriteCompilerJson(file, shader);
    file << "},\n  \"sampler\": ";
    WriteSamplerJson(file);
    file << ",\n  \"debug\": ";
    WriteDebugJson(file, debug);
    file << ",\n  \"cases\": [\n";
    for (size_t index = 0; index < cases.size(); ++index)
    {
        file << "    {\"name\": \"" << render_validation::JsonEscape(cases[index].name)
             << "\", \"passed\": " << (cases[index].passed ? "true" : "false")
             << ", \"detail\": \"" << render_validation::JsonEscape(cases[index].detail)
             << "\"}" << (index + 1 < cases.size() ? "," : "") << "\n";
    }
    file << "  ]\n}\n";
    file.close();
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    return true;
}

struct CaptureInput
{
    std::vector<uint8_t> texture;
    std::vector<Request> requests;
    uint32_t count = 0;
};

bool LoadCaptureInput(const Options& options, CaptureInput& input, std::string& error)
{
    if (!ReadFileBytes(Utf8Path(options.texturePath), input.texture, error)) return false;
    const uint64_t expectedBytes = static_cast<uint64_t>(options.width) * options.height * 4;
    if (expectedBytes != input.texture.size())
    {
        error = "texture byte count is " + std::to_string(input.texture.size())
            + ", expected width*height*4 = " + std::to_string(expectedBytes);
        return false;
    }

    std::vector<uint8_t> requestFile;
    if (!ReadFileBytes(Utf8Path(options.requestsPath), requestFile, error)) return false;
    if (requestFile.size() < 16)
    {
        error = "request file is shorter than the 16-byte header";
        return false;
    }
    uint32_t header[4]{};
    std::memcpy(header, requestFile.data(), sizeof(header));
    if (header[0] != kRequestMagic)
    {
        error = "request magic mismatch";
        return false;
    }
    if (header[1] != kVersion)
    {
        error = "request version is not 1";
        return false;
    }
    if (header[3] != kRequestStride)
    {
        error = "request stride is not 32";
        return false;
    }
    const uint64_t expectedRequestBytes = 16 + static_cast<uint64_t>(header[2]) * kRequestStride;
    if (expectedRequestBytes != requestFile.size())
    {
        error = "request byte count is " + std::to_string(requestFile.size())
            + ", expected 16 + count*32 = " + std::to_string(expectedRequestBytes);
        return false;
    }
    input.count = header[2];
    input.requests.resize(input.count);
    if (input.count)
        std::memcpy(input.requests.data(), requestFile.data() + 16,
            static_cast<size_t>(input.count) * kRequestStride);
    for (uint32_t index = 0; index < input.count; ++index)
    {
        const Request& request = input.requests[index];
        if (request.op > 1)
        {
            error = "request " + std::to_string(index) + " has unsupported op";
            return false;
        }
        if (request.op == 1)
        {
            if (request.x < 0 || request.y < 0
                || static_cast<uint32_t>(request.x) >= options.width
                || static_cast<uint32_t>(request.y) >= options.height)
            {
                error = "request " + std::to_string(index) + " load coordinate is out of range";
                return false;
            }
        }
        else if (!std::isfinite(request.u) || !std::isfinite(request.v))
        {
            error = "request " + std::to_string(index) + " sample UV is not finite";
            return false;
        }
    }
    return true;
}

bool WriteSamples(const std::filesystem::path& path, const std::vector<uint8_t>& records,
    uint32_t count, std::string& error)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    const uint32_t header[4] = {kRequestMagic, kVersion, count, kRecordStride};
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    if (!records.empty())
        file.write(reinterpret_cast<const char*>(records.data()),
            static_cast<std::streamsize>(records.size()));
    file.close();
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    return true;
}

bool WriteMeta(const std::filesystem::path& directory, const Options& options,
    const DeviceContext& context, const ShaderInfo& shader, const CaptureInput& input,
    const std::vector<uint8_t>& records, const DebugReport& debug, std::string& error)
{
    const std::filesystem::path samplesPath = directory / L"samples.bin";
    const std::string textureHash = render_validation::Sha256File(
        Utf8Path(options.texturePath));
    const std::string requestHash = render_validation::Sha256File(
        Utf8Path(options.requestsPath));
    const std::string samplesHash = render_validation::Sha256File(samplesPath);
    const uint64_t textureBytes = static_cast<uint64_t>(options.width) * options.height * 4;
    const uint64_t requestBytes = 16 + static_cast<uint64_t>(input.count) * kRequestStride;
    const uint64_t outputBytes = 16 + static_cast<uint64_t>(input.count) * kRecordStride;

    const std::filesystem::path path = directory / L"meta.json";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    file << "{\n  \"schema\": \"xefg.sampler_probe/1\",\n";
    file << "  \"tool\": \"XefgSamplerProbe\",\n";
    file << "  \"adapter\": \"" << render_validation::JsonEscape(context.adapterName) << "\",\n";
    file << "  \"adapter_luid\": \"" << LuidText(context.luid) << "\",\n";
    file << "  \"vendor_id\": " << context.description.VendorId << ",\n";
    file << "  \"device_id\": " << context.description.DeviceId << ",\n";
    file << "  \"sub_sys_id\": " << context.description.SubSysId << ",\n";
    file << "  \"revision\": " << context.description.Revision << ",\n";
    file << "  \"warp\": " << (context.warp ? "true" : "false") << ",\n";
    file << "  \"texture\": {\"path\": \"" << render_validation::JsonEscape(
        options.texturePath) << "\", \"width\": " << options.width
         << ", \"height\": " << options.height
         << ", \"format\": \"DXGI_FORMAT_B8G8R8A8_UNORM\", \"bytes\": " << textureBytes
         << ", \"sha256\": \"" << textureHash << "\"},\n";
    file << "  \"requests\": {\"path\": \"" << render_validation::JsonEscape(
        options.requestsPath) << "\", \"count\": " << input.count
         << ", \"stride\": 32, \"bytes\": " << requestBytes
         << ", \"sha256\": \"" << requestHash << "\"},\n";
    file << "  \"sampler\": ";
    WriteSamplerJson(file);
    file << ",\n  \"shader\": {\"path\": \"" << render_validation::JsonEscape(
        shader.path.string()) << "\", \"sha256\": \"" << shader.sourceHash
         << "\", \"compiler\": ";
    WriteCompilerJson(file, shader);
    file << "},\n  \"debug\": ";
    WriteDebugJson(file, debug);
    file << ",\n  \"output\": {\"file\": \"samples.bin\", \"count\": " << input.count
         << ", \"stride\": 80, \"bytes\": " << outputBytes
         << ", \"sha256\": \"" << samplesHash << "\"}\n}\n";
    file.close();
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    Options options;
    std::string error;
    if (!ParseArgs(argc, argv, options, error))
    {
        std::fprintf(stderr, "%s\n\n%s", error.c_str(), kUsage);
        return 2;
    }
    if (options.help)
    {
        std::printf("%s", kUsage);
        return 0;
    }
    const std::filesystem::path outputDirectory = Utf8Path(options.outputDirectory);
    if (!PrepareNewDirectory(outputDirectory, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    ShaderInfo shader;
    if (!CompileShader(shader, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    DeviceContext context;
    if (!CreateDeviceContext(options, context, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("adapter: %s\nluid: %s\n", context.adapterName.c_str(),
        LuidText(context.luid).c_str());

    Probe probe;
    probe.SetFixture(context.fixture.get());
    if (!probe.Initialize(context.fixture->device.Get(), shader.blob.Get(), error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    if (options.selfTest)
    {
        std::vector<SelfTestCase> cases;
        const bool casesOkay = RunSelfTest(probe, cases, error);
        DebugReport debug;
        CollectDebugMessages(context.fixture->device.Get(), options.debug, debug);
        const bool debugOkay = !debug.layer || (debug.errors == 0 && debug.warnings == 0);
        const bool passed = casesOkay && debugOkay;
        if (!WriteSelfTestJson(outputDirectory, passed, cases, context, shader, debug, error))
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        for (const SelfTestCase& test : cases)
        {
            std::printf("%s %s: %s\n", test.passed ? "PASS" : "FAIL",
                test.name.c_str(), test.detail.c_str());
        }
        std::printf("selftest: %s (gpu debug errors=%u warnings=%u)\n",
            passed ? "PASS" : "FAIL", debug.errors, debug.warnings);
        if (!passed && error.empty()) error = "selftest failed";
        if (!error.empty()) std::fprintf(stderr, "%s\n", error.c_str());
        return passed ? 0 : 1;
    }

    CaptureInput input;
    if (!LoadCaptureInput(options, input, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::vector<uint8_t> records;
    if (!probe.Run(input.texture.data(), options.width, options.height,
        input.requests.data(), input.count, records, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const std::filesystem::path samplesPath = outputDirectory / L"samples.bin";
    if (!WriteSamples(samplesPath, records, input.count, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    DebugReport debug;
    CollectDebugMessages(context.fixture->device.Get(), options.debug, debug);
    if (!WriteMeta(outputDirectory, options, context, shader, input, records, debug, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("samples: %s (%u records, %llu bytes)\n", samplesPath.string().c_str(),
        input.count, static_cast<unsigned long long>(16 + static_cast<uint64_t>(input.count) * 80));
    std::printf("debug: layer=%s errors=%u warnings=%u\n",
        debug.layer ? "on" : "off", debug.errors, debug.warnings);
    if (debug.layer && debug.errors)
    {
        std::fprintf(stderr, "GPU debug layer reported errors\n");
        return 1;
    }
    return 0;
}
