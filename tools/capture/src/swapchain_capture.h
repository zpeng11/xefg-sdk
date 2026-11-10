#pragma once
// A forwarding IDXGISwapChain4 that captures every frame XeFG presents.
//
// XeFG's proxy swap chain issues one native Present per displayed frame, so
// wrapping the application swap chain is enough to see the base frame and the
// generated 2x frame. Each capture copies the current back buffer to a readback
// heap on the same direct queue XeFG uses, then writes tightly packed pixels.
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace oracle
{
using Microsoft::WRL::ComPtr;

// Global native-Present counter. The oracle ignores it; XefgProbe stamps each
// dispatch with the present it precedes, so a trace and a capture share one
// ordering even when the pacing thread presents asynchronously.
inline std::atomic<uint32_t>& NativePresentOrdinal()
{
    static std::atomic<uint32_t> value{0};
    return value;
}

struct CapturedPresent
{
    uint32_t sequence = 0;
    uint32_t frameIndex = 0;
    uint32_t ordinal = 0; // 0 = first native Present of this application frame
    uint32_t width = 0;
    uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::wstring file;
};

inline const wchar_t* FormatExtension(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return L"bgra";
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return L"rgba";
    case DXGI_FORMAT_R10G10B10A2_UNORM: return L"r10g10b10a2";
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return L"rgba16f";
    default: return L"raw";
    }
}

class CaptureContext
{
public:
    bool Initialize(ID3D12Device* device, ID3D12CommandQueue* queue,
        std::wstring directory, std::string& error)
    {
        device_ = device;
        queue_ = queue;
        directory_ = std::move(directory);
        fenceEvent_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!fenceEvent_)
        {
            error = "CreateEventW failed";
            return false;
        }
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)))
            || FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                IID_PPV_ARGS(&list_)))
            || FAILED(list_->Close())
            || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_))))
        {
            error = "D3D12 capture objects could not be created";
            return false;
        }
        std::error_code ignored;
        std::filesystem::create_directories(std::filesystem::path(directory_), ignored);
        return true;
    }

    ~CaptureContext()
    {
        if (fenceEvent_) CloseHandle(fenceEvent_);
    }

    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool Enabled() const { return enabled_; }
    void SetFrameIndex(uint32_t frame) { frameIndex_ = frame; }

    std::vector<CapturedPresent> Captures()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return captures_;
    }

    std::string Error()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_;
    }

    void Capture(IDXGISwapChain4* native, UINT index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_ || !native) return;

        ComPtr<ID3D12Resource> buffer;
        if (FAILED(native->GetBuffer(index, IID_PPV_ARGS(&buffer))) || !buffer)
        {
            Fail("GetBuffer failed");
            return;
        }
        const D3D12_RESOURCE_DESC desc = buffer->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1)
        {
            Fail("swap-chain buffer is not a 2D texture");
            return;
        }

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 total = 0;
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
        if (!total)
        {
            Fail("empty copyable footprint");
            return;
        }
        if (!readback_ || readbackBytes_ < total)
        {
            D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
            D3D12_RESOURCE_DESC bufferDesc{};
            bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bufferDesc.Width = total;
            bufferDesc.Height = 1;
            bufferDesc.DepthOrArraySize = 1;
            bufferDesc.MipLevels = 1;
            bufferDesc.SampleDesc.Count = 1;
            bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            readback_.Reset();
            if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_))))
            {
                Fail("readback buffer creation failed");
                return;
            }
            readbackBytes_ = total;
        }

        if (FAILED(allocator_->Reset()) || FAILED(list_->Reset(allocator_.Get(), nullptr)))
        {
            Fail("capture command list reset failed");
            return;
        }
        D3D12_RESOURCE_BARRIER toCopy{};
        toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toCopy.Transition.pResource = buffer.Get();
        toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list_->ResourceBarrier(1, &toCopy);
        D3D12_TEXTURE_COPY_LOCATION source{buffer.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        source.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION destination{readback_.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        destination.PlacedFootprint = footprint;
        list_->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        D3D12_RESOURCE_BARRIER back = toCopy;
        back.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        back.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        list_->ResourceBarrier(1, &back);
        if (FAILED(list_->Close()))
        {
            Fail("capture command list close failed");
            return;
        }
        ID3D12CommandList* lists[]{list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        const UINT64 value = ++fenceValue_;
        if (FAILED(queue_->Signal(fence_.Get(), value)))
        {
            Fail("capture fence signal failed");
            return;
        }
        if (fence_->GetCompletedValue() < value)
        {
            if (FAILED(fence_->SetEventOnCompletion(value, fenceEvent_))
                || WaitForSingleObject(fenceEvent_, 10000) != WAIT_OBJECT_0)
            {
                Fail("capture fence wait timed out");
                return;
            }
        }

        void* mapped = nullptr;
        if (FAILED(readback_->Map(0, nullptr, &mapped)))
        {
            Fail("readback map failed");
            return;
        }
        const uint32_t width = static_cast<uint32_t>(desc.Width);
        const uint32_t height = desc.Height;
        const size_t rowBytes = static_cast<size_t>(footprint.Footprint.Width) * BytesPerPixel(desc.Format);
        std::vector<uint8_t> tight(rowBytes * height);
        for (uint32_t y = 0; y < height; ++y)
        {
            std::memcpy(tight.data() + static_cast<size_t>(y) * rowBytes,
                static_cast<const uint8_t*>(mapped) + footprint.Offset
                    + static_cast<size_t>(y) * footprint.Footprint.RowPitch,
                rowBytes);
        }
        readback_->Unmap(0, nullptr);

        CapturedPresent record;
        record.sequence = sequence_++;
        record.frameIndex = frameIndex_;
        record.width = width;
        record.height = height;
        record.format = desc.Format;
        record.ordinal = 0;
        for (const auto& existing : captures_)
            if (existing.frameIndex == record.frameIndex) ++record.ordinal;

        wchar_t name[128]{};
        swprintf_s(name, L"present_%05u_f%05u_o%u.%s", record.sequence, record.frameIndex, record.ordinal,
            FormatExtension(desc.Format));
        record.file = std::filesystem::path(directory_) / name;
        std::ofstream file(record.file, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            Fail("capture file could not be opened");
            return;
        }
        file.write(reinterpret_cast<const char*>(tight.data()), static_cast<std::streamsize>(tight.size()));
        captures_.push_back(std::move(record));
    }

private:
    static size_t BytesPerPixel(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
        default: return 4;
        }
    }

    void Fail(const char* what)
    {
        if (error_.empty()) error_ = what;
        enabled_ = false;
        std::fprintf(stderr, "capture disabled: %s\n", what);
    }

    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    UINT64 fenceValue_ = 0;
    ComPtr<ID3D12Resource> readback_;
    UINT64 readbackBytes_ = 0;
    std::wstring directory_;
    bool enabled_ = true;
    uint32_t frameIndex_ = 0;
    uint32_t sequence_ = 0;
    std::mutex mutex_;
    std::vector<CapturedPresent> captures_;
    std::string error_;
};

class CaptureChain final : public IDXGISwapChain4
{
public:
    CaptureChain(ComPtr<IDXGISwapChain4> native, std::shared_ptr<CaptureContext> context)
        : native_(std::move(native)), context_(std::move(context))
    {
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IDXGIObject) || iid == __uuidof(IDXGIDeviceSubObject)
            || iid == __uuidof(IDXGISwapChain) || iid == __uuidof(IDXGISwapChain1)
            || iid == __uuidof(IDXGISwapChain2) || iid == __uuidof(IDXGISwapChain3)
            || iid == __uuidof(IDXGISwapChain4))
        {
            *out = static_cast<IDXGISwapChain4*>(this);
            AddRef();
            return S_OK;
        }
        return native_->QueryInterface(iid, out);
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG left = --references_;
        if (!left) delete this;
        return left;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID name, UINT size, const void* data) override
    { return native_->SetPrivateData(name, size, data); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID name, const IUnknown* object) override
    { return native_->SetPrivateDataInterface(name, object); }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID name, UINT* size, void* data) override
    { return native_->GetPrivateData(name, size, data); }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID iid, void** parent) override
    { return native_->GetParent(iid, parent); }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID iid, void** device) override
    { return native_->GetDevice(iid, device); }

    HRESULT STDMETHODCALLTYPE Present(UINT interval, UINT flags) override
    {
        const UINT index = native_->GetCurrentBackBufferIndex();
        if (!(flags & DXGI_PRESENT_TEST)) NativePresentOrdinal().fetch_add(1, std::memory_order_relaxed);
        const HRESULT result = native_->Present(interval, flags);
        if (!(flags & DXGI_PRESENT_TEST)) context_->Capture(native_.Get(), index);
        return result;
    }
    HRESULT STDMETHODCALLTYPE Present1(UINT interval, UINT flags, const DXGI_PRESENT_PARAMETERS* parameters) override
    {
        const UINT index = native_->GetCurrentBackBufferIndex();
        if (!(flags & DXGI_PRESENT_TEST)) NativePresentOrdinal().fetch_add(1, std::memory_order_relaxed);
        const HRESULT result = native_->Present1(interval, flags, parameters);
        if (!(flags & DXGI_PRESENT_TEST)) context_->Capture(native_.Get(), index);
        return result;
    }

    HRESULT STDMETHODCALLTYPE GetBuffer(UINT index, REFIID iid, void** surface) override
    { return native_->GetBuffer(index, iid, surface); }
    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL fullscreen, IDXGIOutput* target) override
    { return native_->SetFullscreenState(fullscreen, target); }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* fullscreen, IDXGIOutput** target) override
    { return native_->GetFullscreenState(fullscreen, target); }
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* description) override
    { return native_->GetDesc(description); }
    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags) override
    { return native_->ResizeBuffers(count, width, height, format, flags); }
    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC* parameters) override
    { return native_->ResizeTarget(parameters); }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput** output) override
    { return native_->GetContainingOutput(output); }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* statistics) override
    { return native_->GetFrameStatistics(statistics); }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* count) override
    { return native_->GetLastPresentCount(count); }
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* description) override
    { return native_->GetDesc1(description); }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* description) override
    { return native_->GetFullscreenDesc(description); }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* window) override
    { return native_->GetHwnd(window); }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID iid, void** window) override
    { return native_->GetCoreWindow(iid, window); }
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override
    { return native_->IsTemporaryMonoSupported(); }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** output) override
    { return native_->GetRestrictToOutput(output); }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA* color) override
    { return native_->SetBackgroundColor(color); }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA* color) override
    { return native_->GetBackgroundColor(color); }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION rotation) override
    { return native_->SetRotation(rotation); }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* rotation) override
    { return native_->GetRotation(rotation); }
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT width, UINT height) override
    { return native_->SetSourceSize(width, height); }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* width, UINT* height) override
    { return native_->GetSourceSize(width, height); }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT latency) override
    { return native_->SetMaximumFrameLatency(latency); }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* latency) override
    { return native_->GetMaximumFrameLatency(latency); }
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override
    { return native_->GetFrameLatencyWaitableObject(); }
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F* matrix) override
    { return native_->SetMatrixTransform(matrix); }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F* matrix) override
    { return native_->GetMatrixTransform(matrix); }
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override
    { return native_->GetCurrentBackBufferIndex(); }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE space, UINT* support) override
    { return native_->CheckColorSpaceSupport(space, support); }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE space) override
    { return native_->SetColorSpace1(space); }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags,
        const UINT* nodeMasks, IUnknown* const* queues) override
    { return native_->ResizeBuffers1(count, width, height, format, flags, nodeMasks, queues); }
    HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE type, UINT size, void* metadata) override
    { return native_->SetHDRMetaData(type, size, metadata); }

private:
    ~CaptureChain() = default;
    std::atomic<ULONG> references_{1};
    ComPtr<IDXGISwapChain4> native_;
    std::shared_ptr<CaptureContext> context_;
};

// Takes ownership of `native` (releases the caller's reference).
inline ComPtr<IDXGISwapChain1> WrapForCapture(IDXGISwapChain1* native, std::shared_ptr<CaptureContext> context)
{
    ComPtr<IDXGISwapChain4> chain4;
    if (!native || FAILED(native->QueryInterface(IID_PPV_ARGS(&chain4)))) return nullptr;
    auto* wrapper = new CaptureChain(std::move(chain4), std::move(context));
    ComPtr<IDXGISwapChain1> result;
    wrapper->QueryInterface(IID_PPV_ARGS(&result));
    wrapper->Release();
    native->Release();
    return result;
}

// XeFG creates its own swap chain from the factory passed to
// InitFromSwapChainDesc. Wrapping that factory is how the oracle guarantees
// every internal chain is a CaptureChain.
class CaptureFactory final : public IDXGIFactory2
{
public:
    CaptureFactory(ComPtr<IDXGIFactory2> native, std::shared_ptr<CaptureContext> context)
        : native_(std::move(native)), context_(std::move(context))
    {
    }

    ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG left = --references_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IDXGIObject) || iid == __uuidof(IDXGIFactory)
            || iid == __uuidof(IDXGIFactory1) || iid == __uuidof(IDXGIFactory2))
        {
            *out = static_cast<IDXGIFactory2*>(this);
            AddRef();
            return S_OK;
        }
        return native_->QueryInterface(iid, out);
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID name, UINT size, const void* data) override
    { return native_->SetPrivateData(name, size, data); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID name, const IUnknown* object) override
    { return native_->SetPrivateDataInterface(name, object); }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID name, UINT* size, void* data) override
    { return native_->GetPrivateData(name, size, data); }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID iid, void** parent) override
    { return native_->GetParent(iid, parent); }

    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT index, IDXGIAdapter** adapter) override
    { return native_->EnumAdapters(index, adapter); }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND window, UINT flags) override
    { return native_->MakeWindowAssociation(window, flags); }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND* window) override
    { return native_->GetWindowAssociation(window); }

    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown* device, DXGI_SWAP_CHAIN_DESC* description,
        IDXGISwapChain** chain) override
    {
        return native_->CreateSwapChain(device, description, chain);
    }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE module, IDXGIAdapter** adapter) override
    { return native_->CreateSoftwareAdapter(module, adapter); }
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override { return native_->IsWindowedStereoEnabled(); }

    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown* device, HWND window,
        const DXGI_SWAP_CHAIN_DESC1* description, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
        IDXGIOutput* restrict, IDXGISwapChain1** chain) override
    {
        const HRESULT result = native_->CreateSwapChainForHwnd(device, window, description, fullscreen,
            restrict, chain);
        if (SUCCEEDED(result)) Wrap(chain);
        return result;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForCoreWindow(IUnknown* device, IUnknown* window,
        const DXGI_SWAP_CHAIN_DESC1* description, IDXGIOutput* restrict, IDXGISwapChain1** chain) override
    {
        const HRESULT result = native_->CreateSwapChainForCoreWindow(device, window, description, restrict, chain);
        if (SUCCEEDED(result)) Wrap(chain);
        return result;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(IUnknown* device,
        const DXGI_SWAP_CHAIN_DESC1* description, IDXGIOutput* restrict, IDXGISwapChain1** chain) override
    {
        const HRESULT result = native_->CreateSwapChainForComposition(device, description, restrict, chain);
        if (SUCCEEDED(result)) Wrap(chain);
        return result;
    }
    HRESULT STDMETHODCALLTYPE GetSharedResourceAdapterLuid(HANDLE resource, LUID* luid) override
    { return native_->GetSharedResourceAdapterLuid(resource, luid); }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusWindow(HWND window, UINT message, DWORD* cookie) override
    { return native_->RegisterStereoStatusWindow(window, message, cookie); }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusWindow(HWND window, UINT message, DWORD* cookie) override
    { return native_->RegisterOcclusionStatusWindow(window, message, cookie); }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusEvent(HANDLE event, DWORD* cookie) override
    { return native_->RegisterStereoStatusEvent(event, cookie); }
    void STDMETHODCALLTYPE UnregisterStereoStatus(DWORD cookie) override
    { native_->UnregisterStereoStatus(cookie); }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusEvent(HANDLE event, DWORD* cookie) override
    { return native_->RegisterOcclusionStatusEvent(event, cookie); }
    void STDMETHODCALLTYPE UnregisterOcclusionStatus(DWORD cookie) override
    { native_->UnregisterOcclusionStatus(cookie); }
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT index, IDXGIAdapter1** adapter) override
    { return native_->EnumAdapters1(index, adapter); }
    BOOL STDMETHODCALLTYPE IsCurrent() override { return native_->IsCurrent(); }

private:
    ~CaptureFactory() = default;

    void Wrap(IDXGISwapChain1** chain)
    {
        if (!chain || !*chain) return;
        ComPtr<IDXGISwapChain4> chain4;
        if (FAILED((*chain)->QueryInterface(IID_PPV_ARGS(&chain4)))) return;
        auto* wrapper = new CaptureChain(std::move(chain4), context_);
        ComPtr<IDXGISwapChain1> replacement;
        wrapper->QueryInterface(IID_PPV_ARGS(&replacement));
        wrapper->Release();
        if (!replacement) return;
        (*chain)->Release();
        *chain = replacement.Detach();
    }

    std::atomic<ULONG> references_{1};
    ComPtr<IDXGIFactory2> native_;
    std::shared_ptr<CaptureContext> context_;
};

inline ComPtr<IDXGIFactory2> WrapFactory(IDXGIFactory2* native, std::shared_ptr<CaptureContext> context)
{
    if (!native) return nullptr;
    auto* wrapper = new CaptureFactory(native, std::move(context));
    ComPtr<IDXGIFactory2> result;
    wrapper->QueryInterface(IID_PPV_ARGS(&result));
    wrapper->Release();
    return result;
}
} // namespace oracle
