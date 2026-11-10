#pragma once
// Descriptor -> resource map for the probe.
//
// XeFG binds its tensors through shader-visible descriptor tables, so a dispatch
// only exposes a GPU descriptor handle. Hooking CreateShaderResourceView /
// CreateUnorderedAccessView / CreateConstantBufferView / CopyDescriptors gives
// the CPU handle -> resource mapping; the probe converts the table's GPU handle
// to a CPU handle with the heap bases recorded at SetDescriptorHeaps. That is
// how a dispatch is turned into "these textures and buffers", without
// decompiling anything. A constant buffer view carries only a GPU virtual
// address, so the resource map also records each buffer's address range at
// creation time and resolves the view back to resource + offset.
#include <Windows.h>
#include <d3d12.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace probe
{
struct ViewInfo
{
    ID3D12Resource* resource = nullptr;
    bool unorderedAccess = false;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    D3D12_RESOURCE_DIMENSION resourceDimension = D3D12_RESOURCE_DIMENSION_UNKNOWN;
    UINT64 width = 0;
    UINT height = 0;
    UINT16 depthOrArraySize = 0;
    UINT16 mipLevels = 0;
    UINT firstMip = 0;
    UINT firstSlice = 0;
    UINT64 bufferBytes = 0;
    UINT64 bufferOffset = 0;
    UINT structureStride = 0;
};

struct ResourceInitialState
{
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
    UINT64 width = 0;
    bool buffer = false;
    bool known = false;
};

class ResourceMap
{
public:
    void ResourceCreated(ID3D12Resource* resource, D3D12_RESOURCE_STATES initialState);
    void ShaderResourceView(D3D12_CPU_DESCRIPTOR_HANDLE destination, ID3D12Resource* resource,
        const D3D12_SHADER_RESOURCE_VIEW_DESC* description);
    void UnorderedAccessView(D3D12_CPU_DESCRIPTOR_HANDLE destination, ID3D12Resource* resource,
        const D3D12_UNORDERED_ACCESS_VIEW_DESC* description);
    void ConstantBufferView(D3D12_CPU_DESCRIPTOR_HANDLE destination,
        const D3D12_CONSTANT_BUFFER_VIEW_DESC* description);
    // increment = device->GetDescriptorHandleIncrementSize(heapType)
    void DescriptorsCopied(const D3D12_CPU_DESCRIPTOR_HANDLE& destination,
        const D3D12_CPU_DESCRIPTOR_HANDLE& source, UINT count, UINT increment);

    bool View(D3D12_CPU_DESCRIPTOR_HANDLE handle, ViewInfo& out) const;
    bool InitialState(uintptr_t resource, ResourceInitialState& out) const;

    size_t ViewCount() const;
    void Clear();

private:
    void Store(D3D12_CPU_DESCRIPTOR_HANDLE destination, ID3D12Resource* resource,
        bool unorderedAccess, DXGI_FORMAT format, UINT firstMip, UINT firstSlice,
        UINT64 bufferBytes, UINT64 bufferOffset, UINT structureStride);
    bool ResolveBuffer(D3D12_GPU_VIRTUAL_ADDRESS address, ID3D12Resource*& resource,
        UINT64& offset) const;

    mutable std::mutex mutex_;
    std::unordered_map<uintptr_t, ViewInfo> views_;
    std::unordered_map<uintptr_t, ResourceInitialState> resources_;
};
} // namespace probe
