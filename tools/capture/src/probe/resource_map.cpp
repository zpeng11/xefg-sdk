#include "resource_map.h"

namespace probe
{
namespace
{
UINT64 BufferBytes(const D3D12_RESOURCE_DESC& description)
{
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) return 0;
    return description.Width;
}
} // namespace

void ResourceMap::Store(D3D12_CPU_DESCRIPTOR_HANDLE destination, ID3D12Resource* resource,
    bool unorderedAccess, DXGI_FORMAT format, UINT firstMip, UINT firstSlice,
    UINT64 bufferBytes, UINT64 bufferOffset, UINT structureStride)
{
    if (!destination.ptr) return;
    ViewInfo info;
    info.resource = resource;
    info.unorderedAccess = unorderedAccess;
    info.format = format;
    info.firstMip = firstMip;
    info.firstSlice = firstSlice;
    info.bufferBytes = bufferBytes;
    info.bufferOffset = bufferOffset;
    info.structureStride = structureStride;
    if (!resource)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        views_[static_cast<uintptr_t>(destination.ptr)] = info;
        return;
    }
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    info.resourceDimension = description.Dimension;
    info.width = description.Width;
    info.height = description.Height;
    info.depthOrArraySize = description.DepthOrArraySize;
    info.mipLevels = description.MipLevels;
    if (description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && !info.bufferBytes)
        info.bufferBytes = BufferBytes(description);

    std::lock_guard<std::mutex> lock(mutex_);
    views_[static_cast<uintptr_t>(destination.ptr)] = info;
}

void ResourceMap::ResourceCreated(ID3D12Resource* resource, D3D12_RESOURCE_STATES initialState)
{
    if (!resource) return;
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    ResourceInitialState state;
    state.state = initialState;
    state.known = true;
    state.buffer = description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    if (state.buffer)
    {
        state.gpuAddress = resource->GetGPUVirtualAddress();
        state.width = description.Width;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    resources_[reinterpret_cast<uintptr_t>(resource)] = state;
}

bool ResourceMap::ResolveBuffer(D3D12_GPU_VIRTUAL_ADDRESS address, ID3D12Resource*& resource,
    UINT64& offset) const
{
    if (!address) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    // Placed buffers do not overlap, so the first containing range is the match.
    for (const auto& [pointer, state] : resources_)
    {
        if (!state.buffer || !state.gpuAddress || !state.width) continue;
        if (address < state.gpuAddress || address >= state.gpuAddress + state.width) continue;
        resource = reinterpret_cast<ID3D12Resource*>(pointer);
        offset = address - state.gpuAddress;
        return true;
    }
    return false;
}

void ResourceMap::ConstantBufferView(D3D12_CPU_DESCRIPTOR_HANDLE destination,
    const D3D12_CONSTANT_BUFFER_VIEW_DESC* description)
{
    // A CBV descriptor only names a GPU virtual address. Resolve it to the
    // buffer that contains it and keep the view's offset and size, so the
    // snapshot copies exactly the provider's constant data.
    if (!description)
    {
        Store(destination, nullptr, false, DXGI_FORMAT_UNKNOWN, 0, 0, 0, 0, 0);
        return;
    }
    ID3D12Resource* resource = nullptr;
    UINT64 offset = 0;
    ResolveBuffer(description->BufferLocation, resource, offset);
    Store(destination, resource, false, DXGI_FORMAT_UNKNOWN, 0, 0,
        description->SizeInBytes, offset, 0);
}

void ResourceMap::ShaderResourceView(D3D12_CPU_DESCRIPTOR_HANDLE destination, ID3D12Resource* resource,
    const D3D12_SHADER_RESOURCE_VIEW_DESC* description)
{
    if (!description)
    {
        Store(destination, resource, false, DXGI_FORMAT_UNKNOWN, 0, 0, 0, 0, 0);
        return;
    }
    DXGI_FORMAT format = description->Format;
    if (format == DXGI_FORMAT_UNKNOWN && description->ViewDimension == D3D12_SRV_DIMENSION_BUFFER)
        format = DXGI_FORMAT_R32_TYPELESS;
    UINT firstMip = 0, firstSlice = 0;
    UINT64 bufferBytes = 0;
    UINT stride = 0;
    switch (description->ViewDimension)
    {
    case D3D12_SRV_DIMENSION_TEXTURE1D: firstMip = description->Texture1D.MostDetailedMip; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
        firstMip = description->Texture1DArray.MostDetailedMip;
        firstSlice = description->Texture1DArray.FirstArraySlice;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2D: firstMip = description->Texture2D.MostDetailedMip; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
        firstMip = description->Texture2DArray.MostDetailedMip;
        firstSlice = description->Texture2DArray.FirstArraySlice;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: firstMip = description->Texture3D.MostDetailedMip; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE: firstMip = description->TextureCube.MostDetailedMip; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
        firstMip = description->TextureCubeArray.MostDetailedMip;
        firstSlice = description->TextureCubeArray.First2DArrayFace;
        break;
    case D3D12_SRV_DIMENSION_BUFFER:
        bufferBytes = static_cast<UINT64>(description->Buffer.NumElements)
            * (description->Buffer.StructureByteStride ? description->Buffer.StructureByteStride : 4u);
        stride = description->Buffer.StructureByteStride;
        break;
    default: break;
    }
    Store(destination, resource, false, format, firstMip, firstSlice, bufferBytes, 0, stride);
}

void ResourceMap::UnorderedAccessView(D3D12_CPU_DESCRIPTOR_HANDLE destination, ID3D12Resource* resource,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC* description)
{
    if (!description)
    {
        Store(destination, resource, true, DXGI_FORMAT_UNKNOWN, 0, 0, 0, 0, 0);
        return;
    }
    DXGI_FORMAT format = description->Format;
    if (format == DXGI_FORMAT_UNKNOWN && description->ViewDimension == D3D12_UAV_DIMENSION_BUFFER)
        format = DXGI_FORMAT_R32_TYPELESS;
    UINT firstMip = 0, firstSlice = 0;
    UINT64 bufferBytes = 0;
    UINT stride = 0;
    switch (description->ViewDimension)
    {
    case D3D12_UAV_DIMENSION_TEXTURE1D: firstMip = description->Texture1D.MipSlice; break;
    case D3D12_UAV_DIMENSION_TEXTURE1DARRAY:
        firstMip = description->Texture1DArray.MipSlice;
        firstSlice = description->Texture1DArray.FirstArraySlice;
        break;
    case D3D12_UAV_DIMENSION_TEXTURE2D: firstMip = description->Texture2D.MipSlice; break;
    case D3D12_UAV_DIMENSION_TEXTURE2DARRAY:
        firstMip = description->Texture2DArray.MipSlice;
        firstSlice = description->Texture2DArray.FirstArraySlice;
        break;
    case D3D12_UAV_DIMENSION_TEXTURE3D: firstMip = description->Texture3D.MipSlice; break;
    case D3D12_UAV_DIMENSION_BUFFER:
        bufferBytes = static_cast<UINT64>(description->Buffer.NumElements)
            * (description->Buffer.StructureByteStride ? description->Buffer.StructureByteStride : 4u);
        stride = description->Buffer.StructureByteStride;
        break;
    default: break;
    }
    Store(destination, resource, true, format, firstMip, firstSlice, bufferBytes, 0, stride);
}

void ResourceMap::DescriptorsCopied(const D3D12_CPU_DESCRIPTOR_HANDLE& destination,
    const D3D12_CPU_DESCRIPTOR_HANDLE& source, UINT count, UINT increment)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (UINT i = 0; i < count; ++i)
    {
        const auto found = views_.find(static_cast<uintptr_t>(source.ptr) + size_t(i) * increment);
        if (found == views_.end()) continue;
        views_[static_cast<uintptr_t>(destination.ptr) + size_t(i) * increment] = found->second;
    }
}

bool ResourceMap::View(D3D12_CPU_DESCRIPTOR_HANDLE handle, ViewInfo& out) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = views_.find(static_cast<uintptr_t>(handle.ptr));
    if (found == views_.end()) return false;
    out = found->second;
    return true;
}

bool ResourceMap::InitialState(uintptr_t resource, ResourceInitialState& out) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = resources_.find(resource);
    if (found == resources_.end()) return false;
    out = found->second;
    return true;
}

size_t ResourceMap::ViewCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return views_.size();
}

void ResourceMap::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    views_.clear();
    resources_.clear();
}
} // namespace probe
