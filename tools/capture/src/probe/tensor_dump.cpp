#include "tensor_dump.h"

#include "resource_map.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace probe
{
bool TensorDump::Initialize(ID3D12Device* device, std::filesystem::path directory, uint64_t budgetBytes)
{
    device_ = device;
    directory_ = std::move(directory);
    budgetBytes_ = budgetBytes;
    std::error_code ignored;
    std::filesystem::create_directories(directory_, ignored);
    enabled_ = device_ != nullptr && budgetBytes_ != 0;
    if (!enabled_)
    {
        std::fprintf(stderr, "tensor dump disabled: %s\n",
            device_ ? "zero byte budget (check --tensor-budget-mb)" : "no device");
    }
    return enabled_;
}

void TensorDump::ResourceCreated(void* resource, D3D12_RESOURCE_STATES state)
{
    if (!enabled_ || !resource) return;
    std::lock_guard<std::mutex> lock(mutex_);
    states_[reinterpret_cast<uintptr_t>(resource)] = state;
}

void TensorDump::Barrier(void* resource, D3D12_RESOURCE_STATES /*before*/, D3D12_RESOURCE_STATES after)
{
    if (!enabled_ || !resource) return;
    std::lock_guard<std::mutex> lock(mutex_);
    states_[reinterpret_cast<uintptr_t>(resource)] = after;
}

D3D12_RESOURCE_STATES TensorDump::StateOf(void* resource, bool& known) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = states_.find(reinterpret_cast<uintptr_t>(resource));
    if (found == states_.end())
    {
        known = false;
        return D3D12_RESOURCE_STATE_COMMON;
    }
    known = true;
    return found->second;
}

void TensorDump::Snapshot(ID3D12GraphicsCommandList* list, const ViewInfo& view, const std::string& slot,
    const char* role, uint64_t dispatch, uint32_t present, std::string& file)
{
    file.clear();
    if (!enabled_ || !list || !view.resource) return;
    ID3D12Resource* resource = view.resource;
    const D3D12_RESOURCE_DESC description = resource->GetDesc();

    bool known = false;
    const D3D12_RESOURCE_STATES state = StateOf(resource, known);
    if (!known) return;

    // Footprints for what this view exposes (mip 0, every slice when the
    // resource is single-mip; otherwise just subresource 0).
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints;
    std::vector<UINT> rows;
    std::vector<UINT64> rowBytes;
    UINT64 total = 0;
    UINT64 offset = 0;
    std::vector<UINT> subresources;
    if (description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        // A constant buffer view names a window inside a (possibly much
        // larger) buffer. Copy only that window.
        offset = view.bufferOffset;
        total = view.bufferBytes ? view.bufferBytes : description.Width;
        if (offset > description.Width) return;
        total = (std::min)(total, description.Width - offset);
        if (!total) return;
    }
    else if (description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
    {
        UINT count = 1;
        if (description.MipLevels == 1 && view.firstMip == 0)
            count = (std::min)(UINT(description.DepthOrArraySize), 64u);
        footprints.resize(count);
        rows.resize(count);
        rowBytes.resize(count);
        subresources.resize(count);
        for (UINT i = 0; i < count; ++i)
            subresources[i] = description.MipLevels == 1 ? i : 0;
        device_->GetCopyableFootprints(&description, subresources.empty() ? 0 : subresources[0], count, 0,
            footprints.data(), rows.data(), rowBytes.data(), &total);
        if (!total) return;
    }
    else return;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (bytes_ + total > budgetBytes_)
        {
            enabled_ = false;
            std::fprintf(stderr, "tensor dump stopped: budget reached (%llu bytes)\n",
                static_cast<unsigned long long>(bytes_));
            return;
        }
    }

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))))
    {
        enabled_ = false;
        return;
    }

    // Transition to COPY_SOURCE, copy, then back to the provider's state.
    D3D12_RESOURCE_BARRIER barrier{};
    if (state != D3D12_RESOURCE_STATE_COPY_SOURCE)
    {
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = state;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &barrier);
    }

    Pending record;
    record.readback = readback;
    record.width = static_cast<uint32_t>(description.Width);
    record.height = description.Height;
    record.arraySize = description.DepthOrArraySize;
    record.format = static_cast<uint32_t>(description.Format);

    if (description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        list->CopyBufferRegion(readback.Get(), 0, resource, offset, total);
        Slice slice;
        slice.offset = 0;
        slice.rowPitch = static_cast<uint32_t>(total);
        slice.rowBytes = static_cast<uint32_t>(total);
        slice.rows = 1;
        record.slices.push_back(slice);
    }
    else
    {
        for (size_t i = 0; i < footprints.size(); ++i)
        {
            D3D12_TEXTURE_COPY_LOCATION destination{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            destination.PlacedFootprint = footprints[i];
            D3D12_TEXTURE_COPY_LOCATION source{resource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            source.SubresourceIndex = subresources[i];
            list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
            Slice slice;
            slice.offset = footprints[i].Offset;
            slice.rowPitch = footprints[i].Footprint.RowPitch;
            slice.rowBytes = static_cast<uint32_t>(rowBytes[i]);
            slice.rows = rows[i];
            record.slices.push_back(slice);
        }
    }

    if (state != D3D12_RESOURCE_STATE_COPY_SOURCE)
    {
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = state;
        list->ResourceBarrier(1, &barrier);
    }

    char name[128]{};
    std::snprintf(name, sizeof(name), "d%04llu_%s_%s.bin", static_cast<unsigned long long>(dispatch),
        slot.c_str(), role);
    record.path = directory_ / name;
    file = record.path.filename().string();

    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(std::move(record));
    bytes_ += total;
    ++count_;
}

bool TensorDump::Finish(ID3D12CommandQueue* const* queues, size_t queueCount, uint32_t timeoutMs,
    std::string& error)
{
    error.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_.empty()) return true;
    if (!device_ || !queues || !queueCount)
    {
        error = "no queue to fence; tensor dumps are unverified";
        return false;
    }

    // A copy is complete only once the queue that carried it has passed it.
    // Signal every observed provider queue and wait, instead of sleeping.
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    if (FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
    {
        error = "tensor fence creation failed";
        return false;
    }
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event)
    {
        error = "tensor fence event creation failed";
        return false;
    }

    UINT64 value = 0;
    for (size_t i = 0; i < queueCount; ++i)
    {
        if (!queues[i]) continue;
        const UINT64 signal = ++value;
        if (FAILED(queues[i]->Signal(fence.Get(), signal)))
        {
            CloseHandle(event);
            error = "tensor fence signal failed";
            return false;
        }
        if (fence->GetCompletedValue() < signal)
        {
            if (FAILED(fence->SetEventOnCompletion(signal, event))
                || WaitForSingleObject(event, timeoutMs) != WAIT_OBJECT_0)
            {
                CloseHandle(event);
                error = "provider queue did not drain before the tensor timeout";
                return false;
            }
        }
    }
    CloseHandle(event);

    size_t failed = 0;
    for (auto& record : pending_)
    {
        void* mapped = nullptr;
        if (FAILED(record.readback->Map(0, nullptr, &mapped)))
        {
            std::fprintf(stderr, "tensor readback map failed: %s\n", record.path.string().c_str());
            ++failed;
            continue;
        }
        std::ofstream file(record.path, std::ios::binary | std::ios::trunc);
        if (file)
        {
            for (const auto& slice : record.slices)
                for (uint32_t y = 0; y < slice.rows; ++y)
                    file.write(static_cast<const char*>(mapped) + slice.offset + size_t(y) * slice.rowPitch,
                        slice.rowBytes);
        }
        record.readback->Unmap(0, nullptr);
        if (!file) ++failed;
    }
    pending_.clear();
    if (failed)
    {
        error = std::to_string(failed) + " tensor readback(s) could not be written";
        return false;
    }
    return true;
}

size_t TensorDump::Count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

uint64_t TensorDump::Bytes() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_;
}
} // namespace probe
