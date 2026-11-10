#pragma once
// Optional per-dispatch tensor snapshots.
//
// Data is captured by inserting a copy into the provider's own command list:
// an input is transitioned to COPY_SOURCE just before the Dispatch, an output
// just after it, then each returns to the state the provider left it in. The
// state model is maintained by hooking ResourceBarrier, so the transitions this
// module inserts are legal and invisible to the provider.
//
// Readback buffers stay alive until Finish(); the probe calls it after the queue
// has been signalled and waited, so no copy is ever read early.
#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace probe
{
struct ViewInfo;

class TensorDump
{
public:
    bool Initialize(ID3D12Device* device, std::filesystem::path directory, uint64_t budgetBytes);
    void ResourceCreated(void* resource, D3D12_RESOURCE_STATES state);
    void Barrier(void* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    void Snapshot(ID3D12GraphicsCommandList* list, const ViewInfo& view, const std::string& slot,
        const char* role, uint64_t dispatch, uint32_t present, std::string& file);
    // Writes every pending readback after every observed provider queue has
    // signalled completion. Returns false (with error) when no queue was
    // observed, a queue did not drain, or a readback could not be mapped, so a
    // run records unverified tensors instead of silently writing stale bytes.
    bool Finish(ID3D12CommandQueue* const* queues, size_t queueCount, uint32_t timeoutMs,
        std::string& error);

    bool Enabled() const { return enabled_; }
    size_t Count() const;
    uint64_t Bytes() const;

private:
    struct Slice
    {
        uint64_t offset = 0;
        uint32_t rowPitch = 0;
        uint32_t rowBytes = 0;
        uint32_t rows = 0;
    };

    struct Pending
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> readback;
        std::filesystem::path path;
        std::vector<Slice> slices;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t arraySize = 0;
        uint32_t format = 0;
    };

    D3D12_RESOURCE_STATES StateOf(void* resource, bool& known) const;

    ID3D12Device* device_ = nullptr;
    std::filesystem::path directory_;
    uint64_t budgetBytes_ = 0;
    bool enabled_ = false;

    mutable std::mutex mutex_;
    std::unordered_map<uintptr_t, D3D12_RESOURCE_STATES> states_;
    std::vector<Pending> pending_;
    uint64_t bytes_ = 0;
    size_t count_ = 0;
};
} // namespace probe
