#pragma once
// Dispatch-level trace of the provider's frame generation.
//
// Patches the device and command-list vtables (the same technique as
// source/native/xess/cuda/cuda_route.cpp) and records, for every Dispatch:
// the pipeline's DXIL SHA-256 (resolved to a program key by the decode relay),
// the root signature, every bound descriptor resolved to its resource through
// the descriptor map, and the layer's root constants. A copy of the provider's
// default state is untouched, so the frame is identical with and without it.
#include <Windows.h>
#include <d3d12.h>

#include "root_signature.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace probe
{
class DecodeRelay;
class ResourceMap;
class TraceWriter;
class TensorDump;

struct DispatchTraceOptions
{
    bool tensors = false;
    uint32_t tensorBudgetMB = 512;
    uint32_t tensorDispatches = 0; // 0 = every dispatch
    std::filesystem::path tensorDirectory;
    // Optional inclusive frame window for tensor snapshots.  Dispatch events
    // are still written for every frame; only the large snapshots are limited
    // to the selected frames.  Both unset preserves the original behaviour of
    // snapshotting every frame.
    std::optional<uint32_t> tensorFirstFrame;
    std::optional<uint32_t> tensorLastFrame;
};

class DispatchTrace
{
public:
    DispatchTrace();
    ~DispatchTrace();

    bool Install(ID3D12Device* device, ID3D12GraphicsCommandList* ownList, TraceWriter& writer,
        DecodeRelay& relay, std::string& error);
    void Detach();
    void SetFrame(uint32_t frame);
    void SetOptions(const DispatchTraceOptions& options) { options_ = options; }
    // Drains every observed provider queue with a fence, then writes the tensor
    // dumps. False means the dumps must not be trusted (recorded in meta).
    bool FinishTensors(std::string& error);
    size_t ProviderQueues() const;

    uint64_t DispatchCount() const;
    size_t DistinctPrograms() const;
    size_t PatchedListTables() const;
    size_t TensorFiles() const;
    uint64_t TensorBytes() const;
    // Tensor capture can stop early when the byte budget or a readback
    // allocation fails; this reports whether it is still active.
    bool TensorEnabled() const;
    uint64_t TensorBudgetBytes() const;
    // Inclusive frame predicate; true when tensorFirstFrame/tensorLastFrame
    // leave `frame` selected.  Unset bounds select every frame.
    static constexpr bool TensorFrameSelected(const DispatchTraceOptions& options, uint32_t frame)
    {
        if (options.tensorFirstFrame && frame < *options.tensorFirstFrame) return false;
        if (options.tensorLastFrame && frame > *options.tensorLastFrame) return false;
        return true;
    }
    const std::string& Error() const { return error_; }

    // Hook implementations (public so the static stubs can reach them).
    void RecordPipeline(void* pso, const void* bytecode, size_t bytes, void* rootSignature);
    void RecordRootSignature(void* rootSignature, const void* blob, size_t bytes);
    void RecordCommandList(void* list);
    void RecordCommandQueue(void* queue);
    void ResetList(void* list);
    void RecordPipelineState(void* list, void* pso);
    void RecordDescriptorHeaps(void* list, UINT count, ID3D12DescriptorHeap* const* heaps);
    void RecordRootSignatureBinding(void* list, void* rootSignature);
    void RecordRootTable(void* list, UINT parameter, D3D12_GPU_DESCRIPTOR_HANDLE handle);
    void RecordRootConstant(void* list, UINT parameter, UINT value, UINT offset);
    void RecordRootConstants(void* list, UINT parameter, UINT count, const void* data, UINT offset);
    void RecordBarrier(void* list, UINT count, const D3D12_RESOURCE_BARRIER* barriers);
    void ResourceCreated(void* resource, D3D12_RESOURCE_STATES state);
    using DispatchCall = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
    void Dispatch(ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z, DispatchCall original);

    ResourceMap& Descriptors() { return *descriptors_; }

    static DispatchTrace* Current();

private:
    struct HeapRecord
    {
        ID3D12DescriptorHeap* heap = nullptr;
        D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
        D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
        UINT count = 0;
        UINT increment = 0;
        D3D12_DESCRIPTOR_HEAP_TYPE type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    };

    struct ListState
    {
        void* rootSignature = nullptr;
        void* pipelineState = nullptr;
        std::array<D3D12_GPU_DESCRIPTOR_HANDLE, 16> tables{};
        std::array<bool, 16> tableSet{};
        // Sized from the root signature's num32, not a fixed cap: d25 declares
        // 51 dwords and d38 declares 18, while most programs use fewer than 16.
        std::array<std::vector<uint32_t>, 16> constants;
        std::vector<HeapRecord> heaps;
    };

    struct RootSignatureEntry
    {
        std::string sha256;
        RootLayout layout;
    };

    struct PipelineEntry
    {
        std::string sha256;
        size_t bytes = 0;
        void* rootSignature = nullptr;
    };

    ListState& State(void* list);
    void DropState(void* list);

    bool CpuDescriptor(const ListState& state, D3D12_GPU_DESCRIPTOR_HANDLE gpu, D3D12_CPU_DESCRIPTOR_HANDLE& cpu) const;

    void Fail(const std::string& message);

    ID3D12Device* device_ = nullptr;
    TraceWriter* writer_ = nullptr;
    DecodeRelay* relay_ = nullptr;
    std::unique_ptr<ResourceMap> descriptors_;
    std::unique_ptr<TensorDump> tensors_;
    DispatchTraceOptions options_;

    mutable std::mutex mutex_;
    std::unordered_map<void*, ListState> lists_;
    std::map<std::string, RootSignatureEntry> rootSignatures_;
    std::map<std::string, PipelineEntry> pipelines_;
    std::unordered_map<uintptr_t, std::string> rootSignatureByPointer_;
    std::unordered_map<uintptr_t, std::string> pipelineByPointer_;
    std::set<std::string> dispatchedPrograms_;

    std::atomic<uint32_t> frame_{0};
    std::atomic<uint64_t> dispatches_{0};
    std::vector<ID3D12CommandQueue*> providerQueues_; // AddRef'd, released in Detach/dtor
    std::string error_;
};
} // namespace probe
