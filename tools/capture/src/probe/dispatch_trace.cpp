#include "dispatch_trace.h"

#include "decode_relay.h"
#include "resource_map.h"
#include "sha256_util.h"
#include "swapchain_capture.h"
#include "tensor_dump.h"
#include "trace_writer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace probe
{
namespace
{
// D3D12 ABI slots (the same constants source/native/xess/cuda/cuda_route.cpp uses).
constexpr int kCreateCommandQueue = 8;
constexpr int kCreateComputePipelineState = 11;
constexpr int kCreateCommandList = 12;
constexpr int kCreateRootSignature = 16;
constexpr int kCreateConstantBufferView = 17;
constexpr int kCreateShaderResourceView = 18;
constexpr int kCreateUnorderedAccessView = 19;
constexpr int kCopyDescriptors = 23;
constexpr int kCopyDescriptorsSimple = 24;
constexpr int kCreateCommittedResource = 27;
constexpr int kCreatePlacedResource = 29;
constexpr int kCreatePipelineState = 47;
constexpr int kCreateCommandList1 = 51;

constexpr int kReset = 10;
constexpr int kDispatch = 14;
constexpr int kSetPipelineState = 25;
constexpr int kResourceBarrier = 26;
constexpr int kSetDescriptorHeaps = 28;
constexpr int kSetComputeRootSignature = 29;
constexpr int kSetComputeRootDescriptorTable = 31;
constexpr int kSetComputeRoot32BitConstant = 33;
constexpr int kSetComputeRoot32BitConstants = 35;

struct TableEntry
{
    void** table = nullptr;
    std::array<void*, 64> original{};
    char kind = 0; // 'D' device, 'L' command list
};

bool DebugHooks()
{
    static const bool enabled = [] {
        char value[8]{};
        return GetEnvironmentVariableA("PROBE_DEBUG", value, sizeof(value)) != 0 && value[0] == '1';
    }();
    return enabled;
}

std::mutex gTableMutex;
std::vector<TableEntry> gTables;
DispatchTrace* gInstance = nullptr;

void** TableOf(void* object)
{
    return *reinterpret_cast<void***>(object);
}

void* OriginalSlot(void** table, int slot)
{
    std::lock_guard<std::mutex> lock(gTableMutex);
    for (const auto& entry : gTables)
        if (entry.table == table) return entry.original[slot];
    for (const auto& entry : gTables)
        if (entry.original[slot]) return entry.original[slot];
    return nullptr;
}

bool AdoptTable(void** table, const std::vector<std::pair<int, void*>>& hooks, char kind)
{
    if (!table) return false;
    std::lock_guard<std::mutex> lock(gTableMutex);
    TableEntry* entry = nullptr;
    for (auto& candidate : gTables)
        if (candidate.table == table)
        {
            entry = &candidate;
            break;
        }
    if (!entry)
    {
        if (gTables.size() >= 32) return false;
        gTables.push_back({});
        entry = &gTables.back();
        entry->table = table;
        entry->kind = kind;
    }
    // A device table and a command-list table have different layouts; patching
    // one with the other's slots would corrupt unrelated methods.
    else if (entry->kind != kind) return false;
    for (const auto& [slot, hook] : hooks)
        if (!entry->original[slot]) entry->original[slot] = table[slot];
    for (const auto& [slot, hook] : hooks)
    {
        if (table[slot] == hook) continue;
        DWORD old{};
        const bool changed = VirtualProtect(&table[slot], sizeof(void*), PAGE_READWRITE, &old) != FALSE;
        InterlockedExchangePointer(&table[slot], hook);
        if (changed) VirtualProtect(&table[slot], sizeof(void*), old, &old);
    }
    return true;
}

std::string HashBytes(const void* data, size_t bytes)
{
    if (!data || !bytes) return {};
    sha256_util::Digest digest{};
    if (!sha256_util::Compute(data, bytes, digest)) return {};
    return sha256_util::LowerHex(digest);
}

const char* DimensionName(D3D12_RESOURCE_DIMENSION dimension)
{
    switch (dimension)
    {
    case D3D12_RESOURCE_DIMENSION_BUFFER: return "buffer";
    case D3D12_RESOURCE_DIMENSION_TEXTURE1D: return "texture1d";
    case D3D12_RESOURCE_DIMENSION_TEXTURE2D: return "texture2d";
    case D3D12_RESOURCE_DIMENSION_TEXTURE3D: return "texture3d";
    default: return "unknown";
    }
}

const char* RangeKind(uint32_t type)
{
    switch (type)
    {
    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV: return "srv";
    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV: return "uav";
    case D3D12_DESCRIPTOR_RANGE_TYPE_CBV: return "cbv";
    case D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER: return "sampler";
    default: return "unknown";
    }
}

// ---- device hooks -----------------------------------------------------------
using CreateComputeFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID,
    void**);
using CreateListFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE,
    ID3D12CommandAllocator*, ID3D12PipelineState*, REFIID, void**);
using CreateList1Fn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device4*, UINT, D3D12_COMMAND_LIST_TYPE,
    D3D12_COMMAND_LIST_FLAGS, REFIID, void**);
using CreateRootFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, UINT, const void*, SIZE_T, REFIID, void**);
using CreateCbvFn = void (STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_CONSTANT_BUFFER_VIEW_DESC*,
    D3D12_CPU_DESCRIPTOR_HANDLE);
using CreateSrvFn = void (STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*,
    D3D12_CPU_DESCRIPTOR_HANDLE);
using CreateUavFn = void (STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, ID3D12Resource*,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using CopyDescriptorsFn = void (STDMETHODCALLTYPE*)(ID3D12Device*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*,
    UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, D3D12_DESCRIPTOR_HEAP_TYPE);
using CopyDescriptorsSimpleFn = void (STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_TYPE);
using CreateCommittedFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
    const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
using CreatePlacedFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Heap*, UINT64, const D3D12_RESOURCE_DESC*,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
using CreateQueueFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
using CreateStreamFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device2*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID,
    void**);

HRESULT STDMETHODCALLTYPE HookCreateComputePipelineState(ID3D12Device* self,
    const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID iid, void** out)
{
    const auto original = reinterpret_cast<CreateComputeFn>(OriginalSlot(TableOf(self), kCreateComputePipelineState));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, desc, iid, out);
    if (SUCCEEDED(result) && out && *out && desc && desc->CS.pShaderBytecode && desc->CS.BytecodeLength && gInstance)
        gInstance->RecordPipeline(*out, desc->CS.pShaderBytecode, desc->CS.BytecodeLength, desc->pRootSignature);
    return result;
}

HRESULT STDMETHODCALLTYPE HookCreateCommandList(ID3D12Device* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
    ID3D12CommandAllocator* allocator, ID3D12PipelineState* state, REFIID iid, void** out)
{
    const auto original = reinterpret_cast<CreateListFn>(OriginalSlot(TableOf(self), kCreateCommandList));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, mask, type, allocator, state, iid, out);
    if (SUCCEEDED(result) && out && *out && gInstance) gInstance->RecordCommandList(*out);
    return result;
}

HRESULT STDMETHODCALLTYPE HookCreateCommandList1(ID3D12Device4* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
    D3D12_COMMAND_LIST_FLAGS flags, REFIID iid, void** out)
{
    const auto original = reinterpret_cast<CreateList1Fn>(OriginalSlot(TableOf(self), kCreateCommandList1));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, mask, type, flags, iid, out);
    if (SUCCEEDED(result) && out && *out && gInstance) gInstance->RecordCommandList(*out);
    return result;
}

HRESULT STDMETHODCALLTYPE HookCreateRootSignature(ID3D12Device* self, UINT nodeMask, const void* blob, SIZE_T bytes,
    REFIID iid, void** out)
{
    const auto original = reinterpret_cast<CreateRootFn>(OriginalSlot(TableOf(self), kCreateRootSignature));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, nodeMask, blob, bytes, iid, out);
    if (SUCCEEDED(result) && out && *out && blob && bytes && gInstance)
        gInstance->RecordRootSignature(*out, blob, bytes);
    return result;
}

HRESULT STDMETHODCALLTYPE HookCreateCommandQueue(ID3D12Device* self, const D3D12_COMMAND_QUEUE_DESC* desc,
    REFIID iid, void** out)
{
    const auto original = reinterpret_cast<CreateQueueFn>(OriginalSlot(TableOf(self), kCreateCommandQueue));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, desc, iid, out);
    if (SUCCEEDED(result) && out && *out && gInstance)
        gInstance->RecordCommandQueue(*out);
    return result;
}

void STDMETHODCALLTYPE HookCreateConstantBufferView(ID3D12Device* self,
    const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
    const auto original = reinterpret_cast<CreateCbvFn>(OriginalSlot(TableOf(self), kCreateConstantBufferView));
    if (!original) return;
    original(self, desc, destination);
    if (gInstance) gInstance->Descriptors().ConstantBufferView(destination, desc);
}

void STDMETHODCALLTYPE HookCreateShaderResourceView(ID3D12Device* self, ID3D12Resource* resource,
    const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
    const auto original = reinterpret_cast<CreateSrvFn>(OriginalSlot(TableOf(self), kCreateShaderResourceView));
    if (!original) return;
    original(self, resource, desc, destination);
    if (gInstance) gInstance->Descriptors().ShaderResourceView(destination, resource, desc);
}

void STDMETHODCALLTYPE HookCreateUnorderedAccessView(ID3D12Device* self, ID3D12Resource* resource,
    ID3D12Resource* counter, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
    D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
    const auto original = reinterpret_cast<CreateUavFn>(OriginalSlot(TableOf(self), kCreateUnorderedAccessView));
    if (!original) return;
    // The counter must reach the provider unchanged: dropping it would create a
    // counter-less view and change what the provider reads back.
    original(self, resource, counter, desc, destination);
    if (gInstance) gInstance->Descriptors().UnorderedAccessView(destination, resource, desc);
}

void STDMETHODCALLTYPE HookCopyDescriptors(ID3D12Device* self, UINT destinationRanges,
    const D3D12_CPU_DESCRIPTOR_HANDLE* destinationStarts, const UINT* destinationSizes, UINT sourceRanges,
    const D3D12_CPU_DESCRIPTOR_HANDLE* sourceStarts, const UINT* sourceSizes, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    const auto original = reinterpret_cast<CopyDescriptorsFn>(OriginalSlot(TableOf(self), kCopyDescriptors));
    if (!original) return;
    original(self, destinationRanges, destinationStarts, destinationSizes, sourceRanges, sourceStarts, sourceSizes, type);
    if (!gInstance) return;
    const UINT increment = self->GetDescriptorHandleIncrementSize(type);
    const UINT ranges = (std::min)(destinationRanges, sourceRanges);
    for (UINT i = 0; i < ranges; ++i)
    {
        const UINT count = sourceSizes ? sourceSizes[i] : (destinationSizes ? destinationSizes[i] : 1);
        gInstance->Descriptors().DescriptorsCopied(destinationStarts[i], sourceStarts[i], count, increment);
    }
}

void STDMETHODCALLTYPE HookCopyDescriptorsSimple(ID3D12Device* self, UINT count,
    D3D12_CPU_DESCRIPTOR_HANDLE destination, D3D12_CPU_DESCRIPTOR_HANDLE source, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    const auto original = reinterpret_cast<CopyDescriptorsSimpleFn>(OriginalSlot(TableOf(self), kCopyDescriptorsSimple));
    if (!original) return;
    original(self, count, destination, source, type);
    if (gInstance)
        gInstance->Descriptors().DescriptorsCopied(destination, source, count,
            self->GetDescriptorHandleIncrementSize(type));
}

HRESULT STDMETHODCALLTYPE HookCreateCommittedResource(ID3D12Device* self, const D3D12_HEAP_PROPERTIES* heap,
    D3D12_HEAP_FLAGS flags, const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state,
    const D3D12_CLEAR_VALUE* clear, REFIID iid, void** out)
{
    const auto original = reinterpret_cast<CreateCommittedFn>(OriginalSlot(TableOf(self), kCreateCommittedResource));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, heap, flags, desc, state, clear, iid, out);
    if (SUCCEEDED(result) && out && *out && gInstance) gInstance->ResourceCreated(*out, state);
    return result;
}

HRESULT STDMETHODCALLTYPE HookCreatePlacedResource(ID3D12Device* self, ID3D12Heap* heap, UINT64 offset,
    const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear, REFIID iid,
    void** out)
{
    const auto original = reinterpret_cast<CreatePlacedFn>(OriginalSlot(TableOf(self), kCreatePlacedResource));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, heap, offset, desc, state, clear, iid, out);
    if (SUCCEEDED(result) && out && *out && gInstance) gInstance->ResourceCreated(*out, state);
    return result;
}

template <class Inner>
constexpr size_t InnerOffset()
{
    return (sizeof(UINT) + alignof(Inner) - 1) / alignof(Inner) * alignof(Inner);
}

template <class Inner>
constexpr size_t SubobjectSize()
{
    constexpr size_t align = (std::max)(alignof(Inner), alignof(void*));
    return (InnerOffset<Inner>() + sizeof(Inner) + align - 1) / align * align;
}

const D3D12_SHADER_BYTECODE* StreamComputeShader(const D3D12_PIPELINE_STATE_STREAM_DESC& desc,
    ID3D12RootSignature** rootSignature)
{
    const auto* stream = static_cast<const unsigned char*>(desc.pPipelineStateSubobjectStream);
    if (!stream || reinterpret_cast<uintptr_t>(stream) % alignof(void*)) return nullptr;
    const D3D12_SHADER_BYTECODE* compute{};
    for (size_t at = 0; at + sizeof(UINT) <= desc.SizeInBytes;)
    {
        size_t size{};
        const UINT type = *reinterpret_cast<const UINT*>(stream + at);
        switch (type)
        {
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS:
            if (at + SubobjectSize<D3D12_SHADER_BYTECODE>() > desc.SizeInBytes) return nullptr;
            compute = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + at + InnerOffset<D3D12_SHADER_BYTECODE>());
            size = SubobjectSize<D3D12_SHADER_BYTECODE>();
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE:
            if (at + SubobjectSize<ID3D12RootSignature*>() > desc.SizeInBytes) return nullptr;
            if (rootSignature)
                *rootSignature = *reinterpret_cast<ID3D12RootSignature* const*>(stream + at
                    + InnerOffset<ID3D12RootSignature*>());
            size = SubobjectSize<ID3D12RootSignature*>();
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK: size = SubobjectSize<UINT>(); break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO: size = SubobjectSize<D3D12_CACHED_PIPELINE_STATE>(); break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS: size = SubobjectSize<D3D12_PIPELINE_STATE_FLAGS>(); break;
        default: return nullptr;
        }
        at += size;
    }
    return compute;
}

HRESULT STDMETHODCALLTYPE HookCreatePipelineState(ID3D12Device2* self, const D3D12_PIPELINE_STATE_STREAM_DESC* desc,
    REFIID iid, void** out)
{
    if (DebugHooks()) std::fprintf(stderr, "hook: CreatePipelineState bytes=%zu\n", desc ? desc->SizeInBytes : 0);
    const auto original = reinterpret_cast<CreateStreamFn>(OriginalSlot(TableOf(self), kCreatePipelineState));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, desc, iid, out);
    if (SUCCEEDED(result) && desc && out && *out && gInstance)
    {
        ID3D12RootSignature* rootSignature = nullptr;
        if (const D3D12_SHADER_BYTECODE* shader = StreamComputeShader(*desc, &rootSignature))
            if (shader->pShaderBytecode && shader->BytecodeLength)
                gInstance->RecordPipeline(*out, shader->pShaderBytecode, shader->BytecodeLength, rootSignature);
    }
    return result;
}

// ---- command-list hooks -----------------------------------------------------
using ResetFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*,
    ID3D12PipelineState*);
using SetPipelineFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);
using BarrierFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*);
using SetHeapsFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, ID3D12DescriptorHeap* const*);
using SetRootSigFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12RootSignature*);
using SetRootTableFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE);
using SetConstantFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using SetConstantsFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, const void*, UINT);

HRESULT STDMETHODCALLTYPE HookReset(ID3D12GraphicsCommandList* self, ID3D12CommandAllocator* allocator,
    ID3D12PipelineState* state)
{
    const auto original = reinterpret_cast<ResetFn>(OriginalSlot(TableOf(self), kReset));
    if (!original) return E_FAIL;
    const HRESULT result = original(self, allocator, state);
    if (gInstance) gInstance->ResetList(self);
    return result;
}

void STDMETHODCALLTYPE HookDispatch(ID3D12GraphicsCommandList* self, UINT x, UINT y, UINT z)
{
    const auto original = reinterpret_cast<DispatchTrace::DispatchCall>(
        OriginalSlot(TableOf(self), kDispatch));
    if (!original) return;
    if (gInstance) gInstance->Dispatch(self, x, y, z, original);
    else original(self, x, y, z);
}

void STDMETHODCALLTYPE HookSetPipelineState(ID3D12GraphicsCommandList* self, ID3D12PipelineState* state)
{
    const auto original = reinterpret_cast<SetPipelineFn>(OriginalSlot(TableOf(self), kSetPipelineState));
    if (!original) return;
    original(self, state);
    if (gInstance) gInstance->RecordPipelineState(self, state);
}

void STDMETHODCALLTYPE HookResourceBarrier(ID3D12GraphicsCommandList* self, UINT count,
    const D3D12_RESOURCE_BARRIER* barriers)
{
    const auto original = reinterpret_cast<BarrierFn>(OriginalSlot(TableOf(self), kResourceBarrier));
    if (!original) return;
    if (gInstance) gInstance->RecordBarrier(self, count, barriers);
    original(self, count, barriers);
}

void STDMETHODCALLTYPE HookSetDescriptorHeaps(ID3D12GraphicsCommandList* self, UINT count,
    ID3D12DescriptorHeap* const* heaps)
{
    if (DebugHooks()) std::fprintf(stderr, "hook: SetDescriptorHeaps list=%p count=%u\n", static_cast<void*>(self), count);
    const auto original = reinterpret_cast<SetHeapsFn>(OriginalSlot(TableOf(self), kSetDescriptorHeaps));
    if (!original) return;
    original(self, count, heaps);
    if (gInstance) gInstance->RecordDescriptorHeaps(self, count, heaps);
}

void STDMETHODCALLTYPE HookSetComputeRootSignature(ID3D12GraphicsCommandList* self, ID3D12RootSignature* root)
{
    const auto original = reinterpret_cast<SetRootSigFn>(OriginalSlot(TableOf(self), kSetComputeRootSignature));
    if (!original) return;
    original(self, root);
    if (gInstance) gInstance->RecordRootSignatureBinding(self, root);
}

void STDMETHODCALLTYPE HookSetComputeRootDescriptorTable(ID3D12GraphicsCommandList* self, UINT parameter,
    D3D12_GPU_DESCRIPTOR_HANDLE handle)
{
    const auto original = reinterpret_cast<SetRootTableFn>(OriginalSlot(TableOf(self), kSetComputeRootDescriptorTable));
    if (!original) return;
    original(self, parameter, handle);
    if (gInstance) gInstance->RecordRootTable(self, parameter, handle);
}

void STDMETHODCALLTYPE HookSetComputeRoot32BitConstant(ID3D12GraphicsCommandList* self, UINT parameter, UINT value,
    UINT offset)
{
    const auto original = reinterpret_cast<SetConstantFn>(OriginalSlot(TableOf(self), kSetComputeRoot32BitConstant));
    if (!original) return;
    original(self, parameter, value, offset);
    if (gInstance) gInstance->RecordRootConstant(self, parameter, value, offset);
}

void STDMETHODCALLTYPE HookSetComputeRoot32BitConstants(ID3D12GraphicsCommandList* self, UINT parameter, UINT count,
    const void* data, UINT offset)
{
    const auto original = reinterpret_cast<SetConstantsFn>(OriginalSlot(TableOf(self), kSetComputeRoot32BitConstants));
    if (!original) return;
    original(self, parameter, count, data, offset);
    if (gInstance) gInstance->RecordRootConstants(self, parameter, count, data, offset);
}
} // namespace

DispatchTrace::DispatchTrace() : descriptors_(std::make_unique<ResourceMap>()) {}
DispatchTrace::~DispatchTrace() { Detach(); }

DispatchTrace* DispatchTrace::Current() { return gInstance; }

bool DispatchTrace::Install(ID3D12Device* device, ID3D12GraphicsCommandList* ownList, TraceWriter& writer,
    DecodeRelay& relay, std::string& error)
{
    if (!device)
    {
        error = "no device";
        return false;
    }
    device_ = device;
    writer_ = &writer;
    relay_ = &relay;
    if (options_.tensors)
    {
        auto tensors = std::make_unique<TensorDump>();
        const uint64_t budget = uint64_t(options_.tensorBudgetMB ? options_.tensorBudgetMB : 512) << 20;
        if (tensors->Initialize(device, options_.tensorDirectory, budget)) tensors_ = std::move(tensors);
        else std::fprintf(stderr, "trace: tensor dump unavailable, continuing without it\n");
    }

    const std::vector<std::pair<int, void*>> deviceHooks{
        {kCreateCommandQueue, reinterpret_cast<void*>(&HookCreateCommandQueue)},
        {kCreateComputePipelineState, reinterpret_cast<void*>(&HookCreateComputePipelineState)},
        {kCreateCommandList, reinterpret_cast<void*>(&HookCreateCommandList)},
        {kCreateCommandList1, reinterpret_cast<void*>(&HookCreateCommandList1)},
        {kCreateRootSignature, reinterpret_cast<void*>(&HookCreateRootSignature)},
        {kCreateConstantBufferView, reinterpret_cast<void*>(&HookCreateConstantBufferView)},
        {kCreateShaderResourceView, reinterpret_cast<void*>(&HookCreateShaderResourceView)},
        {kCreateUnorderedAccessView, reinterpret_cast<void*>(&HookCreateUnorderedAccessView)},
        {kCopyDescriptors, reinterpret_cast<void*>(&HookCopyDescriptors)},
        {kCopyDescriptorsSimple, reinterpret_cast<void*>(&HookCopyDescriptorsSimple)},
        {kCreateCommittedResource, reinterpret_cast<void*>(&HookCreateCommittedResource)},
        {kCreatePlacedResource, reinterpret_cast<void*>(&HookCreatePlacedResource)},
        {kCreatePipelineState, reinterpret_cast<void*>(&HookCreatePipelineState)}};
    const std::vector<std::pair<int, void*>> listHooks{
        {kReset, reinterpret_cast<void*>(&HookReset)},
        {kDispatch, reinterpret_cast<void*>(&HookDispatch)},
        {kSetPipelineState, reinterpret_cast<void*>(&HookSetPipelineState)},
        {kResourceBarrier, reinterpret_cast<void*>(&HookResourceBarrier)},
        {kSetDescriptorHeaps, reinterpret_cast<void*>(&HookSetDescriptorHeaps)},
        {kSetComputeRootSignature, reinterpret_cast<void*>(&HookSetComputeRootSignature)},
        {kSetComputeRootDescriptorTable, reinterpret_cast<void*>(&HookSetComputeRootDescriptorTable)},
        {kSetComputeRoot32BitConstant, reinterpret_cast<void*>(&HookSetComputeRoot32BitConstant)},
        {kSetComputeRoot32BitConstants, reinterpret_cast<void*>(&HookSetComputeRoot32BitConstants)}};

    // Every distinct vtable of the device's interfaces; patch each.
    const IID interfaces[]{
        __uuidof(ID3D12Device), __uuidof(ID3D12Device1), __uuidof(ID3D12Device2), __uuidof(ID3D12Device3),
        __uuidof(ID3D12Device4), __uuidof(ID3D12Device5), __uuidof(ID3D12Device6), __uuidof(ID3D12Device7),
        __uuidof(ID3D12Device8), __uuidof(ID3D12Device9)};
    size_t patched = 0;
    for (const auto& iid : interfaces)
    {
        void* view = nullptr;
        if (FAILED(device->QueryInterface(iid, &view)) || !view) continue;
        if (AdoptTable(*reinterpret_cast<void***>(view), deviceHooks, 'D')) ++patched;
        static_cast<IUnknown*>(view)->Release();
    }
    if (!patched)
    {
        error = "device vtables could not be patched";
        return false;
    }
    if (ownList) RecordCommandList(ownList);
    gInstance = this;
    if (DebugHooks())
        std::fprintf(stderr, "trace: device=%p table=%p\n", static_cast<void*>(device),
            static_cast<void*>(TableOf(device)));
    error.clear();
    return true;
}

void DispatchTrace::Detach()
{
    if (gInstance == this) gInstance = nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* queue : providerQueues_) queue->Release();
    providerQueues_.clear();
}

void DispatchTrace::SetFrame(uint32_t frame)
{
    frame_.store(frame, std::memory_order_relaxed);
}

void DispatchTrace::RecordCommandQueue(void* queue)
{
    if (!queue) return;
    ID3D12CommandQueue* typed = nullptr;
    if (FAILED(static_cast<IUnknown*>(queue)->QueryInterface(IID_PPV_ARGS(&typed))) || !typed) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* existing : providerQueues_)
        if (existing == typed)
        {
            typed->Release();
            return;
        }
    providerQueues_.push_back(typed); // QueryInterface already added the reference
}

size_t DispatchTrace::ProviderQueues() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return providerQueues_.size();
}

bool DispatchTrace::FinishTensors(std::string& error)
{
    error.clear();
    if (!tensors_) return true;
    std::vector<ID3D12CommandQueue*> queues;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queues = providerQueues_;
    }
    // Without a queue there is no fence to wait on, so the dumps cannot be
    // called complete. Report that instead of mapping them on a timer.
    if (queues.empty())
    {
        error = "no provider command queue observed; tensor dumps are unverified";
        return false;
    }
    return tensors_->Finish(queues.data(), queues.size(), 30000, error);
}

size_t DispatchTrace::TensorFiles() const
{
    return tensors_ ? tensors_->Count() : 0;
}

uint64_t DispatchTrace::TensorBytes() const
{
    return tensors_ ? tensors_->Bytes() : 0;
}

bool DispatchTrace::TensorEnabled() const
{
    return tensors_ && tensors_->Enabled();
}

uint64_t DispatchTrace::TensorBudgetBytes() const
{
    return uint64_t(options_.tensorBudgetMB ? options_.tensorBudgetMB : 512) << 20;
}

uint64_t DispatchTrace::DispatchCount() const { return dispatches_.load(); }

size_t DispatchTrace::DistinctPrograms() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return dispatchedPrograms_.size();
}

size_t DispatchTrace::PatchedListTables() const
{
    std::lock_guard<std::mutex> lock(gTableMutex);
    return gTables.size();
}

void DispatchTrace::Fail(const std::string& message)
{
    if (error_.empty())
    {
        error_ = message;
        std::fprintf(stderr, "trace: %s\n", message.c_str());
    }
}

void DispatchTrace::RecordPipeline(void* pso, const void* bytecode, size_t bytes, void* rootSignature)
{
    try
    {
        const std::string sha = HashBytes(bytecode, bytes);
        if (sha.empty()) return;
        const bool first = [&] {
            std::lock_guard<std::mutex> lock(mutex_);
            pipelineByPointer_[reinterpret_cast<uintptr_t>(pso)] = sha;
            if (pipelines_.count(sha)) return false;
            pipelines_[sha] = {sha, bytes, rootSignature};
            return true;
        }();
        if (!first || !writer_) return;
        std::ostringstream event;
        event << "\"type\":\"pipeline\",\"pso\":\"" << HexPointer(pso) << "\",\"sha256\":\"" << sha
              << "\",\"bytes\":" << bytes << ",\"root_signature\":\"" << HexPointer(rootSignature) << "\"";
        writer_->Event(event.str());
    }
    catch (...) {}
}

void DispatchTrace::RecordRootSignature(void* rootSignature, const void* blob, size_t bytes)
{
    try
    {
        RootLayout layout;
        std::string error;
        if (!ParseRootSignatureBlob(blob, bytes, layout, error)) return;
        const std::string sha = HashBytes(blob, bytes);
        if (sha.empty()) return;
        std::ostringstream parameters;
        parameters << '[';
        for (size_t i = 0; i < layout.parameters.size(); ++i)
        {
            const auto& parameter = layout.parameters[i];
            if (i) parameters << ',';
            parameters << "{\"index\":" << i << ",\"type\":" << parameter.type;
            if (parameter.type == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
            {
                parameters << ",\"ranges\":[";
                for (size_t r = 0; r < parameter.ranges.size(); ++r)
                {
                    const auto& range = parameter.ranges[r];
                    if (r) parameters << ',';
                    parameters << "{\"kind\":\"" << RangeKind(range.type) << "\",\"register\":"
                               << range.baseRegister << ",\"count\":" << range.count << ",\"space\":"
                               << range.space << ",\"offset\":" << range.offset << "}";
                }
                parameters << ']';
            }
            else if (parameter.type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS)
            {
                parameters << ",\"register\":" << parameter.shaderRegister << ",\"num32\":" << parameter.num32;
            }
            else parameters << ",\"register\":" << parameter.shaderRegister;
            parameters << '}';
        }
        parameters << ']';

        std::ostringstream samplers;
        samplers << '[';
        for (size_t i = 0; i < layout.samplers.size(); ++i)
        {
            const auto& sampler = layout.samplers[i];
            if (i) samplers << ',';
            samplers << "{\"filter\":" << sampler.filter << ",\"address_u\":" << sampler.addressU
                     << ",\"address_v\":" << sampler.addressV << ",\"address_w\":" << sampler.addressW
                     << ",\"mip_lod_bias\":" << sampler.mipLodBias << ",\"max_anisotropy\":"
                     << sampler.maxAnisotropy << ",\"comparison_func\":" << sampler.comparisonFunc
                     << ",\"border_color\":" << sampler.borderColor << ",\"min_lod\":" << sampler.minLod
                     << ",\"max_lod\":" << sampler.maxLod << ",\"register\":" << sampler.shaderRegister
                     << ",\"space\":" << sampler.space << ",\"visibility\":" << sampler.visibility << '}';
        }
        samplers << ']';

        bool first = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rootSignatureByPointer_[reinterpret_cast<uintptr_t>(rootSignature)] = sha;
            if (!rootSignatures_.count(sha))
            {
                rootSignatures_[sha] = {sha, layout};
                first = true;
            }
        }
        if (first && writer_)
        {
            std::ostringstream event;
            event << "\"type\":\"root_signature\",\"root_signature\":\"" << HexPointer(rootSignature)
                  << "\",\"sha256\":\"" << sha << "\",\"version\":" << layout.version
                  << ",\"parameters\":" << parameters.str()
                  << ",\"static_samplers\":" << samplers.str();
            writer_->Event(event.str());
        }
    }
    catch (...) {}
}

void DispatchTrace::RecordCommandList(void* list)
{
    if (!list) return;
    if (DebugHooks()) std::fprintf(stderr, "trace: adopt list=%p table=%p\n", list, static_cast<void*>(TableOf(list)));
    const std::vector<std::pair<int, void*>> hooks{
        {kReset, reinterpret_cast<void*>(&HookReset)},
        {kDispatch, reinterpret_cast<void*>(&HookDispatch)},
        {kSetPipelineState, reinterpret_cast<void*>(&HookSetPipelineState)},
        {kResourceBarrier, reinterpret_cast<void*>(&HookResourceBarrier)},
        {kSetDescriptorHeaps, reinterpret_cast<void*>(&HookSetDescriptorHeaps)},
        {kSetComputeRootSignature, reinterpret_cast<void*>(&HookSetComputeRootSignature)},
        {kSetComputeRootDescriptorTable, reinterpret_cast<void*>(&HookSetComputeRootDescriptorTable)},
        {kSetComputeRoot32BitConstant, reinterpret_cast<void*>(&HookSetComputeRoot32BitConstant)},
        {kSetComputeRoot32BitConstants, reinterpret_cast<void*>(&HookSetComputeRoot32BitConstants)}};
    AdoptTable(TableOf(list), hooks, 'L');
}

void DispatchTrace::ResetList(void* list)
{
    // The list may have moved to another vtable during Reset; adopt it again.
    RecordCommandList(list);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = lists_[list];
    state = {};
}

void DispatchTrace::RecordPipelineState(void* list, void* pso)
{
    std::lock_guard<std::mutex> lock(mutex_);
    lists_[list].pipelineState = pso;
}

void DispatchTrace::RecordDescriptorHeaps(void* list, UINT count, ID3D12DescriptorHeap* const* heaps)
{
    if (!heaps || !device_) return;
    std::vector<HeapRecord> records;
    for (UINT i = 0; i < count; ++i)
    {
        if (!heaps[i]) continue;
        const D3D12_DESCRIPTOR_HEAP_DESC description = heaps[i]->GetDesc();
        HeapRecord record;
        record.heap = heaps[i];
        record.gpu = heaps[i]->GetGPUDescriptorHandleForHeapStart();
        record.cpu = heaps[i]->GetCPUDescriptorHandleForHeapStart();
        record.count = description.NumDescriptors;
        record.type = description.Type;
        record.increment = device_->GetDescriptorHandleIncrementSize(description.Type);
        records.push_back(record);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    lists_[list].heaps = std::move(records);
}

void DispatchTrace::RecordRootSignatureBinding(void* list, void* rootSignature)
{
    std::lock_guard<std::mutex> lock(mutex_);
    lists_[list].rootSignature = rootSignature;
}

void DispatchTrace::RecordRootTable(void* list, UINT parameter, D3D12_GPU_DESCRIPTOR_HANDLE handle)
{
    if (parameter >= 16) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = lists_[list];
    state.tables[parameter] = handle;
    state.tableSet[parameter] = true;
}

void DispatchTrace::RecordRootConstant(void* list, UINT parameter, UINT value, UINT offset)
{
    if (parameter >= 16) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& constants = lists_[list].constants[parameter];
    if (constants.size() <= offset) constants.resize(offset + 1);
    constants[offset] = value;
}

void DispatchTrace::RecordRootConstants(void* list, UINT parameter, UINT count, const void* data, UINT offset)
{
    if (parameter >= 16 || !data) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& constants = lists_[list].constants[parameter];
    if (constants.size() < offset + count) constants.resize(offset + count);
    for (UINT i = 0; i < count; ++i)
        constants[offset + i] = static_cast<const uint32_t*>(data)[i];
}

void DispatchTrace::RecordBarrier(void* list, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    (void)list;
    if (!barriers || !tensors_) return;
    for (UINT i = 0; i < count; ++i)
    {
        if (barriers[i].Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) continue;
        auto* resource = barriers[i].Transition.pResource;
        if (!resource) continue;
        if (tensors_->Enabled())
            tensors_->Barrier(resource, barriers[i].Transition.StateBefore, barriers[i].Transition.StateAfter);
    }
}

void DispatchTrace::ResourceCreated(void* resource, D3D12_RESOURCE_STATES state)
{
    // Both consumers need the creation state: the tensor dump for barriers and
    // the descriptor map for the buffer VA range a constant buffer view names.
    if (resource) descriptors_->ResourceCreated(static_cast<ID3D12Resource*>(resource), state);
    if (tensors_ && tensors_->Enabled()) tensors_->ResourceCreated(resource, state);
}

bool DispatchTrace::CpuDescriptor(const ListState& state, D3D12_GPU_DESCRIPTOR_HANDLE gpu,
    D3D12_CPU_DESCRIPTOR_HANDLE& cpu) const
{
    for (const auto& heap : state.heaps)
    {
        if (!heap.increment || !heap.count) continue;
        const uint64_t first = heap.gpu.ptr;
        const uint64_t last = first + uint64_t(heap.count) * heap.increment;
        if (gpu.ptr < first || gpu.ptr >= last) continue;
        cpu.ptr = heap.cpu.ptr + (gpu.ptr - first);
        return true;
    }
    return false;
}

namespace
{
struct Resolved
{
    int parameter = 0;
    uint32_t rangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    bool unorderedAccess = false;
    std::string slot;
    UINT index = 0;
    ViewInfo view;
    std::string file;
};
} // namespace

void DispatchTrace::Dispatch(ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z, DispatchCall original)
{
    const uint64_t id = dispatches_.fetch_add(1) + 1;
    const uint32_t present = oracle::NativePresentOrdinal().load(std::memory_order_relaxed);
    const uint32_t frame = frame_.load(std::memory_order_relaxed);

    ListState state;
    std::string pipelineSha;
    void* pso = nullptr;
    RootLayout layout;
    bool hasLayout = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = lists_.find(list);
        if (found != lists_.end()) state = found->second;
        pso = state.pipelineState;
        const auto pipeline = pipelineByPointer_.find(reinterpret_cast<uintptr_t>(pso));
        if (pipeline != pipelineByPointer_.end())
        {
            pipelineSha = pipeline->second;
            dispatchedPrograms_.insert(pipelineSha);
        }
        if (state.rootSignature)
        {
            const auto pointer = rootSignatureByPointer_.find(reinterpret_cast<uintptr_t>(state.rootSignature));
            if (pointer != rootSignatureByPointer_.end())
            {
                const auto entry = rootSignatures_.find(pointer->second);
                if (entry != rootSignatures_.end())
                {
                    layout = entry->second.layout;
                    hasLayout = true;
                }
            }
        }
    }

    std::vector<Resolved> resolved;
    if (hasLayout)
    {
        const auto heapFor = [&state](uint64_t address) -> const HeapRecord* {
            for (const auto& heap : state.heaps)
            {
                if (!heap.increment || !heap.count) continue;
                const uint64_t last = heap.gpu.ptr + uint64_t(heap.count) * heap.increment;
                if (address >= heap.gpu.ptr && address < last) return &heap;
            }
            return nullptr;
        };
        for (size_t parameter = 0; parameter < layout.parameters.size() && parameter < 16; ++parameter)
        {
            const auto& root = layout.parameters[parameter];
            if (root.type != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE || !state.tableSet[parameter]) continue;
            const HeapRecord* heap = heapFor(state.tables[parameter].ptr);
            if (!heap) continue;
            for (const auto& range : root.ranges)
            {
                for (uint32_t i = 0; i < range.count; ++i)
                {
                    const uint32_t descriptorIndex = range.offset + i;
                    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
                    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
                    gpu.ptr = state.tables[parameter].ptr + uint64_t(descriptorIndex) * heap->increment;
                    if (!CpuDescriptor(state, gpu, cpu)) continue;
                    ViewInfo view;
                    if (!descriptors_->View(cpu, view)) continue;
                    Resolved entry;
                    entry.parameter = static_cast<int>(parameter);
                    entry.rangeType = range.type;
                    entry.unorderedAccess = range.type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                    entry.slot = RegisterName(range.type, range.baseRegister + i);
                    entry.index = descriptorIndex;
                    entry.view = view;
                    resolved.push_back(std::move(entry));
                }
            }
        }
    }

    const bool snapshots = tensors_ && tensors_->Enabled()
        && TensorFrameSelected(options_, frame)
        && (options_.tensorDispatches == 0 || id <= options_.tensorDispatches);
    if (snapshots)
    {
        for (auto& entry : resolved)
            if (!entry.unorderedAccess)
                tensors_->Snapshot(list, entry.view, entry.slot, "in", id, present, entry.file);
    }

    original(list, x, y, z);

    if (snapshots)
    {
        for (auto& entry : resolved)
            if (entry.unorderedAccess)
                tensors_->Snapshot(list, entry.view, entry.slot, "out", id, present, entry.file);
    }

    if (!writer_) return;
    std::ostringstream event;
    event << "\"type\":\"dispatch\",\"dispatch\":" << id << ",\"frame\":" << frame << ",\"present\":" << present
          << ",\"pso\":\"" << HexPointer(pso) << "\",\"sha256\":\"" << pipelineSha << "\",\"key\":\""
          << (relay_ ? relay_->KeyForHash(pipelineSha) : std::string()) << "\""
          << ",\"root_signature\":\"" << HexPointer(state.rootSignature) << "\",\"group\":[" << x << ',' << y << ','
          << z << "],\"constants\":[";
    bool first = true;
    if (hasLayout)
    {
        for (size_t parameter = 0; parameter < layout.parameters.size() && parameter < 16; ++parameter)
        {
            const auto& root = layout.parameters[parameter];
            if (root.type != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) continue;
            if (!first) event << ',';
            first = false;
            event << "{\"parameter\":" << parameter << ",\"values\":[";
            for (uint32_t i = 0; i < root.num32; ++i)
            {
                if (i) event << ',';
                event << (i < state.constants[parameter].size()
                              ? state.constants[parameter][i] : 0u);
            }
            event << "]}";
        }
    }
    event << "],\"resources\":[";
    for (size_t i = 0; i < resolved.size(); ++i)
    {
        const auto& entry = resolved[i];
        if (i) event << ',';
        event << "{\"parameter\":" << entry.parameter << ",\"kind\":\"" << RangeKind(entry.rangeType)
              << "\",\"slot\":\"" << entry.slot << "\",\"index\":" << entry.index << ",\"resource\":\""
              << HexPointer(entry.view.resource) << "\",\"format\":" << int(entry.view.format) << ",\"dimension\":\""
              << DimensionName(entry.view.resourceDimension) << "\",\"width\":" << entry.view.width
              << ",\"height\":" << entry.view.height << ",\"array\":" << entry.view.depthOrArraySize
              << ",\"mips\":" << entry.view.mipLevels << ",\"first_mip\":" << entry.view.firstMip
              << ",\"first_slice\":" << entry.view.firstSlice << ",\"bytes\":" << entry.view.bufferBytes
              << ",\"buffer_offset\":" << entry.view.bufferOffset;
        if (!entry.file.empty()) event << ",\"file\":\"" << entry.file << "\"";
        event << '}';
    }
    event << "]";
    writer_->Event(event.str());
}
} // namespace probe
