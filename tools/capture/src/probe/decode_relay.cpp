#include "decode_relay.h"

#include "layer_params.h"
#include "runtime_layout.h"
#include "sha256_util.h"
#include "trace_writer.h"
#include "xefg_runtime.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace probe
{
namespace
{
DecodeRelay* gRelay = nullptr;

uintptr_t Base(HMODULE module)
{
    return reinterpret_cast<uintptr_t>(module);
}

size_t ImageSize(HMODULE module)
{
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return 0;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(Base(module) + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return 0;
    return nt->OptionalHeader.SizeOfImage;
}

void Read(uintptr_t address, void* out, size_t count)
{
    SIZE_T got = 0;
    if (!address || count > 4000000
        || !ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(address), out, count, &got)
        || got != count)
        throw std::runtime_error("relay_read");
}

template <class T>
T Get(uintptr_t address)
{
    T value{};
    Read(address, &value, sizeof(value));
    return value;
}

std::string String(uintptr_t address)
{
    const auto size = Get<size_t>(address + 16);
    const auto capacity = Get<size_t>(address + 24);
    if (!size || size > 128 || capacity < size || capacity > 65536) throw std::runtime_error("relay_string");
    std::string out(size, '\0');
    Read(capacity >= 16 ? Get<uintptr_t>(address) : address, out.data(), size);
    for (auto c : out)
        if (c < ' ' || c > '~') throw std::runtime_error("relay_string_ascii");
    return out;
}

std::string Hash(const void* data, size_t size)
{
    sha256_util::Digest digest{};
    if (!sha256_util::Compute(data, size, digest)) throw std::runtime_error("relay_hash");
    return sha256_util::LowerHex(digest);
}

bool IsDigits(const std::string& text)
{
    if (text.empty()) return false;
    for (char c : text)
        if (c < '0' || c > '9') return false;
    return true;
}

void CALLBACK RelayStub(void* store, const void* keyObject, void* output)
{
    if (gRelay) gRelay->Relay(store, keyObject, output);
}
} // namespace

DecodeRelay::DecodeRelay() = default;
DecodeRelay::~DecodeRelay() { Restore(); }

bool DecodeRelay::Install(HMODULE fgModule, TraceWriter& writer, const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& parameterDirectory, std::string& error)
{
    try
    {
        if (!fgModule)
        {
            error = "no libxess_fg module";
            return false;
        }
        const auto& layout = xess_layout::Get(fgModule, true);
        if (layout.family != "fg78")
        {
            error = std::string("unsupported libxess_fg family '") + layout.family
                + "' (the probe is calibrated for fg78)";
            return false;
        }
        const std::string moduleHash = oracle::ModuleFileSha256(fgModule);
        if (_stricmp(moduleHash.c_str(), oracle::kProfiledFg78Sha256) != 0)
        {
            error = "libxess_fg SHA-256 " + moduleHash
                + " is not the profiled build " + oracle::kProfiledFg78Sha256;
            return false;
        }
        base_ = Base(fgModule);
        imageEnd_ = base_ + ImageSize(fgModule);
        writer_ = &writer;
        shaderDirectory_ = shaderDirectory;
        parameters_ = std::make_unique<LayerParameters>(base_, imageEnd_, writer, parameterDirectory);

        site_ = reinterpret_cast<unsigned char*>(base_ + layout.Rva(0x1ad217));
        original_ = Get<LONG>(reinterpret_cast<uintptr_t>(site_ + 1));
        MEMORY_BASIC_INFORMATION memory{};
        if (*site_ != 0xe8
            || reinterpret_cast<uintptr_t>(site_ + 5) + original_ != base_ + layout.Rva(0x11230)
            || VirtualQuery(site_, &memory, sizeof(memory)) != sizeof(memory)
            || memory.AllocationBase != fgModule || memory.Type != MEM_IMAGE
            || memory.Protect != PAGE_EXECUTE_READ)
        {
            error = "decoder call site does not match the profiled build";
            return false;
        }
        protection_ = memory.Protect;
        decode_ = reinterpret_cast<void (*)(void*, const void*, void*)>(base_ + layout.Rva(0x19fda0));

        // A relay page inside the signed rel32 range of the call site.
        const auto callEnd = reinterpret_cast<uintptr_t>(site_ + 5);
        const uintptr_t low = callEnd > 0x7ffe0000 ? callEnd - 0x7ffe0000 : 0x10000;
        const uintptr_t high = callEnd + 0x7ffe0000;
        for (uintptr_t p = (low + 0xffff) & ~uintptr_t(0xffff); p < high;)
        {
            MEMORY_BASIC_INFORMATION region{};
            if (VirtualQuery(reinterpret_cast<void*>(p), &region, sizeof(region)) != sizeof(region)) break;
            auto end = reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize;
            if (region.State == MEM_FREE && end - p >= 0x10000)
            {
                relay_ = VirtualAlloc(reinterpret_cast<void*>(p), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                if (relay_) break;
            }
            if (end <= p) break;
            p = (end + 0xffff) & ~uintptr_t(0xffff);
        }
        if (!relay_)
        {
            error = "relay allocation failed";
            return false;
        }
        unsigned char stub[14]{0xff, 0x25, 0, 0, 0, 0};
        const auto destination = reinterpret_cast<uintptr_t>(&RelayStub);
        std::memcpy(stub + 6, &destination, 8);
        std::memcpy(relay_, stub, sizeof(stub));
        DWORD prior{};
        if (!VirtualProtect(relay_, 0x1000, PAGE_EXECUTE_READ, &prior)
            || !FlushInstructionCache(GetCurrentProcess(), relay_, sizeof(stub)))
        {
            VirtualFree(relay_, 0, MEM_RELEASE);
            relay_ = nullptr;
            error = "relay protection failed";
            return false;
        }
        const auto distance = reinterpret_cast<intptr_t>(relay_) - reinterpret_cast<intptr_t>(site_ + 5);
        if (distance < INT32_MIN || distance > INT32_MAX)
        {
            VirtualFree(relay_, 0, MEM_RELEASE);
            relay_ = nullptr;
            error = "relay out of range";
            return false;
        }
        replacement_ = static_cast<LONG>(distance);

        DWORD old{};
        if (!VirtualProtect(site_, 5, PAGE_EXECUTE_READWRITE, &old))
        {
            error = "call site is not writable";
            return false;
        }
        LONG previous = InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(site_ + 1), replacement_, original_);
        const bool flushed = FlushInstructionCache(GetCurrentProcess(), site_, 5) != FALSE;
        const bool restored = VirtualProtect(site_, 5, old, &prior) != FALSE;
        if (previous != original_ || !flushed || !restored)
        {
            error = "call site patch failed";
            return false;
        }
        gRelay = this;
        active_ = true;
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}

void DecodeRelay::Restore()
{
    if (!active_) return;
    gRelay = nullptr;
    DWORD old{}, unused{};
    if (VirtualProtect(site_, 5, PAGE_EXECUTE_READWRITE, &old))
    {
        InterlockedExchange(reinterpret_cast<volatile LONG*>(site_ + 1), original_);
        FlushInstructionCache(GetCurrentProcess(), site_, 5);
        VirtualProtect(site_, 5, old, &unused);
    }
    if (relay_) VirtualFree(relay_, 0, MEM_RELEASE);
    relay_ = nullptr;
    active_ = false;
}

std::string DecodeRelay::KeyForHash(const std::string& sha256) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = hashToKey_.find(sha256);
    return found == hashToKey_.end() ? std::string() : found->second;
}

size_t DecodeRelay::ProgramCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyToHash_.size();
}

size_t DecodeRelay::ParameterCaptureCount() const
{
    return parameters_ ? parameters_->Count() : 0;
}

bool DecodeRelay::RegisterProgram(const std::string& key, const std::string& sha256)
{
    std::lock_guard<std::mutex> lock(mutex_);
    hashToKey_.emplace(sha256, key);
    return keyToHash_.emplace(key, sha256).second;
}

void DecodeRelay::Relay(void* store, const void* keyObject, void* output)
{
    // Nothing may escape into the provider's frame.
    bool decoded = false;
    try
    {
        const auto key = String(reinterpret_cast<uintptr_t>(keyObject));
        if (!IsDigits(key)) throw std::runtime_error("relay_key");

        // The builder frame is still live here, so the parameter capture runs
        // before the provider's own decode.
        if (parameters_) parameters_->Capture(key);

        if (decode_) decode_(store, keyObject, output);
        decoded = true;

        const auto o = reinterpret_cast<uintptr_t>(output);
        const auto start = Get<uintptr_t>(o);
        const auto end = Get<uintptr_t>(o + 8);
        if (!start || end < start || end - start > 4000000) return;
        const size_t size = static_cast<size_t>(end - start);
        std::vector<unsigned char> bytes(size);
        if (size) Read(start, bytes.data(), size);
        const auto sha = Hash(bytes.data(), size);
        const bool first = RegisterProgram(key, sha);
        if (first && writer_)
        {
            std::error_code ignored;
            std::filesystem::create_directories(shaderDirectory_, ignored);
            const auto path = shaderDirectory_ / (key + "-" + sha.substr(0, 16) + ".dxil");
            if (!std::filesystem::exists(path))
            {
                std::ofstream file(path, std::ios::binary | std::ios::trunc);
                if (file) file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(size));
            }
            std::ostringstream event;
            event << "\"type\":\"program\",\"key\":\"" << key << "\",\"bytes\":" << size
                  << ",\"sha256\":\"" << sha << "\",\"file\":\"" << JsonEscape(path.filename().string()) << "\"";
            writer_->Event(event.str());
        }
    }
    catch (...)
    {
        // A capture failure must never disturb the provider.
        if (!decoded && decode_)
        {
            try { decode_(store, keyObject, output); } catch (...) {}
        }
    }
}
} // namespace probe
