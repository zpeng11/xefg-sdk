#include "layer_params.h"

#include "sha256_util.h"
#include "trace_writer.h"

#include <Windows.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace probe
{
namespace
{
void ReadMemory(uintptr_t address, void* out, size_t count)
{
    SIZE_T got = 0;
    if (!address || count > 4000000
        || !ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(address), out, count, &got)
        || got != count)
        throw std::runtime_error("parameters_read");
}

template <class T>
T Get(uintptr_t address)
{
    T value{};
    ReadMemory(address, &value, sizeof(value));
    return value;
}

std::string String(uintptr_t address)
{
    const auto size = Get<size_t>(address + 16);
    const auto capacity = Get<size_t>(address + 24);
    if (!size || size > 128 || capacity < size || capacity > 65536) throw std::runtime_error("parameters_string");
    std::string out(size, '\0');
    ReadMemory(capacity >= 16 ? Get<uintptr_t>(address) : address, out.data(), size);
    for (auto c : out)
        if (c < ' ' || c > '~') throw std::runtime_error("parameters_string_ascii");
    return out;
}

std::string Hash(const void* data, size_t size)
{
    sha256_util::Digest digest{};
    if (!sha256_util::Compute(data, size, digest)) throw std::runtime_error("parameters_hash");
    return sha256_util::LowerHex(digest);
}

bool WriteIfAbsent(const std::filesystem::path& path, const std::vector<unsigned char>& bytes)
{
    std::error_code ignored;
    std::filesystem::create_directories(path.parent_path(), ignored);
    if (std::filesystem::exists(path)) return true;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}
} // namespace

LayerParameters::LayerParameters(uintptr_t moduleBase, uintptr_t imageEnd, TraceWriter& writer,
    std::filesystem::path directory)
    : base_(moduleBase), imageEnd_(imageEnd), writer_(&writer), directory_(std::move(directory))
{
}

size_t LayerParameters::Count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return captures_;
}

bool LayerParameters::Capture(const std::string& key)
{
    try
    {
        CONTEXT context{};
        RtlCaptureContext(&context);
        for (unsigned frame = 0; frame < 16 && context.Rip; ++frame)
        {
            if (context.Rip >= base_ + 0x1aca60 && context.Rip < base_ + 0x1ad611)
            {
                const auto parameterVector = Get<uintptr_t>(context.Rsp + 0x38);
                const auto begin = Get<uintptr_t>(parameterVector);
                const auto end = Get<uintptr_t>(parameterVector + 8);
                const auto capacity = Get<uintptr_t>(parameterVector + 16);
                if (end < begin || capacity < end || end - begin > 72ull * 64 || (end - begin) % 72)
                    throw std::runtime_error("parameters_shape");

                std::ostringstream fields;
                fields << "\"type\":\"layer_parameters\",\"key\":\"" << key << "\",\"parameters\":[";
                for (auto at = begin; at < end; at += 72)
                {
                    if (at != begin) fields << ',';
                    fields << "[\"" << JsonEscape(String(at + 8)) << "\",\"" << JsonEscape(String(at + 40)) << "\"]";
                }
                fields << "]";

                const auto block = Get<uintptr_t>(context.Rsp + 0x40);
                const auto vtable = Get<uintptr_t>(block);
                if (vtable < base_ || vtable >= imageEnd_) throw std::runtime_error("parameters_block_owner");
                fields << ",\"block_vtable_rva\":" << (vtable - base_);

                if (vtable != base_ + 0xf41da8 && vtable != base_ + 0xf41ea8) return false;

                const std::array<std::pair<unsigned, const char*>, 3> tensors{{
                    {0x150, "weights"}, {0x168, "bias"}, {0x180, "activation"}}};
                for (const auto& [offset, suffix] : tensors)
                {
                    if (vtable == base_ + 0xf41ea8 && offset == 0x180) continue;
                    const auto start = Get<uintptr_t>(block + offset);
                    const auto tensorEnd = Get<uintptr_t>(block + offset + 8);
                    const auto tensorCapacity = Get<uintptr_t>(block + offset + 16);
                    if (!start || tensorEnd <= start || tensorCapacity < tensorEnd || tensorEnd - start > 2000000)
                        throw std::runtime_error("parameters_tensor_span");
                    std::vector<unsigned char> bytes(static_cast<size_t>(tensorEnd - start));
                    for (size_t at = 0; at < bytes.size(); at += 500000)
                        ReadMemory(start + at, bytes.data() + at, (std::min)(size_t(500000), bytes.size() - at));
                    const auto sha = Hash(bytes.data(), bytes.size());
                    const auto path = directory_ / (key + "-" + suffix + ".bin");
                    if (!WriteIfAbsent(path, bytes)) throw std::runtime_error("parameters_write");
                    fields << ",\"" << suffix << "_bytes\":" << bytes.size()
                           << ",\"" << suffix << "_sha256\":\"" << sha << "\""
                           << ",\"" << suffix << "_file\":\"" << JsonEscape(path.filename().string()) << "\"";
                }
                const auto metadata = Get<uintptr_t>(block + 0x148);
                fields << ",\"positive_scale\":" << Get<int>(metadata + 0x8c)
                       << ",\"positive_shift\":" << static_cast<unsigned>(Get<unsigned char>(metadata + 0x90))
                       << ",\"negative_shift\":" << static_cast<unsigned>(Get<unsigned char>(metadata + 0x94));
                if (vtable == base_ + 0xf41ea8)
                {
                    const auto descriptor = Get<uintptr_t>(block + 0x140);
                    fields << ",\"output_channels\":" << Get<unsigned>(descriptor + 0x60);
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++captures_;
                }
                if (writer_) writer_->Event(fields.str());
                return true;
            }
            DWORD64 imageBase{};
            auto* entry = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr);
            if (!entry) return false;
            PVOID handlerData{};
            DWORD64 establisher{};
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, entry, &context, &handlerData, &establisher,
                nullptr);
        }
    }
    catch (...)
    {
        // Every other layer family has a different builder shape.
    }
    return false;
}
} // namespace probe
