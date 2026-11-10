#include "model_dump.h"

#include "runtime_layout.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

namespace oracle
{
namespace
{
constexpr size_t kMaxRead = 16u << 20;

class Reader
{
public:
    explicit Reader(HMODULE module) : base_(reinterpret_cast<uintptr_t>(module)) {}

    uintptr_t Base() const { return base_; }

    bool Bytes(uintptr_t address, void* out, size_t size) const
    {
        if (!address || !size || size > kMaxRead) return false;
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) != sizeof(info)) return false;
        if (info.State != MEM_COMMIT) return false;
        const DWORD protect = info.Protect & 0xff;
        if (protect == PAGE_NOACCESS || protect == PAGE_GUARD) return false;
        SIZE_T read = 0;
        return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), out, size, &read)
            && read == size;
    }

    template <class T>
    bool Get(uintptr_t address, T& value) const
    {
        return Bytes(address, &value, sizeof(value));
    }

    // MSVC std::string: size at +16, capacity at +24, heap pointer at +0 when
    // the capacity is 16 or more, otherwise the bytes live inline.
    bool String(uintptr_t address, std::string& value) const
    {
        size_t size = 0, capacity = 0;
        if (!Get(address + 16, size) || !Get(address + 24, capacity)) return false;
        if (!size || size > 256 || capacity < size || capacity > 65536) return false;
        value.assign(size, '\0');
        uintptr_t data = address;
        if (capacity >= 16 && (!Get(address, data) || !data)) return false;
        return Bytes(data, value.data(), size);
    }

    bool Vector(uintptr_t address, uintptr_t& begin, uintptr_t& end, uintptr_t& capacity) const
    {
        return Get(address, begin) && Get(address + 8, end) && Get(address + 16, capacity)
            && begin <= end && end <= capacity;
    }

    bool Map(uintptr_t address, std::vector<std::pair<std::string, uintptr_t>>& entries, size_t expected = 0) const
    {
        float marker = 0.0f;
        uintptr_t head = 0;
        size_t count = 0;
        if (!Get(address, marker) || marker != 1.0f || !Get(address + 8, head) || !Get(address + 16, count))
            return false;
        if (!count || count > 1024 || (expected && count != expected) || !head) return false;
        uintptr_t node = 0, prior = head;
        if (!Get(head, node)) return false;
        std::set<uintptr_t> seen;
        std::set<std::string> keys;
        while (node != head)
        {
            if (entries.size() >= count || seen.count(node)) return false;
            seen.insert(node);
            uintptr_t back = 0;
            if (!Get(node + 8, back) || back != prior) return false;
            std::string key;
            if (!String(node + 16, key) || !keys.insert(key).second) return false;
            entries.emplace_back(std::move(key), node);
            prior = node;
            if (!Get(node, node)) return false;
        }
        uintptr_t tail = 0;
        return entries.size() == count && Get(head + 8, tail) && tail == prior;
    }

private:
    uintptr_t base_ = 0;
};

std::string Escape(const std::string& text)
{
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20)
            {
                char buffer[8]{};
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                out += buffer;
            }
            else
            {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

std::string Narrow(const wchar_t* text)
{
    if (!text || !*text) return {};
    const int count = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<size_t>(count > 0 ? count - 1 : 0), '\0');
    if (count > 1)
        WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), count, nullptr, nullptr);
    return result;
}

std::string Hex(uintptr_t value)
{
    char buffer[24]{};
    std::snprintf(buffer, sizeof(buffer), "0x%llx", static_cast<unsigned long long>(value));
    return buffer;
}

class RawFile
{
public:
    bool Open(const std::wstring& path)
    {
        file_.open(path, std::ios::binary | std::ios::trunc);
        return static_cast<bool>(file_);
    }
    uint64_t Append(const void* data, size_t size)
    {
        const uint64_t at = offset_;
        if (file_) file_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        offset_ += size;
        return at;
    }
    bool Good() const { return static_cast<bool>(file_); }
private:
    std::ofstream file_;
    uint64_t offset_ = 0;
};
} // namespace

bool DumpModelMetadata(HMODULE fgModule, const std::wstring& outputDirectory,
    bool dumpWeightBytes, size_t maxWeightBytes, std::string& error)
{
    if (!fgModule)
    {
        error = "no libxess_fg module";
        return false;
    }
    const Reader reader(fgModule);
    std::string family;
    uint32_t catalogRva = 0;
    try
    {
        const auto& layout = xess_layout::Get(fgModule, true);
        family = layout.family;
        catalogRva = layout.Rva(0x15be920);
    }
    catch (const std::exception& exception)
    {
        error = std::string("model catalog layout unsupported: ") + exception.what();
        return false;
    }
    const uintptr_t catalog = reader.Base() + catalogRva;
    std::vector<std::pair<std::string, uintptr_t>> models;
    if (!reader.Map(catalog, models, 3))
    {
        error = "model catalog shape did not match the profiled family";
        return false;
    }

    std::error_code filesystemError;
    std::filesystem::create_directories(std::filesystem::path(outputDirectory) / L"weights", filesystemError);
    RawFile raw;
    if (!raw.Open((std::filesystem::path(outputDirectory) / L"model_raw.bin").wstring()))
    {
        error = "model_raw.bin could not be opened";
        return false;
    }

    std::ostringstream json;
    json << "{\n";
    json << "  \"family\": \"" << Escape(family) << "\",\n";
    json << "  \"catalog_rva\": \"" << Hex(catalogRva) << "\",\n";
    json << "  \"catalog_address\": \"" << Hex(catalog) << "\",\n";
    json << "  \"models\": [\n";

    size_t weightBytesWritten = 0;
    bool firstModel = true;
    for (const auto& [modelKey, model] : models)
    {
        if (!firstModel) json << ",\n";
        firstModel = false;
        uint32_t modelId = 0;
        reader.Get(model + 0x50, modelId);
        std::vector<unsigned char> record(0x120);
        if (!reader.Bytes(model, record.data(), record.size())) record.assign(record.size(), 0);
        const uint64_t recordOffset = raw.Append(record.data(), record.size());

        json << "  {\n";
        json << "    \"key\": \"" << Escape(modelKey) << "\",\n";
        json << "    \"address\": \"" << Hex(model) << "\",\n";
        json << "    \"id\": " << modelId << ",\n";
        json << "    \"record\": {\"file\": \"model_raw.bin\", \"offset\": " << recordOffset
             << ", \"size\": " << record.size() << "},\n";
        json << "    \"tables\": [\n";
        bool firstTable = true;

        for (unsigned tableIndex = 0; tableIndex < 10; ++tableIndex)
        {
            uintptr_t table = 0;
            if (!reader.Get(model + 0x78 + tableIndex * 8, table)) continue;
            if (!table) continue;
            if (!firstTable) json << ",\n";
            firstTable = false;
            json << "      {\n";
            json << "        \"index\": " << tableIndex << ",\n";
            json << "        \"address\": \"" << Hex(table) << "\",\n";

            std::vector<std::pair<std::string, uintptr_t>> entries;
            const bool isMap = reader.Map(table, entries);
            if (isMap)
            {
                json << "        \"kind\": \"map\",\n";
                json << "        \"entries\": [\n";
                bool firstEntry = true;
                for (const auto& [key, node] : entries)
                {
                    if (!firstEntry) json << ",\n";
                    firstEntry = false;
                    std::string descriptorName;
                    reader.String(node + 0x30, descriptorName);
                    json << "          {\"key\": \"" << Escape(key) << "\", \"node\": \"" << Hex(node) << "\"";
                    if (!descriptorName.empty())
                        json << ", \"descriptor\": \"" << Escape(descriptorName) << "\"";

                    size_t bodySize = 0x80;
                    if (tableIndex == 0) bodySize = 0x100;
                    else if (tableIndex == 1) bodySize = 0xe0;
                    else if (tableIndex == 2) bodySize = 0xd8;
                    std::vector<unsigned char> body(bodySize);
                    if (reader.Bytes(node + 0x50, body.data(), body.size()))
                    {
                        const uint64_t offset = raw.Append(body.data(), body.size());
                        json << ", \"body\": {\"file\": \"model_raw.bin\", \"offset\": " << offset
                             << ", \"size\": " << body.size() << "}";
                    }

                    if (tableIndex == 2)
                    {
                        uintptr_t begin = 0, end = 0, capacity = 0;
                        if (reader.Vector(node + 0xe0, begin, end, capacity) && end >= begin
                            && (end - begin) == 5 * 32 && end - begin <= 4096)
                        {
                            json << ", \"inputs\": [";
                            for (unsigned i = 0; i < 5; ++i)
                            {
                                std::string input;
                                reader.String(begin + i * 32, input);
                                if (i) json << ", ";
                                json << "\"" << Escape(input) << "\"";
                            }
                            json << "]";
                        }
                    }

                    if (tableIndex >= 4 && tableIndex <= 7)
                    {
                        const size_t resourceOffset = tableIndex == 4 ? 0x98 : tableIndex == 5 ? 0x78
                            : tableIndex == 6 ? 0xa0 : 0x80;
                        uintptr_t begin = 0, end = 0, capacity = 0;
                        if (reader.Vector(node + resourceOffset, begin, end, capacity)
                            && end >= begin && (end - begin) % 64 == 0 && (end - begin) <= 128 * 64)
                        {
                            json << ", \"resources\": [";
                            const size_t count = static_cast<size_t>(end - begin) / 64;
                            for (size_t r = 0; r < count; ++r)
                            {
                                const uintptr_t resource = begin + r * 64;
                                std::string name;
                                reader.String(resource, name);
                                uint32_t format = 0, type = 0;
                                unsigned char flag = 0;
                                reader.Get(resource + 32, format);
                                reader.Get(resource + 36, type);
                                reader.Get(resource + 56, flag);
                                std::vector<unsigned char> bytes(64);
                                reader.Bytes(resource, bytes.data(), bytes.size());
                                const uint64_t offset = raw.Append(bytes.data(), bytes.size());
                                if (r) json << ", ";
                                json << "{\"name\": \"" << Escape(name) << "\", \"format\": " << format
                                     << ", \"type\": " << type << ", \"flag\": " << int(flag)
                                     << ", \"raw\": {\"file\": \"model_raw.bin\", \"offset\": " << offset
                                     << ", \"size\": 64}}";
                            }
                            json << "]";
                        }
                    }

                    if (tableIndex == 0 && dumpWeightBytes && weightBytesWritten < maxWeightBytes)
                    {
                        // Candidate weight storage: vector-like begin/end/capacity
                        // triples inside the entry body. Semantics are unknown.
                        json << ", \"candidates\": [";
                        bool firstCandidate = true;
                        std::set<uintptr_t> seen;
                        for (size_t offset = 0; offset + 24 <= body.size(); offset += 8)
                        {
                            uintptr_t begin = 0, end = 0, capacity = 0;
                            std::memcpy(&begin, body.data() + offset, 8);
                            std::memcpy(&end, body.data() + offset + 8, 8);
                            std::memcpy(&capacity, body.data() + offset + 16, 8);
                            if (!begin || begin >= end || end > capacity || capacity - begin > (16u << 20))
                                continue;
                            const size_t size = static_cast<size_t>(end - begin);
                            if (size > (8u << 20) || !seen.insert(begin).second) continue;
                            if (weightBytesWritten + size > maxWeightBytes) break;
                            std::vector<unsigned char> bytes(size);
                            if (!reader.Bytes(begin, bytes.data(), bytes.size())) continue;
                            wchar_t name[160]{};
                            swprintf_s(name, L"%s_%u_%zu.bin", std::wstring(modelKey.begin(), modelKey.end()).c_str(),
                                tableIndex, offset);
                            const auto path = std::filesystem::path(outputDirectory) / L"weights" / name;
                            std::ofstream candidate(path, std::ios::binary | std::ios::trunc);
                            if (!candidate) continue;
                            candidate.write(reinterpret_cast<const char*>(bytes.data()),
                                static_cast<std::streamsize>(bytes.size()));
                            if (!firstCandidate) json << ", ";
                            firstCandidate = false;
                            json << "{\"offset\": " << offset << ", \"address\": \"" << Hex(begin)
                                 << "\", \"size\": " << size << ", \"file\": \"weights/"
                                 << Escape(Narrow(name)) << "\"}";
                            weightBytesWritten += size;
                        }
                        json << "]";
                    }
                    json << "}";
                }
                json << "\n        ]";
            }
            else
            {
                uintptr_t begin = 0, end = 0, capacity = 0;
                if (reader.Vector(table, begin, end, capacity) && begin <= end)
                {
                    const size_t count = tableIndex == 9 ? static_cast<size_t>(end - begin) / 48
                        : static_cast<size_t>(end - begin);
                    json << "        \"kind\": \"vector\",\n";
                    json << "        \"begin\": \"" << Hex(begin) << "\",\n";
                    json << "        \"bytes\": " << (end - begin) << ",\n";
                    json << "        \"elements\": " << count << ",\n";
                    if (tableIndex == 9)
                    {
                        json << "        \"graph\": [";
                        const size_t nodes = static_cast<size_t>(end - begin) / 48;
                        for (size_t i = 0; i < nodes; ++i)
                        {
                            const uintptr_t node = begin + i * 48;
                            uint32_t op = 0;
                            std::string key;
                            unsigned char flag = 0;
                            reader.Get(node, op);
                            reader.String(node + 8, key);
                            reader.Get(node + 40, flag);
                            if (i) json << ", ";
                            json << "{\"op\": " << op << ", \"key\": \"" << Escape(key)
                                 << "\", \"flag\": " << (flag ? "true" : "false") << "}";
                        }
                        json << "]";
                    }
                    else
                    {
                        std::vector<unsigned char> bytes(static_cast<size_t>(end - begin));
                        if (reader.Bytes(begin, bytes.data(), bytes.size()))
                        {
                            const uint64_t offset = raw.Append(bytes.data(), bytes.size());
                            json << "        \"raw\": {\"file\": \"model_raw.bin\", \"offset\": " << offset
                                 << ", \"size\": " << bytes.size() << "}";
                        }
                    }
                }
                else
                {
                    json << "        \"kind\": \"unsupported\"";
                }
            }
            json << "\n      }";
        }
        json << "\n    ]\n";
        json << "  }";
    }
    json << "\n  ]\n";
    json << "}\n";

    std::ofstream manifest(std::filesystem::path(outputDirectory) / L"model_metadata.json",
        std::ios::binary | std::ios::trunc);
    if (!manifest)
    {
        error = "model_metadata.json could not be written";
        return false;
    }
    manifest << json.str();
    if (!raw.Good()) error = "model_raw.bin was not written completely";
    return true;
}
} // namespace oracle
