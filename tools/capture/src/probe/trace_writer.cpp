#include "trace_writer.h"

#include <cstdio>

namespace probe
{
std::string JsonEscape(const std::string& text)
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
            else out += static_cast<char>(c);
        }
    }
    return out;
}

std::string HexPointer(const void* pointer)
{
    char buffer[24]{};
    std::snprintf(buffer, sizeof(buffer), "0x%llx", static_cast<unsigned long long>(
        reinterpret_cast<uintptr_t>(pointer)));
    return buffer;
}

bool TraceWriter::Open(const std::filesystem::path& directory, std::string& error)
{
    std::error_code ignored;
    std::filesystem::create_directories(directory, ignored);
    events_.open(directory / L"trace.jsonl", std::ios::binary | std::ios::trunc);
    meta_.open(directory / L"meta.json", std::ios::binary | std::ios::trunc);
    if (!events_ || !meta_)
    {
        error = "trace files could not be opened";
        return false;
    }
    open_ = true;
    return true;
}

void TraceWriter::Close()
{
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = false;
    if (events_) events_.flush();
    if (meta_) meta_.flush();
}

void TraceWriter::Event(const std::string& body)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_) return;
    events_ << "{\"seq\":" << ++sequence_ << ',' << body << "}\n";
}

void TraceWriter::SetMeta(const std::string& json)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!meta_) return;
    meta_.seekp(0);
    meta_ << json;
    meta_.flush();
}

void TraceWriter::Flush()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_) events_.flush();
}

uint64_t TraceWriter::Sequence() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return sequence_;
}
} // namespace probe
