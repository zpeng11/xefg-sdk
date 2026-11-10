#pragma once
// Line-delimited JSON trace sink. Every event carries a global sequence number
// and both the application frame and the native-present ordinal it belongs to,
// so a trace can be joined with the oracle's captures without trusting timing.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace probe
{
std::string JsonEscape(const std::string& text);

class TraceWriter
{
public:
    bool Open(const std::filesystem::path& directory, std::string& error);
    void Close();

    // body is a JSON object body after "seq": e.g. "type":"dispatch",...
    void Event(const std::string& body);
    // Written to meta.json (one JSON object assembled by the caller).
    void SetMeta(const std::string& json);
    void Flush();

    uint64_t Sequence() const;

private:
    mutable std::mutex mutex_;
    std::ofstream events_;
    std::ofstream meta_;
    uint64_t sequence_ = 0;
    bool open_ = false;
};

std::string HexPointer(const void* pointer);
} // namespace probe
