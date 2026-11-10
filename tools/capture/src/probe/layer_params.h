#pragma once
// Pipeline-builder parameter capture.
//
// A port of the research probe in source/native/xess/port/shader_port.h
// (XESS_INIT_PROBE): while the provider is inside its pipeline builder, the
// builder frame holds the layer's parameter vector and the conv-block descriptor
// (weights / bias / activation spans plus the int8 quantization metadata).
// Everything is read-only and every failure is contained.
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>

namespace probe
{
class TraceWriter;

class LayerParameters
{
public:
    LayerParameters(uintptr_t moduleBase, uintptr_t imageEnd, TraceWriter& writer,
        std::filesystem::path directory);

    // Writes a "layer_parameters" event when the current stack carries a conv
    // block. Returns false silently for every other layer family.
    bool Capture(const std::string& key);

    size_t Count() const;

private:
    uintptr_t base_ = 0;
    uintptr_t imageEnd_ = 0;
    TraceWriter* writer_ = nullptr;
    std::filesystem::path directory_;
    mutable std::mutex mutex_;
    size_t captures_ = 0;
};
} // namespace probe
