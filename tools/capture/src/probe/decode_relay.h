#pragma once
// Read-only relay in front of the provider's program decoder.
//
// The same call site and relay technique as source/native/xess/port/shader_port.h,
// but this one never supplies or replaces a program: it lets the provider decode,
// then records the decoded DXIL bytes and (for the conv layers) the pipeline
// builder's weight/bias tensors and quantization metadata.
#include <Windows.h>

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace probe
{
class LayerParameters;
class TraceWriter;

class DecodeRelay
{
public:
    DecodeRelay();
    ~DecodeRelay();

    // Fails unless the loaded module is a profiled libxess_fg and the decoder
    // call site, its target and the decoder prologue all match.
    bool Install(HMODULE fgModule, TraceWriter& writer, const std::filesystem::path& shaderDirectory,
        const std::filesystem::path& parameterDirectory, std::string& error);
    void Restore();

    // Program key for a DXIL SHA-256, or an empty string.
    std::string KeyForHash(const std::string& sha256) const;
    size_t ProgramCount() const;
    size_t ParameterCaptureCount() const;

    // Called by the relay stub.
    void Relay(void* store, const void* keyObject, void* output);

private:
    bool RegisterProgram(const std::string& key, const std::string& sha256);

    uintptr_t base_ = 0;
    uintptr_t imageEnd_ = 0;
    unsigned char* site_ = nullptr;
    void* relay_ = nullptr;
    LONG original_ = 0;
    LONG replacement_ = 0;
    DWORD protection_ = 0;
    void (*decode_)(void*, const void*, void*) = nullptr;
    bool active_ = false;

    TraceWriter* writer_ = nullptr;
    std::filesystem::path shaderDirectory_;
    std::unique_ptr<LayerParameters> parameters_;

    mutable std::mutex mutex_;
    std::map<std::string, std::string> hashToKey_;
    std::map<std::string, std::string> keyToHash_;
};
} // namespace probe
