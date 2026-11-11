#pragma once
// Small, self-contained JSON/raw-file writers shared by the render-validation
// native harness.  These are deliberately independent of the older capture
// probe so this tool can be built and tested without its relay/trace code.
#include "sha256_util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace render_validation
{
inline std::string JsonEscape(std::string_view text)
{
    std::string result;
    result.reserve(text.size() + 8);
    for (const char raw : text)
    {
        const auto byte = static_cast<unsigned char>(raw);
        switch (byte)
        {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (byte < 0x20)
            {
                constexpr char digits[] = "0123456789abcdef";
                result += "\\u00";
                result += digits[byte >> 4];
                result += digits[byte & 0x0f];
            }
            else
            {
                result += raw;
            }
            break;
        }
    }
    return result;
}

inline std::string JsonFloat(float value)
{
    if (!std::isfinite(value)) return "null";
    std::ostringstream text;
    text << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return text.str();
}

inline bool WriteRawFile(const std::filesystem::path& path, const void* data, size_t bytes,
    std::string& error)
{
    if (!data && bytes)
    {
        error = "null raw data for " + path.string();
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not open " + path.string();
        return false;
    }
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    file.close();
    if (!file)
    {
        error = "could not write " + path.string();
        return false;
    }
    return true;
}

inline bool WriteRawFile(const std::filesystem::path& path, const std::vector<uint8_t>& bytes,
    std::string& error)
{
    return WriteRawFile(path, bytes.data(), bytes.size(), error);
}

inline std::string Sha256File(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    sha256_util::Hasher hasher;
    std::vector<char> buffer(1u << 20);
    while (file)
    {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = static_cast<size_t>(file.gcount());
        if (count && !hasher.Update(buffer.data(), count)) return {};
    }
    sha256_util::Digest digest{};
    if (!hasher.Finish(digest)) return {};
    return sha256_util::LowerHex(digest);
}

// Same schema as the capture probe: xefg.application_inputs/1.  Keep this
// unchanged; replay.py and validate.py both consume it.
inline bool WriteApplicationInputsManifest(const std::filesystem::path& directory, uint32_t width,
    uint32_t height, uint32_t motionWidth, uint32_t motionHeight, uint32_t warmup,
    uint32_t measuredFrames, std::string& error)
{
    const uint32_t totalFrames = warmup + measuredFrames;
    const std::filesystem::path path = directory / L"manifest.json";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write application input manifest " + path.string();
        return false;
    }
    file << "{\n";
    file << "  \"schema\": \"xefg.application_inputs/1\",\n";
    file << "  \"width\": " << width << ",\n";
    file << "  \"height\": " << height << ",\n";
    file << "  \"motion_width\": " << motionWidth << ",\n";
    file << "  \"motion_height\": " << motionHeight << ",\n";
    file << "  \"frame_count\": " << totalFrames << ",\n";
    file << "  \"first_frame\": 0,\n";
    file << "  \"last_frame\": " << (totalFrames ? totalFrames - 1u : 0u) << ",\n";
    file << "  \"includes_warmup\": true,\n";
    file << "  \"warmup\": " << warmup << ",\n";
    file << "  \"traced_frames\": " << measuredFrames << ",\n";
    file << "  \"frame_semantics\": \"index is the application frame including warmup; "
            "color is the hudless color, backbuffer is the premultiplied UI over it\",\n";
    file << "  \"formats\": {\"color\": \"BGRA8\", \"ui\": \"BGRA8\", \"backbuffer\": \"BGRA8\", "
            "\"depth\": \"R32_FLOAT\", \"motion\": \"R16G16_FLOAT\"},\n";
    file << "  \"frames\": [\n";
    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        const auto name = [frame](const char* prefix, const char* extension) {
            char text[64]{};
            std::snprintf(text, sizeof(text), "%s_%05u.%s", prefix, frame, extension);
            return std::string(text);
        };
        file << "    {\"index\": " << frame << ", \"inputs\": {"
             << "\"color\": \"" << name("color", "bgra") << "\", "
             << "\"ui\": \"" << name("ui", "bgra") << "\", "
             << "\"backbuffer\": \"" << name("backbuffer", "bgra") << "\", "
             << "\"depth\": \"" << name("depth", "r32f") << "\", "
             << "\"motion\": \"" << name("mv", "rg16f") << "\"}}"
             << (frame + 1u < totalFrames ? "," : "") << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    file.flush();
    if (!file.good())
    {
        error = "could not write application input manifest " + path.string();
        return false;
    }
    return true;
}

struct SceneObjectRecord
{
    std::string name;
    std::array<float, 16> world{};
    std::array<float, 16> previousWorld{};
};

struct SceneFrameRecord
{
    uint32_t index = 0;
    bool hasPrevious = false;
    uint32_t previousFrame = 0;
    bool resetHistory = false;
    std::string midpoint;
    float time = 0.0f;
    float previousTime = 0.0f;

    // depth, motion, color, ui, constants, present_id
    std::array<int32_t, 6> submissionResults{};

    std::array<float, 16> view{};
    std::array<float, 16> projection{};
    std::vector<SceneObjectRecord> objects;
};

struct SceneManifest
{
    std::string name;
    uint32_t revision = 1;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t motionWidth = 0;
    uint32_t motionHeight = 0;
    uint32_t warmup = 0;
    uint32_t measuredFrames = 0;
    std::string shaderSource;
    std::string shaderSha256;
    std::string rendererVersion;
    std::vector<SceneFrameRecord> frames;
};

inline void WriteFloatArray(std::ostream& stream, const std::array<float, 16>& values)
{
    stream << '[';
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i) stream << ", ";
        stream << JsonFloat(values[i]);
    }
    stream << ']';
}

inline bool WriteSceneManifest(const std::filesystem::path& path, const SceneManifest& manifest,
    std::string& error)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write scene manifest " + path.string();
        return false;
    }
    file << "{\n";
    file << "  \"schema\": \"xefg.render_scene/1\",\n";
    file << "  \"name\": \"" << JsonEscape(manifest.name) << "\",\n";
    file << "  \"revision\": " << manifest.revision << ",\n";
    file << "  \"width\": " << manifest.width << ",\n";
    file << "  \"height\": " << manifest.height << ",\n";
    file << "  \"motion_width\": " << manifest.motionWidth << ",\n";
    file << "  \"motion_height\": " << manifest.motionHeight << ",\n";
    file << "  \"warmup\": " << manifest.warmup << ",\n";
    file << "  \"measured_frames\": " << manifest.measuredFrames << ",\n";
    file << "  \"marker\": {\"x\": 96, \"y\": 32, \"cell_width\": 8, \"cell_height\": 8, "
            "\"bits\": 32, \"encoding\": \"u16-complement-lsb\", \"threshold\": 128},\n";
    file << "  \"excluded_rects\": [[0, 0, 64, 64], [576, 0, 640, 64], [0, 296, 64, 360], "
            "[576, 296, 640, 360], [48, 0, 64, 360], [576, 0, 592, 360], [96, 32, 352, 40]],\n";
    file << "  \"depth_encoding\": \"device-normal-z\",\n";
    file << "  \"color_space\": \"linear-unorm\",\n";
    file << "  \"color_space_note\": \"no sRGB\",\n";
    file << "  \"motion_encoding\": \"current_to_previous_mv_pixels\",\n";
    file << "  \"renderer\": {\"version\": \"" << JsonEscape(manifest.rendererVersion)
         << "\", \"shader_source\": \"" << JsonEscape(manifest.shaderSource)
         << "\", \"shader_sha256\": \"" << manifest.shaderSha256 << "\"},\n";
    file << "  \"frames\": [\n";
    for (size_t i = 0; i < manifest.frames.size(); ++i)
    {
        const auto& frame = manifest.frames[i];
        file << "    {\"index\": " << frame.index << ", \"previous_frame\": ";
        if (frame.hasPrevious) file << frame.previousFrame;
        else file << "null";
        file << ", \"reset_history\": " << (frame.resetHistory ? "true" : "false");
        file << ", \"midpoint\": \"" << JsonEscape(frame.midpoint) << "\"";
        file << ", \"time\": " << JsonFloat(frame.time)
             << ", \"previous_time\": " << JsonFloat(frame.previousTime);
        file << ", \"submission_results\": {\"depth\": " << frame.submissionResults[0]
             << ", \"motion\": " << frame.submissionResults[1]
             << ", \"color\": " << frame.submissionResults[2]
             << ", \"ui\": " << frame.submissionResults[3]
             << ", \"constants\": " << frame.submissionResults[4]
             << ", \"present_id\": " << frame.submissionResults[5] << "}";
        file << ", \"view\": ";
        WriteFloatArray(file, frame.view);
        file << ", \"projection\": ";
        WriteFloatArray(file, frame.projection);
        file << ", \"objects\": [";
        for (size_t objectIndex = 0; objectIndex < frame.objects.size(); ++objectIndex)
        {
            const auto& object = frame.objects[objectIndex];
            if (objectIndex) file << ", ";
            file << "{\"name\": \"" << JsonEscape(object.name) << "\", \"world\": ";
            WriteFloatArray(file, object.world);
            file << ", \"previous_world\": ";
            WriteFloatArray(file, object.previousWorld);
            file << "}";
        }
        file << "]";
        file << "}" << (i + 1 < manifest.frames.size() ? "," : "") << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    file.flush();
    if (!file.good())
    {
        error = "could not write scene manifest " + path.string();
        return false;
    }
    return true;
}

struct PresentStatusRecord
{
    uint32_t applicationFrame = 0;
    int32_t presentResult = 0;
    int32_t statusResult = 0;
    uint32_t framesPresented = 0;
    int32_t frameGenerationResult = 0;
    int32_t isFrameGenEnabled = 0;
    uint64_t presentStartQpc = 0;
    uint64_t presentEndQpc = 0;
};

struct PresentCaptureRecord
{
    uint32_t sequence = 0;
    uint32_t observedApplicationFrame = 0;
    uint32_t ordinal = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t format = 0;
    std::string file;
    uint64_t presentStartQpc = 0;
    uint64_t presentEndQpc = 0;
    uint32_t dxgiPresentCount = 0;
    int32_t dxgiPresentCountResult = 0;
    int32_t nativePresentResult = 0;
};

struct PresentsManifest
{
    bool onlyInterpolationRequested = false;
    int32_t onlyInterpolationResult = 0;
    bool tagInterpolatedRequested = false;
    int32_t tagInterpolatedResult = 0;
    int64_t qpcFrequency = 0;
    std::string captureError;
    std::vector<PresentStatusRecord> statuses;
    std::vector<PresentCaptureRecord> presents;
};

inline bool WritePresentsManifest(const std::filesystem::path& path,
    const PresentsManifest& manifest, std::string& error)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        error = "could not write present manifest " + path.string();
        return false;
    }
    file << "{\n  \"schema\": \"xefg.present_capture/2\",\n"
         << "  \"only_interpolation_requested\": "
         << (manifest.onlyInterpolationRequested ? "true" : "false") << ",\n"
         << "  \"only_interpolation_result\": " << manifest.onlyInterpolationResult << ",\n"
         << "  \"tag_interpolated_requested\": "
         << (manifest.tagInterpolatedRequested ? "true" : "false") << ",\n"
         << "  \"tag_interpolated_result\": " << manifest.tagInterpolatedResult << ",\n"
         << "  \"qpc_frequency\": " << manifest.qpcFrequency << ",\n"
         << "  \"capture_error\": \"" << JsonEscape(manifest.captureError) << "\",\n"
         << "  \"frame_status\": [\n";
    for (size_t i = 0; i < manifest.statuses.size(); ++i)
    {
        const auto& status = manifest.statuses[i];
        file << "    {\"application_frame\": " << status.applicationFrame
             << ", \"present_result\": " << status.presentResult
             << ", \"status_result\": " << status.statusResult
             << ", \"frames_presented\": " << status.framesPresented
             << ", \"frame_generation_result\": " << status.frameGenerationResult
             << ", \"is_frame_gen_enabled\": " << status.isFrameGenEnabled
             << ", \"present_start_qpc\": " << status.presentStartQpc
             << ", \"present_end_qpc\": " << status.presentEndQpc << "}"
             << (i + 1 < manifest.statuses.size() ? "," : "") << '\n';
    }
    file << "  ],\n  \"presents\": [\n";
    for (size_t i = 0; i < manifest.presents.size(); ++i)
    {
        const auto& present = manifest.presents[i];
        file << "    {\"sequence\": " << present.sequence
             << ", \"observed_application_frame\": " << present.observedApplicationFrame
             << ", \"ordinal\": " << present.ordinal
             << ", \"width\": " << present.width << ", \"height\": " << present.height
             << ", \"format\": " << present.format
             << ", \"file\": \"" << JsonEscape(present.file) << "\""
             << ", \"present_start_qpc\": " << present.presentStartQpc
             << ", \"present_end_qpc\": " << present.presentEndQpc
             << ", \"dxgi_present_count\": " << present.dxgiPresentCount
             << ", \"dxgi_present_count_result\": " << present.dxgiPresentCountResult
             << ", \"native_present_result\": " << present.nativePresentResult << "}"
             << (i + 1 < manifest.presents.size() ? "," : "") << '\n';
    }
    file << "  ]\n}\n";
    file.close();
    if (!file)
    {
        error = "could not write present manifest " + path.string();
        return false;
    }
    return true;
}
} // namespace render_validation
