// WARP render-only tests for the render-validation DX12 renderer.
// These do not create an XeSS-FG context and do not run frame generation.

#include "manifest.h"
#include "renderer.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using namespace render_validation;

namespace
{
int g_failures = 0;

void Check(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::printf("FAIL: %s\n", message.c_str());
        ++g_failures;
    }
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 31)
    {
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::filesystem::path FindShader(int argc, char** argv)
{
    if (argc > 1)
    {
        std::filesystem::path candidate = argv[1];
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(std::filesystem::path(__FILE__).parent_path().parent_path()
        / "shaders" / "scene.hlsl");
    candidates.push_back(std::filesystem::current_path() / "shaders" / "scene.hlsl");
    candidates.push_back(std::filesystem::current_path() / ".." / "shaders" / "scene.hlsl");
    candidates.push_back(std::filesystem::current_path() / ".." / ".." / "shaders" / "scene.hlsl");

    wchar_t executable[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, executable, MAX_PATH))
    {
        candidates.push_back(std::filesystem::path(executable).parent_path()
            / "shaders" / "scene.hlsl");
    }

    for (const auto& candidate : candidates)
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    return {};
}

bool CreateWarpDevice(ComPtr<ID3D12Device>& device, ComPtr<ID3D12CommandQueue>& queue)
{
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
        debug->EnableDebugLayer();

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return false;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&device)))) return false;

    D3D12_COMMAND_QUEUE_DESC queueDescription{};
    queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDescription.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    return SUCCEEDED(device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&queue)));
}

bool DecodeUiMarker(const std::vector<uint8_t>& ui, uint32_t width, uint32_t expectedFrame,
    std::string& error)
{
    const size_t stride = static_cast<size_t>(width) * 4;
    uint32_t code = 0;
    for (uint32_t bit = 0; bit < 32; ++bit)
    {
        const uint32_t x = 96 + bit * 8 + 4;
        const size_t offset = static_cast<size_t>(36) * stride + static_cast<size_t>(x) * 4;
        const uint8_t b = ui[offset + 0];
        const uint8_t g = ui[offset + 1];
        const uint8_t r = ui[offset + 2];
        const uint8_t a = ui[offset + 3];
        if (a != 255 || b != g || g != r)
        {
            error = "UI marker pixel is not opaque grayscale";
            return false;
        }
        if (b > 127) code |= (1u << bit);
    }
    const uint32_t id = code & 0xffffu;
    const uint32_t complement = code >> 16;
    if (id != expectedFrame || complement != (expectedFrame ^ 0xffffu))
    {
        char text[128]{};
        std::snprintf(text, sizeof(text),
            "UI marker decoded id=%u complement=0x%04x, expected id=%u",
            id, complement, expectedFrame);
        error = text;
        return false;
    }
    return true;
}

struct DepthStats
{
    bool finite = true;
    bool inRange = true;
    float minimum = 1.0f;
    float maximum = 0.0f;
};

DepthStats ComputeDepthStats(const std::vector<uint8_t>& depth)
{
    DepthStats stats;
    const size_t count = depth.size() / sizeof(float);
    const float* values = reinterpret_cast<const float*>(depth.data());
    for (size_t i = 0; i < count; ++i)
    {
        const float value = values[i];
        if (!std::isfinite(value)) { stats.finite = false; continue; }
        if (value < 0.0f || value > 1.0001f) stats.inRange = false;
        stats.minimum = std::min(stats.minimum, value);
        stats.maximum = std::max(stats.maximum, value);
    }
    return stats;
}

struct MotionStats
{
    bool finite = true;
    double nonZeroPixels = 0;
    double sumX = 0.0;
    double maxMagnitude = 0.0;
};

MotionStats ComputeMotionStats(const std::vector<uint8_t>& motion)
{
    MotionStats stats;
    const size_t count = motion.size() / 4;
    const uint16_t* values = reinterpret_cast<const uint16_t*>(motion.data());
    for (size_t i = 0; i < count; ++i)
    {
        const float x = HalfToFloat(values[i * 2 + 0]);
        const float y = HalfToFloat(values[i * 2 + 1]);
        if (!std::isfinite(x) || !std::isfinite(y)) { stats.finite = false; continue; }
        const double magnitude = std::sqrt(double(x) * x + double(y) * y);
        stats.maxMagnitude = std::max(stats.maxMagnitude, magnitude);
        if (magnitude > 0.01)
        {
            stats.nonZeroPixels += 1.0;
            stats.sumX += x;
        }
    }
    return stats;
}

struct BgraPixel
{
    uint8_t b = 0;
    uint8_t g = 0;
    uint8_t r = 0;
    uint8_t a = 0;
};

BgraPixel UiPixel(const std::vector<uint8_t>& ui, uint32_t width, uint32_t x, uint32_t y)
{
    const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
    return {ui[offset + 0], ui[offset + 1], ui[offset + 2], ui[offset + 3]};
}

// Mirrors the shader's per-frame crosshair offset, which stays a fixed number
// of pixels at every output size.
void CrosshairCenter(uint32_t frame, uint32_t width, uint32_t height, uint32_t& x, uint32_t& y)
{
    const int offsetX = (static_cast<int>(frame) % 7 - 3) * 3;
    const int offsetY = ((static_cast<int>(frame) / 7) % 5 - 2) * 2;
    x = static_cast<uint32_t>(static_cast<int>(width / 2) + offsetX);
    y = static_cast<uint32_t>(static_cast<int>(height / 2) + offsetY);
}

void TestRenderSizeTypes()
{
    const RenderSize defaultSize;
    Check(defaultSize.width == 640 && defaultSize.height == 360,
        "default RenderSize is 640x360");
    Check(defaultSize.GeometryWidth() == 320 && defaultSize.GeometryHeight() == 180,
        "default geometry is half size");

    const RenderSize hd{1280, 720};
    const RenderSize fhd{1920, 1080};
    const RenderSize qhd{2560, 1440};
    const RenderSize uhd{3840, 2160};
    Check(hd.GeometryWidth() == 640 && hd.GeometryHeight() == 360, "720p geometry is half size");
    Check(fhd.GeometryWidth() == 960 && fhd.GeometryHeight() == 540, "1080p geometry is half size");
    Check(qhd.GeometryWidth() == 1280 && qhd.GeometryHeight() == 720, "1440p geometry is half size");
    Check(uhd.GeometryWidth() == 1920 && uhd.GeometryHeight() == 1080, "4K geometry is half size");
}

void TestNewSceneNamesAndStates()
{
    Check(kSceneCount == 9, "nine scenes are registered");
    Check(static_cast<uint32_t>(SceneKind::Translation) == 0
        && static_cast<uint32_t>(SceneKind::Occlusion) == 1
        && static_cast<uint32_t>(SceneKind::Camera) == 2
        && static_cast<uint32_t>(SceneKind::Cut) == 3,
        "original four scene values are unchanged");
    Check(static_cast<uint32_t>(SceneKind::Nonrigid) == 7
        && static_cast<uint32_t>(SceneKind::Transparency) == 8,
        "opt-in scene values are appended");
    const char* names[5] = {"depth_layers", "screen_cross", "camera_turn",
        "nonrigid", "transparency"};
    for (size_t i = 0; i < 5; ++i)
    {
        const SceneKind scene = static_cast<SceneKind>(4 + i);
        SceneKind parsed = SceneKind::Translation;
        Check(IsKnownScene(names[i]), std::string("known scene ") + names[i]);
        Check(SceneFromName(names[i], parsed) && parsed == scene,
            std::string("scene name maps to enum ") + names[i]);

        const FrameState state = MakeFrameState(scene, 5, false);
        Check(state.objectCount > 0 && state.objectCount <= kMaxObjects,
            std::string(names[i]) + " uses a separate bounded object list");
        for (uint32_t object = 0; object < state.objectCount; ++object)
            Check(state.objects[object].name != "ground"
                && state.objects[object].name.rfind("rod", 0) != 0,
                std::string(names[i]) + " does not reuse the old object list");
    }
}

void TestNonrigidContract()
{
    const FrameState state = MakeFrameState(SceneKind::Nonrigid, 5, false);
    Check(state.objectCount == 2, "nonrigid has the grid and one static background");
    Check(state.objects[0].mesh == MeshKind::Grid
        && state.objects[0].name == "deform_panel",
        "nonrigid first object is the subdivided grid");
    Check(state.objects[1].name == "static_backdrop" && !state.objects[1].transparent,
        "nonrigid background is static opaque geometry");
    const Matrix4& world = state.objects[0].world;
    Check(std::abs(world.m[0] - kNonrigidScaleX) < 1e-5f
        && std::abs(world.m[5] - kNonrigidScaleY) < 1e-5f
        && std::abs(world.m[14] - kNonrigidTranslateZ) < 1e-5f,
        "nonrigid grid world transform matches the contract");
    Check(std::memcmp(state.objects[0].world.m, state.objects[0].previousWorld.m,
        sizeof(state.objects[0].world.m)) == 0,
        "nonrigid grid world transform is fixed");

    float current[3], previous[3], resetPosition[3];
    NonrigidWorldPosition(0.35f, -0.25f, state.time, current);
    NonrigidWorldPosition(0.35f, -0.25f, state.previousTime, previous);
    NonrigidWorldPosition(0.35f, -0.25f, state.time, resetPosition);
    const float dx = current[0] - previous[0];
    const float dy = current[1] - previous[1];
    const float dz = current[2] - previous[2];
    const float previousDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    Check(previousDistance > 0.05f,
        "nonrigid previous deformation is a real vertex position change");
    Check(resetPosition[2] == current[2],
        "nonrigid same-time position is stable");

    const FrameState reset = MakeFrameState(SceneKind::Nonrigid, 0, false);
    Check(reset.previousTime == reset.time, "nonrigid reset frame previous time equals current");

    const FrameState midpoint = MakeFrameState(SceneKind::Nonrigid, 4, true);
    Check(std::abs(midpoint.time - 3.5f) < 1e-6f
        && std::abs(midpoint.previousTime - 3.0f) < 1e-6f,
        "nonrigid midpoint preserves MakeFrameState time semantics");
    float midpointPosition[3];
    NonrigidWorldPosition(0.35f, -0.25f, midpoint.time, midpointPosition);
    Check(std::abs(midpointPosition[2] - current[2]) > 1e-3f,
        "nonrigid midpoint samples the continuous deformation");

    float normalA[3], normalB[3];
    NonrigidNormalLocal(0.35f, -0.25f, 4.0f, normalA);
    NonrigidNormalLocal(0.35f, -0.25f, 5.0f, normalB);
    const float length = std::sqrt(normalA[0] * normalA[0] + normalA[1] * normalA[1]
        + normalA[2] * normalA[2]);
    Check(std::abs(length - 1.0f) < 1e-4f && normalA[2] > 0.0f,
        "nonrigid analytic normal is unit length and faces the camera");
    Check(std::abs(normalA[0] - normalB[0]) + std::abs(normalA[1] - normalB[1])
        + std::abs(normalA[2] - normalB[2]) > 1e-3f,
        "nonrigid normals follow the deformation");
}

void TestTransparencyContract()
{
    const FrameState state = MakeFrameState(SceneKind::Transparency, 5, false);
    Check(state.objectCount == 4, "transparency has backdrop, blocker and two planes");
    const char* expected[4] = {"backdrop", "blocker", "plane_far", "plane_near"};
    for (uint32_t i = 0; i < 4; ++i)
        Check(state.objects[i].name == expected[i], "transparency object order");
    Check(!state.objects[0].transparent && !state.objects[1].transparent
        && state.objects[2].transparent && state.objects[3].transparent,
        "transparency draws opaque objects before transparent objects");
    Check(state.objects[2].sortIndex == 2 && state.objects[3].sortIndex == 3
        && state.objects[2].mesh == MeshKind::Panel,
        "transparent planes are sorted far to near");
    Check(state.objects[0].writesDepthMotion && state.objects[1].writesDepthMotion
        && !state.objects[2].writesDepthMotion && !state.objects[3].writesDepthMotion,
        "transparency depth/motion is opaque-only");
    Check(state.objects[0].depthWrite && state.objects[1].depthWrite
        && !state.objects[2].depthWrite && !state.objects[3].depthWrite,
        "transparent depth writes are disabled");
    Check(std::abs(state.objects[2].color[3] - 0.55f) < 1e-6f
        && std::abs(state.objects[3].color[3] - 0.45f) < 1e-6f,
        "transparent material alphas match the contract");

    for (int i = 0; i < 2; ++i)
        Check(std::memcmp(state.objects[i].world.m, state.objects[i].previousWorld.m,
            sizeof(state.objects[i].world.m)) == 0,
            "opaque transparency geometry is static");
    Check(std::memcmp(state.objects[2].world.m, state.objects[2].previousWorld.m,
        sizeof(state.objects[2].world.m)) != 0,
        "far transparent plane moves independently");
    Check(std::memcmp(state.objects[3].world.m, state.objects[3].previousWorld.m,
        sizeof(state.objects[3].world.m)) != 0,
        "near transparent plane moves independently");

    const FrameState reset = MakeFrameState(SceneKind::Transparency, 0, false);
    for (uint32_t i = 0; i < reset.objectCount; ++i)
        Check(std::memcmp(reset.objects[i].world.m, reset.objects[i].previousWorld.m,
            sizeof(reset.objects[i].world.m)) == 0,
            "transparency reset previous transforms equal current");

    const FrameState midpoint = MakeFrameState(SceneKind::Transparency, 4, true);
    Check(std::abs(midpoint.time - 3.5f) < 1e-6f
        && std::abs(midpoint.previousTime - 3.0f) < 1e-6f,
        "transparency midpoint preserves MakeFrameState time semantics");
    const float expectedFarX = -0.15f + 0.35f * std::sin(0.50f * 3.5f);
    const float expectedNearX = 0.10f + 0.30f * std::sin(0.40f * 3.5f + 1.30f);
    Check(std::abs(midpoint.objects[2].world.m[12] - expectedFarX) < 1e-5f
        && std::abs(midpoint.objects[3].world.m[12] - expectedNearX) < 1e-5f,
        "transparency plane motion uses the contract formulas at midpoint");
}

void TestMatrixConventions()
{
    const auto transform = [](const float point[4], const Matrix4& matrix) {
        float result[4]{};
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                result[column] += point[row] * matrix.m[row * 4 + column];
        return std::array<float, 4>{result[0], result[1], result[2], result[3]};
    };

    const float origin[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const auto translated = transform(origin, MakeTranslation(1.0f, 2.0f, 3.0f));
    Check(std::fabs(translated[0] - 1.0f) < 1e-6f &&
          std::fabs(translated[1] - 2.0f) < 1e-6f &&
          std::fabs(translated[2] - 3.0f) < 1e-6f &&
          std::fabs(translated[3] - 1.0f) < 1e-6f,
        "row-vector translation moves origin to xyz");

    const Matrix4 world = Multiply(MakeScale(2.0f, 2.0f, 2.0f),
        MakeTranslation(1.0f, 2.0f, 3.0f));
    const auto worldOrigin = transform(origin, world);
    Check(std::fabs(worldOrigin[0] - 1.0f) < 1e-6f &&
          std::fabs(worldOrigin[1] - 2.0f) < 1e-6f &&
          std::fabs(worldOrigin[2] - 3.0f) < 1e-6f &&
          std::fabs(worldOrigin[3] - 1.0f) < 1e-6f,
        "scale*translation keeps origin translation in row 3");

    const float eye[3] = {0.0f, 0.0f, 5.0f};
    const float target[3] = {0.0f, 0.0f, 0.0f};
    const float up[3] = {0.0f, 1.0f, 0.0f};
    const Matrix4 viewProjection = Multiply(MakeLookAt(eye, target, up),
        MakePerspective(50.0f * 3.14159265f / 180.0f, 16.0f / 9.0f, 0.1f, 100.0f));
    const float front[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const auto clip = transform(front, viewProjection);
    Check(clip[3] > 0.0f, "front point has positive clip w");
    Check(clip[2] / clip[3] > 0.0f && clip[2] / clip[3] < 1.0f,
        "front point depth is in normal-Z range");
}

bool TestMode1NoUi(Renderer& renderer)
{
    std::string error;
    FrameState state = MakeFrameState(SceneKind::Translation, 5, false);
    state.uiEnabled = false;
    std::vector<uint8_t> color;
    std::vector<uint8_t> ui;
    std::vector<uint8_t> backbuffer;
    if (!renderer.Render(state, SceneKind::Translation, false, error) ||
        !Readback(renderer.Color(), color, error) ||
        !Readback(renderer.Ui(), ui, error) ||
        !Readback(renderer.Backbuffer(), backbuffer, error))
    {
        std::printf("FAIL: mode1 render/readback: %s\n", error.c_str());
        return false;
    }
    Check(std::all_of(ui.begin(), ui.end(), [](uint8_t value) { return value == 0; }),
        "mode1 UI target is transparent black");
    Check(backbuffer == color, "mode1 backbuffer equals hudless color byte-for-byte");
    const std::vector<uint8_t> mode1Color = color;

    state.uiEnabled = true;
    std::vector<uint8_t> uiColor;
    std::vector<uint8_t> uiUi;
    std::vector<uint8_t> uiBackbuffer;
    if (!renderer.Render(state, SceneKind::Translation, false, error) ||
        !Readback(renderer.Color(), uiColor, error) ||
        !Readback(renderer.Ui(), uiUi, error) ||
        !Readback(renderer.Backbuffer(), uiBackbuffer, error))
    {
        std::printf("FAIL: UI-enabled render/readback: %s\n", error.c_str());
        return false;
    }
    Check(uiColor == mode1Color, "enabling UI does not change hudless color");
    Check(std::any_of(uiUi.begin(), uiUi.end(), [](uint8_t value) { return value != 0; }),
        "UI-enabled UI target is not empty");
    Check(uiBackbuffer != uiColor, "UI-enabled backbuffer differs from hudless color");
    return true;
}

void TestPairStates()
{
    Check(!IsValidPairFrame(SceneKind::Cut, kCutFrame),
        "cut pair ending at frame 7 crosses the cut and is rejected");
    Check(IsValidPairFrame(SceneKind::Cut, kCutFrame + 1),
        "cut pair ending at frame 8 is valid");
    Check(IsValidPairFrame(SceneKind::Translation, 1), "pair frame 1 is valid");
    Check(!IsValidPairFrame(SceneKind::Translation, 0), "pair frame 0 is invalid");

    const FrameState first = MakeIsolatedPairFrameState(SceneKind::Cut, kCutFrame, true, false);
    const FrameState second = MakeIsolatedPairFrameState(SceneKind::Cut, kCutFrame + 1, false, false);
    Check(first.sourceFrame == kCutFrame && first.time == float(kCutFrame),
        "first pair state keeps its source frame and time");
    Check(first.historyReset && first.previousTime == first.time,
        "first pair submission resets history");
    Check(second.sourceFrame == kCutFrame + 1 && second.time == float(kCutFrame + 1),
        "second pair state keeps its source frame and time");
    Check(!second.historyReset && second.previousTime == float(kCutFrame),
        "second pair submission follows previous source time");

    const FrameState translationFirst = MakeIsolatedPairFrameState(SceneKind::Translation, 4, true, false);
    const FrameState translationSecond = MakeIsolatedPairFrameState(SceneKind::Translation, 5, false, false);
    Check(translationFirst.historyReset && translationFirst.previousTime == 4.0f,
        "non-cut first pair submission is still reset");
    Check(translationSecond.previousTime == 4.0f && translationSecond.time == 5.0f,
        "non-cut second pair submission spans the requested source pair");
}

void TestSceneManifestSchema()
{
    SceneManifest manifest;
    manifest.name = "translation";
    manifest.mode = 1;
    manifest.pairingMethod = "isolated_pair";
    manifest.hasSourcePair = true;
    manifest.sourcePair = {4, 5};
    manifest.warmup = 0;
    manifest.measuredFrames = 2;
    SceneFrameRecord first;
    first.index = 0;
    first.sceneFrame = 4;
    first.resetHistory = true;
    first.time = 4.0f;
    first.previousTime = 4.0f;
    first.submissionResults[0] = 0;
    SceneFrameRecord second;
    second.index = 1;
    second.sceneFrame = 5;
    second.hasPrevious = true;
    second.previousFrame = 0;
    second.time = 5.0f;
    second.previousTime = 4.0f;
    second.submissionResults[0] = 0;
    second.submissionResults[1] = 0;
    manifest.frames = {first, second};

    const std::filesystem::path path = std::filesystem::temp_directory_path()
        / "xefg_render_validation_scene_test.json";
    std::string error;
    if (!WriteSceneManifest(path, manifest, error))
    {
        std::printf("FAIL: scene manifest write: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    std::ifstream file(path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::filesystem::remove(path);

    Check(text.find("\"mode\": 1") != std::string::npos, "scene manifest records mode");
    Check(text.find("\"scene_extension\"") == std::string::npos,
        "original scenes keep the optional extension absent");
    Check(text.find("\"pairing_method\": \"isolated_pair\"") != std::string::npos,
        "scene manifest records isolated pairing method");
    Check(text.find("\"source_pair\": [4, 5]") != std::string::npos,
        "scene manifest records the source pair");
    Check(text.find("\"index\": 0, \"scene_frame\": 4") != std::string::npos,
        "scene manifest keeps local index and real source frame");
    Check(text.find("\"index\": 1, \"scene_frame\": 5") != std::string::npos,
        "second scene submission keeps contiguous local index");
    Check(text.find("\"ui\": null") != std::string::npos,
        "mode 1 scene manifest emits null UI submission result");
    Check(text.find("\"color\": null") != std::string::npos,
        "mode 1 scene manifest emits null color submission result");
    Check(text.find("\"present_id\": null") != std::string::npos,
        "uncalled submission results are null");
    Check(text.find("\"marker\"") == std::string::npos,
        "mode 1 scene manifest omits the absent UI marker");
    Check(text.find("\"excluded_rects\"") == std::string::npos,
        "mode 1 scene manifest omits the absent UI exclusion strip");

    manifest.hasSourcePair = false;
    manifest.pairingMethod = "sequential";
    if (WriteSceneManifest(path, manifest, error))
    {
        std::ifstream sequentialFile(path, std::ios::binary);
        const std::string sequentialText((std::istreambuf_iterator<char>(sequentialFile)),
            std::istreambuf_iterator<char>());
        sequentialFile.close();
        std::filesystem::remove(path);
        Check(sequentialText.find("\"pairing_method\": \"sequential\"") != std::string::npos,
            "sequential scene manifest records its method");
        Check(sequentialText.find("\"source_pair\"") == std::string::npos,
            "sequential scene manifest omits source_pair");
    }
    else
    {
        Check(false, "sequential scene manifest write: " + error);
    }

    const auto readManifestText = [&]() {
        std::ifstream file(path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        std::filesystem::remove(path);
        return text;
    };

    // The calibrated 640x360 layout keeps both the fixed marker description and
    // the exclusion rectangles that Python still owns authoritatively.
    manifest.mode = 3;
    manifest.width = 640;
    manifest.height = 360;
    manifest.motionWidth = 320;
    manifest.motionHeight = 180;
    if (WriteSceneManifest(path, manifest, error))
    {
        const std::string calibrated = readManifestText();
        Check(calibrated.find("\"marker\"") != std::string::npos,
            "640 UI manifest records the fixed marker");
        Check(calibrated.find("\"excluded_rects\"") != std::string::npos,
            "640 UI manifest retains the calibrated exclusions");
    }
    else
    {
        Check(false, "640 UI scene manifest write: " + error);
    }

    // New sizes keep the same fixed marker but must not invent scaled native
    // exclusion rectangles before the official layout is calibrated.
    manifest.width = 1920;
    manifest.height = 1080;
    manifest.motionWidth = 960;
    manifest.motionHeight = 540;
    if (WriteSceneManifest(path, manifest, error))
    {
        const std::string uncalibrated = readManifestText();
        Check(uncalibrated.find("\"marker\"") != std::string::npos,
            "new-size UI manifest keeps the fixed marker");
        Check(uncalibrated.find("\"excluded_rects\"") == std::string::npos,
            "new-size UI manifest omits uncalibrated excluded_rects");
        Check(uncalibrated.find("\"width\": 1920") != std::string::npos
            && uncalibrated.find("\"height\": 1080") != std::string::npos,
            "new-size UI manifest records output dimensions");
    }
    else
    {
        Check(false, "new-size UI scene manifest write: " + error);
    }
}

void TestSceneExtensionManifest()
{
    SceneManifest manifest;
    manifest.name = "nonrigid";
    manifest.mode = 1;
    manifest.warmup = 0;
    manifest.measuredFrames = 1;
    manifest.sceneExtensionJson =
        "{\"schema\":\"xefg.render_scene_extension/1\",\"version\":1,"
        "\"scene\":\"nonrigid\",\"grid\":{\"cells_x\":64}}";
    SceneFrameRecord frame;
    frame.time = 5.0f;
    frame.previousTime = 4.0f;
    frame.hasDeformation = true;
    frame.previousIsCurrent = false;
    SceneObjectRecord object;
    object.name = "plane_far";
    object.hasMaterial = true;
    object.role = "transparent_plane";
    object.mesh = "panel";
    object.flatColor = {0.80f, 0.22f, 0.18f, 0.55f};
    object.premultipliedColor = {0.44f, 0.121f, 0.099f, 0.55f};
    object.transparent = true;
    object.depthWrite = false;
    object.writesDepthMotion = false;
    object.sortIndex = 2;
    frame.objects.push_back(object);
    manifest.frames.push_back(frame);

    const std::filesystem::path path = std::filesystem::temp_directory_path()
        / "xefg_render_validation_extension_test.json";
    std::string error;
    if (!WriteSceneManifest(path, manifest, error))
    {
        std::printf("FAIL: extension manifest write: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    std::ifstream file(path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::filesystem::remove(path);
    Check(text.find("\"scene_extension\": {\"schema\":\"xefg.render_scene_extension/1\"")
        != std::string::npos, "scene manifest emits scene_extension");
    Check(text.find("\"deformation\": {\"time\": 5") != std::string::npos,
        "scene manifest emits per-frame deformation state");
    Check(text.find("\"mesh\": \"panel\"") != std::string::npos
        && text.find("\"transparent\": true") != std::string::npos
        && text.find("\"writes_depth_motion\": false") != std::string::npos
        && text.find("\"sort_index\": 2") != std::string::npos,
        "scene manifest emits material/depth/order fields");
}


bool TestScene(Renderer& renderer, SceneKind scene)
{
    const std::string sceneName = kSceneNames[static_cast<size_t>(scene)];
    std::printf("scene %s\n", sceneName.c_str());
    std::string error;

    std::vector<uint8_t> previousColor;

    for (uint32_t frame = 0; frame < kTotalFrames; ++frame)
    {
        FrameState state = MakeFrameState(scene, frame, false);
        if (scene == SceneKind::Cut && frame == kCutFrame) state.historyReset = true;

        if (!renderer.Render(state, scene, false, error))
        {
            std::printf("FAIL: %s frame %u render: %s\n", sceneName.c_str(), frame, error.c_str());
            return false;
        }

        std::vector<uint8_t> color;
        std::vector<uint8_t> depth;
        std::vector<uint8_t> motion;
        if (!Readback(renderer.Color(), color, error) ||
            !Readback(renderer.Depth(), depth, error) ||
            !Readback(renderer.Motion(), motion, error))
        {
            std::printf("FAIL: %s frame %u readback: %s\n", sceneName.c_str(), frame, error.c_str());
            return false;
        }

        Check(color.size() == size_t(kColorWidth) * kColorHeight * 4,
            sceneName + " color byte size");
        Check(depth.size() == size_t(kGeometryWidth) * kGeometryHeight * 4,
            sceneName + " depth byte size");
        Check(motion.size() == size_t(kGeometryWidth) * kGeometryHeight * 4,
            sceneName + " motion byte size");

        const DepthStats depthStats = ComputeDepthStats(depth);
        Check(depthStats.finite, sceneName + " depth finite");
        Check(depthStats.inRange, sceneName + " depth in [0,1]");
        Check(depthStats.maximum > depthStats.minimum, sceneName + " depth has visible geometry");
        Check(depthStats.minimum < 0.999f, sceneName + " depth contains near geometry");

        const MotionStats motionStats = ComputeMotionStats(motion);
        Check(motionStats.finite, sceneName + " motion finite");
        Check(motionStats.maxMagnitude < 500.0, sceneName + " motion magnitude bounded");

        if (scene == SceneKind::Translation || scene == SceneKind::Occlusion ||
            scene == SceneKind::Camera)
        {
            if (frame > 0)
            {
                Check(motionStats.nonZeroPixels > 0.0, sceneName + " has nonzero motion");
            }
            if (scene == SceneKind::Translation && frame > 0 && motionStats.nonZeroPixels > 0.0)
            {
                Check(motionStats.sumX / motionStats.nonZeroPixels < -0.1,
                    "translation motion points to previous position");
            }
        }
        if (scene == SceneKind::Cut && (frame == 0 || frame == kCutFrame))
        {
            Check(motionStats.maxMagnitude < 0.01, "cut reset frame motion is zero");
        }

        // Camera cut must produce a clearly different image across frame 7.
        if (scene == SceneKind::Cut && frame == kCutFrame)
        {
            size_t changed = 0;
            for (size_t i = 0; i + 3 < color.size(); i += 4)
            {
                const int delta = std::abs(int(color[i]) - int(previousColor[i])) +
                    std::abs(int(color[i + 1]) - int(previousColor[i + 1])) +
                    std::abs(int(color[i + 2]) - int(previousColor[i + 2]));
                if (delta > 24) ++changed;
            }
            const double changedFraction = double(changed) / double(color.size() / 4);
            Check(changedFraction > 0.10, "camera cut changes pixels");
        }

        // Determinism: same state, same bytes.
        if (frame == 5)
        {
            std::vector<uint8_t> secondColor;
            std::vector<uint8_t> secondDepth;
            std::vector<uint8_t> secondMotion;
            if (!renderer.Render(state, scene, false, error) ||
                !Readback(renderer.Color(), secondColor, error) ||
                !Readback(renderer.Depth(), secondDepth, error) ||
                !Readback(renderer.Motion(), secondMotion, error))
            {
                std::printf("FAIL: %s determinism readback: %s\n", sceneName.c_str(), error.c_str());
                return false;
            }
            Check(secondColor == color, sceneName + " color deterministic");
            Check(secondDepth == depth, sceneName + " depth deterministic");
            Check(secondMotion == motion, sceneName + " motion deterministic");
        }

        previousColor = color;
    }

    // Midpoint: independent targets, current UI marker, does not advance state.
    const uint32_t midpointFrame = scene == SceneKind::Cut ? kCutFrame : 5;
    FrameState current = MakeFrameState(scene, midpointFrame, false);
    if (scene == SceneKind::Cut && midpointFrame == kCutFrame) current.historyReset = true;
    FrameState midpoint = MakeFrameState(scene, midpointFrame, true);

    std::vector<uint8_t> currentColor;
    std::vector<uint8_t> currentBackbuffer;
    if (!renderer.Render(current, scene, false, error) ||
        !Readback(renderer.Color(), currentColor, error) ||
        !Readback(renderer.Backbuffer(), currentBackbuffer, error))
    {
        std::printf("FAIL: %s midpoint baseline: %s\n", sceneName.c_str(), error.c_str());
        return false;
    }

    std::vector<uint8_t> midpointColor;
    std::vector<uint8_t> midpointBackbuffer;
    std::vector<uint8_t> midpointUi;
    if (!renderer.Render(midpoint, scene, true, error) ||
        !Readback(renderer.Color(), midpointColor, error) ||
        !Readback(renderer.Backbuffer(), midpointBackbuffer, error) ||
        !Readback(renderer.Ui(), midpointUi, error))
    {
        std::printf("FAIL: %s midpoint render: %s\n", sceneName.c_str(), error.c_str());
        return false;
    }
    Check(renderer.MidpointActive(), sceneName + " midpoint target active");
    Check(midpointColor.size() == size_t(kColorWidth) * kColorHeight * 4,
        sceneName + " midpoint color size");
    Check(midpointUi.size() == size_t(kColorWidth) * kColorHeight * 4,
        sceneName + " midpoint UI size");
    std::string markerError;
    Check(DecodeUiMarker(midpointUi, kColorWidth, midpointFrame, markerError),
        sceneName + " midpoint UI marker: " + markerError);

    double midpointDifference = 0.0;
    for (size_t i = 0; i < midpointBackbuffer.size(); ++i)
        midpointDifference += std::abs(int(midpointBackbuffer[i]) - int(currentBackbuffer[i]));
    Check(midpointDifference / double(midpointBackbuffer.size()) > 0.05,
        sceneName + " midpoint differs from current");

    std::vector<uint8_t> restoredColor;
    std::vector<uint8_t> restoredBackbuffer;
    if (!renderer.Render(current, scene, false, error) ||
        !Readback(renderer.Color(), restoredColor, error) ||
        !Readback(renderer.Backbuffer(), restoredBackbuffer, error))
    {
        std::printf("FAIL: %s restore after midpoint: %s\n", sceneName.c_str(), error.c_str());
        return false;
    }
    Check(!renderer.MidpointActive(), sceneName + " current target restored");
    Check(restoredColor == currentColor, sceneName + " midpoint did not mutate current color");
    Check(restoredBackbuffer == currentBackbuffer, sceneName + " midpoint did not mutate current backbuffer");

    return true;
}

bool PreviousClipPositive(const FrameState& state, std::string& error)
{
    for (uint32_t object = 0; object < state.objectCount; ++object)
    {
        const ObjectState& item = state.objects[object];
        for (int corner = 0; corner < 8; ++corner)
        {
            const float extent = item.mesh == MeshKind::Cube ? 0.5f : 0.0f;
            const float point[4] = {
                (corner & 1) ? extent : -extent,
                (corner & 2) ? extent : -extent,
                (corner & 4) ? extent : -extent,
                1.0f};
            float world[4]{};
            for (int column = 0; column < 4; ++column)
                for (int row = 0; row < 4; ++row)
                    world[column] += point[row] * item.previousWorld.m[row * 4 + column];
            float clip[4]{};
            for (int column = 0; column < 4; ++column)
                for (int row = 0; row < 4; ++row)
                    clip[column] += world[row] * state.previousViewProjection.m[row * 4 + column];
            if (clip[3] <= 0.05f)
            {
                error = item.name + " previous clip w is not positive";
                return false;
            }
        }
    }
    return true;
}

bool TestDepthLayersCoverage(Renderer& renderer)
{
    std::string error;
    double foreground = 0.0;
    double nearPixels = 0.0;
    double farPixels = 0.0;
    for (uint32_t frame = 0; frame < kTotalFrames; ++frame)
    {
        const FrameState state = MakeFrameState(SceneKind::DepthLayers, frame, false);
        std::vector<uint8_t> depth;
        std::vector<uint8_t> motion;
        if (!renderer.Render(state, SceneKind::DepthLayers, false, error)
            || !Readback(renderer.Depth(), depth, error)
            || !Readback(renderer.Motion(), motion, error))
        {
            std::printf("FAIL: depth_layers frame %u: %s\n", frame, error.c_str());
            return false;
        }
        if (!PreviousClipPositive(state, error))
        {
            std::printf("FAIL: depth_layers frame %u: %s\n", frame, error.c_str());
            return false;
        }
        const MotionStats motionStats = ComputeMotionStats(motion);
        Check(motionStats.finite, "depth_layers motion finite");
        if (frame > 0) Check(motionStats.nonZeroPixels > 0.0,
            "depth_layers has foreground motion");

        const float* values = reinterpret_cast<const float*>(depth.data());
        for (size_t i = 0; i < depth.size() / sizeof(float); ++i)
        {
            if (values[i] >= 0.9999f) continue;
            foreground += 1.0;
            if (values[i] < 0.9f) nearPixels += 1.0;
            if (values[i] > 0.99f) farPixels += 1.0;
        }
    }
    Check(foreground > 0.0, "depth_layers has foreground geometry");
    Check(nearPixels / foreground >= 0.01, "depth_layers near <0.9 covers >=1%");
    Check(farPixels / foreground >= 0.01, "depth_layers far >0.99 covers >=1%");
    return true;
}

bool TestScreenCrossCoverage(Renderer& renderer)
{
    std::string error;
    int positiveFrames = 0;
    int negativeFrames = 0;
    int largeFrames = 0;
    double edgeMinimum = 1.0;
    double edgeMaximum = 0.0;
    double farGeometry = 0.0;
    double staticFar = 0.0;
    double previousNearX = 0.0;
    bool hasPrevious = false;
    bool continuous = true;

    for (uint32_t frame = 4; frame <= 11; ++frame)
    {
        const FrameState state = MakeFrameState(SceneKind::ScreenCross, frame, false);
        std::vector<uint8_t> depth;
        std::vector<uint8_t> motion;
        if (!renderer.Render(state, SceneKind::ScreenCross, false, error)
            || !Readback(renderer.Depth(), depth, error)
            || !Readback(renderer.Motion(), motion, error))
        {
            std::printf("FAIL: screen_cross frame %u: %s\n", frame, error.c_str());
            return false;
        }
        if (!PreviousClipPositive(state, error))
        {
            std::printf("FAIL: screen_cross frame %u: %s\n", frame, error.c_str());
            return false;
        }

        const float nearX = state.objects[0].world.m[12];
        if (hasPrevious)
        {
            const float delta = std::fabs(nearX - previousNearX);
            if (delta > 2.0f) continuous = false;
        }
        previousNearX = nearX;
        hasPrevious = true;

        const float* depthValues = reinterpret_cast<const float*>(depth.data());
        const uint16_t* motionValues = reinterpret_cast<const uint16_t*>(motion.data());
        double positive = 0.0;
        double negative = 0.0;
        double large = 0.0;
        double edgePixels = 0.0;
        double edgeMoving = 0.0;
        for (uint32_t y = 0; y < kGeometryHeight; ++y)
        {
            for (uint32_t x = 0; x < kGeometryWidth; ++x)
            {
                const size_t index = static_cast<size_t>(y) * kGeometryWidth + x;
                const float mvX = HalfToFloat(motionValues[index * 2 + 0]);
                const float mvY = HalfToFloat(motionValues[index * 2 + 1]);
                const double magnitude = std::sqrt(double(mvX) * mvX + double(mvY) * mvY);
                if (mvX > 0.01f * kGeometryWidth) positive += 1.0;
                if (mvX < -0.01f * kGeometryWidth) negative += 1.0;
                if (magnitude > 0.02 * kGeometryWidth) large += 1.0;
                if (depthValues[index] > 0.99f && depthValues[index] < 0.9999f)
                {
                    farGeometry += 1.0;
                    if (magnitude < 0.5) staticFar += 1.0;
                }
                if (x < 2 || x >= kGeometryWidth - 2 || y < 2 || y >= kGeometryHeight - 2)
                {
                    edgePixels += 1.0;
                    if (magnitude > 1.0) edgeMoving += 1.0;
                }
            }
        }
        if (positive >= 0.01 * kGeometryWidth * kGeometryHeight) ++positiveFrames;
        if (negative >= 0.01 * kGeometryWidth * kGeometryHeight) ++negativeFrames;
        if (large >= 0.01 * kGeometryWidth * kGeometryHeight) ++largeFrames;
        const double edgeFraction = edgePixels > 0.0 ? edgeMoving / edgePixels : 0.0;
        edgeMinimum = std::min(edgeMinimum, edgeFraction);
        edgeMaximum = std::max(edgeMaximum, edgeFraction);
    }

    Check(continuous, "screen_cross horizontal trajectory is continuous");
    Check(positiveFrames > 0, "screen_cross has +x dominant frames");
    Check(negativeFrames > 0, "screen_cross has -x dominant frames");
    Check(largeFrames > 0, "screen_cross has a frame with >=1% large motion");
    Check(edgeMaximum - edgeMinimum >= 0.01,
        "screen_cross moving-edge fraction changes by >=1%");
    Check(farGeometry > 0.0 && staticFar / farGeometry >= 0.5,
        "screen_cross has a static far background");
    return true;
}

bool TestCameraTurnCoverage(Renderer& renderer)
{
    std::string error;
    double nearPixels = 0.0;
    double farPixels = 0.0;
    double nearMotion = 0.0;
    double farMotion = 0.0;
    double broadMaximum = 0.0;

    for (uint32_t frame = 0; frame < kTotalFrames; ++frame)
    {
        const FrameState state = MakeFrameState(SceneKind::CameraTurn, frame, false);
        std::vector<uint8_t> depth;
        std::vector<uint8_t> motion;
        if (!renderer.Render(state, SceneKind::CameraTurn, false, error)
            || !Readback(renderer.Depth(), depth, error)
            || !Readback(renderer.Motion(), motion, error))
        {
            std::printf("FAIL: camera_turn frame %u: %s\n", frame, error.c_str());
            return false;
        }
        if (!PreviousClipPositive(state, error))
        {
            std::printf("FAIL: camera_turn frame %u: %s\n", frame, error.c_str());
            return false;
        }

        const float* depthValues = reinterpret_cast<const float*>(depth.data());
        const uint16_t* motionValues = reinterpret_cast<const uint16_t*>(motion.data());
        double foreground = 0.0;
        double broad = 0.0;
        for (size_t i = 0; i < size_t(kGeometryWidth) * kGeometryHeight; ++i)
        {
            const float mvX = HalfToFloat(motionValues[i * 2 + 0]);
            const float mvY = HalfToFloat(motionValues[i * 2 + 1]);
            const double normalized = std::sqrt(double(mvX) * mvX + double(mvY) * mvY)
                / kGeometryWidth;
            if (depthValues[i] < 0.9999f)
            {
                foreground += 1.0;
                if (normalized > 0.005) broad += 1.0;
            }
            if (depthValues[i] < 0.97f)
            {
                nearPixels += 1.0;
                nearMotion += normalized;
            }
            else if (depthValues[i] > 0.99f && depthValues[i] < 0.9999f)
            {
                farPixels += 1.0;
                farMotion += normalized;
            }
        }
        if (foreground > 0.0) broadMaximum = std::max(broadMaximum, broad / foreground);
    }

    Check(nearPixels > 0.0, "camera_turn near band depth <0.97 is nonempty");
    Check(farPixels > 0.0, "camera_turn far band depth >0.99 is nonempty");
    const double nearMean = nearPixels > 0.0 ? nearMotion / nearPixels : 0.0;
    const double farMean = farPixels > 0.0 ? farMotion / farPixels : 0.0;
    Check(nearMean - farMean >= 0.001, "camera_turn near/far mean motion gap >=0.001");
    Check(broadMaximum > 0.30, "camera_turn broad motion covers >30% foreground");
    return true;
}

BgraPixel ColorPixel(const std::vector<uint8_t>& color, uint32_t width, uint32_t x, uint32_t y)
{
    const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
    return {color[offset + 0], color[offset + 1], color[offset + 2], color[offset + 3]};
}

void CheckColorNear(const BgraPixel& pixel, float red, float green, float blue,
    const std::string& message)
{
    const float tolerance = 0.04f * 255.0f;
    Check(std::abs(float(pixel.r) - red * 255.0f) <= tolerance
        && std::abs(float(pixel.g) - green * 255.0f) <= tolerance
        && std::abs(float(pixel.b) - blue * 255.0f) <= tolerance
        && pixel.a == 255,
        message);
}

bool TestNonrigidRender(Renderer& renderer)
{
    std::string error;
    const FrameState state = MakeFrameState(SceneKind::Nonrigid, 5, false);
    std::vector<uint8_t> depth;
    std::vector<uint8_t> motion;
    if (!renderer.Render(state, SceneKind::Nonrigid, false, error)
        || !Readback(renderer.Depth(), depth, error)
        || !Readback(renderer.Motion(), motion, error))
    {
        std::printf("FAIL: nonrigid render/readback: %s\n", error.c_str());
        return false;
    }
    const MotionStats motionStats = ComputeMotionStats(motion);
    Check(motionStats.finite, "nonrigid motion finite");
    Check(motionStats.nonZeroPixels > 500.0,
        "nonrigid grid is visible and produces deformed motion");
    Check(motionStats.maxMagnitude > 1.0, "nonrigid deformation produces real movement");
    const DepthStats depthStats = ComputeDepthStats(depth);
    Check(depthStats.finite && depthStats.inRange, "nonrigid depth finite and in range");
    Check(depthStats.minimum < 0.99f && depthStats.maximum > 0.99f,
        "nonrigid deforming panel and static background both contribute depth");
    return true;
}

bool TestTransparencyRender(Renderer& renderer)
{
    std::string error;
    const FrameState state = MakeFrameState(SceneKind::Transparency, 5, false);
    std::vector<uint8_t> color;
    std::vector<uint8_t> depth;
    std::vector<uint8_t> motion;
    if (!renderer.Render(state, SceneKind::Transparency, false, error)
        || !Readback(renderer.Color(), color, error)
        || !Readback(renderer.Depth(), depth, error)
        || !Readback(renderer.Motion(), motion, error))
    {
        std::printf("FAIL: transparency render/readback: %s\n", error.c_str());
        return false;
    }
    const MotionStats motionStats = ComputeMotionStats(motion);
    Check(motionStats.finite && motionStats.maxMagnitude < 0.01,
        "transparency geometry pass is opaque-only and static");

    const float* values = reinterpret_cast<const float*>(depth.data());
    size_t blockerPixels = 0;
    size_t backdropPixels = 0;
    for (size_t i = 0; i < depth.size() / sizeof(float); ++i)
    {
        if (values[i] < 0.975f) ++blockerPixels;
        if (values[i] > 0.98f && values[i] < 0.9999f) ++backdropPixels;
    }
    const double total = double(depth.size() / sizeof(float));
    Check(double(blockerPixels) / total > 0.05, "transparency blocker covers a distinct region");
    Check(double(backdropPixels) / total > 0.50, "transparency opaque backdrop covers the frame");

    // Independently computed premultiplied blend samples at frame 5.
    CheckColorNear(ColorPixel(color, kColorWidth, 60, 60), 0.10f, 0.15f, 0.24f,
        "transparency backdrop-only color");
    CheckColorNear(ColorPixel(color, kColorWidth, 200, 80),
        0.10f * 0.55f + 0.16f * 0.45f,
        0.15f * 0.55f + 0.60f * 0.45f,
        0.24f * 0.55f + 0.78f * 0.45f,
        "transparency single-near blend color");
    CheckColorNear(ColorPixel(color, kColorWidth, 200, 280),
        0.10f * 0.45f + 0.80f * 0.55f,
        0.15f * 0.45f + 0.22f * 0.55f,
        0.24f * 0.45f + 0.18f * 0.55f,
        "transparency single-far blend color");
    CheckColorNear(ColorPixel(color, kColorWidth, 250, 180),
        0.10f * 0.45f * 0.55f + 0.80f * 0.55f * 0.55f + 0.16f * 0.45f,
        0.15f * 0.45f * 0.55f + 0.22f * 0.55f * 0.55f + 0.60f * 0.45f,
        0.24f * 0.45f * 0.55f + 0.18f * 0.55f * 0.55f + 0.78f * 0.45f,
        "transparency double-overlap blend color");
    CheckColorNear(ColorPixel(color, kColorWidth, 499, 249), 0.20f, 0.24f, 0.26f,
        "transparency blocker occludes both planes");
    return true;
}

bool TestRenderSizeDimensions(Renderer& renderer640, ID3D12Device* device,
    ID3D12CommandQueue* queue, const std::filesystem::path& shader)
{
    std::string error;
    const RenderSize reference = renderer640.Size();
    Check(reference.width == 640 && reference.height == 360, "renderer reports default 640x360");
    Check(reference.GeometryWidth() == 320 && reference.GeometryHeight() == 180,
        "640 renderer geometry targets are half size");

    const RenderSize wideSize{1920, 1080};
    const uint32_t frame = 5;
    const FrameState state = MakeFrameState(SceneKind::Translation, frame, false);

    // 640 reference: fixed marker and fixed-size crosshair.
    std::vector<uint8_t> ui640;
    if (!renderer640.Render(state, SceneKind::Translation, false, error) ||
        !Readback(renderer640.Ui(), ui640, error))
    {
        std::printf("FAIL: 640 dimension render/readback: %s\n", error.c_str());
        return false;
    }
    Check(ui640.size() == size_t(640) * 360 * 4, "640 UI byte size");
    std::string markerError;
    Check(DecodeUiMarker(ui640, 640, frame, markerError), "640 UI marker: " + markerError);
    uint32_t centerX = 0;
    uint32_t centerY = 0;
    CrosshairCenter(frame, 640, 360, centerX, centerY);
    Check(UiPixel(ui640, 640, centerX, centerY).a != 0, "640 crosshair centred at output/2");
    Check(UiPixel(ui640, 640, centerX + 25, centerY).a == 0,
        "640 crosshair retains a fixed 24 px extent");

    Renderer wide;
    if (!wide.Initialize(device, queue, shader, error, wideSize))
    {
        std::printf("FAIL: 1080 renderer initialize: %s\n", error.c_str());
        return false;
    }
    Check(wide.Size().width == 1920 && wide.Size().height == 1080,
        "renderer reports 1920x1080");

    std::vector<uint8_t> color;
    std::vector<uint8_t> ui;
    std::vector<uint8_t> backbuffer;
    std::vector<uint8_t> depth;
    std::vector<uint8_t> motion;
    if (!wide.Render(state, SceneKind::Translation, false, error) ||
        !Readback(wide.Color(), color, error) ||
        !Readback(wide.Ui(), ui, error) ||
        !Readback(wide.Backbuffer(), backbuffer, error) ||
        !Readback(wide.Depth(), depth, error) ||
        !Readback(wide.Motion(), motion, error))
    {
        std::printf("FAIL: 1080 render/readback: %s\n", error.c_str());
        return false;
    }
    Check(color.size() == size_t(1920) * 1080 * 4, "1080 color byte size");
    Check(ui.size() == size_t(1920) * 1080 * 4, "1080 UI byte size");
    Check(backbuffer.size() == size_t(1920) * 1080 * 4, "1080 backbuffer byte size");
    Check(depth.size() == size_t(960) * 540 * 4, "1080 depth uses half geometry");
    Check(motion.size() == size_t(960) * 540 * 4, "1080 motion uses half geometry");

    const DepthStats depthStats = ComputeDepthStats(depth);
    Check(depthStats.finite && depthStats.inRange, "1080 depth finite and in range");
    Check(depthStats.maximum > depthStats.minimum, "1080 depth has visible geometry");
    const MotionStats motionStats = ComputeMotionStats(motion);
    Check(motionStats.finite, "1080 motion finite");
    Check(motionStats.nonZeroPixels > 0.0, "1080 motion has nonzero pixels");

    markerError.clear();
    Check(DecodeUiMarker(ui, 1920, frame, markerError),
        "1080 UI marker stays at 96,32: " + markerError);
    CrosshairCenter(frame, 1920, 1080, centerX, centerY);
    Check(UiPixel(ui, 1920, centerX, centerY).a != 0, "1080 crosshair centred at output/2");
    Check(UiPixel(ui, 1920, centerX + 25, centerY).a == 0,
        "1080 crosshair retains a fixed 24 px extent");
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    const std::filesystem::path shader = FindShader(argc, argv);
    if (shader.empty())
    {
        std::printf("FAIL: scene.hlsl not found; pass its path or build beside shaders/\n");
        return 1;
    }
    std::printf("shader: %s\n", shader.string().c_str());
    TestMatrixConventions();
    TestPairStates();
    TestSceneManifestSchema();
    TestSceneExtensionManifest();
    TestRenderSizeTypes();
    TestNewSceneNamesAndStates();
    TestNonrigidContract();
    TestTransparencyContract();

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    if (!CreateWarpDevice(device, queue))
    {
        std::printf("FAIL: WARP D3D12 device/queue creation failed\n");
        return 1;
    }

    Renderer renderer;
    std::string error;
    if (!renderer.Initialize(device.Get(), queue.Get(), shader, error))
    {
        std::printf("FAIL: renderer initialize: %s\n", error.c_str());
        return 1;
    }
    if (!renderer.Ready())
    {
        std::printf("FAIL: renderer not ready\n");
        return 1;
    }
    if (!TestMode1NoUi(renderer)) return 1;

    for (size_t i = 0; i < kSceneCount; ++i)
        TestScene(renderer, static_cast<SceneKind>(i));

    if (!TestDepthLayersCoverage(renderer)) return 1;
    if (!TestScreenCrossCoverage(renderer)) return 1;
    if (!TestCameraTurnCoverage(renderer)) return 1;
    if (!TestNonrigidRender(renderer)) return 1;
    if (!TestTransparencyRender(renderer)) return 1;
    if (!TestRenderSizeDimensions(renderer, device.Get(), queue.Get(), shader)) return 1;

    ComPtr<ID3D12InfoQueue> info;
    if (SUCCEEDED(device.As(&info)))
    {
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i)
        {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<char> bytes(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
            if (SUCCEEDED(info->GetMessage(i, message, &size)))
                Check(message->Severity != D3D12_MESSAGE_SEVERITY_ERROR
                    && message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION,
                    message->pDescription);
        }
    }

    if (g_failures)
    {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("all renderer tests passed\n");
    return 0;
}
