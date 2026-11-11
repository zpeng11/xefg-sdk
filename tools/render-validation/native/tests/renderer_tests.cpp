// WARP render-only tests for the render-validation DX12 renderer.
// These do not create an XeSS-FG context and do not run frame generation.

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

bool DecodeUiMarker(const std::vector<uint8_t>& ui, uint32_t expectedFrame, std::string& error)
{
    const size_t stride = static_cast<size_t>(kColorWidth) * 4;
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

bool TestScene(Renderer& renderer, SceneKind scene){
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
    Check(DecodeUiMarker(midpointUi, midpointFrame, markerError),
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

    for (size_t i = 0; i < kSceneCount; ++i)
        TestScene(renderer, static_cast<SceneKind>(i));

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
