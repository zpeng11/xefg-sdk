#pragma once
// Deterministic synthetic frames for the XeFG oracle. The scene is chosen so
// that the expected 2x midpoint is known analytically: a rigid checkerboard
// square translates over a static background at a constant velocity.
#include <cstdint>
#include <string>
#include <vector>

namespace oracle
{
struct SceneOptions
{
    uint32_t width = 1920;
    uint32_t height = 1080;
    bool highResMotionVectors = false;
    uint32_t motionWidth = 960;
    uint32_t motionHeight = 540;

    bool staticScene = false;
    bool smoothContent = false; // Gradients instead of the grid/checker; geometry and motion stay unchanged.
    // Object displacement per frame in target pixels. The motion vector handed
    // to XeFG is the negated displacement (current pixel -> previous pixel).
    float velocityX = 12.0f;
    float velocityY = 4.0f;

    bool ui = true;
    // Move the crosshair per frame so the captured UI is not frame-invariant.
    // Off by default: every byte of the default scene stays unchanged.
    bool animateUi = false;
    bool calibration = false; // Opaque frame-ID strip, separate from the motion ROI.
    // XeFG expects current -> previous; flip only for diagnosing sign conventions.
    bool flipMotionVectors = false;
    uint32_t frameIndex = 0;
    float jitterX = 0.0f;
    float jitterY = 0.0f;
};

struct SceneFrame
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t motionWidth = 0;
    uint32_t motionHeight = 0;

    std::vector<uint8_t> hudlessColor; // BGRA8, width x height
    std::vector<uint8_t> uiColor;      // BGRA8, premultiplied alpha
    std::vector<uint8_t> backbuffer;   // BGRA8, ui over hudless
    std::vector<uint8_t> expectedMid;  // BGRA8, the analytic 2x midpoint
    std::vector<float> depth;          // R32_FLOAT, motionWidth x motionHeight
    std::vector<uint16_t> motion;      // R16G16_FLOAT, motionWidth x motionHeight

    float velocityX = 0.0f;
    float velocityY = 0.0f;
};

void GenerateScene(const SceneOptions& options, SceneFrame& frame);

uint16_t FloatToHalf(float value);
bool WriteRawFile(const std::wstring& path, const void* data, size_t bytes);
} // namespace oracle
