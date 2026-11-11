#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace oracle
{
namespace
{
constexpr float kObjectSize = 320.0f;

inline uint8_t ClampByte(float value)
{
    return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
}

void StoreBgra(uint8_t* pixel, float r, float g, float b, float a)
{
    pixel[0] = ClampByte(b);
    pixel[1] = ClampByte(g);
    pixel[2] = ClampByte(r);
    pixel[3] = ClampByte(a);
}

struct Rect
{
    float left, top, right, bottom;
    bool Contains(float x, float y) const { return x >= left && x < right && y >= top && y < bottom; }
};

Rect ObjectRect(const SceneOptions& options, float frameOffset)
{
    const float velocityX = options.staticScene ? 0.0f : options.velocityX;
    const float velocityY = options.staticScene ? 0.0f : options.velocityY;
    const float x = options.width * 0.15f + velocityX * frameOffset;
    const float y = options.height * 0.15f + velocityY * frameOffset;
    return {x, y, x + kObjectSize, y + kObjectSize};
}

// Background: gradient plus a coarse grid; deterministic and cheap.
void Background(float x, float y, uint32_t width, uint32_t height, bool smooth, float& r, float& g, float& b)
{
    r = 20.0f + 45.0f * (x / std::max(1.0f, static_cast<float>(width)));
    g = 28.0f + 52.0f * (y / std::max(1.0f, static_cast<float>(height)));
    b = 70.0f;
    const int gx = static_cast<int>(x) & 63;
    const int gy = static_cast<int>(y) & 63;
    if (!smooth && (gx < 1 || gy < 1))
    {
        r += 18.0f;
        g += 18.0f;
        b += 18.0f;
    }
}

void ObjectPattern(float x, float y, const Rect& rect, bool smooth, float& r, float& g, float& b)
{
    const float lx = x - rect.left;
    const float ly = y - rect.top;
    if (smooth)
    {
        r = 80.0f + 120.0f * (lx / kObjectSize);
        g = 100.0f + 80.0f * (ly / kObjectSize);
        b = 160.0f - 60.0f * (lx / kObjectSize);
        return;
    }
    const bool border = lx < 2.0f || ly < 2.0f || lx >= kObjectSize - 2.0f || ly >= kObjectSize - 2.0f;
    if (border)
    {
        r = 245.0f;
        g = 245.0f;
        b = 245.0f;
        return;
    }
    const bool cell = ((static_cast<int>(lx) >> 4) + (static_cast<int>(ly) >> 4)) & 1;
    if (cell)
    {
        r = 240.0f;
        g = 180.0f;
        b = 40.0f;
    }
    else
    {
        r = 40.0f;
        g = 160.0f;
        b = 220.0f;
    }
}

void RenderColor(const SceneOptions& options, float frameOffset, std::vector<uint8_t>& output)
{
    const Rect object = ObjectRect(options, frameOffset);
    output.resize(static_cast<size_t>(options.width) * options.height * 4);
    for (uint32_t y = 0; y < options.height; ++y)
    {
        for (uint32_t x = 0; x < options.width; ++x)
        {
            float r, g, b;
            Background(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                options.width, options.height, options.smoothContent, r, g, b);
            if (object.Contains(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f))
                ObjectPattern(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                    object, options.smoothContent, r, g, b);
            StoreBgra(&output[(static_cast<size_t>(y) * options.width + x) * 4], r, g, b, 255.0f);
        }
    }
}

// UI is a fixed crosshair plus a frame; RGB is premultiplied so the XeFG
// composition formula (UI + (1 - alpha) * hudless) holds exactly.
void RenderUi(const SceneOptions& options, std::vector<uint8_t>& output)
{
    output.assign(static_cast<size_t>(options.width) * options.height * 4, 0);
    if (!options.ui) return;
    float cx = options.width * 0.5f;
    float cy = options.height * 0.5f;
    if (options.animateUi)
    {
        // A small, deterministic crosshair motion: visible per frame, bounded
        // for the default sizes, and applied only to the UI/backbuffer inputs.
        const uint32_t phase = options.frameIndex % 8u;
        const uint32_t vertical = (options.frameIndex / 8u) % 6u;
        cx += (static_cast<float>(phase) - 3.5f) * 4.0f;
        cy += (static_cast<float>(vertical) - 2.5f) * 3.0f;
    }
    for (uint32_t y = 0; y < options.height; ++y)
    {
        for (uint32_t x = 0; x < options.width; ++x)
        {
            const float px = static_cast<float>(x) + 0.5f;
            const float py = static_cast<float>(y) + 0.5f;
            float alpha = 0.0f;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (std::fabs(px - cx) < 2.0f && std::fabs(py - cy) < 24.0f)
            {
                alpha = 0.80f;
                r = 255.0f;
                g = 40.0f;
                b = 40.0f;
            }
            else if (std::fabs(py - cy) < 2.0f && std::fabs(px - cx) < 24.0f)
            {
                alpha = 0.80f;
                r = 255.0f;
                g = 40.0f;
                b = 40.0f;
            }
            else if (x < 6 || y < 6 || x >= options.width - 6 || y >= options.height - 6)
            {
                alpha = 0.35f;
                r = 255.0f;
                g = 255.0f;
                b = 255.0f;
            }
            if (alpha > 0.0f)
                StoreBgra(&output[(static_cast<size_t>(y) * options.width + x) * 4],
                    r * alpha, g * alpha, b * alpha, alpha * 255.0f);
        }
    }
}

void Composite(const std::vector<uint8_t>& hudless, const std::vector<uint8_t>& ui, std::vector<uint8_t>& output)
{
    output.resize(hudless.size());
    for (size_t i = 0; i < hudless.size(); i += 4)
    {
        const float alpha = ui[i + 3] / 255.0f;
        const float inverse = 1.0f - alpha;
        output[i + 0] = ClampByte(ui[i + 0] + inverse * hudless[i + 0]);
        output[i + 1] = ClampByte(ui[i + 1] + inverse * hudless[i + 1]);
        output[i + 2] = ClampByte(ui[i + 2] + inverse * hudless[i + 2]);
        output[i + 3] = 255;
    }
}

void RenderDepthAndMotion(const SceneOptions& options, SceneFrame& frame)
{
    const uint32_t mvWidth = frame.motionWidth;
    const uint32_t mvHeight = frame.motionHeight;
    const Rect object = ObjectRect(options, static_cast<float>(options.frameIndex));
    const float scaleX = static_cast<float>(frame.motionWidth) / std::max(1u, frame.width);
    const float scaleY = static_cast<float>(frame.motionHeight) / std::max(1u, frame.height);
    const float vx = options.staticScene ? 0.0f : options.velocityX;
    const float vy = options.staticScene ? 0.0f : options.velocityY;

    frame.depth.resize(static_cast<size_t>(mvWidth) * mvHeight);
    frame.motion.resize(static_cast<size_t>(mvWidth) * mvHeight * 2);
    for (uint32_t y = 0; y < mvHeight; ++y)
    {
        for (uint32_t x = 0; x < mvWidth; ++x)
        {
            const float sampleX = (static_cast<float>(x) + 0.5f) / scaleX;
            const float sampleY = (static_cast<float>(y) + 0.5f) / scaleY;
            const bool inside = object.Contains(sampleX, sampleY);
            const size_t index = static_cast<size_t>(y) * mvWidth + x;
            // Smaller depth is closer; the default convention is not inverted.
            frame.depth[index] = inside ? 1.0f : 100.0f;
            // Motion vector: current pixel -> previous pixel, in render-resolution pixels.
            const float sign = options.flipMotionVectors ? 1.0f : -1.0f;
            frame.motion[index * 2 + 0] = FloatToHalf(inside ? sign * vx * scaleX : 0.0f);
            frame.motion[index * 2 + 1] = FloatToHalf(inside ? sign * vy * scaleY : 0.0f);
        }
    }
}
} // namespace

uint16_t FloatToHalf(float value)
{
    const uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    const int32_t shift = static_cast<int32_t>(exponent) - 127 + 15;
    if (shift <= 0)
    {
        if (shift < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const uint32_t shiftRight = static_cast<uint32_t>(14 - shift);
        uint32_t half = mantissa >> shiftRight;
        if (mantissa & (1u << (shiftRight - 1))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    if (shift >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half = sign | (static_cast<uint32_t>(shift) << 10) | (mantissa >> 13);
    if (mantissa & 0x1000u) ++half;
    return static_cast<uint16_t>(half);
}

void GenerateScene(const SceneOptions& options, SceneFrame& frame)
{
    frame.width = options.width;
    frame.height = options.height;
    frame.motionWidth = options.highResMotionVectors ? options.width : std::min(options.motionWidth, options.width);
    frame.motionHeight = options.highResMotionVectors ? options.height : std::min(options.motionHeight, options.height);
    frame.velocityX = options.staticScene ? 0.0f : options.velocityX;
    frame.velocityY = options.staticScene ? 0.0f : options.velocityY;

    RenderColor(options, static_cast<float>(options.frameIndex), frame.hudlessColor);
    RenderUi(options, frame.uiColor);
    Composite(frame.hudlessColor, frame.uiColor, frame.backbuffer);
    RenderColor(options, static_cast<float>(options.frameIndex) - 0.5f, frame.expectedMid);
    std::vector<uint8_t> midpointComposite;
    Composite(frame.expectedMid, frame.uiColor, midpointComposite);
    frame.expectedMid.swap(midpointComposite);
    RenderDepthAndMotion(options, frame);
}

bool WriteRawFile(const std::wstring& path, const void* data, size_t bytes)
{
    if (!data && bytes) return false;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    file.flush();
    return file.good();
}
} // namespace oracle
