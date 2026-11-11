#include "scene.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace
{
void Check(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}
}

int main()
{
    oracle::SceneOptions options;
    options.width = 640;
    options.height = 360;
    options.motionWidth = 320;
    options.motionHeight = 180;
    options.frameIndex = 4;
    options.ui = false;

    oracle::SceneFrame standard, smooth, repeat;
    oracle::GenerateScene(options, standard);
    options.smoothContent = true;
    oracle::GenerateScene(options, smooth);
    oracle::GenerateScene(options, repeat);
    Check(smooth.hudlessColor == repeat.hudlessColor, "smooth content is deterministic");
    Check(smooth.expectedMid == repeat.expectedMid, "smooth midpoint is deterministic");
    Check(smooth.hudlessColor != standard.hudlessColor, "smooth changes the color input");
    Check(smooth.depth == standard.depth, "smooth preserves geometry/depth");
    Check(smooth.motion == standard.motion, "smooth preserves motion vectors");
    Check(smooth.backbuffer == smooth.hudlessColor, "UI-off presents the hudless color");
    Check(smooth.expectedMid != smooth.backbuffer, "moving smooth midpoint differs from current frame");

    options.ui = true;
    oracle::GenerateScene(options, repeat);
    Check(repeat.hudlessColor == smooth.hudlessColor, "UI does not change scene content");
    Check(repeat.backbuffer != smooth.backbuffer, "UI-on adds the overlay");
    Check(repeat.motion == smooth.motion, "UI does not change motion vectors");

    options.staticScene = true;
    oracle::GenerateScene(options, repeat);
    Check(repeat.expectedMid == repeat.backbuffer, "static smooth midpoint equals current frame");
    for (uint16_t value : repeat.motion)
        Check((value & 0x7fff) == 0, "static scene has zero motion");
    options.frameIndex = 5;
    oracle::SceneFrame next;
    oracle::GenerateScene(options, next);
    Check(next.backbuffer == repeat.backbuffer, "static content does not change across frames");

    oracle::SceneOptions defaults;
    Check(!defaults.animateUi, "animateUI is off by default");

    // The default path must remain frame-invariant for UI: no hidden animation.
    oracle::SceneOptions uiOptions;
    uiOptions.width = 640;
    uiOptions.height = 360;
    uiOptions.motionWidth = 320;
    uiOptions.motionHeight = 180;
    uiOptions.staticScene = true;
    uiOptions.ui = true;
    uiOptions.frameIndex = 4;
    oracle::SceneFrame defaultUiA, defaultUiB;
    oracle::GenerateScene(uiOptions, defaultUiA);
    uiOptions.frameIndex = 5;
    oracle::GenerateScene(uiOptions, defaultUiB);
    Check(defaultUiA.uiColor == defaultUiB.uiColor, "default UI is frame-invariant");
    Check(defaultUiA.backbuffer == defaultUiB.backbuffer, "default static backbuffer is frame-invariant");

    // An animated UI changes only UI/backbuffer bytes; geometry, colour and
    // motion stay identical on a static scene.
    uiOptions.animateUi = true;
    uiOptions.frameIndex = 4;
    oracle::SceneFrame animatedUiA, sameFrameStatic;
    oracle::GenerateScene(uiOptions, animatedUiA);
    uiOptions.animateUi = false;
    oracle::GenerateScene(uiOptions, sameFrameStatic);
    Check(animatedUiA.hudlessColor == sameFrameStatic.hudlessColor, "animated UI preserves colour");
    Check(animatedUiA.depth == sameFrameStatic.depth, "animated UI preserves depth");
    Check(animatedUiA.motion == sameFrameStatic.motion, "animated UI preserves motion");
    Check(animatedUiA.uiColor != sameFrameStatic.uiColor, "animated UI changes the UI input");
    Check(animatedUiA.backbuffer != sameFrameStatic.backbuffer, "animated UI changes the backbuffer");

    uiOptions.animateUi = true;
    uiOptions.frameIndex = 5;
    oracle::SceneFrame animatedUiB;
    oracle::GenerateScene(uiOptions, animatedUiB);
    Check(animatedUiA.hudlessColor == animatedUiB.hudlessColor, "animated UI keeps colour across frames");
    Check(animatedUiA.depth == animatedUiB.depth, "animated UI keeps depth across frames");
    Check(animatedUiA.motion == animatedUiB.motion, "animated UI keeps motion across frames");
    Check(animatedUiA.uiColor != animatedUiB.uiColor, "animated UI differs across frames");
    Check(animatedUiA.backbuffer != animatedUiB.backbuffer, "animated backbuffer differs across frames");

    const std::filesystem::path rawPath =
        std::filesystem::temp_directory_path() / "xefg_scene_write_test.bin";
    const uint8_t payload[4] = {0xde, 0xad, 0xbe, 0xef};
    Check(oracle::WriteRawFile(rawPath.wstring(), payload, sizeof(payload)),
        "WriteRawFile writes a raw input");
    Check(std::filesystem::file_size(rawPath) == sizeof(payload), "WriteRawFile writes every byte");
    std::filesystem::remove(rawPath);
    const std::filesystem::path missingPath =
        std::filesystem::temp_directory_path() / "xefg_scene_write_missing_dir" / "input.bin";
    Check(!oracle::WriteRawFile(missingPath.wstring(), payload, sizeof(payload)),
        "WriteRawFile reports an unavailable destination");

    std::puts("scene tests passed: standard/smooth, static/motion, UI on/off, animated UI, raw writes");
    return 0;
}
