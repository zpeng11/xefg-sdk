#include "scene.h"

#include <cstdio>
#include <cstdlib>

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

    std::puts("scene tests passed: standard/smooth, static/motion, UI on/off");
    return 0;
}
