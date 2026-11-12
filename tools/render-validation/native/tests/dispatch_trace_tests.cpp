// Unit tests for the optional tensor-snapshot frame predicate.
// These do not create a device, provider context or tensor dump.

#include "dispatch_trace.h"

#include <cstdint>
#include <cstdio>
#include <string>

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
} // namespace

int main()
{
    using probe::DispatchTrace;
    using probe::DispatchTraceOptions;
    constexpr uint32_t kMaxFrame = UINT32_MAX;

    {
        const DispatchTraceOptions options;
        Check(DispatchTrace::TensorFrameSelected(options, 0),
            "default options select the first frame");
        Check(DispatchTrace::TensorFrameSelected(options, 8),
            "default options select frame 8");
        Check(DispatchTrace::TensorFrameSelected(options, kMaxFrame),
            "default options select the maximum frame");
    }

    {
        DispatchTraceOptions options;
        options.tensorFirstFrame = 8;
        options.tensorLastFrame = 8;
        Check(!DispatchTrace::TensorFrameSelected(options, 7), "single frame rejects earlier frame");
        Check(DispatchTrace::TensorFrameSelected(options, 8), "single frame accepts selected frame");
        Check(!DispatchTrace::TensorFrameSelected(options, 9), "single frame rejects later frame");
    }

    {
        DispatchTraceOptions options;
        options.tensorFirstFrame = 4;
        Check(!DispatchTrace::TensorFrameSelected(options, 3), "first bound rejects lower frame");
        Check(DispatchTrace::TensorFrameSelected(options, 4), "first bound accepts boundary frame");
        Check(DispatchTrace::TensorFrameSelected(options, kMaxFrame), "first bound accepts later frame");
    }

    {
        DispatchTraceOptions options;
        options.tensorLastFrame = 6;
        Check(DispatchTrace::TensorFrameSelected(options, 0), "last bound accepts earlier frame");
        Check(DispatchTrace::TensorFrameSelected(options, 6), "last bound accepts boundary frame");
        Check(!DispatchTrace::TensorFrameSelected(options, 7), "last bound rejects later frame");
    }

    {
        DispatchTraceOptions options;
        options.tensorFirstFrame = 5;
        options.tensorLastFrame = 3;
        Check(!DispatchTrace::TensorFrameSelected(options, 0), "inverted range rejects low frame");
        Check(!DispatchTrace::TensorFrameSelected(options, 4), "inverted range rejects middle frame");
        Check(!DispatchTrace::TensorFrameSelected(options, 5), "inverted range rejects high frame");
    }

    {
        DispatchTraceOptions options;
        options.tensorFirstFrame = 0;
        options.tensorLastFrame = kMaxFrame;
        Check(DispatchTrace::TensorFrameSelected(options, 0), "full range accepts first frame");
        Check(DispatchTrace::TensorFrameSelected(options, 8), "full range accepts frame 8");
        Check(DispatchTrace::TensorFrameSelected(options, kMaxFrame), "full range accepts last frame");
    }

    if (g_failures)
    {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("all dispatch trace tests passed\n");
    return 0;
}
