#pragma once
// Loads the profiled XeSS-FG runtime pair (libxess_fg 3.0.2 / libxell) and
// resolves the public exports the oracle drives. No MFGUnlocker, no patching:
// on a non-Intel GPU the public API generates at most one interpolated frame.
#include <Windows.h>
#include <xefg_swapchain.h>
#include <xefg_swapchain_d3d12.h>
#include <xefg_swapchain_debug.h>
#include <xell.h>
#include <xell_d3d12.h>

#include <mutex>
#include <string>
#include <vector>

namespace oracle
{
#define ORACLE_XEFG_FUNCTIONS(X)                                        \
    X(xefgSwapChainGetVersion)                                          \
    X(xefgSwapChainD3D12CreateContext)                                  \
    X(xefgSwapChainD3D12GetProperties)                                  \
    X(xefgSwapChainD3D12InitFromSwapChainDesc)                          \
    X(xefgSwapChainD3D12InitFromSwapChain)                              \
    X(xefgSwapChainD3D12GetSwapChainPtr)                                \
    X(xefgSwapChainSetLatencyReduction)                                 \
    X(xefgSwapChainSetNumInterpolatedFrames)                            \
    X(xefgSwapChainSetEnabled)                                          \
    X(xefgSwapChainSetPresentId)                                        \
    X(xefgSwapChainSetUiCompositionState)                               \
    X(xefgSwapChainD3D12TagFrameResource)                               \
    X(xefgSwapChainTagFrameConstants)                                   \
    X(xefgSwapChainGetLastPresentStatus)                                \
    X(xefgSwapChainSetLoggingCallback)                                  \
    X(xefgSwapChainEnableDebugFeature)                                  \
    X(xefgSwapChainDestroy)

#define ORACLE_XELL_FUNCTIONS(X)                                        \
    X(xellD3D12CreateContext)                                           \
    X(xellSetSleepMode)                                                 \
    X(xellSleep)                                                        \
    X(xellDestroyContext)

struct Runtime
{
    HMODULE fg = nullptr;
    HMODULE ll = nullptr;

#define ORACLE_DECLARE_FN(name) decltype(&::name) name = nullptr;
    ORACLE_XEFG_FUNCTIONS(ORACLE_DECLARE_FN)
    ORACLE_XELL_FUNCTIONS(ORACLE_DECLARE_FN)
#undef ORACLE_DECLARE_FN

    std::mutex logMutex;
    std::vector<std::string> logLines;

    bool Load(const std::wstring& directory, std::string& error);
    void Unload();
    std::vector<std::string> TakeLog();

    // XeFG logging callback target (thread-safe).
    void AppendLog(const char* message, int level);
};

// The one libxess_fg build the probe's offsets and captures are calibrated
// against: XeSS SDK 3.0.2, bin/libxess_fg.dll.
inline constexpr const char* kProfiledFg78Sha256 =
    "ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27";

// Family ("fg78"/"fg68"/"fg95") detected by the repository's witness tables,
// or empty when the loaded DLL is not one of the profiled releases.
std::string DetectFamily(HMODULE fgModule, std::string& error);

// SHA-256 of a loaded module's on-disk image.
std::string ModuleFileSha256(HMODULE module);
} // namespace oracle
