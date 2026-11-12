#pragma once
// Real DX12 renderer for XeFG render validation.  The main/CMake harness owns
// the device, swap chain and official XeSS-FG context; this class owns only
// the scene targets and a small synchronous command list/fence.
//
// Public targets after Render()/Readback() are always COPY_SOURCE:
//   Color/Ui/Backbuffer width x height BGRA8
//   Depth width/2 x height/2 R32_FLOAT (normal Z, no jitter)
//   Motion width/2 x height/2 R16G16_FLOAT (current -> previous, motion pixels)
// The output size defaults to 640x360; the geometry targets are always half
// size.  All supported output sizes are 16:9.
//
// Midpoint Render draws frame - 0.5 into independent targets using the current
// application frame in the UI strip and never advances current-frame state.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace render_validation
{
constexpr uint32_t kColorWidth = 640;
constexpr uint32_t kColorHeight = 360;
constexpr uint32_t kGeometryWidth = kColorWidth / 2;
constexpr uint32_t kGeometryHeight = kColorHeight / 2;

// Output (color/UI/backbuffer) size.  Depth and motion targets are always
// half size in each dimension.
struct RenderSize
{
    uint32_t width = kColorWidth;
    uint32_t height = kColorHeight;

    constexpr uint32_t GeometryWidth() const { return width / 2; }
    constexpr uint32_t GeometryHeight() const { return height / 2; }
};
constexpr uint32_t kWarmupFrames = 4;
constexpr uint32_t kMeasuredFrames = 8;
constexpr uint32_t kTotalFrames = kWarmupFrames + kMeasuredFrames;
constexpr uint32_t kCutFrame = 7;

constexpr DXGI_FORMAT kColorFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
constexpr DXGI_FORMAT kDepthFormat = DXGI_FORMAT_R32_FLOAT;
constexpr DXGI_FORMAT kMotionFormat = DXGI_FORMAT_R16G16_FLOAT;

enum class SceneKind : uint32_t
{
    Translation = 0,
    Occlusion = 1,
    Camera = 2,
    Cut = 3,
    DepthLayers = 4,
    ScreenCross = 5,
    CameraTurn = 6,
    // Opt-in scenes.  Appended so the original scene numbers stay stable.
    Nonrigid = 7,
    Transparency = 8,
};

constexpr size_t kSceneCount = 9;
extern const char* const kSceneNames[kSceneCount];

bool IsKnownScene(const std::string& scene);
bool SceneFromName(const std::string& name, SceneKind& scene);
bool IsCutResetFrame(uint32_t frame, SceneKind scene);

// An isolated source pair ends at pairFrame and contains pairFrame - 1 and
// pairFrame.  The camera cut is a discontinuity between source frame 6 and 7,
// so a pair ending at frame 7 would span the cut and is rejected.
bool IsValidPairFrame(SceneKind scene, uint32_t pairFrame);

// Row-major, m[row * 4 + column], consumed as p_row * M.
struct Matrix4
{
    float m[16];
};

Matrix4 IdentityMatrix();
Matrix4 Multiply(const Matrix4& left, const Matrix4& right);
Matrix4 MakeTranslation(float x, float y, float z);
Matrix4 MakeScale(float x, float y, float z);
Matrix4 MakeLookAt(const float eye[3], const float target[3], const float up[3]);
Matrix4 MakePerspective(float fovYRadians, float aspect, float nearZ, float farZ);

enum class MeshKind : uint32_t
{
    Cube = 0,
    Ground = 1,
    Rod = 2,
    // Subdivided xy grid used by the nonrigid scene.  Its shader entry
    // displaces local z with the analytic wave in Nonrigid* below.
    Grid = 3,
    // Flat quad [-1,1]^2 used by the transparency scene and drawn with the
    // dedicated flat-color entry (no lighting or checker).
    Panel = 4,
};

const char* MeshName(MeshKind mesh);

struct ObjectState
{
    std::string name;
    std::string role;
    MeshKind mesh = MeshKind::Cube;
    Matrix4 world = IdentityMatrix();
    Matrix4 previousWorld = IdentityMatrix();
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    // Transparent objects are drawn after opaque ones, back to front, with
    // premultiplied-alpha blending, depth test on and depth write off.  They
    // never contribute to the half-resolution depth/motion pass.
    bool transparent = false;
    bool depthWrite = true;
    bool writesDepthMotion = true;
    int32_t sortIndex = 0;
};

constexpr size_t kMaxObjects = 8;

// Nonrigid scene contract.  The same constants are emitted in scene.json and
// consumed by the dedicated deformation shader entry.  Grid vertices are
// local (x,y) in [-1,1] with index = iy * (cells + 1) + ix; world positions
// are p = (scaleX * x, scaleY * y, z(x,y,t) + translateZ).
constexpr uint32_t kNonrigidGridCells = 64;
constexpr uint32_t kNonrigidGridVertices = kNonrigidGridCells + 1;
constexpr float kNonrigidScaleX = 2.7f;
constexpr float kNonrigidScaleY = 1.5f;
constexpr float kNonrigidTranslateZ = -4.2f;
constexpr float kNonrigidPrimaryAmplitude = 0.34f;
constexpr float kNonrigidPrimaryKx = 2.1f;
constexpr float kNonrigidPrimaryKy = 1.6f;
constexpr float kNonrigidPrimaryOmegaX = 1.20f;
constexpr float kNonrigidPrimaryOmegaY = 0.75f;
constexpr float kNonrigidSecondaryAmplitude = 0.10f;
constexpr float kNonrigidSecondaryK = 1.3f;
constexpr float kNonrigidSecondaryOmega = 0.90f;
constexpr float kNonrigidSecondaryPhase = 0.40f;

float NonrigidDisplacement(float localX, float localY, float time);
void NonrigidNormalLocal(float localX, float localY, float time, float out[3]);
// Analytic world position for one local grid coordinate at a given scene time.
void NonrigidWorldPosition(float localX, float localY, float time, float out[3]);

struct FrameState
{
    SceneKind scene = SceneKind::Translation;
    // Rendering source identity.  In isolated-pair mode this is the real scene
    // frame (for example 4 and 5); the submission frame id used for tags,
    // present status and capture labels is a separate local index (0 and 1).
    uint32_t sourceFrame = 0;
    uint32_t frame = 0;
    bool midpoint = false;
    bool historyReset = false;
    // Render UI into the UI target and composite it onto the backbuffer.  Mode
    // 1 disables UI, making the backbuffer byte-identical to the hudless color.
    bool uiEnabled = true;
    float time = 0.0f;
    float previousTime = 0.0f;

    Matrix4 view;
    Matrix4 previousView;
    Matrix4 projection;
    Matrix4 previousProjection;
    Matrix4 viewProjection;
    Matrix4 previousViewProjection;

    uint32_t uiFrameId = 0;

    ObjectState objects[kMaxObjects];
    uint32_t objectCount = 0;
};

// Pure: builds the deterministic state for one application frame.  midpoint
// uses current time = frame - 0.5 but uiFrameId = frame.
FrameState MakeFrameState(SceneKind scene, uint32_t frame, bool midpoint = false);

// Pure: state for one source frame of a two-frame isolated pair.  The first
// local submission (source frame pairFrame - 1) always resets history; the
// second keeps contiguous history from the first.  sourceFrame (and time) stay
// the real scene frame so metadata can record the actual source times.
FrameState MakeIsolatedPairFrameState(SceneKind scene, uint32_t sourceFrame,
    bool firstOfPair, bool midpoint = false);

class Renderer
{
public:
    Renderer();
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;

    // shaderPath may be scene.hlsl or the directory containing it.  renderSize
    // defaults to 640x360; geometry targets are always half of it.
    bool Initialize(ID3D12Device* device,
                    ID3D12CommandQueue* queue,
                    const std::filesystem::path& shaderPath,
                    std::string& error,
                    RenderSize renderSize = {});

    // state may be produced by MakeFrameState; main may override historyReset.
    // midpoint renders into independent targets and leaves current state alone.
    bool Render(const FrameState& state,
                SceneKind scene,
                bool midpoint,
                std::string& error);

    bool WaitForGpu(std::string& error);

    ID3D12Resource* Color() const;
    ID3D12Resource* Ui() const;
    ID3D12Resource* Backbuffer() const;
    ID3D12Resource* Depth() const;
    ID3D12Resource* Motion() const;

    const FrameState& State() const;
    const RenderSize& Size() const;
    bool MidpointActive() const;
    bool Ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Copies a buffer or 2D texture to tightly packed CPU bytes with a short-lived
// DIRECT queue + fence. The resource must be COPY_SOURCE and idle; using a
// COPY queue here would require a COMMON-state queue-type handoff.
bool Readback(ID3D12Resource* resource, std::vector<uint8_t>& pixels, std::string& error);
std::vector<uint8_t> Readback(ID3D12Resource* resource, std::string& error);

} // namespace render_validation
