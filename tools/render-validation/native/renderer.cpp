#include "renderer.h"

#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace render_validation
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kCubeSize = 1.6f;

std::string HrText(const char* what, HRESULT hr)
{
    char text[160]{};
    std::snprintf(text, sizeof(text), "%s failed (hr=0x%08lx)", what,
        static_cast<unsigned long>(hr));
    return text;
}

std::string BlobText(ID3DBlob* blob)
{
    if (!blob || !blob->GetBufferPointer()) return {};
    return std::string(static_cast<const char*>(blob->GetBufferPointer()), blob->GetBufferSize());
}

struct Vertex
{
    float position[3];
    float normal[3];
};

struct MeshRange
{
    UINT indexOffset = 0;
    UINT indexCount = 0;
};

struct GpuObjectConstants
{
    float model[16];
    float previousModel[16];
    float color[4];
    float params[4];
};

struct GpuFrameConstants
{
    float viewProjection[16];
    float previousViewProjection[16];
    float viewport[4];
    float frameData[4];
};

static_assert(sizeof(GpuObjectConstants) == 160, "object constants must be 40 dwords");
static_assert(sizeof(GpuFrameConstants) == 160, "frame constants must be 40 dwords");

Matrix4 Mat(float m00, float m01, float m02, float m03,
            float m10, float m11, float m12, float m13,
            float m20, float m21, float m22, float m23,
            float m30, float m31, float m32, float m33)
{
    Matrix4 result{};
    result.m[0] = m00; result.m[1] = m01; result.m[2] = m02; result.m[3] = m03;
    result.m[4] = m10; result.m[5] = m11; result.m[6] = m12; result.m[7] = m13;
    result.m[8] = m20; result.m[9] = m21; result.m[10] = m22; result.m[11] = m23;
    result.m[12] = m30; result.m[13] = m31; result.m[14] = m32; result.m[15] = m33;
    return result;
}

void AppendBox(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
    float hx, float hy, float hz, MeshRange& range)
{
    range.indexOffset = static_cast<UINT>(indices.size());
    const uint32_t base = static_cast<uint32_t>(vertices.size());

    const auto face = [&](float ax, float ay, float az,
                          float bx, float by, float bz,
                          float cx, float cy, float cz,
                          float dx, float dy, float dz,
                          float nx, float ny, float nz) {
        const uint32_t start = static_cast<uint32_t>(vertices.size());
        vertices.push_back({{ax, ay, az}, {nx, ny, nz}});
        vertices.push_back({{bx, by, bz}, {nx, ny, nz}});
        vertices.push_back({{cx, cy, cz}, {nx, ny, nz}});
        vertices.push_back({{dx, dy, dz}, {nx, ny, nz}});
        indices.push_back(start + 0); indices.push_back(start + 1); indices.push_back(start + 2);
        indices.push_back(start + 0); indices.push_back(start + 2); indices.push_back(start + 3);
    };

    // Counter-clockwise when seen from outside.
    face( hx, -hy,  hz,  hx, -hy, -hz,  hx,  hy, -hz,  hx,  hy,  hz,  1, 0, 0);
    face(-hx, -hy, -hz, -hx, -hy,  hz, -hx,  hy,  hz, -hx,  hy, -hz, -1, 0, 0);
    face(-hx,  hy,  hz,  hx,  hy,  hz,  hx,  hy, -hz, -hx,  hy, -hz,  0, 1, 0);
    face(-hx, -hy, -hz,  hx, -hy, -hz,  hx, -hy,  hz, -hx, -hy,  hz,  0, -1, 0);
    face(-hx, -hy,  hz,  hx, -hy,  hz,  hx,  hy,  hz, -hx,  hy,  hz,  0, 0, 1);
    face( hx, -hy, -hz, -hx, -hy, -hz, -hx,  hy, -hz,  hx,  hy, -hz,  0, 0, -1);

    range.indexCount = static_cast<UINT>(indices.size()) - range.indexOffset;
    (void)base;
}

struct CameraPose
{
    float eye[3];
    float target[3];
};

CameraPose CameraFor(SceneKind scene, float time)
{
    CameraPose pose{};
    switch (scene)
    {
    case SceneKind::Translation:
        pose.eye[0] = 0.0f; pose.eye[1] = 1.60f; pose.eye[2] = 6.0f;
        pose.target[0] = 0.0f; pose.target[1] = 0.10f; pose.target[2] = 0.0f;
        break;
    case SceneKind::Occlusion:
        pose.eye[0] = 0.65f; pose.eye[1] = 1.50f; pose.eye[2] = 5.0f;
        pose.target[0] = 0.0f; pose.target[1] = 0.10f; pose.target[2] = 0.0f;
        break;
    case SceneKind::Camera:
        pose.eye[0] = -1.8f + 0.30f * time;
        pose.eye[1] = 1.50f + 0.10f * std::sin(time * 0.35f);
        pose.eye[2] = 6.0f - 0.05f * time;
        pose.target[0] = 0.45f * std::sin(time * 0.18f);
        pose.target[1] = 0.10f;
        pose.target[2] = 0.0f;
        break;
    case SceneKind::Cut:
    default:
        if (time < static_cast<float>(kCutFrame))
        {
            pose.eye[0] = -2.7f; pose.eye[1] = 1.50f; pose.eye[2] = 5.2f;
            pose.target[0] = 0.0f; pose.target[1] = 0.10f; pose.target[2] = 0.0f;
        }
        else
        {
            pose.eye[0] = 2.9f; pose.eye[1] = 1.80f; pose.eye[2] = 4.8f;
            pose.target[0] = 0.20f; pose.target[1] = 0.20f; pose.target[2] = 0.0f;
        }
        break;
    }
    return pose;
}

void CubePosition(SceneKind scene, float time, float position[3])
{
    switch (scene)
    {
    case SceneKind::Translation:
        position[0] = -2.2f + 0.45f * time;
        position[1] = -0.2f;
        position[2] = 0.0f;
        break;
    case SceneKind::Occlusion:
        position[0] = 2.4f - 0.42f * time;
        position[1] = -0.2f;
        position[2] = -0.9f;
        break;
    case SceneKind::Camera:
        position[0] = 0.35f * std::sin(time * 0.28f);
        position[1] = -0.2f;
        position[2] = 0.0f;
        break;
    case SceneKind::Cut:
    default:
        position[0] = 0.0f;
        position[1] = -0.2f;
        position[2] = 0.0f;
        break;
    }
}

float RodDepth(SceneKind scene)
{
    switch (scene)
    {
    case SceneKind::Translation: return 1.6f;
    case SceneKind::Occlusion: return 0.9f;
    case SceneKind::Camera: return 1.5f;
    case SceneKind::Cut:
    default: return 1.3f;
    }
}

Matrix4 CubeWorld(SceneKind scene, float time)
{
    float p[3];
    CubePosition(scene, time, p);
    return Multiply(MakeScale(kCubeSize, kCubeSize, kCubeSize), MakeTranslation(p[0], p[1], p[2]));
}

void AddObject(FrameState& state, const char* name, MeshKind mesh, const Matrix4& world,
    const Matrix4& previousWorld, const float color[4])
{
    if (state.objectCount >= kMaxObjects) return;
    ObjectState& object = state.objects[state.objectCount++];
    object.name = name;
    object.mesh = mesh;
    object.world = world;
    object.previousWorld = previousWorld;
    std::memcpy(object.color, color, sizeof(object.color));
}

D3D12_VIEWPORT Viewport(UINT width, UINT height)
{
    D3D12_VIEWPORT viewport{};
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    return viewport;
}

D3D12_RECT Scissor(UINT width, UINT height)
{
    D3D12_RECT rect{};
    rect.right = static_cast<LONG>(width);
    rect.bottom = static_cast<LONG>(height);
    return rect;
}

UINT BytesPerPixel(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
        return 1;
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
        return 2;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT:
        return 4;
    case DXGI_FORMAT_R32G32_FLOAT:
        return 8;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return 16;
    default:
        return 0;
    }
}
} // namespace

const char* const kSceneNames[kSceneCount] = {
    "translation", "occlusion", "camera", "cut",
};

bool IsKnownScene(const std::string& scene)
{
    for (const char* name : kSceneNames)
        if (scene == name) return true;
    return false;
}

bool SceneFromName(const std::string& name, SceneKind& scene)
{
    for (size_t i = 0; i < kSceneCount; ++i)
    {
        if (name == kSceneNames[i])
        {
            scene = static_cast<SceneKind>(i);
            return true;
        }
    }
    return false;
}

bool IsCutResetFrame(uint32_t frame, SceneKind scene)
{
    return scene == SceneKind::Cut && frame == kCutFrame;
}

Matrix4 IdentityMatrix()
{
    return Mat(1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}

Matrix4 Multiply(const Matrix4& left, const Matrix4& right)
{
    Matrix4 result{};
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += left.m[row * 4 + k] * right.m[k * 4 + column];
            result.m[row * 4 + column] = sum;
        }
    }
    return result;
}

Matrix4 MakeTranslation(float x, float y, float z)
{
    Matrix4 result = IdentityMatrix();
    // Row-vector convention: translation lives in the last row.
    result.m[12] = x;
    result.m[13] = y;
    result.m[14] = z;
    return result;
}

Matrix4 MakeScale(float x, float y, float z)
{
    Matrix4 result = IdentityMatrix();
    result.m[0] = x;
    result.m[5] = y;
    result.m[10] = z;
    return result;
}

Matrix4 MakeLookAt(const float eye[3], const float target[3], const float up[3])
{
    float zaxis[3] = {eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]};
    float zlength = std::sqrt(zaxis[0] * zaxis[0] + zaxis[1] * zaxis[1] + zaxis[2] * zaxis[2]);
    if (zlength < 1.0e-6f) zlength = 1.0f;
    zaxis[0] /= zlength; zaxis[1] /= zlength; zaxis[2] /= zlength;

    float xaxis[3] = {
        up[1] * zaxis[2] - up[2] * zaxis[1],
        up[2] * zaxis[0] - up[0] * zaxis[2],
        up[0] * zaxis[1] - up[1] * zaxis[0],
    };
    float xlength = std::sqrt(xaxis[0] * xaxis[0] + xaxis[1] * xaxis[1] + xaxis[2] * xaxis[2]);
    if (xlength < 1.0e-6f) xlength = 1.0f;
    xaxis[0] /= xlength; xaxis[1] /= xlength; xaxis[2] /= xlength;

    const float yaxis[3] = {
        zaxis[1] * xaxis[2] - zaxis[2] * xaxis[1],
        zaxis[2] * xaxis[0] - zaxis[0] * xaxis[2],
        zaxis[0] * xaxis[1] - zaxis[1] * xaxis[0],
    };

    return Mat(
        xaxis[0], yaxis[0], zaxis[0], 0.0f,
        xaxis[1], yaxis[1], zaxis[1], 0.0f,
        xaxis[2], yaxis[2], zaxis[2], 0.0f,
        -(xaxis[0] * eye[0] + xaxis[1] * eye[1] + xaxis[2] * eye[2]),
        -(yaxis[0] * eye[0] + yaxis[1] * eye[1] + yaxis[2] * eye[2]),
        -(zaxis[0] * eye[0] + zaxis[1] * eye[1] + zaxis[2] * eye[2]),
        1.0f);
}

Matrix4 MakePerspective(float fovYRadians, float aspect, float nearZ, float farZ)
{
    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    // Right-handed view (camera looks -Z), row-vector, D3D normal-Z:
    // w_clip = -z_view; row 2 stores A and -1, row 3 stores B.
    return Mat(
        f / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, -farZ / (farZ - nearZ), -1.0f,
        0.0f, 0.0f, -(farZ * nearZ) / (farZ - nearZ), 0.0f);
}

FrameState MakeFrameState(SceneKind scene, uint32_t frame, bool midpoint)
{
    FrameState state{};
    state.scene = scene;
    state.frame = frame;
    state.midpoint = midpoint;
    state.time = midpoint ? static_cast<float>(frame) - 0.5f : static_cast<float>(frame);
    state.historyReset = frame == 0 || IsCutResetFrame(frame, scene);
    state.previousTime = state.historyReset
        ? state.time
        : (midpoint ? static_cast<float>(frame) - 1.0f : state.time - 1.0f);
    state.uiFrameId = frame;

    const CameraPose current = CameraFor(scene, state.time);
    const CameraPose previous = CameraFor(scene, state.previousTime);
    const float up[3] = {0.0f, 1.0f, 0.0f};
    const float fov = 50.0f * kPi / 180.0f;
    const float aspect = static_cast<float>(kColorWidth) / static_cast<float>(kColorHeight);

    state.view = MakeLookAt(current.eye, current.target, up);
    state.projection = MakePerspective(fov, aspect, 0.1f, 100.0f);
    state.previousView = state.historyReset
        ? state.view
        : MakeLookAt(previous.eye, previous.target, up);
    state.previousProjection = state.projection;
    state.viewProjection = Multiply(state.view, state.projection);
    state.previousViewProjection = Multiply(state.previousView, state.previousProjection);

    const float cubeColor[4] = {0.95f, 0.55f, 0.12f, 1.0f};
    const float groundColor[4] = {0.16f, 0.20f, 0.26f, 1.0f};
    const float rodColor[4] = {0.78f, 0.80f, 0.84f, 1.0f};

    const Matrix4 cubeCurrent = CubeWorld(scene, state.time);
    const Matrix4 cubePrevious = state.historyReset ? cubeCurrent : CubeWorld(scene, state.previousTime);
    AddObject(state, "cube", MeshKind::Cube, cubeCurrent, cubePrevious, cubeColor);

    const Matrix4 ground = MakeTranslation(0.0f, -1.1f, 0.0f);
    AddObject(state, "ground", MeshKind::Ground, ground, ground, groundColor);

    const float rodX[5] = {-2.4f, -1.2f, 0.0f, 1.2f, 2.4f};
    const float rodZ = RodDepth(scene);
    for (int i = 0; i < 5; ++i)
    {
        const Matrix4 rod = MakeTranslation(rodX[i], 0.3f, rodZ);
        char name[16]{};
        std::snprintf(name, sizeof(name), "rod%d", i);
        AddObject(state, name, MeshKind::Rod, rod, rod, rodColor);
    }
    return state;
}

struct Renderer::Impl
{
    struct TargetSet
    {
        ComPtr<ID3D12Resource> color;
        ComPtr<ID3D12Resource> ui;
        ComPtr<ID3D12Resource> backbuffer;
        ComPtr<ID3D12Resource> depth;
        ComPtr<ID3D12Resource> motion;
        UINT rtvBase = 0;
        UINT srvBase = 0;
    };

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> colorPipeline;
    ComPtr<ID3D12PipelineState> geometryPipeline;
    ComPtr<ID3D12PipelineState> uiPipeline;
    ComPtr<ID3D12PipelineState> compositePipeline;

    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    uint64_t fenceValue = 0;

    ComPtr<ID3D12Resource> frameConstantBuffer;
    void* frameConstantMapped = nullptr;

    ComPtr<ID3D12Resource> vertexBuffer;
    ComPtr<ID3D12Resource> indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexView{};
    D3D12_INDEX_BUFFER_VIEW indexView{};
    MeshRange cubeRange;
    MeshRange groundRange;
    MeshRange rodRange;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    ComPtr<ID3D12DescriptorHeap> srvHeap;
    UINT rtvDescriptorSize = 0;
    UINT srvDescriptorSize = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE colorDepthDsv{};
    D3D12_CPU_DESCRIPTOR_HANDLE geometryDepthDsv{};
    ComPtr<ID3D12Resource> colorDepth;
    ComPtr<ID3D12Resource> geometryDepth;

    TargetSet currentTargets;
    TargetSet midpointTargets;
    TargetSet* activeTargets = nullptr;

    FrameState state;
    bool midpointActive = false;
    bool ready = false;

    D3D12_CPU_DESCRIPTOR_HANDLE RtvCpu(UINT index) const
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(index) * rtvDescriptorSize;
        return handle;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE SrvCpu(UINT index) const
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = srvHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(index) * srvDescriptorSize;
        return handle;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE SrvGpu(UINT index) const
    {
        D3D12_GPU_DESCRIPTOR_HANDLE handle = srvHeap->GetGPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<UINT64>(index) * srvDescriptorSize;
        return handle;
    }

    ComPtr<ID3D12Resource> CreateTexture(UINT width, UINT height, DXGI_FORMAT format,
        D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, std::string& error)
    {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        description.Width = width;
        description.Height = height;
        description.DepthOrArraySize = 1;
        description.MipLevels = 1;
        description.Format = format;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        description.Flags = flags;

        ComPtr<ID3D12Resource> resource;
        HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &description, initialState, nullptr, IID_PPV_ARGS(&resource));
        if (FAILED(hr))
        {
            error = HrText("CreateCommittedResource(texture)", hr);
            return nullptr;
        }
        return resource;
    }

    ComPtr<ID3D12Resource> CreateUploadBuffer(const void* data, size_t bytes, std::string& error)
    {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = bytes;
        description.Height = 1;
        description.DepthOrArraySize = 1;
        description.MipLevels = 1;
        description.Format = DXGI_FORMAT_UNKNOWN;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        ComPtr<ID3D12Resource> resource;
        HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &description, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&resource));
        if (FAILED(hr))
        {
            error = HrText("CreateCommittedResource(upload)", hr);
            return nullptr;
        }

        void* mapped = nullptr;
        hr = resource->Map(0, nullptr, &mapped);
        if (FAILED(hr))
        {
            error = HrText("Map(upload)", hr);
            return nullptr;
        }
        std::memcpy(mapped, data, bytes);
        resource->Unmap(0, nullptr);
        return resource;
    }

    void CreateRtv(ID3D12Resource* resource, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE handle)
    {
        D3D12_RENDER_TARGET_VIEW_DESC description{};
        description.Format = format;
        description.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        description.Texture2D.MipSlice = 0;
        description.Texture2D.PlaneSlice = 0;
        device->CreateRenderTargetView(resource, &description, handle);
    }

    void CreateSrv(ID3D12Resource* resource, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE handle)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC description{};
        description.Format = format;
        description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        description.Texture2D.MostDetailedMip = 0;
        description.Texture2D.MipLevels = 1;
        description.Texture2D.PlaneSlice = 0;
        description.Texture2D.ResourceMinLODClamp = 0.0f;
        device->CreateShaderResourceView(resource, &description, handle);
    }

    bool CreateTargetSet(TargetSet& targets, UINT rtvBase, UINT srvBase, std::string& error)
    {
        targets.rtvBase = rtvBase;
        targets.srvBase = srvBase;
        targets.color = CreateTexture(kColorWidth, kColorHeight, kColorFormat,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE, error);
        targets.ui = CreateTexture(kColorWidth, kColorHeight, kColorFormat,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE, error);
        targets.backbuffer = CreateTexture(kColorWidth, kColorHeight, kColorFormat,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE, error);
        targets.depth = CreateTexture(kGeometryWidth, kGeometryHeight, kDepthFormat,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE, error);
        targets.motion = CreateTexture(kGeometryWidth, kGeometryHeight, kMotionFormat,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE, error);
        if (!targets.color || !targets.ui || !targets.backbuffer || !targets.depth || !targets.motion)
            return false;

        CreateRtv(targets.color.Get(), kColorFormat, RtvCpu(rtvBase + 0));
        CreateRtv(targets.ui.Get(), kColorFormat, RtvCpu(rtvBase + 1));
        CreateRtv(targets.backbuffer.Get(), kColorFormat, RtvCpu(rtvBase + 2));
        CreateRtv(targets.depth.Get(), kDepthFormat, RtvCpu(rtvBase + 3));
        CreateRtv(targets.motion.Get(), kMotionFormat, RtvCpu(rtvBase + 4));
        CreateSrv(targets.color.Get(), kColorFormat, SrvCpu(srvBase + 0));
        CreateSrv(targets.ui.Get(), kColorFormat, SrvCpu(srvBase + 1));
        return true;
    }

    bool CompileShader(const std::wstring& file, const char* entry, const char* target,
        ComPtr<ID3DBlob>& blob, std::string& error)
    {
        ComPtr<ID3DBlob> errors;
        const UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_PACK_MATRIX_ROW_MAJOR;
        HRESULT hr = D3DCompileFromFile(file.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entry, target, flags, 0, &blob, &errors);
        if (FAILED(hr))
        {
            error = std::string("shader ") + entry + ": " + BlobText(errors.Get());
            return false;
        }
        return true;
    }

    bool CreatePipeline(const std::filesystem::path& shaderPath, const char* vsEntry,
        const char* psEntry, const D3D12_INPUT_ELEMENT_DESC* layout, UINT layoutCount,
        DXGI_FORMAT rtv0, DXGI_FORMAT rtv1, DXGI_FORMAT dsvFormat, bool depth,
        D3D12_CULL_MODE cull, ComPtr<ID3D12PipelineState>& pipeline, std::string& error)
    {
        ComPtr<ID3DBlob> vertexShader;
        ComPtr<ID3DBlob> pixelShader;
        const std::wstring file = shaderPath.wstring();
        if (!CompileShader(file, vsEntry, "vs_5_1", vertexShader, error)) return false;
        if (!CompileShader(file, psEntry, "ps_5_1", pixelShader, error)) return false;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC description{};
        description.pRootSignature = rootSignature.Get();
        description.VS = {vertexShader->GetBufferPointer(), vertexShader->GetBufferSize()};
        description.PS = {pixelShader->GetBufferPointer(), pixelShader->GetBufferSize()};
        description.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        if (rtv1 != DXGI_FORMAT_UNKNOWN)
            description.BlendState.RenderTarget[1].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        description.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        description.RasterizerState.CullMode = cull;
        description.RasterizerState.FrontCounterClockwise = FALSE;
        description.RasterizerState.DepthClipEnable = TRUE;
        description.DepthStencilState.DepthEnable = depth ? TRUE : FALSE;
        description.DepthStencilState.DepthWriteMask = depth
            ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        description.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        description.DepthStencilState.StencilEnable = FALSE;
        description.SampleMask = UINT_MAX;
        description.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        description.NumRenderTargets = rtv1 == DXGI_FORMAT_UNKNOWN ? 1 : 2;
        description.RTVFormats[0] = rtv0;
        description.RTVFormats[1] = rtv1;
        description.DSVFormat = dsvFormat;
        description.SampleDesc.Count = 1;
        description.InputLayout.pInputElementDescs = layout;
        description.InputLayout.NumElements = layoutCount;

        HRESULT hr = device->CreateGraphicsPipelineState(&description, IID_PPV_ARGS(&pipeline));
        if (FAILED(hr))
        {
            error = HrText("CreateGraphicsPipelineState", hr);
            return false;
        }
        return true;
    }

    bool CreateRootSignature(std::string& error)
    {
        D3D12_ROOT_PARAMETER parameters[3]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.ShaderRegister = 0;
        parameters[0].Constants.RegisterSpace = 0;
        parameters[0].Constants.Num32BitValues = 40;
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        // Frame constants are a root CBV so the root signature stays under
        // the 64-DWORD limit (object constants 40 + frame CBV 1 + SRV table 1).
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[1].Descriptor.ShaderRegister = 1;
        parameters[1].Descriptor.RegisterSpace = 0;
        parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_DESCRIPTOR_RANGE range{};
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = 2;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = 0;
        range.OffsetInDescriptorsFromTableStart = 0;

        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[2].DescriptorTable.NumDescriptorRanges = 1;
        parameters[2].DescriptorTable.pDescriptorRanges = &range;
        parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC description{};
        description.NumParameters = 3;
        description.pParameters = parameters;
        description.NumStaticSamplers = 0;
        description.pStaticSamplers = nullptr;
        description.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> errors;
        HRESULT hr = D3D12SerializeRootSignature(&description, D3D_ROOT_SIGNATURE_VERSION_1_0,
            &signature, &errors);
        if (FAILED(hr))
        {
            error = std::string("D3D12SerializeRootSignature: ") + BlobText(errors.Get());
            return false;
        }
        hr = device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
            IID_PPV_ARGS(&rootSignature));
        if (FAILED(hr))
        {
            error = HrText("CreateRootSignature", hr);
            return false;
        }
        return true;
    }

    bool CreateMesh(std::string& error)
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        vertices.reserve(72);
        indices.reserve(108);
        AppendBox(vertices, indices, 0.5f, 0.5f, 0.5f, cubeRange);
        AppendBox(vertices, indices, 9.0f, 0.1f, 9.0f, groundRange);        AppendBox(vertices, indices, 0.045f, 1.3f, 0.045f, rodRange);

        vertexBuffer = CreateUploadBuffer(vertices.data(), vertices.size() * sizeof(Vertex), error);
        indexBuffer = CreateUploadBuffer(indices.data(), indices.size() * sizeof(uint32_t), error);
        if (!vertexBuffer || !indexBuffer) return false;

        vertexView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
        vertexView.SizeInBytes = static_cast<UINT>(vertices.size() * sizeof(Vertex));
        vertexView.StrideInBytes = sizeof(Vertex);

        indexView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
        indexView.SizeInBytes = static_cast<UINT>(indices.size() * sizeof(uint32_t));
        indexView.Format = DXGI_FORMAT_R32_UINT;
        return true;
    }

    bool CreateDescriptorHeaps(std::string& error)
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtvDescription{};
        rtvDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvDescription.NumDescriptors = 10;
        rtvDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        HRESULT hr = device->CreateDescriptorHeap(&rtvDescription, IID_PPV_ARGS(&rtvHeap));
        if (FAILED(hr)) { error = HrText("CreateDescriptorHeap(RTV)", hr); return false; }
        rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_DESCRIPTOR_HEAP_DESC dsvDescription{};
        dsvDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvDescription.NumDescriptors = 2;
        dsvDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        hr = device->CreateDescriptorHeap(&dsvDescription, IID_PPV_ARGS(&dsvHeap));
        if (FAILED(hr)) { error = HrText("CreateDescriptorHeap(DSV)", hr); return false; }

        D3D12_DESCRIPTOR_HEAP_DESC srvDescription{};
        srvDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvDescription.NumDescriptors = 4;
        srvDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = device->CreateDescriptorHeap(&srvDescription, IID_PPV_ARGS(&srvHeap));
        if (FAILED(hr)) { error = HrText("CreateDescriptorHeap(SRV)", hr); return false; }
        srvDescriptorSize = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return true;
    }

    bool CreateDepthTargets(std::string& error)
    {
        colorDepth = CreateTexture(kColorWidth, kColorHeight, DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, error);
        geometryDepth = CreateTexture(kGeometryWidth, kGeometryHeight, DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, error);
        if (!colorDepth || !geometryDepth) return false;

        D3D12_DEPTH_STENCIL_VIEW_DESC description{};
        description.Format = DXGI_FORMAT_D32_FLOAT;
        description.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        description.Flags = D3D12_DSV_FLAG_NONE;
        description.Texture2D.MipSlice = 0;

        colorDepthDsv = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        geometryDepthDsv = colorDepthDsv;
        geometryDepthDsv.ptr += device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        device->CreateDepthStencilView(colorDepth.Get(), &description, colorDepthDsv);
        device->CreateDepthStencilView(geometryDepth.Get(), &description, geometryDepthDsv);
        return true;
    }

    bool Initialize(ID3D12Device* inputDevice, ID3D12CommandQueue* inputQueue,
        const std::filesystem::path& shaderPath, std::string& error)
    {
        if (!inputDevice || !inputQueue)
        {
            error = "device and queue are required";
            return false;
        }

        std::filesystem::path resolvedShader = shaderPath;
        if (std::filesystem::is_directory(resolvedShader))
            resolvedShader /= "scene.hlsl";
        if (!std::filesystem::exists(resolvedShader))
        {
            error = "shader file not found: " + resolvedShader.string();
            return false;
        }

        device = inputDevice;
        queue = inputQueue;

        HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator));
        if (FAILED(hr)) { error = HrText("CreateCommandAllocator", hr); return false; }
        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
            nullptr, IID_PPV_ARGS(&commandList));
        if (FAILED(hr)) { error = HrText("CreateCommandList", hr); return false; }
        commandList->Close();

        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(hr)) { error = HrText("CreateFence", hr); return false; }
        fenceEvent = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
        if (!fenceEvent) { error = "CreateEventEx failed"; return false; }

        if (!CreateDescriptorHeaps(error)) return false;
        if (!CreateRootSignature(error)) return false;

        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
                D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,
                D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };
        if (!CreatePipeline(resolvedShader, "ColorVS", "ColorPS", layout, 2,
                kColorFormat, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_D32_FLOAT, true,
                D3D12_CULL_MODE_BACK, colorPipeline, error)) return false;
        if (!CreatePipeline(resolvedShader, "GeometryVS", "GeometryPS", layout, 2,
                kDepthFormat, kMotionFormat, DXGI_FORMAT_D32_FLOAT, true,
                D3D12_CULL_MODE_BACK, geometryPipeline, error)) return false;
        if (!CreatePipeline(resolvedShader, "FullscreenVS", "UiPS", nullptr, 0,
                kColorFormat, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, false,
                D3D12_CULL_MODE_NONE, uiPipeline, error)) return false;
        if (!CreatePipeline(resolvedShader, "FullscreenVS", "CompositePS", nullptr, 0,
                kColorFormat, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, false,
                D3D12_CULL_MODE_NONE, compositePipeline, error)) return false;

        if (!CreateDepthTargets(error)) return false;
        if (!CreateTargetSet(currentTargets, 0, 0, error)) return false;
        if (!CreateTargetSet(midpointTargets, 5, 2, error)) return false;
        if (!CreateMesh(error)) return false;

        const uint8_t zeroConstants[256]{};
        frameConstantBuffer = CreateUploadBuffer(zeroConstants, sizeof(zeroConstants), error);
        if (!frameConstantBuffer) return false;
        hr = frameConstantBuffer->Map(0, nullptr, &frameConstantMapped);
        if (FAILED(hr))
        {
            error = HrText("Map(frame constants)", hr);
            return false;
        }

        activeTargets = &currentTargets;
        ready = true;
        return true;
    }

    bool WaitForGpu(std::string& error)
    {
        if (!fence) return true;
        if (fence->GetCompletedValue() < fenceValue)
        {
            HRESULT hr = fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (FAILED(hr)) { error = HrText("SetEventOnCompletion", hr); return false; }
            if (WaitForSingleObject(fenceEvent, 30000) != WAIT_OBJECT_0)
            {
                error = "GPU wait timed out";
                return false;
            }
        }
        return true;
    }

    void Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after)
    {
        if (!resource || before == after) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        commandList->ResourceBarrier(1, &barrier);
    }

    void SetFrameConstants(const FrameState& frameState)
    {
        GpuFrameConstants constants{};
        std::memcpy(constants.viewProjection, frameState.viewProjection.m,
            sizeof(constants.viewProjection));
        std::memcpy(constants.previousViewProjection, frameState.previousViewProjection.m,
            sizeof(constants.previousViewProjection));
        constants.viewport[0] = static_cast<float>(kColorWidth);
        constants.viewport[1] = static_cast<float>(kColorHeight);
        constants.viewport[2] = static_cast<float>(kGeometryWidth);
        constants.viewport[3] = static_cast<float>(kGeometryHeight);
        constants.frameData[0] = static_cast<float>(frameState.uiFrameId);
        constants.frameData[1] = frameState.midpoint ? 1.0f : 0.0f;
        constants.frameData[2] = frameState.historyReset ? 1.0f : 0.0f;
        constants.frameData[3] = static_cast<float>(frameState.scene);
        std::memcpy(frameConstantMapped, &constants, sizeof(constants));
        commandList->SetGraphicsRootConstantBufferView(1,
            frameConstantBuffer->GetGPUVirtualAddress());
    }

    void DrawObjects(const FrameState& frameState, bool geometryPass)
    {
        for (uint32_t i = 0; i < frameState.objectCount; ++i)
        {
            const ObjectState& object = frameState.objects[i];
            GpuObjectConstants constants{};
            std::memcpy(constants.model, object.world.m, sizeof(constants.model));
            std::memcpy(constants.previousModel, object.previousWorld.m,
                sizeof(constants.previousModel));
            std::memcpy(constants.color, object.color, sizeof(constants.color));
            constants.params[0] = static_cast<float>(i);
            commandList->SetGraphicsRoot32BitConstants(0, 40, &constants, 0);

            const MeshRange* range = &cubeRange;
            if (object.mesh == MeshKind::Ground) range = &groundRange;
            else if (object.mesh == MeshKind::Rod) range = &rodRange;
            commandList->DrawIndexedInstanced(range->indexCount, 1, range->indexOffset, 0, 0);
        }
        (void)geometryPass;
    }

    bool RenderFrame(FrameState frameState, bool midpoint, std::string& error)
    {
        if (!ready || !activeTargets)
        {
            error = "renderer is not initialized";
            return false;
        }
        if (!WaitForGpu(error)) return false;

        // Honor an explicit history reset from the harness by zeroing motion.
        if (frameState.historyReset)
        {
            frameState.previousView = frameState.view;
            frameState.previousProjection = frameState.projection;
            frameState.previousViewProjection = frameState.viewProjection;
            for (uint32_t i = 0; i < frameState.objectCount; ++i)
                frameState.objects[i].previousWorld = frameState.objects[i].world;
        }

        TargetSet& targets = midpoint ? midpointTargets : currentTargets;
        activeTargets = &targets;

        HRESULT hr = allocator->Reset();
        if (FAILED(hr)) { error = HrText("CommandAllocator::Reset", hr); return false; }
        hr = commandList->Reset(allocator.Get(), nullptr);
        if (FAILED(hr)) { error = HrText("CommandList::Reset", hr); return false; }

        commandList->SetGraphicsRootSignature(rootSignature.Get());
        ID3D12DescriptorHeap* heaps[] = {srvHeap.Get()};
        commandList->SetDescriptorHeaps(1, heaps);
        commandList->SetGraphicsRootDescriptorTable(2, SrvGpu(targets.srvBase));

        Transition(targets.color.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        Transition(targets.ui.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        Transition(targets.backbuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        Transition(targets.depth.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        Transition(targets.motion.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET);

        SetFrameConstants(frameState);
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 1, &vertexView);
        commandList->IASetIndexBuffer(&indexView);

        const float colorClear[4] = {0.02f, 0.03f, 0.05f, 1.0f};
        const float depthClear[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        const float motionClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float uiClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Color pass at 640x360, depth tested with the full-resolution depth.
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtv = RtvCpu(targets.rtvBase + 0);
            commandList->OMSetRenderTargets(1, &rtv, FALSE, &colorDepthDsv);
            D3D12_VIEWPORT viewport = Viewport(kColorWidth, kColorHeight);
            D3D12_RECT scissor = Scissor(kColorWidth, kColorHeight);
            commandList->RSSetViewports(1, &viewport);
            commandList->RSSetScissorRects(1, &scissor);
            commandList->ClearRenderTargetView(rtv, colorClear, 0, nullptr);
            commandList->ClearDepthStencilView(colorDepthDsv, D3D12_CLEAR_FLAG_DEPTH,
                1.0f, 0, 0, nullptr);
            commandList->SetPipelineState(colorPipeline.Get());
            DrawObjects(frameState, false);
        }

        // Geometry pass at 320x180, writing normalised depth and motion.
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2] = {
                RtvCpu(targets.rtvBase + 3), RtvCpu(targets.rtvBase + 4)};
            commandList->OMSetRenderTargets(2, rtvs, FALSE, &geometryDepthDsv);
            D3D12_VIEWPORT viewport = Viewport(kGeometryWidth, kGeometryHeight);
            D3D12_RECT scissor = Scissor(kGeometryWidth, kGeometryHeight);
            commandList->RSSetViewports(1, &viewport);
            commandList->RSSetScissorRects(1, &scissor);
            commandList->ClearRenderTargetView(rtvs[0], depthClear, 0, nullptr);
            commandList->ClearRenderTargetView(rtvs[1], motionClear, 0, nullptr);
            commandList->ClearDepthStencilView(geometryDepthDsv, D3D12_CLEAR_FLAG_DEPTH,
                1.0f, 0, 0, nullptr);
            commandList->SetPipelineState(geometryPipeline.Get());
            DrawObjects(frameState, true);
        }

        // UI pass at 640x360: frame marker plus premultiplied-alpha crosshair.
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtv = RtvCpu(targets.rtvBase + 1);
            commandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            D3D12_VIEWPORT viewport = Viewport(kColorWidth, kColorHeight);
            D3D12_RECT scissor = Scissor(kColorWidth, kColorHeight);
            commandList->RSSetViewports(1, &viewport);
            commandList->RSSetScissorRects(1, &scissor);
            commandList->ClearRenderTargetView(rtv, uiClear, 0, nullptr);
            commandList->SetPipelineState(uiPipeline.Get());
            commandList->DrawInstanced(3, 1, 0, 0);
        }

        // GPU composite: backbuffer = ui + (1 - ui.a) * color.
        Transition(targets.color.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        Transition(targets.ui.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtv = RtvCpu(targets.rtvBase + 2);
            commandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            D3D12_VIEWPORT viewport = Viewport(kColorWidth, kColorHeight);
            D3D12_RECT scissor = Scissor(kColorWidth, kColorHeight);
            commandList->RSSetViewports(1, &viewport);
            commandList->RSSetScissorRects(1, &scissor);
            commandList->ClearRenderTargetView(rtv, colorClear, 0, nullptr);
            commandList->SetPipelineState(compositePipeline.Get());
            commandList->DrawInstanced(3, 1, 0, 0);
        }

        Transition(targets.color.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(targets.ui.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(targets.backbuffer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(targets.depth.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(targets.motion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE);

        hr = commandList->Close();
        if (FAILED(hr)) { error = HrText("CommandList::Close", hr); return false; }
        ID3D12CommandList* lists[] = {commandList.Get()};
        queue->ExecuteCommandLists(1, lists);
        hr = queue->Signal(fence.Get(), ++fenceValue);
        if (FAILED(hr)) { error = HrText("CommandQueue::Signal", hr); return false; }

        state = frameState;
        state.midpoint = midpoint;
        midpointActive = midpoint;
        return WaitForGpu(error);
    }
};

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}
Renderer::~Renderer() = default;
Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

bool Renderer::Initialize(ID3D12Device* device, ID3D12CommandQueue* queue,
    const std::filesystem::path& shaderPath, std::string& error)
{
    return impl_->Initialize(device, queue, shaderPath, error);
}

bool Renderer::Render(const FrameState& state, SceneKind scene, bool midpoint, std::string& error)
{
    FrameState frameState = state;
    if (frameState.scene != scene)
    {
        // Keep the harness-provided frame index; only normalise the scene tag.
        frameState.scene = scene;
    }
    return impl_->RenderFrame(frameState, midpoint, error);
}

bool Renderer::WaitForGpu(std::string& error)
{
    return impl_->WaitForGpu(error);
}

ID3D12Resource* Renderer::Color() const
{
    return impl_ && impl_->activeTargets ? impl_->activeTargets->color.Get() : nullptr;
}

ID3D12Resource* Renderer::Ui() const
{
    return impl_ && impl_->activeTargets ? impl_->activeTargets->ui.Get() : nullptr;
}

ID3D12Resource* Renderer::Backbuffer() const
{
    return impl_ && impl_->activeTargets ? impl_->activeTargets->backbuffer.Get() : nullptr;
}

ID3D12Resource* Renderer::Depth() const
{
    return impl_ && impl_->activeTargets ? impl_->activeTargets->depth.Get() : nullptr;
}

ID3D12Resource* Renderer::Motion() const
{
    return impl_ && impl_->activeTargets ? impl_->activeTargets->motion.Get() : nullptr;
}

const FrameState& Renderer::State() const
{
    return impl_->state;
}

bool Renderer::MidpointActive() const
{
    return impl_->midpointActive;
}

bool Renderer::Ready() const
{
    return impl_->ready;
}

bool Readback(ID3D12Resource* resource, std::vector<uint8_t>& pixels, std::string& error)
{
    pixels.clear();
    if (!resource)
    {
        error = "readback resource is null";
        return false;
    }

    ComPtr<ID3D12Device> device;
    HRESULT hr = resource->GetDevice(IID_PPV_ARGS(&device));
    if (FAILED(hr))
    {
        error = HrText("ID3D12Resource::GetDevice", hr);
        return false;
    }

    const D3D12_RESOURCE_DESC description = resource->GetDesc();

    D3D12_RESOURCE_DESC bufferDescription{};
    bufferDescription.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDescription.Height = 1;
    bufferDescription.DepthOrArraySize = 1;
    bufferDescription.MipLevels = 1;
    bufferDescription.Format = DXGI_FORMAT_UNKNOWN;
    bufferDescription.SampleDesc.Count = 1;
    bufferDescription.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    UINT64 totalBytes = 0;
    UINT rowCount = 0;
    UINT64 rowPitch = 0;
    UINT64 tightRowBytes = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};

    if (description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        totalBytes = description.Width;
    }
    else if (description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
    {
        const UINT bytesPerPixel = BytesPerPixel(description.Format);
        if (!bytesPerPixel)
        {
            error = "unsupported readback texture format";
            return false;
        }
        UINT64 rowSize = 0;
        device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rowCount,
            &rowSize, &totalBytes);
        rowPitch = rowSize;
        tightRowBytes = static_cast<UINT64>(description.Width) * bytesPerPixel;
    }
    else
    {
        error = "unsupported readback resource dimension";
        return false;
    }

    bufferDescription.Width = totalBytes;

    D3D12_HEAP_PROPERTIES readbackHeap{};
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> readbackBuffer;
    hr = device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE,
        &bufferDescription, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readbackBuffer));
    if (FAILED(hr)) { error = HrText("CreateCommittedResource(readback)", hr); return false; }

    D3D12_COMMAND_QUEUE_DESC queueDescription{};
    queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDescription.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ComPtr<ID3D12CommandQueue> copyQueue;
    hr = device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&copyQueue));
    if (FAILED(hr)) { error = HrText("CreateCommandQueue(copy)", hr); return false; }

    ComPtr<ID3D12CommandAllocator> copyAllocator;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&copyAllocator));
    if (FAILED(hr)) { error = HrText("CreateCommandAllocator(copy)", hr); return false; }

    ComPtr<ID3D12GraphicsCommandList> copyList;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copyAllocator.Get(),
        nullptr, IID_PPV_ARGS(&copyList));
    if (FAILED(hr)) { error = HrText("CreateCommandList(copy)", hr); return false; }

    if (description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        copyList->CopyBufferRegion(readbackBuffer.Get(), 0, resource, 0, totalBytes);
    }
    else
    {
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readbackBuffer.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = footprint;

        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = resource;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = 0;

        D3D12_BOX box{};
        box.right = static_cast<UINT>(description.Width);
        box.bottom = description.Height;
        box.back = 1;
        copyList->CopyTextureRegion(&destination, 0, 0, 0, &source, &box);
    }

    hr = copyList->Close();
    if (FAILED(hr)) { error = HrText("CommandList::Close(copy)", hr); return false; }
    ID3D12CommandList* lists[] = {copyList.Get()};
    copyQueue->ExecuteCommandLists(1, lists);

    ComPtr<ID3D12Fence> copyFence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copyFence));
    if (FAILED(hr)) { error = HrText("CreateFence(copy)", hr); return false; }
    HANDLE event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    if (!event) { error = "CreateEventEx(copy) failed"; return false; }
    hr = copyQueue->Signal(copyFence.Get(), 1);
    if (FAILED(hr)) { CloseHandle(event); error = HrText("Signal(copy)", hr); return false; }
    if (copyFence->GetCompletedValue() < 1)
    {
        hr = copyFence->SetEventOnCompletion(1, event);
        if (FAILED(hr) || WaitForSingleObject(event, 30000) != WAIT_OBJECT_0)
        {
            CloseHandle(event);
            error = "readback GPU wait failed or timed out";
            return false;
        }
    }
    CloseHandle(event);

    if (description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        pixels.resize(static_cast<size_t>(totalBytes));
    }
    else
    {
        pixels.resize(static_cast<size_t>(tightRowBytes) * description.Height);
    }

    void* mapped = nullptr;
    hr = readbackBuffer->Map(0, nullptr, &mapped);
    if (FAILED(hr)) { error = HrText("Map(readback)", hr); return false; }

    if (description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        std::memcpy(pixels.data(), mapped, static_cast<size_t>(totalBytes));
    }
    else
    {
        const uint8_t* base = static_cast<const uint8_t*>(mapped) + footprint.Offset;
        for (UINT row = 0; row < rowCount; ++row)
        {
            std::memcpy(pixels.data() + static_cast<size_t>(row) * tightRowBytes,
                base + static_cast<size_t>(row) * rowPitch,
                static_cast<size_t>(tightRowBytes));
        }
    }
    readbackBuffer->Unmap(0, nullptr);
    return true;
}

std::vector<uint8_t> Readback(ID3D12Resource* resource, std::string& error)
{
    std::vector<uint8_t> pixels;
    if (!Readback(resource, pixels, error)) pixels.clear();
    return pixels;
}

} // namespace render_validation
