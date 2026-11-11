// Scene shader for the render-validation DX12 renderer.
// Compiled at runtime with D3DCompileFromFile.  Matrices are row-major and
// multiplied row-vector style: p_clip = p_world * ViewProj.

cbuffer ObjectConstants : register(b0)
{
    row_major float4x4 Model;
    row_major float4x4 PreviousModel;
    float4 ObjectColor;
    float4 ObjectParams;
};

cbuffer FrameConstants : register(b1)
{
    row_major float4x4 ViewProj;
    row_major float4x4 PreviousViewProj;
    float4 Viewport;  // xy = color size (640x360), zw = motion size (320x180)
    float4 FrameData; // x = application frame id, y = midpoint, z = history reset
};

struct SceneVertex
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct ColorVSOut
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float3 worldPosition : WORLDPOSITION;
    float3 localPosition : LOCALPOSITION;
};

ColorVSOut ColorVS(SceneVertex input)
{
    ColorVSOut output;
    float4 world = mul(float4(input.position, 1.0), Model);
    output.position = mul(world, ViewProj);
    output.normal = normalize(mul(float4(input.normal, 0.0), Model).xyz);
    output.worldPosition = world.xyz;
    output.localPosition = input.position;
    return output;
}

float4 ColorPS(ColorVSOut input) : SV_Target
{
    float3 normal = normalize(input.normal);
    float3 lightDirection = normalize(float3(0.35, 0.80, 0.45));
    float diffuse = saturate(dot(normal, lightDirection));
    float rim = pow(saturate(1.0 - abs(normal.z)), 2.0) * 0.10;

    // Object-local checker so the pattern moves with the object, not the world.
    float checker = frac((floor(input.localPosition.x * 4.0) +
        floor(input.localPosition.y * 4.0) +
        floor(input.localPosition.z * 4.0)) * 0.5);
    float shade = checker < 0.25 ? 1.0 : 0.78;

    float3 rgb = ObjectColor.rgb * shade * (0.22 + 0.78 * diffuse) + rim * shade;
    return float4(rgb, 1.0);
}

struct GeometryVSOut
{
    float4 position : SV_Position;
    float4 previousClip : PREVIOUSCLIP;
};

GeometryVSOut GeometryVS(SceneVertex input)
{
    GeometryVSOut output;
    float4 world = mul(float4(input.position, 1.0), Model);
    float4 previousWorld = mul(float4(input.position, 1.0), PreviousModel);
    output.position = mul(world, ViewProj);
    output.previousClip = mul(previousWorld, PreviousViewProj);
    return output;
}

struct GeometryPSOut
{
    float depth : SV_Target0;
    float2 motion : SV_Target1;
};

GeometryPSOut GeometryPS(GeometryVSOut input)
{
    GeometryPSOut output;
    output.depth = saturate(input.position.z);

    // Current pixel -> previous pixel, in motion-target pixels.  Previous
    // clip is projected with the previous object and camera matrices and
    // converted from NDC (y up) to pixel space (y down).
    float2 previousPixel = input.position.xy;
    if (input.previousClip.w > 0.05)
    {
        float2 ndc = input.previousClip.xy / input.previousClip.w;
        previousPixel = (ndc * float2(0.5, -0.5) + 0.5) * Viewport.zw;
    }
    output.motion = clamp(previousPixel - input.position.xy, -Viewport.zw, Viewport.zw);
    return output;
}

struct FullscreenVSOut
{
    float4 position : SV_Position;
};

FullscreenVSOut FullscreenVS(uint vertexId : SV_VertexID)
{
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    FullscreenVSOut output;
    output.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

float4 UiPS(FullscreenVSOut input) : SV_Target
{
    uint2 pixel = uint2(input.position.xy);
    // Per-frame UI motion; the calibration marker below stays fixed.
    float2 offset = float2(
        (float)((int)FrameData.x % 7 - 3) * 3.0,
        (float)(((int)FrameData.x / 7) % 5 - 2) * 2.0);
    float2 centered = abs(float2(pixel) + 0.5 - (float2(320.0, 180.0) + offset));

    // Premultiplied-alpha crosshair.
    float4 ui = float4(0.0, 0.0, 0.0, 0.0);
    if ((centered.x < 1.5 && centered.y < 24.0) ||
        (centered.y < 1.5 && centered.x < 24.0))
    {
        ui = float4(0.76, 0.11, 0.09, 0.80);
    }

    // Calibration strip: x=96,y=32, 32 bits, 8x8 pixels per bit, opaque.
    // Each 16-bit application id is followed by its 16-bit complement.
    if (pixel.x >= 96 && pixel.x < 352 && pixel.y >= 32 && pixel.y < 40)
    {
        uint bit = (pixel.x - 96) / 8;
        uint id = (uint)FrameData.x & 0xffffu;
        uint code = id | ((id ^ 0xffffu) << 16);
        float value = (code & (1u << bit)) ? 1.0 : 0.0;
        ui = float4(value, value, value, 1.0);
    }
    return ui;
}

Texture2D ColorTexture : register(t0);
Texture2D UiTexture : register(t1);

float4 CompositePS(FullscreenVSOut input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    float4 color = ColorTexture.Load(int3(pixel, 0));
    float4 ui = UiTexture.Load(int3(pixel, 0));
    return float4(ui.rgb + color.rgb * (1.0 - ui.a), 1.0);
}
