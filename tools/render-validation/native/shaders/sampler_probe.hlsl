// Standalone one-shot texture sampler probe.
//
// One dispatch evaluates every request.  Requests and results are structured
// buffers so the CPU writes the exact record layout to samples.bin without any
// intermediate format conversion: normalized is the raw f32x4 returned by the
// texture fetch (no FP16 or RGBA8 quantization), u/v are copied as raw bits,
// and units/truncated use the simple saturate(x) * 255 contract.
//
// The sampler itself is a root-signature static sampler configured by the CPU
// as filter 0x15 (MIN_MAG_MIP_LINEAR), CLAMP/CLAMP/WRAP, bias 0, anisotropy 16,
// LESS_EQUAL, OPAQUE_WHITE, minLOD 0, maxLOD FLT_MAX.  This is a non-comparison
// filter; the comparison function is inert configuration.

struct RequestRecord
{
    uint id;
    uint op;      // 0 = SampleLevel(uv, 0), 1 = Load(x, y)
    float u;
    float v;
    int x;
    int y;
    uint reserved0;
    uint reserved1;
};

struct OutputRecord
{
    uint id;
    uint op;
    float u;
    float v;
    float4 normalized;
    float4 units;
    int4 truncated;
    uint4 padding;
};

Texture2D<float4> gTexture : register(t0);
StructuredBuffer<RequestRecord> gRequests : register(t1);
RWStructuredBuffer<OutputRecord> gResults : register(u0);
SamplerState gSampler : register(s0);

cbuffer RootConstants : register(b0)
{
    uint gCount;
};

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchId : SV_DispatchThreadID)
{
    const uint index = dispatchId.x;
    if (index >= gCount)
        return;

    const RequestRecord request = gRequests[index];
    float4 color;
    if (request.op == 0u)
        color = gTexture.SampleLevel(gSampler, float2(request.u, request.v), 0.0f);
    else
        color = gTexture.Load(int3(request.x, request.y, 0));

    OutputRecord result;
    result.id = request.id;
    result.op = request.op;
    result.u = request.u;
    result.v = request.v;
    result.normalized = color;
    result.units = saturate(color) * 255.0f;
    result.truncated = int4(result.units);
    result.padding = uint4(0, 0, 0, 0);
    gResults[index] = result;
}
