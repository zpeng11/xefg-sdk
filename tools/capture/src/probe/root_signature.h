#pragma once
// Minimal serialized-root-signature parser (the RTS0 layout the repo already
// parses in source/native/xess/vulkan/xvk_programs.cpp). Knowing the descriptor
// ranges lets the probe name each bound resource (t0/t1/... , u0, b0) instead of
// guessing table sizes.
#include <cstdint>
#include <string>
#include <vector>

namespace probe
{
struct RootRange
{
    uint32_t type = 0; // D3D12_DESCRIPTOR_RANGE_TYPE
    uint32_t baseRegister = 0;
    uint32_t count = 0;
    uint32_t space = 0;
    uint32_t offset = 0;
};

struct RootParameter
{
    uint32_t type = 0; // D3D12_ROOT_PARAMETER_TYPE
    std::vector<RootRange> ranges;
    uint32_t shaderRegister = 0;
    uint32_t space = 0;
    uint32_t num32 = 0;
};

struct StaticSampler
{
    uint32_t filter = 0;
    uint32_t addressU = 0;
    uint32_t addressV = 0;
    uint32_t addressW = 0;
    float mipLodBias = 0;
    uint32_t maxAnisotropy = 0;
    uint32_t comparisonFunc = 0;
    uint32_t borderColor = 0;
    float minLod = 0;
    float maxLod = 0;
    uint32_t shaderRegister = 0;
    uint32_t space = 0;
    uint32_t visibility = 0;
};

struct RootLayout
{
    uint32_t version = 0;
    std::vector<RootParameter> parameters;
    std::vector<StaticSampler> samplers;
};

bool ParseRootSignatureBlob(const void* blob, size_t size, RootLayout& out, std::string& error);

// "t0" / "u3" / "b0" for a descriptor range, or an empty string.
std::string RegisterName(uint32_t rangeType, uint32_t baseRegister);
} // namespace probe
