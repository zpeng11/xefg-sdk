#include "root_signature.h"

#include <d3d12.h>

#include <cstring>

namespace probe
{
namespace
{
template <class T>
bool Read(const uint8_t* data, size_t size, size_t offset, T& out)
{
    if (offset > size || size - offset < sizeof(T)) return false;
    std::memcpy(&out, data + offset, sizeof(T));
    return true;
}

bool FindPart(const uint8_t* data, size_t size, uint32_t fourcc, const uint8_t*& part, size_t& partSize)
{
    uint32_t magic = 0, count = 0;
    if (!Read(data, size, 0, magic) || magic != 0x43425844 /* DXBC */) return false;
    if (!Read(data, size, 28, count)) return false;
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t offset = 0, code = 0, length = 0;
        if (!Read(data, size, 32 + 4 * i, offset) || !Read(data, size, offset, code)
            || !Read(data, size, offset + 4, length)) return false;
        if (code != fourcc) continue;
        if (offset + 8 > size || size - offset - 8 < length) return false;
        part = data + offset + 8;
        partSize = length;
        return true;
    }
    return false;
}
} // namespace

bool ParseRootSignatureBlob(const void* blob, size_t size, RootLayout& out, std::string& error)
{
    if (!blob || size < 24)
    {
        error = "empty root signature";
        return false;
    }
    const auto* data = static_cast<const uint8_t*>(blob);
    const uint8_t* rts = data;
    size_t rtsSize = size;
    if (size >= 4 && std::memcmp(data, "DXBC", 4) == 0
        && !FindPart(data, size, 0x30535452 /* RTS0 */, rts, rtsSize))
    {
        error = "container has no RTS0 part";
        return false;
    }
    uint32_t version = 0, parameterCount = 0, parameterOffset = 0, samplerCount = 0, samplerOffset = 0, flags = 0;
    if (!Read(rts, rtsSize, 0, version) || !Read(rts, rtsSize, 4, parameterCount)
        || !Read(rts, rtsSize, 8, parameterOffset) || !Read(rts, rtsSize, 12, samplerCount)
        || !Read(rts, rtsSize, 16, samplerOffset) || !Read(rts, rtsSize, 20, flags))
    {
        error = "truncated RTS0 header";
        return false;
    }
    if (version < 1 || version > 3)
    {
        error = "unsupported RTS0 version " + std::to_string(version);
        return false;
    }
    out = {};
    out.version = version;
    for (uint32_t i = 0; i < parameterCount; ++i)
    {
        uint32_t type = 0, visibility = 0, payload = 0;
        if (!Read(rts, rtsSize, parameterOffset + 12ull * i, type)
            || !Read(rts, rtsSize, parameterOffset + 12ull * i + 4, visibility)
            || !Read(rts, rtsSize, parameterOffset + 12ull * i + 8, payload))
        {
            error = "truncated parameter";
            return false;
        }
        RootParameter parameter;
        parameter.type = type;
        if (type == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
        {
            uint32_t rangeCount = 0, rangeOffset = 0;
            if (!Read(rts, rtsSize, payload, rangeCount) || !Read(rts, rtsSize, payload + 4, rangeOffset))
            {
                error = "truncated table";
                return false;
            }
            const uint32_t stride = version == 1 ? 20 : 24;
            uint32_t running = 0;
            for (uint32_t r = 0; r < rangeCount; ++r)
            {
                const size_t at = rangeOffset + size_t(stride) * r;
                uint32_t rangeType = 0, count = 0, base = 0, space = 0, offset = 0;
                if (!Read(rts, rtsSize, at, rangeType) || !Read(rts, rtsSize, at + 4, count)
                    || !Read(rts, rtsSize, at + 8, base) || !Read(rts, rtsSize, at + 12, space)
                    || !Read(rts, rtsSize, at + (version == 1 ? 16 : 20), offset))
                {
                    error = "truncated range";
                    return false;
                }
                if (offset == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) offset = running;
                if (count == UINT_MAX)
                {
                    error = "unbounded range";
                    return false;
                }
                parameter.ranges.push_back({rangeType, base, count, space, offset});
                running = offset + count;
            }
        }
        else if (type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS)
        {
            if (!Read(rts, rtsSize, payload, parameter.shaderRegister)
                || !Read(rts, rtsSize, payload + 4, parameter.space)
                || !Read(rts, rtsSize, payload + 8, parameter.num32))
            {
                error = "truncated constants";
                return false;
            }
        }
        else
        {
            if (!Read(rts, rtsSize, payload, parameter.shaderRegister)
                || !Read(rts, rtsSize, payload + 4, parameter.space))
            {
                error = "truncated root descriptor";
                return false;
            }
        }
        out.parameters.push_back(std::move(parameter));
    }

    // Static samplers follow the parameters. The layout is the same one
    // source/native/xess/vulkan/xvk_programs.cpp already parses: version 3
    // entries are 56 bytes, older ones 52. A truncated tail is non-fatal: the
    // descriptor ranges are the part a dispatch needs, and a provider that
    // writes an unusual sampler block should not lose the whole signature.
    const size_t samplerStride = version == 3 ? 56 : 52;
    for (uint32_t i = 0; i < samplerCount; ++i)
    {
        D3D12_STATIC_SAMPLER_DESC description{};
        if (!Read(rts, rtsSize, samplerOffset + samplerStride * i, description)) break;
        StaticSampler sampler;
        sampler.filter = static_cast<uint32_t>(description.Filter);
        sampler.addressU = static_cast<uint32_t>(description.AddressU);
        sampler.addressV = static_cast<uint32_t>(description.AddressV);
        sampler.addressW = static_cast<uint32_t>(description.AddressW);
        sampler.mipLodBias = description.MipLODBias;
        sampler.maxAnisotropy = description.MaxAnisotropy;
        sampler.comparisonFunc = static_cast<uint32_t>(description.ComparisonFunc);
        sampler.borderColor = static_cast<uint32_t>(description.BorderColor);
        sampler.minLod = description.MinLOD;
        sampler.maxLod = description.MaxLOD;
        sampler.shaderRegister = description.ShaderRegister;
        sampler.space = description.RegisterSpace;
        sampler.visibility = static_cast<uint32_t>(description.ShaderVisibility);
        out.samplers.push_back(sampler);
    }
    return true;
}

std::string RegisterName(uint32_t rangeType, uint32_t baseRegister)
{
    const char prefix = rangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SRV ? 't'
        : rangeType == D3D12_DESCRIPTOR_RANGE_TYPE_UAV ? 'u'
        : rangeType == D3D12_DESCRIPTOR_RANGE_TYPE_CBV ? 'b'
        : rangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER ? 's'
        : '\0';
    if (!prefix) return {};
    return std::string(1, prefix) + std::to_string(baseRegister);
}
} // namespace probe
