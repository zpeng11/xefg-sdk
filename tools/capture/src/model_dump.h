#pragma once
// Read-only dump of the model catalog inside the loaded libxess_fg.
// Shapes follow source/native/xess/port/model_port.h (3.0.2 family tables);
// nothing is patched, and unsupported shapes are reported instead of guessed.
#include <Windows.h>

#include <string>

namespace oracle
{
bool DumpModelMetadata(HMODULE fgModule, const std::wstring& outputDirectory,
    bool dumpWeightBytes, size_t maxWeightBytes, std::string& error);
} // namespace oracle
