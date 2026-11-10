#include "xefg_runtime.h"

#include "runtime_layout.h"
#include "sha256_util.h"

#include <fstream>
#include <sstream>

namespace oracle
{
namespace
{
std::wstring Join(const std::wstring& directory, const wchar_t* name)
{
    std::wstring path = directory;
    if (!path.empty() && path.back() != L'\\' && path.back() != L'/') path.push_back(L'\\');
    path += name;
    return path;
}

void CALLBACK LogTrampoline(const char* message, xefg_swapchain_logging_level_t level, void* userData)
{
    static_cast<Runtime*>(userData)->AppendLog(message, static_cast<int>(level));
}
} // namespace

bool Runtime::Load(const std::wstring& directory, std::string& error)
{
    if (directory.empty())
    {
        error = "empty --xess-dir";
        return false;
    }
    // The pair resolves its own dependencies from the same folder.
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    AddDllDirectory(directory.c_str());

    const auto load = [&](const wchar_t* name, HMODULE& module) {
        const std::wstring path = Join(directory, name);
        module = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module)
        {
            std::ostringstream text;
            text << "LoadLibraryExW failed for " << std::string(name, name + wcslen(name))
                 << " (error " << GetLastError() << ")";
            error = text.str();
            return false;
        }
        return true;
    };
    if (!load(L"libxell.dll", ll)) return false;
    if (!load(L"libxess_fg.dll", fg)) return false;

    std::vector<std::string> missing;
    const auto resolve = [&](HMODULE module, const char* name) -> FARPROC {
        const FARPROC address = GetProcAddress(module, name);
        if (!address) missing.emplace_back(name);
        return address;
    };

#define ORACLE_RESOLVE_FN(name) name = reinterpret_cast<decltype(name)>(resolve(fg, #name));
    ORACLE_XEFG_FUNCTIONS(ORACLE_RESOLVE_FN)
#undef ORACLE_RESOLVE_FN
#define ORACLE_RESOLVE_LL_FN(name) name = reinterpret_cast<decltype(name)>(resolve(ll, #name));
    ORACLE_XELL_FUNCTIONS(ORACLE_RESOLVE_LL_FN)
#undef ORACLE_RESOLVE_LL_FN

    if (!missing.empty())
    {
        std::ostringstream text;
        text << "missing XeSS exports:";
        for (const auto& name : missing) text << ' ' << name;
        error = text.str();
        return false;
    }
    if (xefgSwapChainGetVersion)
    {
        xefg_swapchain_version_t version{};
        if (xefgSwapChainGetVersion(&version) != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        {
            error = "xefgSwapChainGetVersion failed";
            return false;
        }
        if (version.major != 1)
        {
            std::ostringstream text;
            text << "unsupported XeFG major version " << version.major << '.' << version.minor << '.' << version.patch;
            error = text.str();
            return false;
        }
    }
    return true;
}

void Runtime::Unload()
{
    // XeLL first: XeFG's destroy path may still reference it.
    if (fg) { FreeLibrary(fg); fg = nullptr; }
    if (ll) { FreeLibrary(ll); ll = nullptr; }
}

std::vector<std::string> Runtime::TakeLog()
{
    std::lock_guard<std::mutex> lock(logMutex);
    return std::move(logLines);
}

void Runtime::AppendLog(const char* message, int level)
{
    std::lock_guard<std::mutex> lock(logMutex);
    if (logLines.size() >= 4096) return;
    std::ostringstream line;
    line << '[' << level << "] " << (message ? message : "");
    logLines.push_back(line.str());
}

std::string DetectFamily(HMODULE fgModule, std::string& error)
{
    try
    {
        return xess_layout::Get(fgModule, true).family;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return {};
    }
}

std::string ModuleFileSha256(HMODULE module)
{
    wchar_t path[MAX_PATH]{};
    if (!GetModuleFileNameW(module, path, MAX_PATH)) return {};
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    sha256_util::Hasher hasher;
    std::vector<char> buffer(1u << 20);
    while (file)
    {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = static_cast<size_t>(file.gcount());
        if (count && !hasher.Update(buffer.data(), count)) return {};
    }
    sha256_util::Digest digest{};
    if (!hasher.Finish(digest)) return {};
    return sha256_util::LowerHex(digest);
}
} // namespace oracle
