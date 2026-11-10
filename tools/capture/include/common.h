#pragma once
// D3D12 test fixture helpers shared by the native capture tools.
//
// This is a trimmed down copy of tests/release_regression/common.h: only the
// WARP/device fixture and the test window used by the probe remain. The
// MFGUnlocker status transport and candidate-DLL helpers are intentionally
// absent so the capture tools depend on nothing but the Windows SDK.
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <string>

using Microsoft::WRL::ComPtr;

namespace harness
{
inline HMODULE SystemModule(const wchar_t* name)
{
    wchar_t path[32768]{};
    GetSystemDirectoryW(path, 32768);
    wcscat_s(path, L"\\");
    wcscat_s(path, name);
    return LoadLibraryExW(path, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
}

// ---------------------------------------------------------------------------
// WARP presentation fixture
// ---------------------------------------------------------------------------
struct WarpFixture
{
    ComPtr<IDXGIFactory2> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 0;
    bool debugLayer = false;

    bool Initialize(bool requestDebugLayer,IDXGIAdapter* selectedAdapter=nullptr,HMODULE deviceModule=nullptr)
    {
        if (requestDebugLayer)
        {
            HMODULE d3d12 = SystemModule(L"d3d12.dll");
            using DebugFn = HRESULT (WINAPI*)(REFIID, void**);
            if (auto getDebug = reinterpret_cast<DebugFn>(
                    GetProcAddress(d3d12, "D3D12GetDebugInterface")))
            {
                ComPtr<struct ID3D12Debug> debug;
                if (SUCCEEDED(getDebug(IID_PPV_ARGS(&debug))))
                {
                    debug->EnableDebugLayer();
                    debugLayer = true;
                }
            }
        }
        ComPtr<IDXGIFactory4> factory4;
        ComPtr<IDXGIAdapter> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4)))) return false;
        if(selectedAdapter)warp=selectedAdapter;
        else if(FAILED(factory4->EnumWarpAdapter(IID_PPV_ARGS(&warp))))return false;
        if (FAILED(factory4.As(&factory))) return false;
        using CreateDevice=HRESULT(WINAPI*)(IUnknown*,D3D_FEATURE_LEVEL,REFIID,void**);
        const auto create=selectedAdapter?reinterpret_cast<CreateDevice>(GetProcAddress(deviceModule?deviceModule:SystemModule(L"d3d12.dll"),"D3D12CreateDevice")):&D3D12CreateDevice;
        if (!create||FAILED(create(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
            return false;
        if (debugLayer)
        {
            // The debug layer terminates the process on its final-release
            // corruption report. Releasing backbuffer references before
            // ResizeBuffers while queued presentations reference them is the
            // documented flip-model resize pattern and is handled by the
            // runtime, so the harness collects those reports through the
            // info queue instead of dying on them.
            ComPtr<struct ID3D12InfoQueue> info;
            if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&info))) && info)
            {
                info->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
                info->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
                info->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, FALSE);
            }
        }
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)))
            || FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&allocator)))
            || FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(), nullptr, IID_PPV_ARGS(&commands)))
            || FAILED(commands->Close())
            || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            return false;
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return fenceEvent != nullptr;
    }

    bool SubmitAndWait()
    {
        if (FAILED(allocator->Reset()) || FAILED(commands->Reset(allocator.Get(), nullptr)))
            return false;
        return true;
    }

    bool SignalAndWait()
    {
        const UINT64 value = ++fenceValue;
        if (FAILED(queue->Signal(fence.Get(), value))) return false;
        if (fence->GetCompletedValue() >= value) return true;
        if (FAILED(fence->SetEventOnCompletion(value, fenceEvent))) return false;
        return WaitForSingleObject(fenceEvent, 5000) == WAIT_OBJECT_0;
    }

    ~WarpFixture()
    {
        if (fenceEvent) CloseHandle(fenceEvent);
    }
};

struct TestWindow
{
    HWND window = nullptr;
    ATOM registered = 0;
    HINSTANCE instance = nullptr;
    std::wstring className;
    WNDPROC proc = DefWindowProcW;

    TestWindow(const wchar_t* name, LONG width, LONG height, WNDPROC custom = DefWindowProcW)
        : instance(GetModuleHandleW(nullptr)), className(name), proc(custom)
    {
        WNDCLASSW description{};
        description.lpfnWndProc = custom;
        description.lpszClassName = name;
        description.hInstance = instance;
        registered = RegisterClassW(&description);
        if (registered) window = CreateWindowExW(0, name, name, WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT, width, height, nullptr, nullptr, instance, nullptr);
    }
    ~TestWindow()
    {
        if (window) DestroyWindow(window);
        if (registered) UnregisterClassW(className.c_str(), instance);
    }
    TestWindow(const TestWindow&) = delete;
    TestWindow& operator=(const TestWindow&) = delete;
    void Show() { ShowWindow(window, SW_SHOWNOACTIVATE); UpdateWindow(window); }
    void Pump(unsigned iterations = 8)
    {
        MSG message;
        for (unsigned i = 0; i < iterations; ++i)
        {
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            Sleep(10);
        }
    }
};
}
