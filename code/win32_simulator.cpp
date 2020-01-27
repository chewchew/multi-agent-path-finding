#include <windows.h>

#include "simulator.h"

struct MousePosition
{
    f32 x;
    f32 y;
};

struct MouseButton
{
    b32 ended_down;
    u32 half_transition_count;
};
    
struct Input
{
    union
    {
        MouseButton mouse_buttons[3];
        struct
        {
            MouseButton mouse_right;
            MouseButton mouse_middle;
            MouseButton mouse_left;
        };
    };
    
    MousePosition mouse_position;
    
    char DEBUG_last_character_typed;
};

global_variable b32 g_running = false;
global_variable WINDOWPLACEMENT g_window_position = {sizeof(g_window_position)};
global_variable int64_t g_perf_count_frequency;

static void
get_mouse_position(HWND window, POINT* mouse_position)
{
    RECT window_rect;
    GetWindowRect(window, &window_rect);

    RECT client_rect;
    GetClientRect(window, &client_rect);
    u32 client_height = client_rect.bottom - client_rect.top;
    u32 window_height = window_rect.bottom - window_rect.top;
    
    GetCursorPos(mouse_position);
    mouse_position->x = mouse_position->x - window_rect.left;
    mouse_position->y = client_height - (mouse_position->y - window_rect.top - (window_height - client_height));
}

inline LARGE_INTEGER
win32_get_wall_clock()
{    
    LARGE_INTEGER result;
    QueryPerformanceCounter(&result);
    return result;
}

inline f32
win32_get_seconds_elapsed(LARGE_INTEGER start, LARGE_INTEGER end)
{
    f32 result = ((f32)(end.QuadPart - start.QuadPart) /
                     (f32)g_perf_count_frequency);
    
    return result;
}

struct ViewportDimensions
{
    u32 width;
    u32 height;
};
    
static ViewportDimensions
win32_get_viewport_dimensions(HWND window)
{
    ViewportDimensions result;

    RECT client_rect;
    GetClientRect(window, &client_rect);

    result.width = client_rect.right - client_rect.left;
    result.height = client_rect.bottom - client_rect.top;

    return result;
}

static void
win32_toggle_fullscreen(HWND window, u32* window_width, u32* window_height)
{
    DWORD style = GetWindowLong(window, GWL_STYLE);
    if(style & WS_OVERLAPPEDWINDOW)
    {
        MONITORINFO monitor_info = {sizeof(monitor_info)};
        if(GetWindowPlacement(window, &g_window_position) &&
           GetMonitorInfo(MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY), &monitor_info))
        {
            SetWindowLong(window, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
            SetWindowPos(window, HWND_TOP,
                         monitor_info.rcMonitor.left, monitor_info.rcMonitor.top,
                         monitor_info.rcMonitor.right - monitor_info.rcMonitor.left,
                         monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top,
                         SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        }
    }
    else
    {
        
        SetWindowLong(window, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(window, &g_window_position);
        SetWindowPos(window, 0, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                     SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
    
    ViewportDimensions viewport_dimensions = win32_get_viewport_dimensions(window);
    
    *window_width = viewport_dimensions.width;
    *window_height = viewport_dimensions.height;
}

DEBUG_PLATFORM_FREE_FILE_MEMORY(DEBUG_win32_platform_free_file_memory)
{
    if (memory)
    {
        VirtualFree(memory, 0, MEM_RELEASE);
    }
}

DEBUG_PLATFORM_READ_ENTIRE_FILE(DEBUG_win32_platform_read_entire_file)
{
    DebugReadFileResult read_file_result = {};
    void* contents = 0;
    
    HANDLE file_handle = CreateFileA(filename, GENERIC_READ, 0, 0,
                                    OPEN_EXISTING, 0, 0);
    if (file_handle != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER file_size;
        if (GetFileSizeEx(file_handle, &file_size))
        {
            u32 file_size_32 = safe_truncate_size_64(file_size.QuadPart);
            contents = VirtualAlloc(0, file_size_32,
                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (contents)
            {
                DWORD bytes_read;
                if (ReadFile(file_handle, contents,
                             file_size_32, &bytes_read, 0) &&
                    (file_size_32 == bytes_read))
                {
                    // File read successfully
                    read_file_result.contents_size = file_size_32;
                    read_file_result.contents = contents;
                }
                else
                {
                    // TODO: Logging
                    DEBUG_platform_free_file_memory(contents);
                    contents = 0;
                }
            }
            else
            {
                // TODO: Logging
            }
        }
        else
        {
            // TODO: Logging
        }

        CloseHandle(file_handle);
    }
    else
    {
        // TODO: Logging
    }

    Assert(read_file_result.contents);
    return read_file_result;
}

LRESULT CALLBACK
window_callback(HWND window,
                UINT message,
                WPARAM w_param,
                LPARAM l_param)
{
    LRESULT result = 0;
    
    switch (message)
    {
        case WM_ACTIVATEAPP:
        {
            if (w_param == TRUE)
            {
                SetLayeredWindowAttributes(window, RGB(0, 0, 0), 255, LWA_ALPHA);
            }
            else
            {
                SetLayeredWindowAttributes(window, RGB(0, 0, 0), 64, LWA_ALPHA);
            }
        }
        break;
        
        case WM_DESTROY: // TODO(Quit): Error has happened?
        case WM_CLOSE:   // TODO(Quit): Message to user?
        {
            g_running = false;
        }
        break;
            
        default:
        {
            result = DefWindowProc(window, message, w_param, l_param);
        }
        break;
    
    }

    return result;
}

int CALLBACK
WinMain(HINSTANCE instance,
        HINSTANCE prev_instance,
        LPSTR args,
        int cmd_show)
{
    WNDCLASSA window_class = {};
    window_class.style         = CS_VREDRAW | CS_HREDRAW | CS_OWNDC;
    window_class.lpfnWndProc   = window_callback;
    window_class.hInstance     = instance;
    window_class.hCursor       = LoadCursor(0, IDC_ARROW);
    window_class.lpszClassName = "SimulatorClientwindow_class";

    DEBUG_platform_read_entire_file = DEBUG_win32_platform_read_entire_file;
    DEBUG_platform_free_file_memory = DEBUG_win32_platform_free_file_memory;
    
    LARGE_INTEGER perf_count_frequency;
    QueryPerformanceFrequency(&perf_count_frequency);
    g_perf_count_frequency = perf_count_frequency.QuadPart;
    
    if (RegisterClassA(&window_class))
    {
        HWND window = CreateWindowExA(
            0,
            // WS_EX_TOPMOST | WS_EX_LAYERED,
            window_class.lpszClassName,
            "Simulator",
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            0,
            0,
            instance,
            0);

        if (window)
        {
            ViewportDimensions viewport_dimensions = win32_get_viewport_dimensions(window);
            u32 window_width = viewport_dimensions.width;
            u32 window_height = viewport_dimensions.height;
            win32_toggle_fullscreen(window, &window_width, &window_height);
            
            HDC device_ctx = GetDC(window);

            LARGE_INTEGER last_counter = win32_get_wall_clock();

            int monitor_refresh_rate = 60;
            int win32_refresh_rate = GetDeviceCaps(device_ctx, VREFRESH);
            if (win32_refresh_rate > 1)
            {
                monitor_refresh_rate = win32_refresh_rate;
            }
            f32 update_hz = (f32)monitor_refresh_rate;
            u32 expected_frames_per_update = 1;
            f32 target_seconds_per_frame = expected_frames_per_update / update_hz;

            Input inputs[2] = {};
            Input* new_input = &inputs[0];
            Input* old_input = &inputs[1];

            u32 memory_size = Megabytes(1000);
            MemoryArena memory_arena;
            memory_arena.size = memory_size;
            memory_arena.base = (u8*)VirtualAlloc(
                0, memory_size,
                MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
            memory_arena.used = 0;
            
            g_running = true;
            while (g_running)
            {
                Input zero_input = {};
                *new_input = zero_input;

                for (u32 button_index = 0;
                     button_index < ArrayCount(old_input->mouse_buttons);
                     button_index++)
                {
                    new_input->mouse_buttons[button_index].ended_down =
                        old_input->mouse_buttons[button_index].ended_down;
                }
    
                MSG message;
                while(PeekMessageA(&message, 0, 0, 0, PM_REMOVE))
                {
                    switch(message.message)
                    {
                        case WM_QUIT:
                        {
                            g_running = false;
                        }
                        break;
                        
                        case WM_SYSKEYDOWN:
                        case WM_SYSKEYUP:
                        case WM_KEYDOWN:
                        case WM_KEYUP:
                        {
                            u32 vk_code = (u32)message.wParam;
                            b32 was_down = ((message.lParam & (1 << 30)) != 0);
                            b32 is_down = (message.lParam & (1 << 31)) == 0;

                            if (was_down != is_down)
                            {
                                switch(vk_code)
                                {
                                    case VK_ESCAPE:
                                    {
                                        g_running = false;
                                    }
                                    break;
                                }
                            }
                            
                            if (is_down)
                            {
                                b32 alt_key_was_down = message.lParam & (1 << 29);
                                if (alt_key_was_down)
                                {
                                    if (vk_code == VK_F4)
                                    {
                                        g_running = false;
                                    }
                                    if (vk_code == VK_RETURN)
                                    {
                                        win32_toggle_fullscreen(message.hwnd, &window_width, &window_height);
                                    }
                                }
                            }
                        }
                        break;

                        case WM_LBUTTONDOWN:
                        {
                            new_input->mouse_left.ended_down = true;
                            new_input->mouse_left.half_transition_count += 1;
                        }
                        break;

                        case WM_LBUTTONUP:
                        {
                            new_input->mouse_left.ended_down = false;
                            new_input->mouse_left.half_transition_count += 1;
                        }
                        break;

                        case WM_MBUTTONDOWN:
                        {
                            new_input->mouse_middle.ended_down = true;
                            new_input->mouse_middle.half_transition_count += 1;
                        }
                        break;

                        case WM_MBUTTONUP:
                        {
                            new_input->mouse_middle.ended_down = false;
                            new_input->mouse_middle.half_transition_count += 1;
                        }
                        break;
                        
                        case WM_RBUTTONDOWN:
                        {
                            new_input->mouse_right.ended_down = true;
                            new_input->mouse_right.half_transition_count += 1;
                        }
                        break;

                        case WM_RBUTTONUP:
                        {
                            new_input->mouse_right.ended_down = false;
                            new_input->mouse_right.half_transition_count += 1;
                        }
                        break;
                    }

                    TranslateMessage(&message);
                    DispatchMessageA(&message);
                }

                POINT mouse_position;
                get_mouse_position(window, &mouse_position);
                new_input->mouse_position.x = (f32)mouse_position.x;
                new_input->mouse_position.y = (f32)mouse_position.y;

                LARGE_INTEGER end_counter = win32_get_wall_clock();
                f32 measured_seconds_per_frame = win32_get_seconds_elapsed(last_counter, end_counter);
                last_counter = end_counter;

                LARGE_INTEGER start = win32_get_wall_clock();
                simulate(&memory_arena);
                LARGE_INTEGER end = win32_get_wall_clock();
                f32 time = win32_get_seconds_elapsed(end, start);

#ifdef LINES_DEBUG
                FILETIME new_file_time = win32_get_last_write_time(source_dll_full_path);
                if (CompareFileTime(&new_file_time, &game_code.last_write_time) != 0)
                {
                    win32_unload_game_code(&game_code);
                    game_code = win32_load_game_code(source_dll_full_path, tmp_dll_full_path);
                }
#endif

                Input* tmp = old_input;
                old_input = new_input;
                new_input = tmp;
            }
        }
    }
    
    return 0;
}
