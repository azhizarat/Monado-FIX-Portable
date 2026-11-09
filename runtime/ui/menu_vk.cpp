#include "../fps_meter.h"
#include "gpu_picker.h"
// [XCDE] In-game settings overlay (runtime/ui/menu.h; Windows / Vulkan): F1 opens and closes it (Esc closes), the
// launcher's options over the running game. Dear ImGui drawn by its Vulkan backend into the window image after the
// game image (device.c calls menu_render inside that rendering). Events arrive on the main thread and are queued for
// the render thread, which owns the ImGui context; while the menu is open the game sees no keyboard / mouse / pad
// input. Options that the renderer reads every frame apply at once (resolution, upscaler and its quality, SMAA, depth
// of field, volume, FPS in the title, V-Sync); the others are saved to xcde.ini and apply at the next start
// (marked "al reiniciar"). "Guardar" writes xcde.ini (settings_save) for the launcher and the next start.
#include <SDL3/SDL.h>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <string.h>

#include "imgui.h"
#include "theme.h"
#include "quality_controls.h"
#include "imgui_impl_vulkan.h"

extern "C" {
#include "../nvn/vulkan/vk.h"
#include "../settings.h"
#include "menu.h"
bool vk_imgui_backend_init(void);
void input_reset(void);
void vk_request_swapchain_recreate(void);
bool dlss_available(void);
const char *dlss_status(void);
bool diag_deterministic(void);
}

static XcdeFpsMeter fps_meter;
extern "C" void menu_fps_tick(uint64_t now_ns) { xcde_fps_tick(&fps_meter, now_ns); }
static std::mutex q_lock;
static SDL_Event q[256];
static int q_n;
static volatile bool open_, want_window;
static bool fs_applied = false, fs_init;

static bool is_input(Uint32 t) {
    switch (t) {
    case SDL_EVENT_KEY_DOWN: case SDL_EVENT_KEY_UP: case SDL_EVENT_TEXT_INPUT: case SDL_EVENT_MOUSE_MOTION:
    case SDL_EVENT_MOUSE_BUTTON_DOWN: case SDL_EVENT_MOUSE_BUTTON_UP: case SDL_EVENT_MOUSE_WHEEL:
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN: case SDL_EVENT_GAMEPAD_BUTTON_UP: case SDL_EVENT_GAMEPAD_AXIS_MOTION:
        return true;
    default: return false;
    }
}

extern "C" bool menu_handle_event(const SDL_Event *e) {
    if (e->type == SDL_EVENT_KEY_DOWN && !e->key.repeat &&
        (e->key.scancode == SDL_SCANCODE_F1 || (open_ && e->key.scancode == SDL_SCANCODE_ESCAPE))) {
        open_ = !open_ && e->key.scancode == SDL_SCANCODE_F1;
        input_reset();  // (keys held when the menu opened / closed are released for the game)
        return true;
    }
    if (!open_ || !is_input(e->type)) return false;
    std::lock_guard<std::mutex> g(q_lock);
    if (q_n < 256) q[q_n++] = *e;
    return true;
}

extern "C" bool menu_is_open(void) { return open_; }

extern "C" void menu_apply_window_requests(void *sdl_window) {
    if (!fs_init) { fs_init = true; fs_applied = g_settings.fullscreen; }
    if (!want_window || !sdl_window) return;
    want_window = false;
    if (fs_applied != g_settings.fullscreen) {
        SDL_SetWindowFullscreen((SDL_Window *)sdl_window, g_settings.fullscreen);
        fs_applied = g_settings.fullscreen;
    }
}

extern "C" void menu_test_open(void) { open_ = true; }

static ImGuiKey key_of(SDL_Scancode s) {
    switch (s) {
    case SDL_SCANCODE_TAB: return ImGuiKey_Tab;
    case SDL_SCANCODE_LEFT: return ImGuiKey_LeftArrow;
    case SDL_SCANCODE_RIGHT: return ImGuiKey_RightArrow;
    case SDL_SCANCODE_UP: return ImGuiKey_UpArrow;
    case SDL_SCANCODE_DOWN: return ImGuiKey_DownArrow;
    case SDL_SCANCODE_RETURN: return ImGuiKey_Enter;
    case SDL_SCANCODE_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDL_SCANCODE_SPACE: return ImGuiKey_Space;
    case SDL_SCANCODE_ESCAPE: return ImGuiKey_Escape;
    case SDL_SCANCODE_BACKSPACE: return ImGuiKey_Backspace;
    case SDL_SCANCODE_LSHIFT: return ImGuiKey_LeftShift;
    case SDL_SCANCODE_RSHIFT: return ImGuiKey_RightShift;
    case SDL_SCANCODE_LCTRL: return ImGuiKey_LeftCtrl;
    case SDL_SCANCODE_RCTRL: return ImGuiKey_RightCtrl;
    default: return ImGuiKey_None;
    }
}
static ImGuiKey pad_key(Uint8 b) {
    switch (b) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return ImGuiKey_GamepadFaceDown;
    case SDL_GAMEPAD_BUTTON_EAST: return ImGuiKey_GamepadFaceRight;
    case SDL_GAMEPAD_BUTTON_WEST: return ImGuiKey_GamepadFaceLeft;
    case SDL_GAMEPAD_BUTTON_NORTH: return ImGuiKey_GamepadFaceUp;
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return ImGuiKey_GamepadL1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return ImGuiKey_GamepadR1;
    case SDL_GAMEPAD_BUTTON_START: return ImGuiKey_GamepadStart;
    default: return ImGuiKey_None;
    }
}

// Events queued by the main thread -> ImGui IO (render thread). Mouse positions: window points -> pixels.
static void feed(ImGuiIO &io, uint32_t w, uint32_t h) {
    SDL_Event ev[256];
    int n;
    {
        std::lock_guard<std::mutex> g(q_lock);
        n = q_n;
        memcpy(ev, q, sizeof(SDL_Event) * n);
        q_n = 0;
    }
    for (int i = 0; i < n; i++) {
        const SDL_Event &e = ev[i];
        float sx = 1, sy = 1;
        SDL_Window *win = SDL_GetWindowFromID(e.motion.windowID);
        if (win) {
            int ww, wh;
            SDL_GetWindowSize(win, &ww, &wh);
            if (ww > 0 && wh > 0) { sx = (float)w / ww; sy = (float)h / wh; }
        }
        switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION: io.AddMousePosEvent(e.motion.x * sx, e.motion.y * sy); break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN: case SDL_EVENT_MOUSE_BUTTON_UP: {
            int b = e.button.button == SDL_BUTTON_LEFT ? 0 : e.button.button == SDL_BUTTON_RIGHT ? 1 : 2;
            io.AddMousePosEvent(e.button.x * sx, e.button.y * sy);
            io.AddMouseButtonEvent(b, e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            break;
        }
        case SDL_EVENT_MOUSE_WHEEL: io.AddMouseWheelEvent(e.wheel.x, e.wheel.y); break;
        case SDL_EVENT_KEY_DOWN: case SDL_EVENT_KEY_UP: {
            ImGuiKey k = key_of(e.key.scancode);
            if (k != ImGuiKey_None) io.AddKeyEvent(k, e.type == SDL_EVENT_KEY_DOWN);
            break;
        }
        case SDL_EVENT_TEXT_INPUT: io.AddInputCharactersUTF8(e.text.text); break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN: case SDL_EVENT_GAMEPAD_BUTTON_UP: {
            ImGuiKey k = pad_key(e.gbutton.button);
            if (k != ImGuiKey_None) io.AddKeyEvent(k, e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN);
            if (e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST && e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) open_ = false;  // (B: back to the game)
            break;
        }
        default: break;
        }
    }
}

// [XCDE] the menu's language follows the launcher's (xcde.ini launcher_lang; g_settings.ui_lang 1 = English): the
// Spanish text is the key, T() gives the English one.
static const char *const k_en[][2] = {
    {"Igual que la ventana (recomendado)", "Match the window (recommended)"}, {"1x - 720p (como Switch)", "1x - 720p (like Switch)"},
    {"75 % de la ventana", "75 % of the window"}, {"50 % de la ventana", "50 % of the window"},
    {"Dinámica (según objetivo FPS)", "Dynamic (follows FPS target)"}, {"Opciones  (F1 / Esc para volver al juego)", "Options  (F1 / Esc to go back to the game)"},
    {"Gráficos", "Graphics"}, {"Pantalla", "Display"}, {"Pantalla completa", "Fullscreen"},
    {"Resolución y escalado", "Resolution and scaling"}, {"Resolución interna", "Internal resolution"},
    {"Bilineal (suave)", "Bilinear (soft)"}, {"AMD FSR 1.0 (nítido)", "AMD FSR 1.0 (sharp)"}, {"Escalado / antialiasing", "Upscaling / anti-aliasing"},
    {"DLSS se carga al iniciar el juego: hasta reiniciar se usa FSR 3.1.", "DLSS loads when the game starts: FSR 3.1 is used until a restart."},
    {"Nativo (solo antialiasing)", "Native (anti-aliasing only)"}, {"Calidad (67 %)", "Quality (67 %)"}, {"Equilibrado (59 %)", "Balanced (59 %)"},
    {"Rendimiento (50 %)", "Performance (50 %)"}, {"Ultra rendimiento (33 %)", "Ultra performance (33 %)"}, {"Calidad del escalado", "Upscaling quality"},
    {"Generación de frames (FSR 3; activarla por primera vez: al reiniciar)", "Frame generation (FSR 3; the first time it is enabled: after a restart)"},
    {"Calidad", "Quality"}, {"Antialiasing SMAA", "SMAA anti-aliasing"}, {"(no se usa con FSR 3.1 / DLSS)", "(not used with FSR 3.1 / DLSS)"},
    {"Profundidad de campo", "Depth of field"},
    {"Antialiasing y desenfoque de movimiento del juego (al reiniciar)", "The game's anti-aliasing and motion blur (after a restart)"},
    {"Antialiasing temporal (TAA) del juego (al reiniciar)", "The game's temporal anti-aliasing (TAA) (after a restart)"},
    {"Mostrar FPS en la barra de título", "Show FPS in the title bar"}, {"Velocidad", "Speed"}, {"30 (como en Switch)", "30 (like Switch)"},
    {"Sin límite", "Unlímited"}, {"FPS (al reiniciar)", "FPS (after a restart)"}, {"V-Sync (con 30 / 60 FPS)", "V-Sync (at 30 / 60 FPS)"},
    {"Volumen", "Volume"}, {"Audio de los vídeos (al reiniciar)", "Video sound (after a restart)"}, {"Controles", "Controls"},
    {"La asignación de teclas y botones se cambia en el launcher (Play Monado.bat), pestaña Controles.",
     "Keys and buttons are remapped in the launcher, Controls tab."},
    {"GPU / Vulkan", "GPU / Vulkan"}, {"Adaptador Vulkan activo", "Active Vulkan adapter"},
    {"Índice de GPU (-1 = automático)", "GPU index (-1 = automatic)"},
    {"Perfil gráfico", "Graphics profile"}, {"Automático (recomendado)", "Automatic (recommended)"},
    {"Compatibilidad (desactiva rutas opcionales rápidas)", "Compatibility (disables optional fast paths)"},
    {"Bibliotecas de pipelines", "Pipeline libraries"}, {"Según el controlador", "Driver default"},
    {"Desactivadas", "Disabled"}, {"Activadas si son compatibles", "Enabled when supported"},
    {"Presupuesto de memoria Vulkan", "Vulkan memory budget"},
    {"Estos cambios requieren reiniciar el juego.", "These changes require restarting the game."},
    {"Memoria local reportada", "Reported local memory"}, {"Memoria compartida / no local", "Shared / non-local memory"},
    {"Guardar", "Save"}, {"Volver al juego", "Back to the game"}, {"Guardado en xcde.ini", "Saved to xcde.ini"},
};
static const char *T(const char *es) {
    if (g_settings.ui_lang != 1) return es;
    for (const auto &e : k_en) if (!strcmp(e[0], es)) return e[1];
    return es;
}
struct TList { const char *p[8]; };
static TList TL(const char *const *a, int n) { TList t{}; for (int i = 0; i < n && i < 8; i++) t.p[i] = T(a[i]); return t; }

static const char *const k_res_ui[] = {"Igual que la ventana (recomendado)", "1x - 720p (como Switch)", "2x - 1440p", "3x - 2160p",
                                       "75 % de la ventana", "50 % de la ventana", "Dinámica (según objetivo FPS)"};
static const int k_res_mode[] = {RES_MATCH, 1, 2, 3, RES_Q75, RES_Q50, RES_DYNAMIC};

static void ui(void) {
    static bool saved_note;
    Settings opening=g_settings, quality_before=g_settings;
    ImGuiViewport *vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->Size.x * 0.5f, vp->Size.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(fminf(vp->Size.x-24,940),fminf(vp->Size.y-24,720)), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::Begin(T("Opciones  (F1 / Esc para volver al juego)"), nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings);
    ImGui::BeginChild("options_body",ImVec2(0,-ImGui::GetFrameHeightWithSpacing()*2),ImGuiChildFlags_Borders);
    if (ImGui::BeginTabBar("tabs")) {
        if (ImGui::BeginTabItem(T("Gráficos"))) {
            monado_quality_controls(&g_settings,g_settings.ui_lang!=0);
            quality_before=g_settings;
            ImGui::SeparatorText(T("Pantalla"));
            if (ImGui::Checkbox(T("Pantalla completa"), &g_settings.fullscreen)) want_window = true;
            const char *aspects[] = {g_settings.ui_lang ? "Automatic" : "Automático",
                                     g_settings.ui_lang ? "Original 16:9" : "16:9 original"};
            monado_combo(g_settings.ui_lang ? "Aspect ratio" : "Relación de aspecto", &g_settings.aspect, aspects, 2);
            ImGui::SeparatorText(T("Resolución y escalado"));
            int ri = 0;
            for (int i = 0; i < 7; i++) if (k_res_mode[i] == g_settings.res_mode) ri = i;
            if (monado_combo(T("Resolución interna"), &ri, TL(k_res_ui, 7).p, 7)) g_settings.res_mode = k_res_mode[ri];
            static const char *ups[] = {"Bilineal (suave)", "AMD FSR 1.0 (nítido)", "AMD FSR 3.1 (temporal)", "NVIDIA DLSS (temporal, RTX)"};
            static const int upv[] = {UPSCALE_BILINEAR, UPSCALE_FSR1, UPSCALE_FSR3, UPSCALE_DLSS};
            int ui_ = 0;
            for (int i = 0; i < 4; i++) if (upv[i] == g_settings.upscaler) ui_ = i;
            const char *preview = g_settings.upscaler_auto ? (g_settings.ui_lang ? "Automatic (DLSS on supported RTX)" : "Automático (DLSS en RTX compatible)") : T(ups[ui_]);
            if (monado_begin_combo(T("Escalado / antialiasing"), preview)) {
                if (ImGui::Selectable(g_settings.ui_lang ? "Automatic (restart)" : "Automático (reiniciar)", g_settings.upscaler_auto)) g_settings.upscaler_auto = true;
                for (int i = 0; i < 4; ++i) {
                    bool unavailable = i == 3 && !dlss_available();
                    ImGui::BeginDisabled(unavailable);
                    if (ImGui::Selectable(T(ups[i]), !g_settings.upscaler_auto && ui_ == i)) { g_settings.upscaler_auto = false; g_settings.upscaler = upv[i]; }
                    ImGui::EndDisabled();
                    if (unavailable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", dlss_status());
                }
                ImGui::EndCombo();
            }
            if (g_settings.upscaler == UPSCALE_FSR3 || g_settings.upscaler == UPSCALE_DLSS) {
                static const char *uq[] = {"Nativo (solo antialiasing)", "Calidad (67 %)", "Equilibrado (59 %)", "Rendimiento (50 %)",
                                           "Ultra rendimiento (33 %)"};
                monado_combo(T("Calidad del escalado"), &g_settings.upscale_quality, TL(uq, 5).p, 5);
                ImGui::BeginDisabled(g_settings.upscaler != UPSCALE_FSR3);
                ImGui::Checkbox(T("Generación de frames (FSR 3; activarla por primera vez: al reiniciar)"), &g_settings.frame_gen);
                ImGui::EndDisabled();
            }
            ImGui::SeparatorText(T("Calidad"));
            bool smaa = g_settings.antialias == AA_SMAA;
            if (ImGui::Checkbox(T("Antialiasing SMAA"), &smaa)) g_settings.antialias = smaa ? AA_SMAA : AA_OFF;
            if (g_settings.upscaler == UPSCALE_FSR3 || g_settings.upscaler == UPSCALE_DLSS) ImGui::SameLine(), ImGui::TextDisabled(T("(no se usa con FSR 3.1 / DLSS)"));
            bool dof = !g_settings.dof_off;
            if (ImGui::Checkbox(T("Profundidad de campo"), &dof)) g_settings.dof_off = !dof;
            ImGui::Checkbox(T("Antialiasing y desenfoque de movimiento del juego (al reiniciar)"), &g_settings.game_blur);
            bool taa = !g_settings.taa_off;
            if (ImGui::Checkbox(T("Antialiasing temporal (TAA) del juego (al reiniciar)"), &taa)) g_settings.taa_off = !taa;
            ImGui::Checkbox(g_settings.ui_lang ? "On-screen FPS counter" : "Contador FPS en pantalla", &g_settings.fps_overlay);
            ImGui::Checkbox(T("Mostrar FPS en la barra de título"), &g_settings.show_fps);
            ImGui::SeparatorText(T("Velocidad"));
            if (ImGui::Checkbox(T("V-Sync (con 30 / 60 FPS)"), &g_settings.vsync)) vk_request_swapchain_recreate();
            monado_effect_controls(&g_settings,g_settings.ui_lang!=0);
            if (ImGui::Button(g_settings.ui_lang ? "Restore graphics" : "Restaurar gráficos")) { xcde_quality_apply(&g_settings,QUALITY_BALANCED); quality_before=g_settings; }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(T("GPU / Vulkan"))) {
            ImGui::SeparatorText(T("Adaptador Vulkan activo"));
            ImGui::TextWrapped("%s", g_vk.props.deviceName);
            const char *vendor = g_vk.props.vendorID == 0x1002 ? "AMD" : g_vk.props.vendorID == 0x10de ? "NVIDIA" :
                                 g_vk.props.vendorID == 0x8086 ? "Intel" : "Unknown";
            ImGui::Text("%s | vendor 0x%04x | device 0x%04x", vendor, g_vk.props.vendorID, g_vk.props.deviceID);
            ImGui::TextWrapped("Driver %u: %s (%s)", g_vk.props.driverVersion,
                               g_vk.driver_props.driverName, g_vk.driver_props.driverInfo);
            ImGui::Text("Vulkan %u.%u.%u | adapter index %u", VK_API_VERSION_MAJOR(g_vk.props.apiVersion),
                        VK_API_VERSION_MINOR(g_vk.props.apiVersion), VK_API_VERSION_PATCH(g_vk.props.apiVersion), g_vk.phys_index);
            VkPhysicalDeviceMemoryProperties mp=g_vk.memory_props;
            VkDeviceSize local = 0, shared = 0;
            for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
                bool host_visible = false;
                for (uint32_t t = 0; t < mp.memoryTypeCount; ++t)
                    if (mp.memoryTypes[t].heapIndex == i && (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                        host_visible = true;
                if ((mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && !host_visible) local += mp.memoryHeaps[i].size;
                else shared += mp.memoryHeaps[i].size;
            }
            ImGui::Text("%s: %.0f MiB", T("Memoria local reportada"), (double)local / (1024.0 * 1024.0));
            ImGui::Text("%s: %.0f MiB", T("Memoria compartida / no local"), (double)shared / (1024.0 * 1024.0));
            if (g_vk.vma) {
                static VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
                static uint64_t budget_at; uint64_t now=SDL_GetTicksNS();
                if (!budget_at || now-budget_at>=500000000ull) { vmaGetHeapBudgets(g_vk.vma, budgets); budget_at=now; }
                ImGui::SeparatorText(T("Presupuesto de memoria (VMA / controlador)"));
                for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
                    const double used_mb = (double)budgets[i].usage / (1024.0 * 1024.0);
                    const double budget_mb = (double)budgets[i].budget / (1024.0 * 1024.0);
                    const double ours_mb = (double)budgets[i].statistics.allocationBytes / (1024.0 * 1024.0);
                    if (budget_mb > 0.0) {
                        ImGui::Text("Heap %u%s: %.0f / %.0f MiB | Monado: %.0f MiB", i,
                                    (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " (local)" : " (host)",
                                    used_mb, budget_mb, ours_mb);
                        const float pressure = (float)(budgets[i].usage / (double)budgets[i].budget);
                        ImGui::ProgressBar(pressure < 0.0f ? 0.0f : pressure > 1.0f ? 1.0f : pressure,
                                           ImVec2(-1.0f, 0.0f), "");
                    } else {
                        ImGui::Text("Heap %u: presupuesto no disponible | Monado: %.0f MiB", i, ours_mb);
                    }
                }
            }
            if (g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                ImGui::TextWrapped(T("GPU integrada: la memoria del sistema es compartida; el tamaño del heap Vulkan no equivale a VRAM dedicada."));
            ImGui::SeparatorText(T("Selección y compatibilidad (al reiniciar)"));
            xcde_gpu_picker(g_settings.ui_lang ? "Graphics adapter" : "Adaptador gráfico", g_settings.gpu_uuid,
                            sizeof g_settings.gpu_uuid, &g_settings.gpu_index, g_vk_adapters, g_vk_adapter_count, g_settings.ui_lang != 0);
            const XcdeGpuInfo *selected_gpu = xcde_gpu_preference(g_vk_adapters, g_vk_adapter_count, g_settings.gpu_uuid, g_settings.gpu_index);
            xcde_vulkan_picker(selected_gpu, g_vk.loader_api_version, &g_settings.vulkan_api, g_settings.ui_lang != 0);
            ImGui::Text(g_settings.ui_lang ? "Active route: Vulkan %u.%u" : "Ruta activa: Vulkan %u.%u",
                        VK_API_VERSION_MAJOR(g_vk.api_version), VK_API_VERSION_MINOR(g_vk.api_version));
            ImGui::TextWrapped("DLSS: %s", dlss_status());
            ImGui::Checkbox(g_settings.ui_lang ? "Restore heavy AMD volume effects (restart)" : "Restaurar efectos volumétricos AMD (reiniciar)", &g_settings.gpu_keep_heavy);
            if (g_vk.props.vendorID == 0x1002 && g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                ImGui::TextWrapped(g_settings.ui_lang ? "Three expensive volume effects are disabled by default on AMD integrated GPUs." :
                    "Tres efectos volumétricos costosos se desactivan por defecto en las GPU integradas AMD.");
            static const char *profiles[] = {"Automático (recomendado)", "Compatibilidad (desactiva rutas opcionales rápidas)"};
            int profile = g_settings.gpu_profile == GPU_PROFILE_COMPATIBILITY ? 1 : 0;
            if (monado_combo(T("Perfil gráfico"), &profile, TL(profiles, 2).p, 2))
                g_settings.gpu_profile = profile ? GPU_PROFILE_COMPATIBILITY : GPU_PROFILE_AUTO;
            static const char *gpl_modes[] = {"Según el controlador", "Desactivadas", "Activadas si son compatibles"};
            int gpl = g_settings.gpu_gpl == GPU_GPL_OFF ? 1 : g_settings.gpu_gpl == GPU_GPL_ON ? 2 : 0;
            if (monado_combo(T("Bibliotecas de pipelines"), &gpl, TL(gpl_modes, 3).p, 3))
                g_settings.gpu_gpl = gpl == 1 ? GPU_GPL_OFF : gpl == 2 ? GPU_GPL_ON : GPU_GPL_AUTO;
            ImGui::Checkbox(T("Presupuesto de memoria Vulkan"), &g_settings.gpu_memory_budget);
            ImGui::TextDisabled(T("Estos cambios requieren reiniciar el juego."));
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(T("Audio"))) {
            int vol = (int)lroundf(g_settings.volume * 100);
            if (ImGui::SliderInt(T("Volumen"), &vol, 0, 200, "%d %%")) g_settings.volume = vol / 100.0f;
            ImGui::Checkbox(T("Audio de los vídeos (al reiniciar)"), &g_settings.movie_audio);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(T("Controles"))) {
            ImGui::TextWrapped(T("La asignación de teclas y botones se cambia en el launcher (Play Monado.bat), pestaña Controles."));
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
    if (g_settings.quality_preset==quality_before.quality_preset && !xcde_quality_equal(xcde_quality_current(&quality_before),xcde_quality_current(&g_settings))) { g_settings.quality_preset=QUALITY_CUSTOM; saved_note=false; }
    if (memcmp(&opening,&g_settings,sizeof g_settings)!=0) saved_note=false;
    ImGui::Separator();
    if (ImGui::Button(g_settings.ui_lang ? "Apply" : "Aplicar")) { settings_save(); saved_note = true; }
    ImGui::SameLine();
    if (ImGui::Button(T("Volver al juego"))) { open_ = false; input_reset(); }
    if (saved_note) { ImGui::SameLine(); ImGui::TextDisabled(T("Guardado en xcde.ini")); }
    ImGui::End();
}

extern "C" void menu_render(void *cmd, uint32_t w, uint32_t h) {
    vk_submit_drain();  // [XCDE] the backend may submit on the queue itself (texture uploads): the submission thread idles first
    static ImGuiContext *ctx;
    static float font_h;
    static uint64_t t_last;
    bool show_counter = g_settings.fps_overlay && !diag_deterministic();
    if (!open_ && !show_counter) return;
    if (!ctx) {
        ctx = ImGui::CreateContext();
        ImGui::SetCurrentContext(ctx);
        ImGuiIO &io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
        io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
        font_h = fmaxf(16.0f, h / 1080.0f * 22.0f);
        // [XCDE] the first font that exists (Windows; then SteamOS / Linux: Noto Sans, DejaVu Sans), else ImGui's own
        static const char *const fonts[] = {"C:\\Windows\\Fonts\\segoeui.ttf", "/usr/share/fonts/noto/NotoSans-Regular.ttf",
                                            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"};
        ImFont *font = nullptr;
        for (const char *f : fonts) {
            FILE *fp = fopen(f, "rb");
            if (!fp) continue;
            fclose(fp);
            if ((font = io.Fonts->AddFontFromFileTTF(f, font_h, nullptr, io.Fonts->GetGlyphRangesDefault()))) break;
        }
        if (!font) io.Fonts->AddFontDefault();
        monado_theme(font_h / 16.0f);
        if (!vk_imgui_backend_init()) { ImGui::DestroyContext(ctx); ctx = nullptr; open_ = false; return; }
        ImGui_ImplVulkan_CreateFontsTexture();
    }
    ImGui::SetCurrentContext(ctx);
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)w, (float)h);
    uint64_t now = SDL_GetTicksNS();
    io.DeltaTime = t_last ? (float)((now - t_last) / 1e9) : 1.0f / 60;
    if (io.DeltaTime <= 0) io.DeltaTime = 1.0f / 60;
    t_last = now;
    if (open_) feed(io, w, h);
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    if (open_) ui();
    if (show_counter) {
        ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.65f);
        ImGui::Begin("##native-fps", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
        if (fps_meter.fps > 0) ImGui::Text("FPS: %.1f  |  %.2f ms", fps_meter.fps, fps_meter.ms);
        else ImGui::TextUnformatted("FPS: --");
        ImGui::End();
    }
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), (VkCommandBuffer)cmd);
}
