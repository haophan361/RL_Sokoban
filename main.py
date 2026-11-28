import os
import sys
import copy
import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame

import assets
from game import Game
from RL.ai_solver import dqn_step_generator

HEADER_HEIGHT = 85
TILE_SIZE = 64
BG_COLOR = (30, 32, 44)
HEADER_BG = (22, 24, 34)
TEXT_COLOR = (240, 240, 240)
SUBTEXT_COLOR = (180, 185, 200)

def load_map(level_path):
    with open(level_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["map_matrix"]

class Button:
    def __init__(self, rect, text, bg_color, hover_color, text_color=(255, 255, 255), border_radius=6):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.is_hovered = False

    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(screen, color, self.rect, border_radius=self.border_radius)
        txt_surf = font.render(self.text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

def open_level_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    levels_dir = Path(__file__).resolve().parent / "levels"
    if not levels_dir.exists():
        levels_dir = Path.cwd()

    file_path = filedialog.askopenfilename(
        title="Chọn file Level Sokoban (.json)",
        initialdir=str(levels_dir),
        filetypes=[("JSON Level", "*.json"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def find_default_level():
    levels_dir = Path(__file__).resolve().parent / "levels"
    if levels_dir.exists():
        for p in levels_dir.rglob("*.json"):
            return str(p)
    return None

def main():
    pygame.init()
    pygame.font.init()
    assets.load_sprites()

    font_btn = pygame.font.SysFont("Segoe UI, Arial", 14, bold=True)
    font_info = pygame.font.SysFont("Segoe UI, Arial", 13)
    font_status = pygame.font.SysFont("Segoe UI, Arial", 14, bold=True)
    font_large = pygame.font.SysFont("Segoe UI, Arial", 26, bold=True)

    default_level_path = find_default_level()
    current_level_path = default_level_path

    if current_level_path and os.path.exists(current_level_path):
        initial_matrix = load_map(current_level_path)
        level_name = Path(current_level_path).name
    else:
        messagebox.showerror("Lỗi", "Không tìm thấy file level mặc định. Vui lòng chọn file level.")
        selected_file = open_level_file_dialog()
        if selected_file:
            current_level_path = selected_file
            initial_matrix = load_map(selected_file)
            level_name = Path(selected_file).name
        else:
            sys.exit()
    
    current_matrix = copy.deepcopy(initial_matrix)
    game_sokoban = Game(current_matrix)

    def calculate_window_size(matrix):
        h = len(matrix)
        w = max(len(row) for row in matrix) if h > 0 else 0
        board_w = max(w * TILE_SIZE, 680)
        board_h = h * TILE_SIZE + HEADER_HEIGHT
        win_w = max(board_w, 700)
        win_h = max(board_h, 540)
        return win_w, win_h

    win_w, win_h = calculate_window_size(current_matrix)
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DQN Visualize")

    btn_load = Button((15, 12, 120, 36), "Level", (41, 98, 255), (68, 120, 255))
    btn_start = Button((145, 12, 125, 36), "Bắt đầu", (46, 125, 50), (67, 160, 71))
    btn_reset = Button((280, 12, 95, 36), "Reset", (211, 47, 47), (239, 83, 80))

    ai_generator = None
    ai_running = False
    ai_step_delay = 0.35
    last_step_time = 0.0

    status_text = "Sẵn sàng (Ready)"
    status_color = (130, 220, 130)
    pushes_count = 0
    last_action_desc = "None"
    is_completed = False

    clock = pygame.time.Clock()
    running = True

    while running:
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if btn_load.handle_event(event):
                selected_file = open_level_file_dialog()
                if selected_file:
                    try:
                        current_level_path = selected_file
                        level_name = Path(selected_file).name
                        initial_matrix = load_map(selected_file)
                        current_matrix = copy.deepcopy(initial_matrix)
                        game_sokoban.set_matrix(current_matrix)

                        ai_generator = None
                        ai_running = False
                        pushes_count = 0
                        last_action_desc = "None"
                        is_completed = False
                        status_text = f"Đã nạp {level_name} - Sẵn sàng"
                        status_color = (130, 220, 130)

                        win_w, win_h = calculate_window_size(current_matrix)
                        screen = pygame.display.set_mode((win_w, win_h))
                    except Exception as e:
                        print(f"Lỗi nạp file level: {e}")
                        status_text = f"Lỗi đọc file: {e}"
                        status_color = (255, 100, 100)

            if btn_start.handle_event(event):
                if not ai_running and not is_completed:
                    ai_generator = dqn_step_generator(current_matrix,max_pushes=100)
                    ai_running = True
                    status_text = "AI thinking..."
                    status_color = (255, 215, 64)
                    last_step_time = current_time

            if btn_reset.handle_event(event):
                current_matrix = copy.deepcopy(initial_matrix)
                game_sokoban.set_matrix(current_matrix)
                ai_generator = None
                ai_running = False
                pushes_count = 0
                is_completed = False
                status_text = "Đã Reset về ban đầu"
                status_color = (180, 200, 240)

        if ai_running and ai_generator is not None:
            if current_time - last_step_time >= ai_step_delay:
                last_step_time = current_time
                try:
                    step_info = next(ai_generator)
                    pushes_count = step_info["step"]
                    current_matrix = step_info["matrix"]
                    game_sokoban.set_matrix(current_matrix)

                    if step_info.get("box_pos") and step_info.get("direction_name"):
                        b_pos = step_info["box_pos"]
                        d_name = step_info["direction_name"]
                        last_action_desc = f"Hộp {b_pos} -> {d_name}"

                    if step_info.get("solved", False):
                        ai_running = False
                        is_completed = True
                        status_text = f"Hoàn thành! Game Solved ({pushes_count} pushes) 🎉"
                        status_color = (76, 217, 100)
                    elif step_info.get("deadlock", False):
                        ai_running = False
                        status_text = "Phát hiện Deadlock (Bế tắc)!"
                        status_color = (255, 99, 71)
                    elif step_info.get("done", False):
                        ai_running = False
                        status_text = "Đã dừng (Hết lượt đẩy hoặc không có action hợp lệ)"
                        status_color = (255, 179, 0)
                    else:
                        status_text = f"AI đang đẩy hộp... (Push #{pushes_count})"
                        status_color = (255, 215, 64)

                except StopIteration:
                    ai_running = False
                    if not is_completed:
                        status_text = "AI đã kết thúc chuỗi hành động"
                        status_color = (200, 200, 200)

        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, HEADER_BG, (0, 0, win_w, HEADER_HEIGHT))
        pygame.draw.line(screen, (50, 55, 75), (0, HEADER_HEIGHT), (win_w, HEADER_HEIGHT), 2)

        btn_load.draw(screen, font_btn)
        btn_start.draw(screen, font_btn)
        btn_reset.draw(screen, font_btn)

        lbl_status = font_status.render(status_text, True, status_color)
        screen.blit(lbl_status, (15, 55))

        map_w, map_h = game_sokoban.load_size(tile_size=TILE_SIZE)
        avail_h = win_h - HEADER_HEIGHT
        offset_x = max(0, (win_w - map_w) // 2)
        offset_y = HEADER_HEIGHT + max(0, (avail_h - map_h) // 2)

        game_sokoban.print_game(screen, offset_x=offset_x, offset_y=offset_y, tile_size=TILE_SIZE)

        if is_completed:
            overlay = pygame.Surface((win_w, 60), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, win_h // 2 - 30))
            win_surf = font_large.render("GAME SOLVED BY DQN AGENT!", True, (76, 217, 100))
            win_rect = win_surf.get_rect(center=(win_w // 2, win_h // 2))
            screen.blit(win_surf, win_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
