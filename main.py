import time
import pygame
import assets
from game import Game
import os
import tkinter as tk
from tkinter import ttk, messagebox
from solver import bfs, dfs, astar, backtracking, simulated_annealing
from RL.ai_solver import dqn_solver
from state import State
import copy

def load_map(level):
    list_map = []
    path = f"levels\\{level}.txt"
    if not os.path.exists(path):
        print(f"Thư mục {path} không tồn tại.")
        return
    else:
        with open(path, "r") as file:
            for line in file:
                row = [char for char in line.rstrip('\n')]
                list_map.append(row)
    return list_map

def show_win_game(screen):
    font = pygame.font.SysFont("Arial", 36)
    win_text = font.render("You Win!", True, (0, 255, 0))
    text_rect = win_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(win_text, text_rect)
    pygame.display.flip()

def start_game():
    select_level = combobox.get().lower().replace(" ", "")
    if select_level == "chooselevel":
        messagebox.showinfo("Choose a level", "Please choose a level")
        return

    matrix = load_map(select_level)

    initial_matrix=copy.deepcopy(matrix)

    assets.load_sprites()

    pygame.init()
    sprites = pygame.sprite.LayeredUpdates()

    pygame.display.set_caption("Sokoban")

    gameSokoban = Game(matrix, [])

    size = gameSokoban.load_size()
    screen = pygame.display.set_mode(size)

    Game.fill_screen_with_floor(size, screen)

    running = True
    list_dock = gameSokoban.get_docks()

    machinePlay = False
    path = ""
    state = State(matrix)
    lenPath = 0
    i = 0

    while running:
        gameSokoban.fill_screen_with_floor(size, screen)
        gameSokoban.print_game(screen)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and not machinePlay:
                if event.key == pygame.K_z:
                    if gameSokoban.stack_matrix:
                        prev_matrix = gameSokoban.stack_matrix.pop()
                        gameSokoban.matrix = prev_matrix

                if event.key == pygame.K_r:
                    gameSokoban.matrix = copy.deepcopy(initial_matrix)
                    gameSokoban.stack_matrix=[]

                elif event.key == pygame.K_UP:
                    gameSokoban.move(-1, 0, list_dock)
                elif event.key == pygame.K_DOWN:
                    gameSokoban.move(1, 0, list_dock)
                elif event.key == pygame.K_LEFT:
                    gameSokoban.move(0, -1, list_dock)
                elif event.key == pygame.K_RIGHT:
                    gameSokoban.move(0, 1, list_dock)
                elif event.key == pygame.K_b:
                    state.matrix = gameSokoban.matrix
                    path = bfs(state)
                    lenPath = len(path) if path is not None else 0  
                    machinePlay = True
                elif event.key == pygame.K_d:
                    state.matrix = gameSokoban.matrix
                    path = dfs(state)
                    lenPath = len(path) if path is not None else 0
                    machinePlay = True
                elif event.key == pygame.K_a:
                    state.matrix = gameSokoban.matrix
                    path = astar(state)
                    lenPath = len(path) if path is not None else 0
                    machinePlay = True
                elif event.key == pygame.K_k:
                    state.matrix = gameSokoban.matrix
                    path = backtracking(state)
                    lenPath = len(path) if path is not None else 0
                    machinePlay = True
                elif event.key == pygame.K_s:
                    state.matrix = gameSokoban.matrix
                    path = simulated_annealing(state)
                    lenPath = len(path) if path is not None else 0
                    machinePlay = True
                elif event.key == pygame.K_q:
                    state.matrix = gameSokoban.matrix
                    path = dqn_solver(state)
                    lenPath = len(path) if path is not None else 0
                    machinePlay = True
            if event.type == pygame.QUIT:
                running = False

        if machinePlay:
            if i == lenPath:
                machinePlay = False
            elif i < lenPath:
                move = path[i]
                if move == 'U':
                    state.move(-1, 0)
                elif move == 'D':
                    state.move(1, 0)
                elif move == 'L':
                    state.move(0, -1)
                elif move == 'R':
                    state.move(0, 1)
            i += 1
            time.sleep(0.1)

        gameSokoban.print_game(screen)
        pygame.display.update()

        if gameSokoban.is_completed(list_dock):
            show_win_game(screen)

    pygame.quit()


root = tk.Tk()
root.title("My Sokoban")

window_width = 540
window_height = 420
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = screen_width // 2 - window_width // 2
y = screen_height // 2 - window_height // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(False, False)

top_frame = tk.Frame(root)
top_frame.pack(fill="x", padx=15, pady=15)

label = tk.Label(top_frame, text="Select Level:", font=("Arial", 12, "bold"))
label.pack(side="left", padx=(0, 10))

options = ['Choose Level', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5',
           'Level 6', 'Level 7', 'Level 8', 'Level 9', 'Level 10']
combobox = ttk.Combobox(top_frame, values=options, width=15, font=("Arial", 11), state="readonly")
combobox.current(0)
combobox.pack(side="left", padx=(0, 10))

button_start = tk.Button(top_frame, text="Start Game", command=start_game, font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", padx=10, pady=2, cursor="hand2")
button_start.pack(side="right")

guide_frame = tk.LabelFrame(root, text=" HƯỚNG DẪN CÁC PHÍM TRONG GAME ", font=("Arial", 11, "bold"), padx=15, pady=10)
guide_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

col1_frame = tk.LabelFrame(guide_frame, text=" Thao tác người chơi ", font=("Arial", 10, "bold"), padx=10, pady=8)
col1_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

basic_controls = [
    ("↑ ↓ ← →", "Di chuyển nhân vật"),
    ("Z", "Quay lại bước trước"),
    ("R", "Chơi lại từ đầu")
]

for key, desc in basic_controls:
    row = tk.Frame(col1_frame)
    row.pack(fill="x", pady=4)
    lbl_key = tk.Label(row, text=key, font=("Consolas", 10, "bold"), width=10, relief="groove", bg="#f0f0f0", bd=1)
    lbl_key.pack(side="left")
    lbl_desc = tk.Label(row, text=f": {desc}", font=("Arial", 9), anchor="w")
    lbl_desc.pack(side="left", fill="x", expand=True, padx=5)

col2_frame = tk.LabelFrame(guide_frame, text=" Máy tự chơi ", font=("Arial", 10, "bold"), padx=10, pady=8)
col2_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

ai_controls = [
    ("B", "Breadth-First Search (BFS)"),
    ("D", "Depth-First Search (DFS)"),
    ("A", "A* (A-Star Algorithm)"),
    ("K", "Backtracking"),
    ("S", "Simulated Annealing"),
    ("Q", "DQN (Deep Q-Network)")
]

for key, desc in ai_controls:
    row = tk.Frame(col2_frame)
    row.pack(fill="x", pady=3)
    lbl_key = tk.Label(row, text=key, font=("Consolas", 10, "bold"), width=4, relief="groove", bg="#e8f0fe", fg="#1a73e8", bd=1)
    lbl_key.pack(side="left")
    lbl_desc = tk.Label(row, text=f": {desc}", font=("Arial", 9), anchor="w")
    lbl_desc.pack(side="left", fill="x", expand=True, padx=5)

root.mainloop()