from time import time
import torch
import sys
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parent
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))
_CPP_DIR = _RL_DIR / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))

import sokoban_cpp
from dqn_agent import DQNAgent

_DIR_CHAR = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
CHECKPOINT_PATH = _RL_DIR / "models" / "dqn_sokoban_best.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = _RL_DIR / "models" / "dqn_sokoban_last.pt"

def dqn_solver(state, max_pushes=200):
    start = time()
    print('Processing DQN......')

    matrix = state.get_matrix() if hasattr(state, 'get_matrix') else state.matrix

    env = sokoban_cpp.SokobanEnv(max_pushes=max_pushes, avoid_dead_squares=False)
    env.set_level(matrix, [])
    obs = env.reset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DQNAgent(
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        replay_capacity=1000,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
        target_update_frequency=1000,
        device=device,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=1000,
    )

    if CHECKPOINT_PATH.exists():
        agent.load(str(CHECKPOINT_PATH), load_buffer=False)
    else:
        print(f'[DQN] Khong tim thay checkpoint: {CHECKPOINT_PATH}')
        return None

    agent.policy_net.eval()

    full_path = ''
    node_generated = 0
    done = False
    info = {}

    while not done:
        action_mask = env.get_valid_action_mask()
        if not action_mask.any():
            break
        action_id = agent.select_action(obs, action_mask, training=False)
        if action_id is None:
            break

        pos_dir = env.action_id_to_position_direction(action_id)
        if pos_dir is None:
            break
        box_y, box_x, direction = pos_dir

        path_to_push = env.find_path_to_push(box_y, box_x, direction)
        if path_to_push is not None:
            for move_idx in path_to_push:
                full_path += _DIR_CHAR[move_idx]
                node_generated += 1
        full_path += _DIR_CHAR[direction]
        node_generated += 1

        obs, reward, done, info = env.step(action_id)

    elapsed = round(time() - start, 2)
    if info.get('solved', False):
        print(f'Time to find solution: {elapsed} seconds')
        print(f'Number of visited nodes: {node_generated}')
        print(f'Solution: {full_path}  Number steps: {len(full_path)}')
        return full_path
    else:
        print(f'[DQN] Khong giai duoc sau {env.step_count} pushes ({elapsed}s)')
        return None
