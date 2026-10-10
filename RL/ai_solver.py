
import copy
from time import time
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sokoban_env import SokobanEnv
from dqn_agent import DQNAgent

_DIR_CHAR = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
CHECKPOINT_PATH= "RL/models/dqn_sokoban_best.pt"

def dqn_solver(state,max_pushes=200):

    start = time()

    print('Processing DQN......')

    env = SokobanEnv(
        map_matrix=copy.deepcopy(state.matrix),
        dead_squares=[],
        max_pushes=max_pushes,
        avoid_dead_squares=False,
    )
    obs = env.reset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = DQNAgent(lr=1e-4, gamma=0.99, batch_size=64, replay_capacity=1, device=device)
    agent.load(CHECKPOINT_PATH)
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
        push_action = env.action_id_to_push(action_id)
        if push_action is None:
            break
        path_to_push = env.find_path_to_push_box(
            push_action.box_position[0], push_action.box_position[1], push_action.direction)
        if path_to_push:
            for move_idx in path_to_push:
                full_path += _DIR_CHAR[move_idx]
                node_generated += 1
        full_path += _DIR_CHAR[push_action.direction]
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