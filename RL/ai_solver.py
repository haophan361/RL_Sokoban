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

_DIR_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
_DIR_CHAR = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
_DIR_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

CHECKPOINT_PATH = _RL_DIR / "models" / "dqn_sokoban_last.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = _RL_DIR / "models" / "dqn_sokoban_last.pt"

_cached_agent = None

def get_agent(checkpoint_path=None):
    global _cached_agent
    if _cached_agent is not None and checkpoint_path is None:
        return _cached_agent

    path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device, inference_only=True)

    if path.exists():
        print(f"[DQN] Loading model checkpoint: {path}")
        agent.load(str(path), load_buffer=False)
    else:
        print(f"[DQN] Checkpoint not found: {path}")

    agent.policy_net.eval()
    if checkpoint_path is None:
        _cached_agent = agent
    return agent

def dqn_step_generator(matrix, max_pushes=100, checkpoint_path=None):
    agent = get_agent(checkpoint_path)
    env = sokoban_cpp.SokobanEnv(max_pushes=max_pushes, avoid_dead_squares=False)
    env.set_level(matrix, [])
    obs = env.reset()

    done = False
    step_idx = 0

    while not done:
        action_mask = env.get_valid_action_mask()
        if not action_mask.any():
            yield {
                "step": step_idx,
                "action_id": None,
                "box_pos": None,
                "direction": None,
                "matrix": env.get_matrix(),
                "boxes": env.get_boxes_position(),
                "worker": env.get_worker_position(),
                "docks": env.get_docks_position(),
                "done": True,
                "solved": env.is_complete(),
                "deadlock": True,
                "timeout": False,
                "message": "Không còn hành động đẩy hợp lệ (No valid push actions)"
            }
            break

        action_id = agent.select_action(obs, action_mask, training=False)
        if action_id is None:
            break

        pos_dir = env.action_id_to_position_direction(action_id)
        if pos_dir is None:
            break
        box_y, box_x, direction = pos_dir

        obs, reward, done, info = env.step(action_id)
        step_idx = env.step_count

        current_matrix = env.get_matrix()
        boxes_pos = env.get_boxes_position()
        worker_pos = env.get_worker_position()
        docks_pos = env.get_docks_position()
        is_solved = info.get("solved", False) or env.is_complete()
        is_deadlock = info.get("deadlock", False)
        is_timeout = info.get("timeout", False)

        yield {
            "step": step_idx,
            "action_id": action_id,
            "box_pos": (box_y, box_x),
            "direction": direction,
            "direction_name": _DIR_NAMES.get(direction, str(direction)),
            "direction_char": _DIR_CHAR.get(direction, ""),
            "direction_delta": _DIR_DELTA.get(direction, (0, 0)),
            "matrix": current_matrix,
            "boxes": boxes_pos,
            "worker": worker_pos,
            "docks": docks_pos,
            "reward": reward,
            "done": done,
            "solved": is_solved,
            "deadlock": is_deadlock,
            "timeout": is_timeout,
            "message": "Success" if is_solved else ("Deadlock" if is_deadlock else "Pushing")
        }

        if is_solved or done:
            break

