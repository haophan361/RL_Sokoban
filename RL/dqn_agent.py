from collections import deque

import torch
import numpy as np
from dqn import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer, SILBuffer
import torch.nn as nn

class DQNAgent:

    def __init__(self,
                 device=None,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 replay_capacity: int = 100_000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: int = 500_000,
                 target_update_frequency: int = 1000,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_frames: int = 500_000,
                 solve_boost: float = 3.0,
                 inference_only: bool = False,
                 sil_fraction: float = 0.15,
                 sil_loss_weight: float = 0.01):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_only = inference_only
        self.total_steps = 0
        self.gradient_steps = 0
        self.solve_boost = solve_boost

        self.policy_net = DuelingDQN(in_channels=4).to(self.device)
        self.sil_buffer = None
        self._recent_replay_samples = deque(maxlen=100)

        if not self.inference_only:
            if batch_size < 1:
                raise ValueError("batch_size must be at least 1")
            if not np.isfinite(solve_boost) or solve_boost <= 0:
                raise ValueError("solve_boost must be finite and positive")
            if not 0 <= sil_fraction < 1:
                raise ValueError("sil_fraction must be in [0, 1)")
            if not np.isfinite(sil_loss_weight) or sil_loss_weight < 0:
                raise ValueError("sil_loss_weight must be finite and non-negative")

            self.gamma = gamma
            self.batch_size = batch_size
            self.epsilon_start = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.target_update_frequency = target_update_frequency
            self.solve_boost = solve_boost

            self.target_net = DuelingDQN(in_channels=4).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr)

            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_capacity,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
            )
            self.sil_batch_size = min(batch_size - 1, round(batch_size * sil_fraction))
            self.sil_loss_weight = sil_loss_weight
            if self.sil_batch_size > 0 and self.sil_loss_weight > 0:
                self.sil_buffer = SILBuffer()
        else:
            self.policy_net.eval()
            self.target_net = None
            self.optimizer = None
            self.replay_buffer = None
            self.epsilon_start = 0.0
            self.epsilon_end = 0.0
            self.epsilon_decay = 1
            self.gamma = gamma
            self.batch_size = batch_size

    def get_epsilon(self) -> float:
        fraction = min(self.total_steps / self.epsilon_decay, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def select_action(self, observation, action_mask, training: bool = True):
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            return None

        epsilon = self.get_epsilon() if training else 0.0

        if training and np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        state_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor  = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        q_values = self.policy_net(state_tensor, action_mask=mask_tensor)[0]
        q_values = q_values.masked_fill(~mask_tensor[0], -torch.inf)
        return int(torch.argmax(q_values).item())

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        solved_batch = self.sil_buffer.sample(self.sil_batch_size) if self.sil_buffer is not None else []
        per_batch_size = self.batch_size - len(solved_batch)
        batch, tree_indices, is_weights = self.replay_buffer.sample(per_batch_size)

        current_batch = batch + solved_batch
        states = torch.as_tensor(np.stack([t.state for t in current_batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in current_batch], dtype=torch.long, device=self.device)
        masks = torch.as_tensor(np.stack([t.action_mask for t in current_batch]), dtype=torch.bool, device=self.device)
        rewards= torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor([self.gamma ** t.n_steps for t in batch], dtype=torch.float32, device=self.device)
        next_states= torch.as_tensor(np.stack([t.next_state for t in batch]),dtype=torch.float32, device=self.device)
        dones= torch.as_tensor([t.done for t in batch], dtype=torch.bool, device=self.device)
        next_masks= torch.as_tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.bool, device=self.device)

        q_values = self.policy_net(states, action_mask=masks)
        chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q = chosen_q[:per_batch_size]

        with torch.no_grad():

            next_q_policy = self.policy_net(next_states, action_mask=next_masks)
            next_q_policy = next_q_policy.masked_fill(~next_masks, -1e9)
            best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states, action_mask=next_masks)
            next_q = next_q_target.gather(1, best_next_actions).squeeze(1)

            terminal_mask = dones | ~next_masks.any(dim=1)
            next_q = torch.where(terminal_mask, torch.zeros_like(next_q), next_q)

            target = rewards + discounts * next_q

        element_loss = nn.functional.smooth_l1_loss(current_q, target, reduction="none")

        weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)
        loss = (weights_t * element_loss).mean()
        if solved_batch:
            mc_returns = torch.as_tensor(
                [t.mc_return for t in solved_batch], dtype=torch.float32, device=self.device
            )
            sil_loss = (mc_returns - chosen_q[per_batch_size:]).clamp_min(0).square().mean()
            loss = loss + self.sil_loss_weight * sil_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        td_errors = (current_q - target).detach().cpu().numpy()
        self.replay_buffer.update_priorities(tree_indices, td_errors)
        self._recent_replay_samples.append((
            sum(t.transition_type == "solve" for t in batch),
            per_batch_size,
            len(solved_batch),
        ))

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, episode, best_val_solve_rate, path):
        torch.save({
            "episode": episode,
            "best_val_solve_rate": best_val_solve_rate,
            "epsilon": self.get_epsilon(),
            "batch_size": self.batch_size,
            "epsilon_decay": self.epsilon_decay,
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "sil_buffer": self.sil_buffer.state_dict() if self.sil_buffer is not None else None,
            "total_steps": self.total_steps,
            "gradient_steps": self.gradient_steps,
        }, path)

    def load(self, path, load_buffer: bool = True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"], strict=False)
        if self.target_net is not None:
            self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["policy_net"]), strict=False)

        if not self.inference_only and "optimizer" in checkpoint and load_buffer:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except (ValueError, KeyError):
                print("[LOAD] Optimizer state incompatible with new architecture — RESET")

        self.total_steps    = checkpoint.get("total_steps", 0)
        self.gradient_steps = checkpoint.get("gradient_steps", 0)

        if not self.inference_only and load_buffer and "replay_buffer" in checkpoint:
            if self.replay_buffer.load_state_dict(checkpoint["replay_buffer"]):
                print(f"[LOAD] Replay buffer restored: {len(self.replay_buffer):,} transitions")
            else:
                print("[LOAD] Legacy replay lacks source action masks and n-step lengths (warm-up will run)")
        elif not load_buffer or self.inference_only:
            print("[LOAD] Skipped loading replay buffer (eval/inference mode)")
        else:
            print("[LOAD] No replay buffer in checkpoint (warm-up will run)")

        if load_buffer and self.sil_buffer is not None and checkpoint.get("sil_buffer") is not None:
            self.sil_buffer.load_state_dict(checkpoint["sil_buffer"])
            print(f"[LOAD] SIL buffer restored: {len(self.sil_buffer):,} solved episodes")
            
        print(
            f"[LOAD] episode={checkpoint.get('episode', '?')} | "
            f"solve_rate={checkpoint.get('best_val_solve_rate', '?')} | "
            f"epsilon={self.get_epsilon():.4f} | "
            f"total_steps={self.total_steps:,}"
        )
        return checkpoint

    def buffer_stats(self) -> dict:
        if self.inference_only or self.replay_buffer is None:
            return {}
        stats = self.replay_buffer.buffer_stats()
        solve_count = sum(sample[0] for sample in self._recent_replay_samples)
        per_count = sum(sample[1] for sample in self._recent_replay_samples)
        sil_count = sum(sample[2] for sample in self._recent_replay_samples)
        stats["recent_solve_sample_pct"] = 100 * solve_count / max(per_count, 1)
        stats["recent_sil_sample_pct"] = 100 * sil_count / max(per_count + sil_count, 1)
        stats["sil_episodes"] = len(self.sil_buffer) if self.sil_buffer is not None else 0
        return stats
