import torch
import numpy as np
from replay_buffer import PrioritizedReplayBuffer, SILBuffer
from dqn import DuelingDQN
import torch.nn as nn

class DQNAgent:

    def __init__(self,lr,gamma,batch_size,replay_capacity,epsilon_start,epsilon_end,epsilon_decay,
                target_update_frequency,device,per_alpha,per_beta_start,per_beta_frames,
                sil_capacity=50_000, sil_batch_size=128, sil_coeff=0.1):
                
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.total_steps = 0
        self.gradient_steps = 0
        self.sil_batch_size = sil_batch_size
        self.sil_coeff = sil_coeff

        self.policy_net = DuelingDQN(in_channels=4).to(device)
        self.target_net = DuelingDQN(in_channels=4).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )
        self.sil_buffer = SILBuffer(capacity=sil_capacity)

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
        q_values = self.policy_net(state_tensor)[0]
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
        return int(torch.argmax(q_values).item())

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch, tree_indices, is_weights = self.replay_buffer.sample(self.batch_size)

        states= torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions= torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards= torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states= torch.as_tensor(np.stack([t.next_state for t in batch]),dtype=torch.float32, device=self.device)
        dones= torch.as_tensor([t.done for t in batch], dtype=torch.bool, device=self.device)
        next_masks= torch.as_tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.bool, device=self.device)

        q_values  = self.policy_net(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            next_q_policy = self.policy_net(next_states)
            next_q_policy = next_q_policy.masked_fill(~next_masks, -1e9)
            best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, best_next_actions).squeeze(1)

            terminal_mask = dones | ~next_masks.any(dim=1)
            next_q = torch.where(terminal_mask, torch.zeros_like(next_q), next_q)

            target = rewards + self.gamma * next_q

        element_loss = nn.functional.smooth_l1_loss(current_q, target, reduction="none")

        weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)
        loss = (weights_t * element_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        td_errors = (current_q - target).detach().cpu().numpy()
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def sil_optimize(self) -> float | None:
        if len(self.sil_buffer) < self.sil_batch_size:
            return None

        batch = self.sil_buffer.sample(self.sil_batch_size)
        states_t  = torch.as_tensor(np.stack([x[0] for x in batch]), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
        returns_t = torch.as_tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)

        q_vals = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        sil_loss = (self.sil_coeff * 0.5 * torch.relu(returns_t - q_vals).pow(2)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        sil_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(sil_loss.item())

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
            "sil_buffer": self.sil_buffer.state_dict(),
            "total_steps": self.total_steps,
            "gradient_steps": self.gradient_steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps    = checkpoint.get("total_steps", 0)
        self.gradient_steps = checkpoint.get("gradient_steps", 0)
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
            print(f"[LOAD] Replay buffer restored: {len(self.replay_buffer):,} transitions")
        else:
            print("[LOAD] No replay buffer in checkpoint (warm-up will run)")
        if "sil_buffer" in checkpoint:
            self.sil_buffer.load_state_dict(checkpoint["sil_buffer"])
            print(f"[LOAD] SIL buffer restored: {len(self.sil_buffer):,} transitions")
        print(
            f"[LOAD] episode={checkpoint.get('episode', '?')} | "
            f"solve_rate={checkpoint.get('best_val_solve_rate', '?')} | "
            f"epsilon={self.get_epsilon():.4f} | "
            f"total_steps={self.total_steps:,}"
        )
        return checkpoint

    def buffer_stats(self) -> dict:
        return self.replay_buffer.buffer_stats()