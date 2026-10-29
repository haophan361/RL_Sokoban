import numpy as np
from tqdm import tqdm
from collections import deque
from pathlib import Path
import wandb


def _compute_n_step_return(buf: deque, gamma: float):
    s0, a0 = buf[0][0], buf[0][1]
    n_reward = 0.0
    g = 1.0

    for _, _, r, sn_i, done_i, mask_i in buf:
        n_reward += g * r
        g *= gamma
        if done_i:
            return s0, a0, n_reward, sn_i, True, mask_i

    last = buf[-1]
    return s0, a0, n_reward, last[3], last[4], last[5]


def evaluate(agent, env, dataset):
    solved_count = 0
    total_reward = 0.0
    push_list = []

    for idx in range(len(dataset)):
        map_matrix, dead_squares = dataset[idx]
        env.set_level(map_matrix, dead_squares)
        obs = env.reset()

        episode_reward = 0.0
        done = False
        info = {}

        while not done:
            action_mask = env.get_valid_action_mask()
            if not action_mask.any():
                break

            action = agent.select_action(obs, action_mask, training=False)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        solved = info.get("solved", False)
        pushes = info.get("pushes", env.step_count)
        total_reward += episode_reward
        if solved:
            solved_count += 1
            push_list.append(pushes)

    n = len(dataset)
    return {
        "solve_rate":  solved_count / n,
        "mean_reward": total_reward / n,
        "mean_pushes": float(np.mean(push_list)) if push_list else 0.0,
    }


def warmup_replay_buffer(agent, env, dataset, warm_up_steps: int, n_step: int = 3):
    tqdm.write(f"[Warm-up] Filling replay buffer ({warm_up_steps} random steps)...")

    level_idx = np.random.randint(len(dataset))
    map_matrix, dead_squares = dataset[level_idx]
    env.set_level(map_matrix, dead_squares)
    obs = env.reset()

    wu_buf = deque(maxlen=n_step)
    for _ in range(warm_up_steps):
        action_mask = env.get_valid_action_mask()
        if not action_mask.any():
            level_idx = np.random.randint(len(dataset))
            map_matrix, dead_squares = dataset[level_idx]
            env.set_level(map_matrix, dead_squares)
            obs = env.reset()
            wu_buf.clear()
            continue

        action = int(np.random.choice(np.flatnonzero(action_mask)))
        next_obs, reward, done, _ = env.step(action)
        next_mask = (
            np.zeros(env.num_actions, dtype=np.bool_)
            if done else env.get_valid_action_mask()
        )
        wu_buf.append((obs, action, reward, next_obs, done, next_mask))

        if len(wu_buf) == n_step:
            s0, a0, nr, sn, dn, mn = _compute_n_step_return(wu_buf, agent.gamma)
            agent.replay_buffer.push(s0, a0, nr, sn, dn, mn)

        obs = next_obs
        if done:
            level_idx = np.random.randint(len(dataset))
            map_matrix, dead_squares = dataset[level_idx]
            env.set_level(map_matrix, dead_squares)
            obs = env.reset()
            wu_buf.clear()

    tqdm.write(f"[Warm-up] Done. Buffer size: {len(agent.replay_buffer):,}")


def _store_episode_transitions(agent, transitions, info):
    solved = info.get("solved", False)
    boost = getattr(agent, "solve_boost", 3.0) if solved else 1.0

    if solved:
        ep_type = "solve"
    elif info.get("deadlock"):
        ep_type = "deadlock"
    elif info.get("timeout"):
        ep_type = "timeout"
    else:
        ep_type = "ongoing"

    num_t = len(transitions)
    for idx_t, (s0, a0, nr, sn, dn, mn) in enumerate(transitions):
        ttype = ep_type if (solved or idx_t == num_t - 1) else "ongoing"
        agent.replay_buffer.push(
            s0, a0, nr, sn, dn, mn,
            transition_type=ttype,
            priority_boost=boost
        )


def train_one_episode(agent, env, dataset, n_step: int = 3, optimize_every: int = 4):
    level_idx = np.random.randint(len(dataset))
    map_matrix, dead_squares = dataset[level_idx]
    env.set_level(map_matrix, dead_squares)

    obs = env.reset()
    episode_reward = 0.0
    episode_loss = []
    episode_transitions = []
    done = False
    info = {}
    n_step_buf = deque(maxlen=n_step)

    # Lấy action_mask ban đầu 1 lần duy nhất
    action_mask = env.get_valid_action_mask()

    while not done:
        if not action_mask.any():
            break

        action = agent.select_action(obs, action_mask, training=True)
        next_obs, reward, done, info = env.step(action)

        next_action_mask = (
            np.zeros(env.num_actions, dtype=np.bool_)
            if done
            else env.get_valid_action_mask()
        )

        n_step_buf.append((obs, action, reward, next_obs, done, next_action_mask))

        if len(n_step_buf) == n_step:
            s0, a0, n_reward, sn, done_n, mask_n = _compute_n_step_return(n_step_buf, agent.gamma)
            episode_transitions.append((s0, a0, n_reward, sn, done_n, mask_n))

        # Điều chỉnh tần suất optimize: Cứ optimize_every steps mới train GPU/Buffer 1 lần
        if agent.total_steps % optimize_every == 0:
            loss = agent.optimize()
            if loss is not None:
                episode_loss.append(loss)

        obs = next_obs
        action_mask = next_action_mask  # Tái sử dụng mask cho bước sau, không gọi get_valid_action_mask() thừa
        episode_reward += reward
        agent.total_steps += 1

    while n_step_buf:
        s0, a0, n_reward, sn, done_n, mask_n = _compute_n_step_return(n_step_buf, agent.gamma)
        episode_transitions.append((s0, a0, n_reward, sn, done_n, mask_n))
        n_step_buf.popleft()

    _store_episode_transitions(agent, episode_transitions, info)

    return {
        "reward": episode_reward,
        "loss": float(np.mean(episode_loss)) if episode_loss else 0.0,
        "solved": info.get("solved", False),
        "pushes": info.get("pushes", env.step_count),
        "info": info,
    }


def _log_training_progress(pbar, episode: int, num_episodes: int, ep_stats: dict,
                           recent_rewards: deque, recent_solved: deque,
                           log_every: int, agent):

    recent_rewards.append(ep_stats["reward"])
    recent_solved.append(int(ep_stats["solved"]))

    epsilon = agent.get_epsilon()
    pbar.set_postfix(
        reward=f"{ep_stats['reward']:7.1f}",
        loss=f"{ep_stats['loss']:.4f}",
        eps=f"{epsilon:.3f}",
        solved="Y" if ep_stats["solved"] else "N",
    )

    if episode % log_every == 0:
        recent_reward_mean = float(np.mean(recent_rewards))
        recent_solve_rate = float(np.mean(recent_solved))
        tqdm.write(
            f"\n[Ep {episode:6d}/{num_episodes}] "
            f"AvgReward: {recent_reward_mean:8.2f} | "
            f"SolveRate: {recent_solve_rate*100:5.1f}% | "
            f"Loss: {ep_stats['loss']:.5f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Steps: {agent.total_steps:}"
        )

        if  wandb is not None and wandb.run is not None:
            wandb.log({
                "train/avg_reward": recent_reward_mean,
                "train/recent_solve_rate": recent_solve_rate,
                "train/loss": ep_stats["loss"],
                "train/epsilon": epsilon,
                "train/total_steps": agent.total_steps,
                "train/gradient_steps": agent.gradient_steps,
            }, step=episode)


def _evaluate_and_checkpoint(agent, env, val_dataset, episode: int,
                             best_val_solve_rate: float, checkpoint_path=None) -> float:

    val_metrics = evaluate(agent, env, val_dataset)
    stats = agent.buffer_stats()

    tqdm.write(
        f"[VAL ep {episode}] "
        f"SolveRate: {val_metrics['solve_rate']*100:5.1f}% | "
        f"MeanReward: {val_metrics['mean_reward']:8.2f} | "
        f"MeanPushes: {val_metrics['mean_pushes']:.1f}\n"
        f"[Buffer {stats['total']:,}] "
        f"solve={stats['solve_pct']:.1f}% ({stats['solve']}) | "
        f"deadlock={stats['deadlock_pct']:.1f}% | "
        f"timeout={stats['timeout_pct']:.1f}% | "
        f"ongoing={stats['ongoing_pct']:.1f}%"
    )

    if wandb is not None and wandb.run is not None:
        wandb.log({
            "val/solve_rate": val_metrics["solve_rate"],
            "val/mean_reward": val_metrics["mean_reward"],
            "val/mean_pushes": val_metrics["mean_pushes"],
            "buffer/total": stats["total"],
            "buffer/solve_pct": stats["solve_pct"],
            "buffer/solve_count": stats["solve"],
            "buffer/deadlock_pct": stats["deadlock_pct"],
            "buffer/timeout_pct": stats["timeout_pct"],
            "buffer/ongoing_pct": stats["ongoing_pct"],
        }, step=episode)

    if checkpoint_path is not None and val_metrics["solve_rate"] > best_val_solve_rate:
        best_val_solve_rate = val_metrics["solve_rate"]
        agent.save(episode, best_val_solve_rate, checkpoint_path)
        tqdm.write(
            f"  [SAVED BEST] solve_rate={best_val_solve_rate*100:.1f}% -> {checkpoint_path}"
        )

    return best_val_solve_rate


def train_dqn(agent, env, dataset, num_episodes, val_dataset, val_every,
              checkpoint_path=None, log_every: int = 100, n_step: int = 3,
              warm_up_steps: int = 10_000, save_every: int = 5_000,
              resume_best_solve_rate: float = -1.0, start_episode: int = 1,
              optimize_every: int = 4):

    recent_rewards = deque(maxlen=log_every)
    recent_solved = deque(maxlen=log_every)
    best_val_solve_rate = resume_best_solve_rate

    last_checkpoint_path = (
        Path(checkpoint_path).parent / "dqn_sokoban_last.pt"
        if checkpoint_path else None
    )

    if warm_up_steps > 0 and len(agent.replay_buffer) < agent.batch_size:
        warmup_replay_buffer(agent, env, dataset, warm_up_steps, n_step)

    pbar = tqdm(
        range(start_episode, num_episodes + 1),
        initial=start_episode - 1,
        total=num_episodes,
        desc="Training",
        unit="ep",
        dynamic_ncols=True,
    )

    for episode in pbar:
        ep_stats = train_one_episode(agent, env, dataset, n_step, optimize_every)

        _log_training_progress(
            pbar, episode, num_episodes, ep_stats,
            recent_rewards, recent_solved,
            log_every, agent
        )

        if episode % val_every == 0:
            best_val_solve_rate = _evaluate_and_checkpoint(
                agent, env, val_dataset, episode,
                best_val_solve_rate, checkpoint_path
            )

        if last_checkpoint_path is not None and episode % save_every == 0:
            agent.save(episode, best_val_solve_rate, last_checkpoint_path)
            tqdm.write(f"  [SAVED LAST] ep={episode} -> {last_checkpoint_path}")

    pbar.close()
    return best_val_solve_rate
