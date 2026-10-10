import numpy as np
from tqdm import tqdm
from collections import deque
from pathlib import Path
from level_tracker import LevelTracker
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


def evaluate(agent, env, dataset, tracker=None):
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

        if tracker is not None:
            group, level = dataset.get_level_name(idx)
            tracker.update(group, level, solved, pushes)

    n = len(dataset)
    return {
        "solve_rate":  solved_count / n,
        "mean_reward": total_reward / n,
        "mean_pushes": float(np.mean(push_list)) if push_list else 0.0,
    }


def train_dqn(
    agent, env, dataset, num_episodes,
    val_dataset, val_every,
    checkpoint_path=None, log_every: int = 100, n_step: int = 3,
    warm_up_steps: int = 10_000,
    save_every: int = 5_000,
    resume_best_solve_rate: float = -1.0,
    start_episode: int = 1,
    use_wandb: bool = False,
):
    recent_rewards = deque(maxlen=log_every)
    recent_solved  = deque(maxlen=log_every)

    best_val_solve_rate = resume_best_solve_rate
    last_checkpoint_path = (
        Path(checkpoint_path).parent / "dqn_sokoban_last.pt"
        if checkpoint_path else None
    )

    if warm_up_steps > 0 and len(agent.replay_buffer) < agent.batch_size:
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

    base = Path(checkpoint_path).parent if checkpoint_path else Path(".")
    train_tracker = LevelTracker(dataset, base / "train_levels.csv")
    val_tracker = LevelTracker(val_dataset, base / "val_levels.csv") if val_dataset else None
    t_summary = train_tracker.summary()
    tqdm.write(
        f"[Tracker] Train levels: {t_summary['total']:,} "
        f"(solved: {t_summary['solved']:,} / {t_summary['solve_rate']*100:.1f}%)"
    )
    if val_tracker:
        v_summary = val_tracker.summary()
        tqdm.write(
            f"[Tracker] Val levels:   {v_summary['total']:,} "
            f"(solved: {v_summary['solved']:,} / {v_summary['solve_rate']*100:.1f}%)"
        )

    pbar = tqdm(
        range(start_episode, num_episodes + 1),
        initial=start_episode - 1,
        total=num_episodes,
        desc="Training",
        unit="ep",
        dynamic_ncols=True,
    )

    for episode in pbar:

        level_idx = np.random.randint(len(dataset))
        map_matrix, dead_squares = dataset[level_idx]
        env.set_level(map_matrix, dead_squares)

        obs = env.reset()
        episode_reward = 0.0
        episode_loss   = []
        episode_history = []
        done  = False
        info  = {}
        n_step_buf = deque(maxlen=n_step)

        while not done:
            action_mask = env.get_valid_action_mask()
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
            episode_history.append((obs, action, reward))

            if len(n_step_buf) == n_step:
                s0, a0, n_reward, sn, done_n, mask_n = _compute_n_step_return(n_step_buf, agent.gamma)
                agent.replay_buffer.push(s0, a0, n_reward, sn, done_n, mask_n, "ongoing")

            loss = agent.optimize()
            if loss is not None:
                episode_loss.append(loss)

            obs = next_obs
            episode_reward += reward
            agent.total_steps += 1

        ep_type = "ongoing"
        if info.get("solved"):   ep_type = "solve"
        elif info.get("deadlock"): ep_type = "deadlock"
        elif info.get("timeout"): ep_type = "timeout"

        while n_step_buf:
            s0, a0, n_reward, sn, done_n, mask_n = _compute_n_step_return(n_step_buf, agent.gamma)

            ttype = ep_type if len(n_step_buf) == 1 else "ongoing"
            agent.replay_buffer.push(s0, a0, n_reward, sn, done_n, mask_n, ttype)
            n_step_buf.popleft()

        solved    = info.get("solved", False)
        pushes    = info.get("pushes", env.step_count)
        mean_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
        epsilon   = agent.get_epsilon()

        if solved and episode_history:
            R = 0.0
            for s, a, r in reversed(episode_history):
                R = r + agent.gamma * R
                agent.sil_buffer.push(s, a, R)
            agent.sil_optimize()

        group, level = dataset.get_level_name(level_idx)
        train_tracker.update(group, level, solved, pushes)

        recent_rewards.append(episode_reward)
        recent_solved.append(int(solved))

        pbar.set_postfix(
            reward=f"{episode_reward:7.1f}",
            loss=f"{mean_loss:.4f}",
            eps=f"{epsilon:.3f}",
            solved="Y" if solved else "N",
        )

        if episode % log_every == 0:
            ts = train_tracker.summary()
            recent_reward_mean = float(np.mean(recent_rewards))
            recent_solve_rate = float(np.mean(recent_solved))
            tqdm.write(
                f"\n[Ep {episode:6d}/{num_episodes}] "
                f"AvgReward: {recent_reward_mean:8.2f} | "
                f"SolveRate: {recent_solve_rate*100:5.1f}% | "
                f"Loss: {mean_loss:.5f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Steps: {agent.total_steps:,} | "
                f"Levels: {ts['solved']:,}/{ts['total']:,} ({ts['solve_rate']*100:.1f}%)"
            )

            if use_wandb and wandb is not None and wandb.run is not None:
                wandb.log({
                    "train/avg_reward": recent_reward_mean,
                    "train/recent_solve_rate": recent_solve_rate,
                    "train/loss": mean_loss,
                    "train/epsilon": epsilon,
                    "train/total_steps": agent.total_steps,
                    "train/gradient_steps": agent.gradient_steps,
                    "train/levels_solved": ts["solved"],
                    "train/levels_total": ts["total"],
                    "train/level_solve_rate": ts["solve_rate"],
                }, step=episode)

        if episode % val_every == 0:
            val_metrics = evaluate(agent, env, val_dataset, val_tracker)


            stats = agent.buffer_stats()
            tqdm.write(
                f"[VAL ep {episode}] "
                f"SolveRate: {val_metrics['solve_rate']*100:5.1f}% | "
                f"MeanReward: {val_metrics['mean_reward']:8.2f} | "
                f"MeanPushes: {val_metrics['mean_pushes']:.1f}\n"
                f"[Buffer {stats['total']:,}] "
                f"solve={stats['solve_pct']:.1f}% | "
                f"deadlock={stats['deadlock_pct']:.1f}% | "
                f"timeout={stats['timeout_pct']:.1f}% | "
                f"ongoing={stats['ongoing_pct']:.1f}% | "
                f"sil_buf={len(agent.sil_buffer):,}"
            )

            train_tracker.flush()
            vs = None
            if val_tracker:
                val_tracker.flush()
                vs = val_tracker.summary()
                tqdm.write(
                    f"  -> [VAL Tracker] Solved: {vs['solved']:,}/{vs['total']:,} "
                    f"({vs['solve_rate']*100:.1f}%)"
                )

            if use_wandb:
                val_log_payload = {
                    "val/solve_rate": val_metrics["solve_rate"],
                    "val/mean_reward": val_metrics["mean_reward"],
                    "val/mean_pushes": val_metrics["mean_pushes"],
                    "buffer/total": stats["total"],
                    "buffer/solve_pct": stats["solve_pct"],
                    "buffer/deadlock_pct": stats["deadlock_pct"],
                    "buffer/timeout_pct": stats["timeout_pct"],
                    "buffer/ongoing_pct": stats["ongoing_pct"],
                    "buffer/sil_buffer_size": len(agent.sil_buffer),
                }
                if vs is not None:
                    val_log_payload.update({
                        "val/tracker_solved": vs["solved"],
                        "val/tracker_total": vs["total"],
                        "val/tracker_solve_rate": vs["solve_rate"],
                    })
                wandb.log(val_log_payload, step=episode)

            if checkpoint_path is not None and val_metrics["solve_rate"] > best_val_solve_rate:
                best_val_solve_rate = val_metrics["solve_rate"]
                agent.save(episode,best_val_solve_rate,checkpoint_path)
                tqdm.write(
                    f"  [SAVED BEST] solve_rate={best_val_solve_rate*100:.1f}% -> {checkpoint_path}"
                )

        if last_checkpoint_path is not None and episode % save_every == 0:
            agent.save(episode,best_val_solve_rate,last_checkpoint_path)
            train_tracker.flush()
            if val_tracker:
                val_tracker.flush()
            tqdm.write(f"  [SAVED LAST] ep={episode} -> {last_checkpoint_path}")

    train_tracker.flush()
    if val_tracker:
        val_tracker.flush()
    pbar.close()
    return best_val_solve_rate