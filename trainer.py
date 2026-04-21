"""
Dream Leech - RL Training Engine
Runs training loops and collects metrics for all agents.
"""

import numpy as np
import time
from typing import List, Dict, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment import DreamLeechEnv, Action, NUM_ACTIONS
from agents import QLearningAgent, SARSAAgent, DQNAgent, RandomAgent, BaseAgent


def run_episode(env: DreamLeechEnv, agent: BaseAgent, training: bool = True) -> Dict:
    """Run a single episode. Returns episode metrics."""
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    won = False

    prev_action = None

    while not done:
        action = agent.select_action(obs, greedy=not training)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if training:
            if isinstance(agent, SARSAAgent):
                next_action = agent.select_action(next_obs) if not done else 0
                agent.update(obs, action, reward, next_obs, done, next_action)
            elif isinstance(agent, DQNAgent):
                agent.store(obs, action, reward, next_obs, done)
                agent.update()
            elif isinstance(agent, QLearningAgent):
                agent.update(obs, action, reward, next_obs, done)

        obs = next_obs

    if training and hasattr(agent, 'decay_epsilon'):
        agent.decay_epsilon()

    won = info.get("terminal_reason") == "win"

    agent.episode_rewards.append(total_reward)
    agent.episode_lengths.append(steps)
    agent.win_history.append(int(won))
    agent.total_episodes += 1
    if hasattr(agent, 'epsilon'):
        agent.epsilon_history.append(agent.epsilon)

    return {
        "reward": total_reward,
        "steps": steps,
        "won": won,
        "terminal_reason": info.get("terminal_reason", ""),
    }


def train_agent(
    agent: BaseAgent,
    n_episodes: int = 500,
    max_turns: int = 150,
    seed: int = 42,
    progress_callback=None,
) -> Dict:
    """
    Train a single agent for n_episodes.
    progress_callback(episode, total, stats) if provided.
    Returns final stats dict.
    """
    env = DreamLeechEnv(max_turns=max_turns, seed=seed)
    start = time.time()

    for ep in range(n_episodes):
        run_episode(env, agent, training=True)

        if progress_callback and (ep % max(1, n_episodes // 100) == 0 or ep == n_episodes - 1):
            progress_callback(ep + 1, n_episodes, agent.get_stats())

    agent.training_time = time.time() - start
    return agent.get_stats()


def train_all_agents(
    n_episodes: int = 500,
    max_turns: int = 150,
    seed: int = 42,
    progress_callback=None,
) -> Dict[str, BaseAgent]:
    """Train Q-Learning, SARSA, DQN, and Random agents."""
    agents = {
        "Q-Learning": QLearningAgent(NUM_ACTIONS, 12, seed=seed),
        "SARSA":      SARSAAgent(NUM_ACTIONS, 12, seed=seed),
        "DQN":        DQNAgent(NUM_ACTIONS, 12, seed=seed),
        "Random":     RandomAgent(NUM_ACTIONS, 12, seed=seed),
    }

    for name, agent in agents.items():
        def cb(ep, total, stats, agent_name=name):
            if progress_callback:
                progress_callback(agent_name, ep, total, stats)

        train_agent(agent, n_episodes=n_episodes, max_turns=max_turns,
                    seed=seed, progress_callback=cb)

    return agents


def evaluate_agent(agent: BaseAgent, n_episodes: int = 100, max_turns: int = 150, seed: int = 99) -> Dict:
    """Greedy evaluation run."""
    env = DreamLeechEnv(max_turns=max_turns, seed=seed)
    rewards, lengths, wins = [], [], []
    death_reasons = {"fear": 0, "sanity": 0, "timeout": 0, "win": 0}

    for _ in range(n_episodes):
        result = run_episode(env, agent, training=False)
        rewards.append(result["reward"])
        lengths.append(result["steps"])
        wins.append(result["won"])
        reason = result.get("terminal_reason", "timeout")
        if reason in death_reasons:
            death_reasons[reason] += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "win_rate": float(np.mean(wins)),
        "mean_length": float(np.mean(lengths)),
        "death_reasons": death_reasons,
        "agent": agent.name,
        "color": agent.color,
    }


def build_training_curves(agents: Dict[str, BaseAgent], window: int = 20) -> Dict:
    """Build smoothed training curves for plotting."""
    curves = {}
    for name, agent in agents.items():
        rewards = np.array(agent.episode_rewards)
        wins = np.array(agent.win_history, dtype=float)

        if len(rewards) >= window:
            smooth_r = np.convolve(rewards, np.ones(window) / window, mode='valid').tolist()
            smooth_w = np.convolve(wins, np.ones(window) / window, mode='valid').tolist()
        else:
            smooth_r = rewards.tolist()
            smooth_w = wins.tolist()

        curves[name] = {
            "rewards": smooth_r,
            "win_rates": smooth_w,
            "color": agent.color,
            "epsilon": agent.epsilon_history if hasattr(agent, 'epsilon_history') else [],
        }
    return curves