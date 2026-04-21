"""
Dream Leech - RL Agents
Implements: Q-Learning, SARSA, Deep Q-Network (DQN)
"""

import numpy as np
import random
import json
import os
import pickle
from collections import deque, defaultdict
from typing import Optional
import time


# ─── Base Agent ─────────────────────────────────────────────────────────────
class BaseAgent:
    name: str = "Base"
    color: str = "#ffffff"

    def __init__(self, n_actions: int, obs_size: int, seed: int = 42):
        self.n_actions = n_actions
        self.obs_size = obs_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Training tracking
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self.win_history: list = []
        self.epsilon_history: list = []
        self.q_value_history: list = []
        self.training_time: float = 0.0
        self.total_episodes: int = 0

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        raise NotImplementedError

    def update(self, *args, **kwargs):
        pass

    def get_stats(self) -> dict:
        n = len(self.episode_rewards)
        if n == 0:
            return {}
        recent = min(50, n)
        wins = sum(self.win_history[-recent:])
        return {
            "total_episodes": n,
            "mean_reward": float(np.mean(self.episode_rewards[-recent:])),
            "max_reward": float(np.max(self.episode_rewards)),
            "win_rate": wins / recent,
            "mean_length": float(np.mean(self.episode_lengths[-recent:])),
            "training_time": self.training_time,
        }


# ─── Q-Learning ─────────────────────────────────────────────────────────────
class QLearningAgent(BaseAgent):
    """Off-policy TD control using discretised observations."""
    name = "Q-Learning"
    color = "#00d4ff"

    def __init__(self, n_actions: int, obs_size: int,
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, seed: int = 42):
        super().__init__(n_actions, obs_size, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict = defaultdict(lambda: np.zeros(n_actions))

    def _discretise(self, obs: np.ndarray) -> tuple:
        """Bucket continuous obs into discrete bins for table lookup."""
        buckets = [10, 10, 5, 5, 2, 4, 2, 2, 2, 2, 2, 5]
        disc = []
        for i, (val, b) in enumerate(zip(obs, buckets)):
            disc.append(int(np.clip(val * b, 0, b - 1)))
        return tuple(disc)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        key = self._discretise(obs)
        return int(np.argmax(self.q_table[key]))

    def update(self, obs, action, reward, next_obs, done):
        key = self._discretise(obs)
        next_key = self._discretise(next_obs)
        current_q = self.q_table[key][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["epsilon"] = self.epsilon
        stats["q_table_size"] = len(self.q_table)
        stats["agent"] = self.name
        stats["color"] = self.color
        return stats


# ─── SARSA ───────────────────────────────────────────────────────────────────
class SARSAAgent(QLearningAgent):
    """On-policy TD control (SARSA)."""
    name = "SARSA"
    color = "#ff6b6b"

    def update(self, obs, action, reward, next_obs, done, next_action: int = None):
        key = self._discretise(obs)
        next_key = self._discretise(next_obs)
        current_q = self.q_table[key][action]
        if done or next_action is None:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_key][next_action]
        self.q_table[key][action] += self.alpha * (target - current_q)


# ─── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ─── Simple Neural Network (numpy, no torch dependency) ──────────────────────
class SimpleNN:
    """2-layer fully connected network implemented in pure NumPy."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.W1 = rng.normal(0, scale1, (hidden_size, input_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, scale2, (output_size, hidden_size))
        self.b2 = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.h_pre = self.W1 @ x + self.b1
        self.h = np.maximum(0, self.h_pre)         # ReLU
        self.out = self.W2 @ self.h + self.b2
        return self.out

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.forward(x) for x in X])

    def copy_weights_from(self, other: "SimpleNN"):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    def backward(self, x: np.ndarray, target_q: np.ndarray, lr: float) -> float:
        """Single sample SGD update. Returns MSE loss."""
        pred = self.forward(x)
        error = pred - target_q
        loss = float(np.mean(error ** 2))

        # Output layer
        dW2 = np.outer(error, self.h)
        db2 = error

        # Hidden layer
        dh = self.W2.T @ error
        dh_pre = dh * (self.h_pre > 0)
        dW1 = np.outer(dh_pre, x)
        db1 = dh_pre

        # Update
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss


# ─── DQN Agent ────────────────────────────────────────────────────────────────
class DQNAgent(BaseAgent):
    """Deep Q-Network with experience replay and target network."""
    name = "DQN"
    color = "#a29bfe"

    def __init__(self, n_actions: int, obs_size: int,
                 hidden_size: int = 64,
                 alpha: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.997,
                 batch_size: int = 32, target_update: int = 50,
                 buffer_capacity: int = 10000, seed: int = 42):
        super().__init__(n_actions, obs_size, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        self.loss_history: list = []

        self.online_net = SimpleNN(obs_size, hidden_size, n_actions, seed)
        self.target_net = SimpleNN(obs_size, hidden_size, n_actions, seed)
        self.target_net.copy_weights_from(self.online_net)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        q_vals = self.online_net.forward(obs)
        return int(np.argmax(q_vals))

    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self, *args, **kwargs):
        if len(self.replay_buffer) < self.batch_size:
            return
        obs_b, act_b, rew_b, next_b, done_b = self.replay_buffer.sample(self.batch_size)

        total_loss = 0.0
        for obs, action, reward, next_obs, done in zip(obs_b, act_b, rew_b, next_b, done_b):
            target_q = self.online_net.forward(obs).copy()
            if done:
                target_q[action] = reward
            else:
                next_q = self.target_net.forward(next_obs)
                target_q[action] = reward + self.gamma * np.max(next_q)
            loss = self.online_net.backward(obs, target_q, self.alpha)
            total_loss += loss

        self.loss_history.append(total_loss / self.batch_size)
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.copy_weights_from(self.online_net)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["epsilon"] = self.epsilon
        stats["buffer_size"] = len(self.replay_buffer)
        stats["avg_loss"] = float(np.mean(self.loss_history[-50:])) if self.loss_history else 0.0
        stats["agent"] = self.name
        stats["color"] = self.color
        return stats


# ─── Random Baseline ─────────────────────────────────────────────────────────
class RandomAgent(BaseAgent):
    name = "Random"
    color = "#fdcb6e"

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        return self.rng.randint(0, self.n_actions - 1)

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["agent"] = self.name
        stats["color"] = self.color
        return stats