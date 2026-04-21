"""
Dream Leech - Core Game Environment
Compatible with OpenAI Gym interface for RL training
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Room Layout ────────────────────────────────────────────────────────────
ROOM_NAMES = {
    0: "Entrance",
    1: "Hallway",
    2: "Kitchen",
    3: "Library",
    4: "Study",
    5: "Bedroom",
    6: "Bathroom",
    7: "Garden",
    8: "Tower",
    9: "Mirror Room",
}

ROOM_DESCRIPTIONS = {
    0: "The front door hangs open. Cold air seeps in. This is where it started.",
    1: "Long shadows stretch down the corridor. The floorboards creak with every step.",
    2: "Pots clatter by themselves. Something cooked here, recently.",
    3: "Shelves of ancient books. The pages whisper when you're not looking.",
    4: "A writing desk covered in notes. The handwriting changes midway through sentences.",
    5: "The bed is warm. Someone was just here. Something still lingers.",
    6: "The mirror is cracked. Your reflection moves a second too slow.",
    7: "Overgrown and dark. The garden grows inward, toward the house.",
    8: "You can see the whole nightmare from here. And it can see you.",
    9: "The Mirror Room. Shards of light. Your way out. You feel it.",
}

# Graph adjacency
CONNECTIONS = {
    0: [1],
    1: [0, 2, 3],
    2: [1, 4, 5],
    3: [1, 6],
    4: [2, 8],
    5: [2, 7],
    6: [3],
    7: [5],
    8: [4, 9],
    9: [8],
}

ITEMS_IN_ROOMS = {
    0: "memory",    # 📖 Restores 25 sanity
    2: "safe_thought",  # ✨ Reduces 20 fear
    3: "candle",    # 🕯️ Extra candle charge
    4: "mirror_shard",  # 🪞 Reveals Leech location
}

ITEM_EMOJIS = {
    "memory": "📖",
    "safe_thought": "✨",
    "candle": "🕯️",
    "mirror_shard": "🪞",
}

# ─── Action Space ────────────────────────────────────────────────────────────
class Action(IntEnum):
    MOVE       = 0
    TOGGLE_CANDLE = 1
    USE_MEMORY    = 2
    USE_SAFE_THOUGHT = 3
    USE_MIRROR_SHARD = 4
    WAIT          = 5

NUM_ACTIONS = len(Action)

# ─── Game State ──────────────────────────────────────────────────────────────
@dataclass
class GameState:
    player_room: int = 0
    leech_room: int = 9
    fear: float = 30.0        # 0–100, death at 100 + leech in room
    sanity: float = 80.0      # 0–100, death at 0
    candle_on: bool = True
    candle_charges: int = 1
    items_collected: dict = field(default_factory=dict)
    rooms_visited: set = field(default_factory=set)
    leech_known: bool = False  # set True after using mirror shard
    turn: int = 0
    is_terminal: bool = False
    terminal_reason: str = ""
    # Atmosphere (cosmetic, used in UI)
    last_event: str = "You wake up. The nightmare begins."
    leech_last_seen: Optional[int] = None

    def __post_init__(self):
        self.rooms_visited.add(self.player_room)
        for room_id in ITEMS_IN_ROOMS:
            self.items_collected[room_id] = False

    def copy(self) -> "GameState":
        s = GameState.__new__(GameState)
        s.__dict__.update(self.__dict__)
        s.items_collected = dict(self.items_collected)
        s.rooms_visited = set(self.rooms_visited)
        return s


# ─── Environment ─────────────────────────────────────────────────────────────
class DreamLeechEnv:
    """
    Gym-style environment for Dream Leech.
    
    Observation vector (12 values, all normalised 0-1):
      [player_room/9, leech_room/9, fear/100, sanity/100,
       candle_on, candle_charges/3,
       has_memory, has_safe_thought, has_candle_extra, has_mirror_shard,
       leech_known, rooms_visited_ratio]
    """

    OBSERVATION_SIZE = 12

    # Reward shaping
    R_WIN           =  200.0
    R_DEATH         = -100.0
    R_MOVE_NEW_ROOM =    2.0
    R_MOVE_OLD_ROOM =   -0.5
    R_ITEM_PICKUP   =    5.0
    R_TOWARD_GOAL   =    1.0
    R_LEECH_NEAR    =   -3.0
    R_SANITY_LOW    =   -1.0
    R_FEAR_HIGH     =   -1.0
    R_WAIT_PENALTY  =   -0.3

    def __init__(self, max_turns: int = 200, seed: int = None):
        self.max_turns = max_turns
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.state: GameState = None
        self.reset()

    # ── Setup ─────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self.state = GameState()
        self.state.leech_room = self._rng.choice([7, 8, 9])
        return self._obs()

    # ── Core Step ─────────────────────────────────────────────────────────
    def step(self, action: int):
        s = self.state
        if s.is_terminal:
            return self._obs(), 0.0, True, self._info()

        reward = 0.0
        s.turn += 1
        event_parts = []

        # ── 1. Player action ──────────────────────────────────────────────
        if action == Action.MOVE:
            neighbors = CONNECTIONS[s.player_room]
            old_room = s.player_room
            new_room = self._rng.choice(neighbors)
            s.player_room = new_room
            s.rooms_visited.add(new_room)

            if new_room not in s.rooms_visited or new_room == old_room:
                reward += self.R_MOVE_NEW_ROOM
            else:
                reward += self.R_MOVE_OLD_ROOM

            # proximity reward toward goal (BFS distance heuristic)
            reward += self._goal_proximity_delta(old_room, new_room)

            event_parts.append(f"You move to the {ROOM_NAMES[new_room]}.")

            # Item pickup
            if new_room in ITEMS_IN_ROOMS and not s.items_collected.get(new_room, False):
                item = ITEMS_IN_ROOMS[new_room]
                s.items_collected[new_room] = True
                reward += self.R_ITEM_PICKUP
                self._apply_item(item, s)
                event_parts.append(f"You find {ITEM_EMOJIS[item]} {item.replace('_', ' ')}!")

        elif action == Action.TOGGLE_CANDLE:
            if s.candle_charges > 0 or s.candle_on:
                s.candle_on = not s.candle_on
                if s.candle_on:
                    event_parts.append("You light the candle. Warm light fills the room.")
                else:
                    event_parts.append("You snuff the candle. Darkness closes in.")
            else:
                event_parts.append("No candle charges left.")

        elif action == Action.USE_MEMORY:
            if 0 in s.items_collected and s.items_collected[0] and s.sanity < 100:
                s.sanity = min(100, s.sanity + 25)
                event_parts.append("📖 You recall a happy memory. Sanity restored.")
            else:
                event_parts.append("Nothing to use here.")

        elif action == Action.USE_SAFE_THOUGHT:
            if 2 in s.items_collected and s.items_collected[2] and s.fear > 0:
                s.fear = max(0, s.fear - 20)
                event_parts.append("✨ A safe thought washes over you. Fear fades.")
            else:
                event_parts.append("Nothing to use here.")

        elif action == Action.USE_MIRROR_SHARD:
            if 4 in s.items_collected and s.items_collected[4]:
                s.leech_known = True
                s.leech_last_seen = s.leech_room
                event_parts.append(f"🪞 The shard reveals: The Leech is in the {ROOM_NAMES[s.leech_room]}!")
            else:
                event_parts.append("You don't have the Mirror Shard.")

        elif action == Action.WAIT:
            reward += self.R_WAIT_PENALTY
            event_parts.append("You wait. The shadows shift.")

        # ── 2. Passive meter changes ──────────────────────────────────────
        if s.candle_on:
            s.sanity = min(100, s.sanity + 3.0)
            s.fear   = min(100, s.fear   + 1.5)  # light attracts leech
        else:
            s.sanity = max(0,   s.sanity - 4.0)
            s.fear   = max(0,   s.fear   - 2.5)

        # ── 3. Leech movement (semi-intelligent) ─────────────────────────
        leech_event = self._move_leech(s)
        if leech_event:
            event_parts.append(leech_event)

        # ── 4. Leech encounter check ──────────────────────────────────────
        if s.leech_room == s.player_room:
            s.fear = min(100, s.fear + 15)
            reward += self.R_LEECH_NEAR
            event_parts.append("⚠️ THE LEECH IS HERE!")
            if s.fear >= 100:
                s.is_terminal = True
                s.terminal_reason = "fear"
                reward += self.R_DEATH
                event_parts.append("Your fear reaches its peak. The Leech takes you.")

        # ── 5. Sanity / fear death checks ────────────────────────────────
        if s.sanity <= 0 and not s.is_terminal:
            s.is_terminal = True
            s.terminal_reason = "sanity"
            reward += self.R_DEATH
            event_parts.append("Your mind shatters. The nightmare wins.")

        if s.sanity < 30:
            reward += self.R_SANITY_LOW
        if s.fear > 70:
            reward += self.R_FEAR_HIGH

        # ── 6. Win check ─────────────────────────────────────────────────
        if s.player_room == 9 and not s.is_terminal:
            s.is_terminal = True
            s.terminal_reason = "win"
            reward += self.R_WIN
            event_parts.append("You step through the mirror. The nightmare dissolves. You're FREE.")

        # ── 7. Turn limit ─────────────────────────────────────────────────
        if s.turn >= self.max_turns and not s.is_terminal:
            s.is_terminal = True
            s.terminal_reason = "timeout"
            reward += self.R_DEATH * 0.5
            event_parts.append("The nightmare consumes all sense of time. You are lost forever.")

        s.last_event = " ".join(event_parts) if event_parts else "Nothing happens."
        return self._obs(), reward, s.is_terminal, self._info()

    # ── Leech AI ──────────────────────────────────────────────────────────
    def _move_leech(self, s: GameState) -> str:
        """Leech moves toward player if candle is ON, wanders if OFF."""
        neighbors = CONNECTIONS[s.leech_room]

        if s.candle_on:
            # Hunt: move toward player along shortest path
            path = self._bfs(s.leech_room, s.player_room)
            if path and len(path) > 1:
                s.leech_room = path[1]
            else:
                s.leech_room = self._rng.choice(neighbors)
        else:
            # Wander randomly (slight bias away from player)
            safe = [n for n in neighbors if n != s.player_room]
            s.leech_room = self._rng.choice(safe if safe else neighbors)

        if abs(s.leech_room - s.player_room) <= 1:
            return "🩸 You hear wet, dragging sounds nearby..."
        return ""

    def _bfs(self, start: int, goal: int) -> list:
        from collections import deque
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                return path
            for nb in CONNECTIONS[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(path + [nb])
        return [start]

    def _goal_proximity_delta(self, old_room: int, new_room: int) -> float:
        old_dist = len(self._bfs(old_room, 9)) - 1
        new_dist = len(self._bfs(new_room, 9)) - 1
        return self.R_TOWARD_GOAL if new_dist < old_dist else 0.0

    # ── Item Effects ──────────────────────────────────────────────────────
    def _apply_item(self, item: str, s: GameState):
        if item == "memory":
            s.sanity = min(100, s.sanity + 25)
        elif item == "safe_thought":
            s.fear = max(0, s.fear - 20)
        elif item == "candle":
            s.candle_charges += 1
        elif item == "mirror_shard":
            s.leech_known = True
            s.leech_last_seen = s.leech_room

    # ── Observation / Info ────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        s = self.state
        items = s.items_collected
        visited_ratio = len(s.rooms_visited) / len(ROOM_NAMES)
        return np.array([
            s.player_room / 9.0,
            s.leech_room  / 9.0,
            s.fear        / 100.0,
            s.sanity      / 100.0,
            float(s.candle_on),
            min(s.candle_charges, 3) / 3.0,
            float(items.get(0, False)),   # memory
            float(items.get(2, False)),   # safe_thought
            float(items.get(3, False)),   # candle_extra
            float(items.get(4, False)),   # mirror_shard
            float(s.leech_known),
            visited_ratio,
        ], dtype=np.float32)

    def _info(self) -> dict:
        s = self.state
        return {
            "turn": s.turn,
            "player_room": s.player_room,
            "leech_room": s.leech_room,
            "fear": s.fear,
            "sanity": s.sanity,
            "candle_on": s.candle_on,
            "terminal_reason": s.terminal_reason,
            "rooms_visited": len(s.rooms_visited),
        }

    # ── Helpers for UI ────────────────────────────────────────────────────
    def get_valid_actions(self) -> list:
        s = self.state
        valid = [Action.MOVE, Action.WAIT]
        if s.candle_on or s.candle_charges > 0:
            valid.append(Action.TOGGLE_CANDLE)
        if s.items_collected.get(0, False):
            valid.append(Action.USE_MEMORY)
        if s.items_collected.get(2, False):
            valid.append(Action.USE_SAFE_THOUGHT)
        if s.items_collected.get(4, False):
            valid.append(Action.USE_MIRROR_SHARD)
        return valid