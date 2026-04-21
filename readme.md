# 🩸 Dream Leech — RL Horror Escape Game

A horror escape game built as a Reinforcement Learning project.  
Navigate a nightmare house, survive The Leech, and find the Mirror Room.

---

## Project Structure

```
dream_leech/
├── app.py                    ← Main Streamlit application (run this)
├── requirements.txt
│

│-- environment.py        ← Core game env (Gym-style)
│                               DreamLeechEnv, GameState, Action
│
│   ├── agents.py             ← Q-Learning, SARSA, DQN, Random agents
│   └── trainer.py            ← Training loop, evaluation, curves


    ├── components.py         ← HTML/CSS UI components
    └── theme.css             ← CSS variables reference
```

---

## Quick Start

```bash
# 1. Install dependencies (no PyTorch needed — DQN is pure NumPy)
pip install streamlit numpy

# 2. Run
cd dream_leech
python -m streamlit run app.py
```

---

## The Game

### Map (10 rooms)
```
Entrance(0) → Hallway(1) → Kitchen(2) → Study(4) → Tower(8) → Mirror Room(9) ✨
                  ↓              ↓            
              Library(3)     Bedroom(5) → Garden(7)
                  ↓
              Bathroom(6)
```

### Health Bars
| Meter | Death Condition | Candle ON | Candle OFF |
|-------|----------------|-----------|------------|
| FEAR (red) | 100% + Leech in same room | +1.5/turn | -2.5/turn |
| SANITY (green) | 0% | +3/turn | -4/turn |

### Items
| Item | Effect | Location |
|------|--------|----------|
| 📖 Memory | +25 sanity | Entrance (0) |
| ✨ Safe Thought | -20 fear | Kitchen (2) |
| 🕯️ Candle | +1 candle charge | Library (3) |
| 🪞 Mirror Shard | Reveal Leech location | Study (4) |

---

## RL Algorithms

### Q-Learning (Off-Policy)
- Tabular Q-table with state discretisation
- Update: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`
- Best for: finding optimal policy faster

### SARSA (On-Policy)
- Same table structure, but updates use the actual next action taken
- Update: `Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]`
- Best for: safer/more conservative behaviour

### DQN (Deep Q-Network)
- Pure NumPy 2-layer neural network (12→64→6)
- Experience replay buffer (10k) + target network
- Best for: generalisation over continuous state spaces

### Random Baseline
- Uniform random action selection
- Establishes a lower-bound comparison

---

## State Observation (12 features)
```
[player_room/9, leech_room/9, fear/100, sanity/100,
 candle_on, candle_charges/3,
 has_memory, has_safe_thought, has_candle_extra, has_mirror_shard,
 leech_known, rooms_visited_ratio]
```

## Reward Shaping
```
Win:              +200
Death:            -100
New room visited: +2
Item pickup:      +5
Move toward goal: +1
Leech encounter:  -3
Sanity < 30:      -1/turn
Fear > 70:        -1/turn
Wait:             -0.3
```

---

## Features
- **Human Mode** — Full playable game with horror UI
- **AI Watch Mode** — Step-by-step or fast-forward AI playback  
- **RL Training** — Train all 4 agents simultaneously with live progress
- **Leaderboard** — Win rates, episode outcomes, death breakdowns
- **Training Curves** — Smoothed reward + win rate comparison charts
- **Epsilon Decay** — Visualise exploration vs exploitation over time
- **Algorithm Cards** — Formulas, pros/cons, hyperparameters

---
