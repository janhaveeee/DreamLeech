"""
Dream Leech — Main Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import sys, os, time, json, random
import streamlit.components.v1 as components
# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from environment import (
    DreamLeechEnv, GameState, Action, NUM_ACTIONS,
    ROOM_NAMES, ROOM_DESCRIPTIONS, CONNECTIONS, ITEMS_IN_ROOMS, ITEM_EMOJIS
)
from agents import QLearningAgent, SARSAAgent, DQNAgent, RandomAgent
from trainer import run_episode, train_all_agents, evaluate_agent, build_training_curves
from components import (
    HORROR_CSS, title_html, room_card_html, meter_html,
    minimap_html, event_log_html, status_badge, game_over_html, agent_card_html
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dream Leech",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(HORROR_CSS, unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
def init_session():
    if "mode" not in st.session_state:
        st.session_state.mode = "menu"         # menu | human | ai_watch | rl_train
    if "env" not in st.session_state:
        st.session_state.env = None
    if "event_log" not in st.session_state:
        st.session_state.event_log = []
    if "agents" not in st.session_state:
        st.session_state.agents = {}
    if "training_done" not in st.session_state:
        st.session_state.training_done = False
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = {}
    if "training_curves" not in st.session_state:
        st.session_state.training_curves = {}
    if "ai_watch_agent" not in st.session_state:
        st.session_state.ai_watch_agent = None
    if "ai_watch_env" not in st.session_state:
        st.session_state.ai_watch_env = None
    if "ai_watch_obs" not in st.session_state:
        st.session_state.ai_watch_obs = None
    if "ai_watch_done" not in st.session_state:
        st.session_state.ai_watch_done = False
    if "human_game_over" not in st.session_state:
        st.session_state.human_game_over = False

init_session()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(title_html(), unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🏠 Main Menu", use_container_width=True):
        st.session_state.mode = "menu"
        st.rerun()

    st.markdown("### Navigate")
    if st.button("👤 Human Play", use_container_width=True):
        env = DreamLeechEnv(max_turns=150)
        st.session_state.env = env
        st.session_state.event_log = ["You wake up. The nightmare begins."]
        st.session_state.human_game_over = False
        st.session_state.mode = "human"
        st.rerun()

    if st.button("🤖 Watch AI Play", use_container_width=True):
        if not st.session_state.training_done:
            st.warning("Train the AI first!")
        else:
            st.session_state.mode = "ai_watch"
            st.rerun()

    if st.button("📊 RL Training & Analysis", use_container_width=True):
        st.session_state.mode = "rl_train"
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="font-family:'VT323',monospace; color:#444; font-size:0.85rem; line-height:1.8;">
🩸 Fear kills at 100%<br>
💚 Sanity kills at 0%<br>
🕯️ Candle: +sanity, +fear<br>
🔴 Leech hunts the light<br>
🪞 Reach room 9 to win
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MENU
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "menu":
    st.markdown(title_html(), unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center; margin:1rem auto; max-width:600px;
            font-family:'Special Elite',serif; color:#8a8580; line-height:1.8; font-size:1rem;">
You wake up in a strange house. The walls breathe. Shadows move.<br>
Somewhere in the darkness, <span style="color:#c0392b">The Leech</span> is hunting you.<br><br>
Find the <span style="color:#27ae60">Mirror Room</span>. Escape the nightmare.
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div style="background:#111118; border:1px solid #2a2a3a; border-radius:12px;
            padding:1.5rem; text-align:center; min-height:200px;">
  <div style="font-size:2.5rem;">👤</div>
  <div style="font-family:'Creepster',cursive; font-size:1.4rem; color:#ddd8cc; margin:0.5rem 0;">
    HUMAN MODE
  </div>
  <div style="font-family:'Special Elite',serif; color:#8a8580; font-size:0.85rem;">
    You control the player. Navigate 10 rooms, manage fear & sanity, avoid The Leech.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("▶ Play Now", use_container_width=True, key="menu_human"):
            env = DreamLeechEnv(max_turns=150)
            st.session_state.env = env
            st.session_state.event_log = ["You wake up. The nightmare begins."]
            st.session_state.human_game_over = False
            st.session_state.mode = "human"
            st.rerun()

    with col2:
        st.markdown("""
<div style="background:#111118; border:1px solid #2a2a3a; border-radius:12px;
            padding:1.5rem; text-align:center; min-height:200px;">
  <div style="font-size:2.5rem;">🤖</div>
  <div style="font-family:'Creepster',cursive; font-size:1.4rem; color:#00d4ff; margin:0.5rem 0;">
    WATCH AI
  </div>
  <div style="font-family:'Special Elite',serif; color:#8a8580; font-size:0.85rem;">
    Watch a trained RL agent navigate the nightmare in real time. Choose your algorithm.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("▶ Watch AI", use_container_width=True, key="menu_ai"):
            if not st.session_state.training_done:
                st.warning("Train agents first in RL Analysis tab.")
            else:
                st.session_state.mode = "ai_watch"
                st.rerun()

    with col3:
        st.markdown("""
<div style="background:#111118; border:1px solid #2a2a3a; border-radius:12px;
            padding:1.5rem; text-align:center; min-height:200px;">
  <div style="font-size:2.5rem;">📊</div>
  <div style="font-family:'Creepster',cursive; font-size:1.4rem; color:#a29bfe; margin:0.5rem 0;">
    RL ANALYSIS
  </div>
  <div style="font-family:'Special Elite',serif; color:#8a8580; font-size:0.85rem;">
    Train Q-Learning, SARSA, DQN & Random. Compare performance curves & win rates.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("▶ Train & Analyse", use_container_width=True, key="menu_rl"):
            st.session_state.mode = "rl_train"
            st.rerun()

    # Quick rules
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div style="background:#111118; border:1px solid #2a2a3a; border-radius:8px; padding:1rem;">
<div style="font-family:'VT323',monospace; color:#e74c3c; font-size:1.1rem;">FEAR (red bar)</div>
<div style="font-family:'Special Elite',serif; color:#666; font-size:0.8rem;">
Rises near The Leech and when candle is ON. At 100% + Leech in same room = death.
</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div style="background:#111118; border:1px solid #2a2a3a; border-radius:8px; padding:1rem;">
<div style="font-family:'VT323',monospace; color:#2ecc71; font-size:1.1rem;">SANITY (green bar)</div>
<div style="font-family:'Special Elite',serif; color:#666; font-size:0.8rem;">
Rises with candle ON. Falls in darkness. At 0% = madness and death.
</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div style="background:#111118; border:1px solid #2a2a3a; border-radius:8px; padding:1rem;">
<div style="font-family:'VT323',monospace; color:#f39c12; font-size:1.1rem;">THE LEECH 🔴</div>
<div style="font-family:'Special Elite',serif; color:#666; font-size:0.8rem;">
Hunts toward light. Wanders in darkness. Use the mirror shard to track it.
</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN PLAY MODE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "human":
    env: DreamLeechEnv = st.session_state.env
    s: GameState = env.state

    if st.session_state.human_game_over:
        st.markdown(game_over_html(
            s.terminal_reason == "win",
            s.terminal_reason,
            s.turn
        ), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Play Again", use_container_width=True):
                new_env = DreamLeechEnv(max_turns=150)
                st.session_state.env = new_env
                st.session_state.event_log = ["You wake up. The nightmare begins."]
                st.session_state.human_game_over = False
                st.rerun()
        with col2:
            if st.button("🏠 Main Menu", use_container_width=True):
                st.session_state.mode = "menu"
                st.rerun()
        st.stop()

    # Header row
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown(f"""
<div style="font-family:'Creepster',cursive; font-size:2rem; color:#ddd8cc; letter-spacing:2px;">
  {ROOM_NAMES[s.player_room]}
  <span style="font-size:1rem; color:#555; font-family:'VT323',monospace; margin-left:1rem;">
    Turn {s.turn}/150
  </span>
</div>
""", unsafe_allow_html=True)
    with col_status:
        candle_label = "🕯️ ON" if s.candle_on else "🕯️ OFF"
        candle_color = "#f39c12" if s.candle_on else "#555"
        st.markdown(status_badge(candle_label, candle_color), unsafe_allow_html=True)

    # Main layout
    left, right = st.columns([2, 1])

    with left:
        # Room card
        room_id = s.player_room
        has_item = room_id in ITEMS_IN_ROOMS and not s.items_collected.get(room_id, False)
        item_name = ITEMS_IN_ROOMS.get(room_id, "") if has_item else ""
        st.markdown(
            room_card_html(room_id, ROOM_DESCRIPTIONS[room_id], has_item, item_name).strip(),
            unsafe_allow_html=True
        )

        # Connections
        neighbors = CONNECTIONS[s.player_room]
        nb_names = ", ".join([ROOM_NAMES[n] for n in neighbors])
        st.markdown(f"""
<div style="font-family:'VT323',monospace; color:#555; font-size:0.95rem; margin-bottom:0.8rem;">
  Exits → {nb_names}
</div>""", unsafe_allow_html=True)

        # Actions
        st.markdown("""
<div style="font-family:'VT323',monospace; color:#8a8580; font-size:1rem; margin-bottom:0.4rem;">
  ── ACTIONS ──
</div>""", unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("🚶 MOVE", use_container_width=True, key="btn_move"):
                obs, rew, done, info = env.step(Action.MOVE)
                st.session_state.event_log.append(s.last_event)
                if done:
                    st.session_state.human_game_over = True
                st.rerun()

        with a2:
            candle_label = "🕯️ EXTINGUISH" if s.candle_on else "🕯️ LIGHT"
            if st.button(candle_label, use_container_width=True, key="btn_candle"):
                obs, rew, done, info = env.step(Action.TOGGLE_CANDLE)
                st.session_state.event_log.append(s.last_event)
                if done:
                    st.session_state.human_game_over = True
                st.rerun()

        with a3:
            if st.button("⏳ WAIT", use_container_width=True, key="btn_wait"):
                obs, rew, done, info = env.step(Action.WAIT)
                st.session_state.event_log.append(s.last_event)
                if done:
                    st.session_state.human_game_over = True
                st.rerun()

        # Item actions
        item_cols = st.columns(3)
        items_ui = [
            (Action.USE_MEMORY,       0, "📖 Memory",      s.items_collected.get(0, False)),
            (Action.USE_SAFE_THOUGHT, 2, "✨ Safe Thought", s.items_collected.get(2, False)),
            (Action.USE_MIRROR_SHARD, 4, "🪞 Mirror Shard", s.items_collected.get(4, False)),
        ]
        for i, (action, room, label, collected) in enumerate(items_ui):
            with item_cols[i]:
                disabled = not collected
                if st.button(label, use_container_width=True,
                             key=f"btn_item_{i}", disabled=disabled):
                    obs, rew, done, info = env.step(action)
                    st.session_state.event_log.append(s.last_event)
                    if done:
                        st.session_state.human_game_over = True
                    st.rerun()

        # Event log
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(event_log_html(st.session_state.event_log), unsafe_allow_html=True)

    with right:
        # Health bars
        st.markdown("""<div style="font-family:'VT323',monospace; color:#8a8580; font-size:1rem; margin-bottom:0.5rem;">── STATUS ──</div>""", unsafe_allow_html=True)
        st.markdown(meter_html("⚡ FEAR", s.fear, 100, "#e74c3c", warning_at=70), unsafe_allow_html=True)
        st.markdown(meter_html("🧠 SANITY", s.sanity, 100, "#2ecc71", warning_at=30), unsafe_allow_html=True)

        # Candle charges
        charge_str = "🕯️" * s.candle_charges + "◻️" * max(0, 3 - s.candle_charges)
        st.markdown(f"""<div style="font-family:'VT323',monospace; color:#f39c12; margin-bottom:1rem;">{charge_str} Candle charges</div>""", unsafe_allow_html=True)

        # Minimap
        st.markdown("""<div style="font-family:'VT323',monospace; color:#8a8580; font-size:1rem; margin-bottom:0.4rem;">── MAP ──</div>""", unsafe_allow_html=True)
        components.html(
            minimap_html(s.player_room, s.leech_room, s.rooms_visited, s.leech_known),
            height=250,
        )

        # Inventory
        st.markdown("""<div style="font-family:'VT323',monospace; color:#8a8580; font-size:1rem; margin:0.8rem 0 0.4rem;">── INVENTORY ──</div>""", unsafe_allow_html=True)
        inv_items = []
        inv_map = {0: ("memory", "📖"), 2: ("safe_thought", "✨"), 3: ("candle", "🕯️"), 4: ("mirror_shard", "🪞")}
        for room_id, (name, emoji) in inv_map.items():
            if s.items_collected.get(room_id, False):
                inv_items.append(f"{emoji} {name.replace('_',' ').title()}")
        if inv_items:
            for item in inv_items:
                st.markdown(f'<div style="font-family:\'VT323\',monospace; color:#f39c12; font-size:0.95rem;">{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'VT323\',monospace; color:#444; font-size:0.9rem;">Empty</div>', unsafe_allow_html=True)

        # Leech location (if known)
        if s.leech_known:
            st.markdown(f"""
<div style="margin-top:0.8rem; background:#1a0000; border:1px solid #8b000066;
            border-radius:6px; padding:0.5rem 0.8rem;
            font-family:'VT323',monospace; color:#c0392b; font-size:1rem;">
  🔴 LEECH: {ROOM_NAMES[s.leech_room]}
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RL TRAINING & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "rl_train":
    st.markdown("""
<div style="font-family:'Creepster',cursive; font-size:2.2rem; color:#a29bfe;
            letter-spacing:3px; margin-bottom:0.5rem;">
  📊 RL ALGORITHM ANALYSIS
</div>""", unsafe_allow_html=True)

    # Training controls
    with st.expander("⚙️ Training Configuration", expanded=not st.session_state.training_done):
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            n_episodes = st.slider("Episodes per agent", 200, 2000, 600, step=100)
        with tc2:
            max_turns = st.slider("Max turns per episode", 50, 200, 120, step=10)
        with tc3:
            seed = st.number_input("Random seed", value=42, step=1)

        if st.button("🚀 TRAIN ALL AGENTS", use_container_width=True):
            progress_container = st.empty()
            status_text = st.empty()
            prog_bar = st.progress(0)

            agents_progress = {
                "Q-Learning": 0,
                "SARSA": 0,
                "DQN": 0,
                "Random": 0,
            }
            agent_order = list(agents_progress.keys())
            current_idx = [0]

            def progress_cb(agent_name, ep, total, stats):
                agents_progress[agent_name] = ep / total
                done_agents = current_idx[0]
                overall = (done_agents + agents_progress[agent_name]) / len(agent_order)
                prog_bar.progress(overall)
                status_text.markdown(
                    f'<div style="font-family:\'VT323\',monospace; color:#00d4ff;">'
                    f'Training {agent_name}... episode {ep}/{total} | '
                    f'win rate: {stats.get("win_rate",0)*100:.1f}%</div>',
                    unsafe_allow_html=True
                )

            trained = {}
            for i, agent_name in enumerate(agent_order):
                current_idx[0] = i
                if agent_name == "Q-Learning":
                    ag = QLearningAgent(NUM_ACTIONS, 12, seed=seed)
                elif agent_name == "SARSA":
                    ag = SARSAAgent(NUM_ACTIONS, 12, seed=seed)
                elif agent_name == "DQN":
                    ag = DQNAgent(NUM_ACTIONS, 12, seed=seed)
                else:
                    ag = RandomAgent(NUM_ACTIONS, 12, seed=seed)

                from trainer import train_agent
                train_agent(ag, n_episodes=n_episodes, max_turns=max_turns,
                            seed=seed, progress_callback=lambda ep, tot, s, an=agent_name: progress_cb(an, ep, tot, s))
                trained[agent_name] = ag

            st.session_state.agents = trained
            st.session_state.training_done = True

            # Evaluate
            status_text.markdown(
                '<div style="font-family:\'VT323\',monospace; color:#f39c12;">Evaluating agents...</div>',
                unsafe_allow_html=True
            )
            eval_results = {}
            for name, agent in trained.items():
                eval_results[name] = evaluate_agent(agent, n_episodes=100)
            st.session_state.eval_results = eval_results
            st.session_state.training_curves = build_training_curves(trained)

            prog_bar.progress(1.0)
            status_text.markdown(
                '<div style="font-family:\'VT323\',monospace; color:#27ae60;">✅ Training complete!</div>',
                unsafe_allow_html=True
            )
            st.rerun()

    if not st.session_state.training_done:
        st.markdown("""
<div style="text-align:center; padding:3rem; color:#444;
            font-family:'Special Elite',serif;">
  Configure training above and click TRAIN ALL AGENTS to begin.
</div>""", unsafe_allow_html=True)
        st.stop()

    # ── Results Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Leaderboard",
        "📈 Training Curves",
        "🎯 Evaluation",
        "🔬 Algorithm Details"
    ])

    agents = st.session_state.agents
    eval_results = st.session_state.eval_results
    curves = st.session_state.training_curves

    # ── Tab 1: Leaderboard ────────────────────────────────────────────────────
    with tab1:
        st.markdown("""<div style="font-family:'VT323',monospace; color:#8a8580; font-size:1rem; margin-bottom:1rem;">Greedy evaluation over 100 episodes</div>""", unsafe_allow_html=True)

        # Sort by win rate
        sorted_agents = sorted(eval_results.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True)

        for rank, (name, stats) in enumerate(sorted_agents):
            medal = ["🥇", "🥈", "🥉", "4️⃣"][rank]
            color = stats.get("color", "#fff")
            win_pct = stats.get("win_rate", 0) * 100
            mean_r = stats.get("mean_reward", 0)
            mean_l = stats.get("mean_length", 0)

            # Death reason breakdown
            dr = stats.get("death_reasons", {})
            total = sum(dr.values()) or 1
            dr_str = " | ".join([f"{k}: {v/total*100:.0f}%" for k, v in dr.items()])

            st.markdown(f"""
<div style="background:#111118; border:1px solid {color}44; border-left:4px solid {color};
            border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div style="font-family:'Creepster',cursive; font-size:1.5rem; color:{color}; letter-spacing:2px;">
      {medal} {name}
    </div>
    <div style="font-family:'VT323',monospace; font-size:1.4rem; color:{color};">
      {win_pct:.1f}% wins
    </div>
  </div>
  <div style="display:flex; gap:2rem; margin-top:0.5rem;">
    <span style="font-family:'VT323',monospace; color:#8a8580; font-size:1rem;">
      Avg Reward: <span style="color:#ddd">{mean_r:.1f}</span>
    </span>
    <span style="font-family:'VT323',monospace; color:#8a8580; font-size:1rem;">
      Avg Length: <span style="color:#ddd">{mean_l:.0f} turns</span>
    </span>
  </div>
  <div style="font-family:'VT323',monospace; color:#444; font-size:0.85rem; margin-top:4px;">
    {dr_str}
  </div>
</div>
""", unsafe_allow_html=True)

        # Win rate comparison bar chart (HTML)
        names = [n for n, _ in sorted_agents]
        win_rates = [eval_results[n].get("win_rate", 0) * 100 for n in names]
        colors = [eval_results[n].get("color", "#fff") for n in names]

        bar_html = '<div style="margin-top:1rem;">'
        for name, wr, color in zip(names, win_rates, colors):
            bar_html += f"""
<div style="margin-bottom:0.6rem;">
  <div style="display:flex; justify-content:space-between; font-family:'VT323',monospace;
              color:#aaa; font-size:1rem; margin-bottom:3px;">
    <span>{name}</span><span style="color:{color}">{wr:.1f}%</span>
  </div>
  <div style="background:#1e1e2e; border-radius:4px; height:20px; overflow:hidden;">
    <div style="width:{wr}%; height:100%; background:linear-gradient(90deg,{color}66,{color});
                border-radius:4px; transition:width 0.5s;"></div>
  </div>
</div>"""
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

    # ── Tab 2: Training Curves ────────────────────────────────────────────────
    with tab2:
        curve_type = st.radio("Show:", ["Smoothed Reward", "Win Rate"], horizontal=True)

        # Build inline SVG chart
        selected_curves = {}
        for name, data in curves.items():
            if curve_type == "Smoothed Reward":
                selected_curves[name] = (data["rewards"], data["color"])
            else:
                selected_curves[name] = (data["win_rates"], data["color"])

        # SVG line chart
        W, H = 800, 300
        PAD = 40
        chart_w = W - PAD * 2
        chart_h = H - PAD * 2

        all_vals = []
        for name, (vals, color) in selected_curves.items():
            all_vals.extend(vals)

        if all_vals:
            y_min = min(all_vals)
            y_max = max(all_vals) + 1e-6
            max_len = max(len(v) for _, (v, _) in selected_curves.items())

            def to_svg(vals, color):
                if not vals:
                    return ""
                pts = []
                for i, v in enumerate(vals):
                    x = PAD + (i / max(max_len - 1, 1)) * chart_w
                    y = PAD + (1 - (v - y_min) / (y_max - y_min)) * chart_h
                    pts.append(f"{x:.1f},{y:.1f}")
                path = " ".join(pts)
                return f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>'

            legend_items = ""
            lines_svg = ""
            for name, (vals, color) in selected_curves.items():
                lines_svg += to_svg(vals, color)
                legend_items += f'<text x="0" y="0" fill="{color}" font-size="12">■ {name}</text>'

            # Grid lines
            grids = ""
            for i in range(5):
                y = PAD + (i / 4) * chart_h
                val = y_max - (i / 4) * (y_max - y_min)
                grids += f'<line x1="{PAD}" y1="{y:.0f}" x2="{W - PAD}" y2="{y:.0f}" stroke="#2a2a3a" stroke-width="1"/>'
                grids += f'<text x="{PAD - 5}" y="{y + 4:.0f}" fill="#555" font-size="10" text-anchor="end">{val:.1f}</text>'

            # Legend (horizontal)
            legend_svg = ""
            lx = PAD
            for name, (vals, color) in selected_curves.items():
                legend_svg += f'<rect x="{lx}" y="{H - 15}" width="12" height="8" fill="{color}"/>'
                legend_svg += f'<text x="{lx + 16}" y="{H - 8}" fill="#aaa" font-size="11">{name}</text>'
                lx += 120

            x_label = "Training Episodes (smoothed)"
            y_label = "Win Rate" if curve_type == "Win Rate" else "Reward"

            svg = f"""
<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg"
     style="background:#111118; border-radius:8px; border:1px solid #2a2a3a;">
  {grids}
  {lines_svg}
  {legend_svg}
  <text x="{W//2}" y="{H - 2}" fill="#555" font-size="11" text-anchor="middle">{x_label}</text>
  <text x="12" y="{H//2}" fill="#555" font-size="11" text-anchor="middle"
        transform="rotate(-90, 12, {H//2})">{y_label}</text>
</svg>"""
            st.markdown(svg, unsafe_allow_html=True)

        # Epsilon decay chart for applicable agents
        st.markdown("""<div style="font-family:'VT323',monospace; color:#8a8580; margin-top:1.2rem; margin-bottom:0.5rem;">ε (Epsilon) Decay over Training</div>""", unsafe_allow_html=True)

        eps_curves = {n: d["epsilon"] for n, d in curves.items() if d["epsilon"]}
        if eps_curves:
            W2, H2 = 800, 180
            all_e = [v for vals in eps_curves.values() for v in vals]
            max_e = max(all_e) if all_e else 1
            max_len2 = max(len(v) for v in eps_curves.values())
            eps_lines = ""
            eps_leg = ""
            lx2 = PAD
            for name, vals in eps_curves.items():
                color = curves[name]["color"]
                pts = []
                for i, v in enumerate(vals):
                    x = PAD + (i / max(max_len2 - 1, 1)) * (W2 - PAD * 2)
                    y = 20 + (1 - v / max_e) * (H2 - 40)
                    pts.append(f"{x:.1f},{y:.1f}")
                eps_lines += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.8"/>'
                eps_leg += f'<rect x="{lx2}" y="{H2 - 14}" width="10" height="7" fill="{color}"/>'
                eps_leg += f'<text x="{lx2 + 14}" y="{H2 - 7}" fill="#aaa" font-size="10">{name}</text>'
                lx2 += 120

            st.markdown(f"""
<svg width="{W2}" height="{H2}" xmlns="http://www.w3.org/2000/svg"
     style="background:#111118; border-radius:8px; border:1px solid #2a2a3a;">
  {eps_lines}
  {eps_leg}
  <text x="{W2//2}" y="{H2 - 2}" fill="#555" font-size="10" text-anchor="middle">Episodes</text>
  <text x="14" y="{H2//2}" fill="#555" font-size="10" text-anchor="middle"
        transform="rotate(-90, 14, {H2//2})">Epsilon</text>
</svg>""", unsafe_allow_html=True)

    # ── Tab 3: Evaluation ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("""<div style="font-family:'VT323',monospace; color:#8a8580; margin-bottom:1rem;">Greedy policy evaluated over 100 episodes per agent</div>""", unsafe_allow_html=True)

        cols = st.columns(2)
        for i, (name, stats) in enumerate(eval_results.items()):
            with cols[i % 2]:
                color = stats.get("color", "#fff")
                dr = stats.get("death_reasons", {})
                total_ep = sum(dr.values()) or 1

                death_bars = ""
                death_colors = {
                    "win": "#27ae60",
                    "fear": "#e74c3c",
                    "sanity": "#8e44ad",
                    "timeout": "#f39c12"
                }
                for reason, count in dr.items():
                    pct = count / total_ep * 100
                    dc = death_colors.get(reason, "#aaa")
                    death_bars += f"""
<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:3px;">
  <span style="font-family:'VT323',monospace; color:{dc}; width:80px; font-size:0.9rem;">{reason}</span>
  <div style="flex:1; background:#1e1e2e; border-radius:3px; height:12px;">
    <div style="width:{pct}%; height:100%; background:{dc}; border-radius:3px;"></div>
  </div>
  <span style="font-family:'VT323',monospace; color:#666; font-size:0.8rem; width:40px;">{pct:.0f}%</span>
</div>"""

                st.markdown(f"""
<div style="background:#111118; border:1px solid {color}44; border-radius:10px;
            padding:1rem; margin-bottom:1rem;">
  <div style="font-family:'Creepster',cursive; font-size:1.4rem; color:{color}; margin-bottom:0.5rem;">
    {name}
  </div>
  <div style="font-family:'VT323',monospace; font-size:1rem; color:#aaa; margin-bottom:0.8rem;">
    Win Rate: <span style="color:{color}; font-size:1.2rem;">{stats.get('win_rate',0)*100:.1f}%</span>
    &nbsp;|&nbsp; Avg Reward: {stats.get('mean_reward',0):.1f}
    &nbsp;|&nbsp; Avg Turns: {stats.get('mean_length',0):.0f}
  </div>
  <div style="font-family:'VT323',monospace; color:#555; font-size:0.85rem; margin-bottom:0.5rem;">
    Episode Outcomes:
  </div>
  {death_bars}
</div>
""", unsafe_allow_html=True)

    # ── Tab 4: Algorithm Details ──────────────────────────────────────────────
    with tab4:
        algo_info = {
            "Q-Learning": {
                "color": "#00d4ff",
                "type": "Off-Policy TD",
                "formula": "Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]",
                "update": "Uses the BEST next action (greedy) regardless of policy",
                "pros": "Faster convergence, finds optimal policy even with random exploration",
                "cons": "Can overestimate Q-values; less stable in stochastic environments",
                "hyperparams": "α=0.1, γ=0.95, ε decay=0.995",
                "memory": "Discrete Q-table (state buckets × actions)"
            },
            "SARSA": {
                "color": "#ff6b6b",
                "type": "On-Policy TD",
                "formula": "Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]",
                "update": "Uses the ACTUAL next action taken (including exploration)",
                "pros": "Safer/more conservative; accounts for exploration in updates",
                "cons": "Slower convergence; may not reach optimal policy with ε-greedy",
                "hyperparams": "α=0.1, γ=0.95, ε decay=0.995",
                "memory": "Discrete Q-table (state buckets × actions)"
            },
            "DQN": {
                "color": "#a29bfe",
                "type": "Deep Q-Network",
                "formula": "Loss = E[(r + γ·max Q̂(s',a') − Q(s,a))²]",
                "update": "Neural network approximates Q-function; experience replay + target net",
                "pros": "Handles continuous/high-dim state; stabilised by replay & target network",
                "cons": "Slower to train on small state spaces; more hyperparameters",
                "hyperparams": "α=0.001, γ=0.95, ε decay=0.997, batch=32, target_update=50",
                "memory": "Neural network (12→64→6) + replay buffer"
            },
            "Random": {
                "color": "#fdcb6e",
                "type": "Baseline",
                "formula": "π(a|s) = uniform(A)",
                "update": "No learning — pure random action selection",
                "pros": "Zero compute; provides lower bound for comparison",
                "cons": "No intelligence whatsoever",
                "hyperparams": "None",
                "memory": "None"
            }
        }

        for name, info in algo_info.items():
            color = info["color"]
            stats = agents[name].get_stats() if name in agents else {}
            st.markdown(f"""
<div style="background:#111118; border:1px solid {color}33;
            border-left:4px solid {color}; border-radius:8px;
            padding:1.2rem; margin-bottom:1rem;">
  <div style="font-family:'Creepster',cursive; font-size:1.5rem; color:{color};
              letter-spacing:2px; margin-bottom:0.5rem;">
    {name} <span style="font-size:0.9rem; color:#555;">({info['type']})</span>
  </div>

  <div style="font-family:'VT323',monospace; background:#0d0d14; border:1px solid #1e1e2e;
              border-radius:4px; padding:0.5rem 0.8rem; margin-bottom:0.8rem;
              color:#00d4ff; font-size:1rem; overflow-x:auto;">
    {info['formula']}
  </div>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
    <div style="font-family:'Special Elite',serif; color:#8a8580; font-size:0.82rem;">
      <span style="color:#aaa;">Update rule:</span> {info['update']}
    </div>
    <div style="font-family:'Special Elite',serif; color:#8a8580; font-size:0.82rem;">
      <span style="color:#aaa;">Memory:</span> {info['memory']}
    </div>
    <div style="font-family:'Special Elite',serif; color:#27ae60; font-size:0.82rem;">
      ✅ {info['pros']}
    </div>
    <div style="font-family:'Special Elite',serif; color:#e74c3c; font-size:0.82rem;">
      ❌ {info['cons']}
    </div>
  </div>

  <div style="font-family:'VT323',monospace; color:#555; font-size:0.85rem; margin-top:0.5rem;">
    Hyperparams: {info['hyperparams']}
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AI WATCH MODE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "ai_watch":
    st.markdown("""
<div style="font-family:'Creepster',cursive; font-size:2rem; color:#00d4ff;
            letter-spacing:3px; margin-bottom:0.5rem;">
  🤖 WATCH AI PLAY
</div>""", unsafe_allow_html=True)

    agents = st.session_state.agents
    agent_names = list(agents.keys())

    col_sel, col_btn = st.columns([2, 1])
    with col_sel:
        chosen = st.selectbox("Choose Algorithm:", agent_names)
    with col_btn:
        if st.button("▶ Start New Episode", use_container_width=True):
            env = DreamLeechEnv(max_turns=150, seed=random.randint(0, 9999))
            obs = env.reset()
            st.session_state.ai_watch_env = env
            st.session_state.ai_watch_obs = obs
            st.session_state.ai_watch_agent = agents[chosen]
            st.session_state.ai_watch_done = False
            st.session_state.event_log = ["AI begins its nightmare..."]
            st.rerun()

    if st.session_state.ai_watch_env is None:
        st.markdown("""
<div style="text-align:center; padding:3rem; color:#444;
            font-family:'Special Elite',serif;">
  Select an agent and click "Start New Episode" to watch the AI play.
</div>""", unsafe_allow_html=True)
        st.stop()

    env: DreamLeechEnv = st.session_state.ai_watch_env
    agent = st.session_state.ai_watch_agent
    obs = st.session_state.ai_watch_obs
    s: GameState = env.state

    # Show current state
    left_ai, right_ai = st.columns([2, 1])

    with left_ai:
        # Room info
        room_id = s.player_room
        has_item = room_id in ITEMS_IN_ROOMS and not s.items_collected.get(room_id, False)
        item_name = ITEMS_IN_ROOMS.get(room_id, "") if has_item else ""
        st.markdown(
            room_card_html(room_id, ROOM_DESCRIPTIONS[room_id], has_item, item_name).strip(),
            unsafe_allow_html=True
        )

        # Agent decision
        if not st.session_state.ai_watch_done:
            q_action = agent.select_action(obs, greedy=True)
            action_names = ["MOVE", "TOGGLE CANDLE", "USE MEMORY", "USE SAFE THOUGHT", "USE MIRROR SHARD", "WAIT"]
            st.markdown(f"""
<div style="background:#111118; border:1px solid #00d4ff33;
            border-radius:6px; padding:0.5rem 0.8rem; margin-bottom:0.8rem;
            font-family:'VT323',monospace; color:#00d4ff; font-size:1rem;">
  🤖 Agent decision: <span style="color:#fff">{action_names[q_action]}</span>
  &nbsp;|&nbsp; Algorithm: <span style="color:{agent.color}">{agent.name}</span>
</div>""", unsafe_allow_html=True)

        # Event log
        st.markdown(event_log_html(st.session_state.event_log), unsafe_allow_html=True)

        # Controls
        if not st.session_state.ai_watch_done:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("⏭ Next Step", use_container_width=True):
                    action = agent.select_action(obs, greedy=True)
                    next_obs, rew, done, info = env.step(action)
                    st.session_state.event_log.append(s.last_event)
                    st.session_state.ai_watch_obs = next_obs
                    if done:
                        st.session_state.ai_watch_done = True
                    st.rerun()
            with btn_col2:
                if st.button("⚡ Run to End", use_container_width=True):
                    while not env.state.is_terminal:
                        action = agent.select_action(obs, greedy=True)
                        obs, rew, done, info = env.step(action)
                        st.session_state.event_log.append(env.state.last_event)
                        if done:
                            break
                    st.session_state.ai_watch_obs = obs
                    st.session_state.ai_watch_done = True
                    st.rerun()
        else:
            st.markdown(
                game_over_html(s.terminal_reason == "win", s.terminal_reason, s.turn),
                unsafe_allow_html=True
            )

    with right_ai:
        st.markdown(meter_html("⚡ FEAR", s.fear, 100, "#e74c3c"), unsafe_allow_html=True)
        st.markdown(meter_html("🧠 SANITY", s.sanity, 100, "#2ecc71"), unsafe_allow_html=True)

        st.markdown("""<div style="font-family:'VT323',monospace; color:#8a8580; font-size:0.9rem; margin:0.5rem 0 0.3rem;">── MAP ──</div>""", unsafe_allow_html=True)
        components.html(
            minimap_html(s.player_room, s.leech_room, s.rooms_visited, True),
            height=250,
        )

        # Agent stats
        stats = agent.get_stats()
        color = stats.get("color", "#fff")
        st.markdown(f"""
<div style="margin-top:0.8rem; background:#111118; border:1px solid {color}33;
            border-radius:8px; padding:0.8rem;">
  <div style="font-family:'VT323',monospace; color:{color}; font-size:1rem; margin-bottom:0.5rem;">
    {agent.name} — Training History
  </div>
  <div style="font-family:'VT323',monospace; color:#666; font-size:0.85rem; line-height:1.8;">
    Episodes: {stats.get('total_episodes',0)}<br>
    Win rate: {stats.get('win_rate',0)*100:.1f}%<br>
    Avg reward: {stats.get('mean_reward',0):.1f}<br>
    Train time: {stats.get('training_time',0):.1f}s
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 0.5rem;
            font-family:'VT323',monospace; color:#2a2a3a; font-size:0.85rem;">
  DREAM LEECH · Reinforcement Learning Project · Q-Learning | SARSA | DQN
</div>
""", unsafe_allow_html=True)