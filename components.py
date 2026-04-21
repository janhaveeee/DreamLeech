"""
Dream Leech - UI Components (HTML strings for st.markdown)
"""

from environment import ROOM_NAMES, CONNECTIONS, ITEMS_IN_ROOMS, ITEM_EMOJIS


HORROR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Creepster&family=Special+Elite&family=VT323:wght@400&display=swap');

/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #ddd8cc;
}
[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid #1e1e2e;
}
.stButton > button {
    background: #16161f !important;
    color: #ddd8cc !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 4px !important;
    font-family: 'Special Elite', serif !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1e1e30 !important;
    border-color: #00d4ff !important;
    box-shadow: 0 0 8px #00d4ff44 !important;
}
h1, h2, h3 { font-family: 'Creepster', cursive !important; letter-spacing: 2px; }
.stProgress > div > div { background: #1e1e2e !important; }
div[data-testid="stMarkdownContainer"] p { font-family: 'Special Elite', serif; }
.stTabs [data-baseweb="tab"] {
    font-family: 'VT323', monospace !important;
    font-size: 1.1rem !important;
    color: #8a8580 !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}
</style>
"""


def title_html() -> str:
    return """
<div style="text-align:center; padding: 1rem 0 0.5rem;">
  <h1 style="
    font-family:'Creepster',cursive;
    font-size:3.5rem;
    color:#c0392b;
    text-shadow: 0 0 20px #8b000088, 0 0 40px #8b000044;
    letter-spacing:6px;
    margin:0;
  ">🩸 DREAM LEECH 🩸</h1>
  <p style="font-family:'Special Elite',serif; color:#8a8580; font-size:0.9rem; margin:4px 0 0;">
    Find the Mirror Room. Escape the nightmare.
  </p>
</div>
"""


def room_card_html(room_id: int, description: str, has_item: bool, item_name: str = "") -> str:
    item_badge = ""
    if has_item:
        emoji = ITEM_EMOJIS.get(item_name, "?")
        item_badge = f"""<div style="margin-top:8px; padding:4px 10px; background:#1e1e2e;
             border:1px solid #2a2a3a; border-radius:20px; display:inline-block;
             font-family:'VT323',monospace; color:#f39c12; font-size:1rem;">
             {emoji} {item_name.replace('_', ' ').title()} — pick up on enter
             </div>"""
    return f"""
<div style="
  background:#16161f;
  border:1px solid #2a2a3a;
  border-radius:8px;
  padding:1rem 1.2rem;
  margin-bottom:0.5rem;
  box-shadow: inset 0 0 30px #00000066;
">
  <div style="font-family:'Creepster',cursive; font-size:1.6rem; color:#ddd8cc; letter-spacing:2px;">
    {ROOM_NAMES[room_id]}
  </div>
  <div style="font-family:'Special Elite',serif; color:#8a8580; font-size:0.88rem; margin-top:4px; line-height:1.5;">
    {description}
  </div>
  {item_badge}
</div>
"""


def meter_html(label: str, value: float, max_val: float, color: str, warning_at: float = None) -> str:
    pct = max(0, min(100, (value / max_val) * 100))
    warn = ""
    if warning_at and ((color == "#e74c3c" and pct >= warning_at) or
                       (color == "#2ecc71" and pct <= warning_at)):
        warn = f"animation: pulse 1s infinite;"
    return f"""
<div style="margin-bottom:0.6rem;">
  <div style="display:flex; justify-content:space-between; font-family:'VT323',monospace;
              font-size:1.1rem; color:#aaa; margin-bottom:2px;">
    <span>{label}</span><span style="color:{color}">{value:.0f} / {max_val:.0f}</span>
  </div>
  <div style="background:#1e1e2e; border-radius:4px; height:14px; overflow:hidden;">
    <div style="
      width:{pct}%;
      height:100%;
      background: linear-gradient(90deg, {color}88, {color});
      border-radius:4px;
      transition: width 0.4s ease;
      {warn}
    "></div>
  </div>
</div>
"""


def minimap_html(player_room: int, leech_room: int, rooms_visited: set,
                 leech_known: bool) -> str:
    # Fixed positions for rooms in a grid layout
    positions = {
        0: (1, 0), 1: (2, 0), 2: (3, 0), 3: (2, 1),
        4: (4, 0), 5: (3, 1), 6: (2, 2), 7: (3, 2),
        8: (5, 0), 9: (6, 0),
    }
    cell = 56
    pad = 20
    cols = 8
    rows = 4
    w = cols * cell + pad * 2
    h = rows * cell + pad * 2

    # SVG connections
    lines = []
    drawn = set()
    for room, neighbors in CONNECTIONS.items():
        for nb in neighbors:
            key = tuple(sorted([room, nb]))
            if key in drawn:
                continue
            drawn.add(key)
            r1 = positions[room]
            r2 = positions[nb]
            x1 = pad + r1[0] * cell + cell // 2
            y1 = pad + r1[1] * cell + cell // 2
            x2 = pad + r2[0] * cell + cell // 2
            y2 = pad + r2[1] * cell + cell // 2
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2a2a3a" stroke-width="2"/>')

    # Room circles
    circles = []
    for room_id, (col, row) in positions.items():
        cx = pad + col * cell + cell // 2
        cy = pad + row * cell + cell // 2
        r = 18

        if room_id not in rooms_visited:
            fill = "#111"
            stroke = "#222"
            label = "❓"
        elif room_id == 9:
            fill = "#1a2a1a"
            stroke = "#27ae60"
            label = "🪞"
        else:
            fill = "#1a1a2a"
            stroke = "#2a2a4a"
            label = str(room_id)

        # Player
        if room_id == player_room:
            stroke = "#00d4ff"
            fill = "#001a2a"
            label = "⭐"
        # Leech
        if room_id == leech_room and leech_known:
            stroke = "#c0392b"
            fill = "#2a0000"
            label = "🔴"

        room_label = ROOM_NAMES[room_id][:4]
        circles.append(f"""
          <circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
          <text x="{cx}" y="{cy + 5}" text-anchor="middle" font-size="12"
                font-family="monospace" fill="#aaa">{label}</text>
          <text x="{cx}" y="{cy + r + 12}" text-anchor="middle" font-size="9"
                font-family="monospace" fill="#555">{room_label}</text>
        """)

    return f"""
<div style="overflow-x:auto; border:1px solid #2a2a3a; border-radius:8px;
            background:#0d0d14; padding:0.5rem;">
  <svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
    {''.join(lines)}
    {''.join(circles)}
  </svg>
</div>
"""


def event_log_html(events: list) -> str:
    items = ""
    for i, ev in enumerate(reversed(events[-8:])):
        opacity = 1.0 - i * 0.1
        color = "#ddd8cc" if i == 0 else "#8a8580"
        items += f'<div style="font-family:\'Special Elite\',serif; color:{color}; opacity:{opacity}; font-size:0.85rem; padding:3px 0; border-bottom:1px solid #1e1e2e;">{ev}</div>'
    return f"""
<div style="background:#111118; border:1px solid #1e1e2e; border-radius:8px;
            padding:0.8rem 1rem; max-height:200px; overflow-y:auto;">
  <div style="font-family:'VT323',monospace; color:#555; font-size:0.8rem; margin-bottom:6px;">
    ── EVENT LOG ──
  </div>
  {items}
</div>
"""


def status_badge(text: str, color: str) -> str:
    return f"""
<span style="background:{color}22; border:1px solid {color}66;
             color:{color}; padding:2px 10px; border-radius:20px;
             font-family:'VT323',monospace; font-size:1rem;">
  {text}
</span>
"""


def game_over_html(won: bool, reason: str, turns: int) -> str:
    if won:
        title = "ESCAPED"
        subtitle = "You found the Mirror Room and fled the nightmare."
        color = "#27ae60"
        emoji = "🪞✨"
    elif reason == "fear":
        title = "CONSUMED BY FEAR"
        subtitle = "The Leech found you at your most terrified. It fed well."
        color = "#c0392b"
        emoji = "🩸"
    elif reason == "sanity":
        title = "MIND SHATTERED"
        subtitle = "The darkness swallowed your last thread of sanity."
        color = "#8e44ad"
        emoji = "💀"
    else:
        title = "LOST FOREVER"
        subtitle = "The nightmare stretched on until time lost meaning."
        color = "#f39c12"
        emoji = "⌛"

    return f"""
<div style="
  text-align:center;
  padding:2rem;
  background:#111118;
  border:2px solid {color};
  border-radius:12px;
  box-shadow: 0 0 30px {color}44;
">
  <div style="font-size:3rem; margin-bottom:0.5rem;">{emoji}</div>
  <div style="font-family:'Creepster',cursive; font-size:2.5rem; color:{color};
              letter-spacing:4px; text-shadow: 0 0 15px {color}88;">
    {title}
  </div>
  <div style="font-family:'Special Elite',serif; color:#8a8580; margin-top:0.5rem;">
    {subtitle}
  </div>
  <div style="font-family:'VT323',monospace; color:#555; margin-top:1rem; font-size:1.1rem;">
    Survived {turns} turns
  </div>
</div>
"""


def agent_card_html(stats: dict) -> str:
    win_pct = stats.get('win_rate', 0) * 100
    color = stats.get('color', '#ffffff')
    eps = f"ε = {stats.get('epsilon', 0):.3f}" if 'epsilon' in stats else ""
    q_size = f"Q-states: {stats.get('q_table_size', 0):,}" if 'q_table_size' in stats else ""
    buf = f"Buffer: {stats.get('buffer_size', 0):,}" if 'buffer_size' in stats else ""
    loss = f"Loss: {stats.get('avg_loss', 0):.4f}" if 'avg_loss' in stats else ""

    detail = " · ".join(filter(None, [eps, q_size, buf, loss]))

    return f"""
<div style="
  background:#111118;
  border:1px solid {color}44;
  border-left: 3px solid {color};
  border-radius:8px;
  padding:0.8rem 1rem;
  margin-bottom:0.5rem;
">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span style="font-family:'Creepster',cursive; font-size:1.3rem; color:{color}; letter-spacing:1px;">
      {stats['agent']}
    </span>
    <span style="font-family:'VT323',monospace; font-size:1.2rem; color:{color};">
      {win_pct:.1f}% win
    </span>
  </div>
  <div style="font-family:'VT323',monospace; color:#555; font-size:0.9rem; margin-top:4px;">
    Reward: {stats.get('mean_reward', 0):.1f} avg · {stats.get('total_episodes', 0)} episodes · {stats.get('training_time', 0):.1f}s
  </div>
  {f'<div style="font-family:monospace; color:#444; font-size:0.8rem; margin-top:2px;">{detail}</div>' if detail else ''}
</div>
"""