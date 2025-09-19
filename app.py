# app.py
import os
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

# ===================== Page & CSS (no top banners) =====================
st.set_page_config(page_title="Fight Health & Fatigue HUD", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 0.6rem; }
h1,h2,h3 { font-weight: 800; letter-spacing: .2px; }
.smallcaps { letter-spacing: .08em; text-transform: uppercase; font-size: .78rem; opacity:.72;}
.kpi { font-size: 1.15rem; font-weight: 800; }
.legend { font-size:.85rem; opacity:.7; }
.frame { border-radius:12px; padding:10px 12px; border:1px solid rgba(0,0,0,.08); }
.stAlert { display:none; }  /* hide Streamlit default success/info banners */
</style>
""", unsafe_allow_html=True)

# ===================== Constants & Defaults =====================
ZONES = ["head", "left_arm", "right_arm", "torso", "legs"]

DEFAULT_ACTION_COSTS = {
    "jab": {"zone": "right_arm", "cost": 0.20},
    "cross": {"zone": "right_arm", "cost": 0.35},
    "hook": {"zone": "left_arm", "cost": 0.40},
    "uppercut": {"zone": "right_arm", "cost": 0.45},
    "advance": {"zone": "legs", "cost": 0.04},
    "retreat": {"zone": "legs", "cost": 0.03},
    "circle": {"zone": "legs", "cost": 0.035},
    "clinch": {"zone": "torso", "cost": 0.25},
}
DEFAULT_IMPACT_COSTS = {"head": 1.2, "torso": 0.9, "left_arm": 0.6, "right_arm": 0.6, "legs": 0.5}
DEFAULT_BASE_RECOVERY = {z: 0.06 for z in ZONES}
DEFAULT_REST_RECOVERY = {z: 0.12 for z in ZONES}
DEFAULT_ROUND_INTENSITY = [1.10, 1.00, 0.95, 0.90, 0.85]
DEFAULT_BASE_RATES = {"jab":0.25,"cross":0.10,"hook":0.08,"uppercut":0.05,"advance":0.40,"retreat":0.25,"circle":0.30,"clinch":0.03}
P_LAND = {"jab":0.35,"cross":0.30,"hook":0.28,"uppercut":0.25}
P_TARGET = {"head":0.6,"torso":0.3,"left_arm":0.05,"right_arm":0.03,"legs":0.02}
DEFAULT_ZONE_WEIGHTS = {"head":0.20,"left_arm":0.15,"right_arm":0.20,"torso":0.25,"legs":0.20}

# ===================== Assets (better placeholder; supports uploads) =====================
def draw_placeholder_silhouette(w=420, h=820):
    """Simple neutral human-ish silhouette (cleaner than the chunky one)."""
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(img)
    base = (25,25,25)
    # head
    d.ellipse((170, 35, 250, 115), fill=base)
    # neck + shoulders + torso
    d.rounded_rectangle((155, 115, 265, 480), radius=40, fill=base)
    d.rounded_rectangle((95, 180, 155, 440), radius=28, fill=base)   # left arm
    d.rounded_rectangle((265,180, 325, 440), radius=28, fill=base)   # right arm
    # hips
    d.polygon([(155,480),(265,480),(295,560),(125,560)], fill=base)
    # legs
    d.rounded_rectangle((145, 560, 195, 780), radius=24, fill=base)
    d.rounded_rectangle((225, 560, 275, 780), radius=24, fill=base)
    return img

def fatigue_to_rgb(v: float, amplify: float = 1.0):
    # 0 fresh -> green; 100 taxed -> red. Amplify = visual boost.
    v = float(np.clip(v * amplify, 0.0, 100.0)) / 100.0
    r = min(1.0, 2.0 * v)
    g = min(1.0, 2.0 * (1.0 - v))
    return int(r*255), int(g*255), 0

def draw_heat_overlay(img: Image.Image, fatigue: dict, alpha=128, amplify=1.6) -> Image.Image:
    """Draw body-part heat rectangles aligned to this placeholder's proportions."""
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)

    # relative boxes (x0,y0,x1,y1), tuned to placeholder above
    boxes = {
        "head":      (0.41, 0.05, 0.59, 0.16),
        "torso":     (0.37, 0.18, 0.63, 0.52),
        "left_arm":  (0.23, 0.22, 0.35, 0.54),
        "right_arm": (0.65, 0.22, 0.77, 0.54),
        "legs":      (0.39, 0.58, 0.61, 0.92),
    }
    for z, (x0, y0, x1, y1) in boxes.items():
        color = fatigue_to_rgb(fatigue[z], amplify=amplify) + (alpha,)
        X0, Y0, X1, Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
        if z == "head":
            d.ellipse((X0, Y0, X1, Y1), fill=color)
        else:
            radius = max(10, int((X1-X0) * 0.2))
            d.rounded_rectangle((X0, Y0, X1, Y1), radius=radius, fill=color)

    return Image.alpha_composite(img.convert("RGBA"), overlay)

# ===================== Simulation =====================
def overall_stamina(fatigue_dict, weights):
    weighted = sum(weights[z] * fatigue_dict[z] for z in ZONES)
    return max(0.0, 100.0 - weighted)

def simulate(rounds=5, round_sec=180, rest_sec=60, hz=5,
             action_costs=None, impact_costs=None,
             base_recovery=None, rest_recovery=None,
             zone_weights=None, base_rates=None,
             round_intensity=None, seed=42):
    np.random.seed(seed)
    action_costs = action_costs or DEFAULT_ACTION_COSTS
    impact_costs = impact_costs or DEFAULT_IMPACT_COSTS
    base_recovery = base_recovery or DEFAULT_BASE_RECOVERY
    rest_recovery = rest_recovery or DEFAULT_REST_RECOVERY
    weights = zone_weights or DEFAULT_ZONE_WEIGHTS
    base_rates = base_rates or DEFAULT_BASE_RATES
    round_intensity = round_intensity or DEFAULT_ROUND_INTENSITY

    dt = 1.0 / hz
    fighters = ["Red","Blue"]
    rows = []
    fatigue = {f: {z: 0.0 for z in ZONES} for f in fighters}
    t_global = 0.0
    targets = list(P_TARGET.keys()); p_targets = list(P_TARGET.values())

    for r in range(1, rounds+1):
        # Active
        for _ in range(int(round_sec*hz)):
            t_global += dt
            for f in fighters:
                opp = "Blue" if f=="Red" else "Red"
                ov_stam = overall_stamina(fatigue[f], weights)
                ov_fatigue_norm = (100.0 - ov_stam) / 100.0
                rate_scale = max(0.25, 1.0 - 0.5*ov_fatigue_norm) * round_intensity[r-1]

                events = []
                for ev, lam in base_rates.items():
                    lam_t = lam * rate_scale * dt
                    if np.random.rand() < lam_t:
                        events.append(ev)

                for ev in events:
                    z = action_costs[ev]["zone"]
                    fatigue[f][z] = min(100.0, fatigue[f][z] + action_costs[ev]["cost"])

                    landed, target, fval = None, None, None
                    if ev in ["jab","cross","hook","uppercut"]:
                        if np.random.rand() < P_LAND[ev]:
                            tgt = np.random.choice(targets, p=p_targets)
                            force = float(np.clip(np.random.normal(0.6,0.2), 0.05, 1.0))
                            impact = impact_costs[tgt] * force
                            fatigue[opp][tgt] = min(100.0, fatigue[opp][tgt] + impact)
                            landed, target, fval = 1, tgt, round(force,3)
                        else:
                            landed = 0

                    rows.append({
                        "t": t_global, "round": r, "phase": "active", "actor": f, "opponent": opp,
                        "event": ev, "landed": landed, "target": target, "force": fval,
                        **{f"Red_{zz}_fatigue": fatigue["Red"][zz] for zz in ZONES},
                        **{f"Blue_{zz}_fatigue": fatigue["Blue"][zz] for zz in ZONES},
                        "Red_stamina": overall_stamina(fatigue["Red"], weights),
                        "Blue_stamina": overall_stamina(fatigue["Blue"], weights),
                    })

                # recovery during active
                for zz in ZONES:
                    fatigue[f][zz] = max(0.0, fatigue[f][zz] - base_recovery[zz]*dt)

        # Rest
        if r < rounds:
            for i in range(int(rest_sec*hz)):
                t_global += dt
                for f in fighters:
                    for zz in ZONES:
                        fatigue[f][zz] = max(0.0, fatigue[f][zz] - rest_recovery[zz]*dt)
                if i % hz == 0:
                    rows.append({
                        "t": t_global, "round": r, "phase": "rest",
                        "actor": None, "opponent": None, "event": "rest",
                        "landed": None, "target": None, "force": None,
                        **{f"Red_{zz}_fatigue": fatigue["Red"][zz] for zz in ZONES},
                        **{f"Blue_{zz}_fatigue": fatigue["Blue"][zz] for zz in ZONES},
                        "Red_stamina": overall_stamina(fatigue["Red"], weights),
                        "Blue_stamina": overall_stamina(fatigue["Blue"], weights),
                    })
    return pd.DataFrame(rows)

# ===================== Sidebar =====================
st.sidebar.header("Simulation Controls")
rounds    = st.sidebar.slider("Rounds", 1, 12, 5)
round_sec = st.sidebar.slider("Round length (sec)", 60, 240, 180, 10)
rest_sec  = st.sidebar.slider("Rest length (sec)", 0, 120, 60, 5)
hz        = st.sidebar.slider("Sampling rate (Hz)", 1, 20, 5)

st.sidebar.subheader("Recovery rates")
base_rec  = st.sidebar.slider("Active recovery (/s)", 0.00, 0.20, 0.06, 0.005)
rest_rec  = st.sidebar.slider("Rest recovery (/s)", 0.00, 0.30, 0.12, 0.005)

st.sidebar.subheader("Impact multipliers")
head_imp  = st.sidebar.slider("Head impact",  0.1, 2.0, 1.2, 0.05)
torso_imp = st.sidebar.slider("Torso impact", 0.1, 2.0, 0.9, 0.05)
arm_imp   = st.sidebar.slider("Arm impact",   0.1, 2.0, 0.6, 0.05)
leg_imp   = st.sidebar.slider("Leg impact",   0.1, 2.0, 0.5, 0.05)

st.sidebar.subheader("Display")
display_intensity = st.sidebar.slider("Body heat intensity", 0.5, 3.0, 1.6, 0.1)
img_width         = st.sidebar.slider("Image width (px)", 260, 440, 330, 10)

st.sidebar.subheader("Silhouette images (optional)")
up_red  = st.sidebar.file_uploader("Red silhouette PNG",  type=["png"], key="red_png")
up_blue = st.sidebar.file_uploader("Blue silhouette PNG", type=["png"], key="blue_png")

# store uploaded imgs in session once
if up_red and "img_red" not in st.session_state:
    st.session_state.img_red = Image.open(up_red).convert("RGBA")
if up_blue and "img_blue" not in st.session_state:
    st.session_state.img_blue = Image.open(up_blue).convert("RGBA")

# ===================== Sim (cached) =====================
@st.cache_data(show_spinner=False)
def run_sim_cached(params):
    base_recovery = {z: params['base_rec'] for z in ZONES}
    rest_recovery = {z: params['rest_rec'] for z in ZONES}
    impact_costs = {
        "head": params['head_imp'], "torso": params['torso_imp'],
        "left_arm": params['arm_imp'], "right_arm": params['arm_imp'],
        "legs": params['leg_imp'],
    }
    weights = {"head": .20, "left_arm": .15, "right_arm": .20, "torso": .25, "legs": .20}
    return simulate(params['rounds'], params['round_sec'], params['rest_sec'], params['hz'],
                    DEFAULT_ACTION_COSTS, impact_costs, base_recovery, rest_recovery, weights)

params = dict(rounds=rounds, round_sec=round_sec, rest_sec=rest_sec, hz=hz,
              base_rec=base_rec, rest_rec=rest_rec,
              head_imp=head_imp, torso_imp=torso_imp, arm_imp=arm_imp, leg_imp=leg_imp)
df = run_sim_cached(params)

# ===================== Playback State =====================
if "t" not in st.session_state:
    st.session_state.t = float(df["t"].min())
if "playing" not in st.session_state:
    st.session_state.playing = False
if "speed" not in st.session_state:
    st.session_state.speed = 1.0

# ===================== Header KPIs & Controls =====================
k1, k2, k3 = st.columns([1.2, 1.2, 2.1])
with k1:
    st.markdown('<div class="smallcaps">Red — Stamina</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{df.iloc[-1]["Red_stamina"]:.1f}</div>', unsafe_allow_html=True)
    st.progress(int(df.iloc[-1]["Red_stamina"]))
with k2:
    st.markdown('<div class="smallcaps">Blue — Stamina</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{df.iloc[-1]["Blue_stamina"]:.1f}</div>', unsafe_allow_html=True)
    st.progress(int(df.iloc[-1]["Blue_stamina"]))
with k3:
    a, b, c = st.columns([1, 2.8, 1])
    with a:
        # Toggle that doesn't reset on rerun
        st.session_state.playing = st.toggle("Play", value=st.session_state.playing)
    with b:
        # Slider always sets time; changing it pauses playback to avoid "fight"
        def _on_scrub():
            st.session_state.playing = False
        st.session_state.t = st.slider("Scrub time (s)",
                                       float(df["t"].min()), float(df["t"].max()),
                                       st.session_state.t, 0.2, on_change=_on_scrub)
    with c:
        st.session_state.speed = st.selectbox("Speed", [0.5,1.0,1.5,2.0,3.0],
                                              index=[0.5,1.0,1.5,2.0,3.0].index(st.session_state.speed))

# ===================== Tabs =====================
tab_live, tab_analytics, tab_data = st.tabs(["🎛️ Live HUD", "📈 Analytics", "📄 Data & Export"])

def frame_at(t):
    return df.iloc[int((df["t"] - t).abs().argmin())]

# -------- Live HUD --------
with tab_live:
    f = frame_at(st.session_state.t)
    red  = {z: float(f[f"Red_{z}_fatigue"]) for z in ZONES}
    blue = {z: float(f[f"Blue_{z}_fatigue"]) for z in ZONES}

    colL, colR = st.columns(2)

    def render(panel, title, img_key, default_placeholder):
        panel.markdown(f"### {title}")
        base = st.session_state.get(img_key, default_placeholder.copy())
        overlay = draw_heat_overlay(base, red if 'Red' in title else blue,
                                    alpha=138, amplify=display_intensity)
        panel.image(overlay, width=img_width, use_column_width=False)
        # compact zone readout
        m1, m2, m3 = panel.columns(3)
        state = red if 'Red' in title else blue
        m1.metric("Head",  f"{state['head']:.1f}")
        m2.metric("Torso", f"{state['torso']:.1f}")
        m3.metric("Legs",  f"{state['legs']:.1f}")
        m4, m5 = panel.columns(2)
        m4.metric("Left arm",  f"{state['left_arm']:.1f}")
        m5.metric("Right arm", f"{state['right_arm']:.1f}")

    with colL:
        render(colL, "🟥 Red — Body Fatigue Map", "img_red", draw_placeholder_silhouette())
    with colR:
        render(colR, "🟦 Blue — Body Fatigue Map", "img_blue", draw_placeholder_silhouette())

    st.markdown('<div class="legend">Color: fresh → green · working → yellow · taxed → red</div>',
                unsafe_allow_html=True)

    # Smooth playback loop (only when Play is on). Keeps one frame step per rerun.
    if st.session_state.playing:
        step = 0.12 * st.session_state.speed  # seconds per tick
        st.session_state.t = min(float(df["t"].max()), st.session_state.t + step)
        time.sleep(0.05)
        st.experimental_rerun()

# -------- Analytics --------
with tab_analytics:
    d2 = df.copy()
    d2["t_sec"] = d2["t"].round().astype(int)
    grp = d2.groupby("t_sec").agg({"Red_stamina":"mean","Blue_stamina":"mean"}).reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grp["t_sec"], grp["Red_stamina"], label="Red")
    ax.plot(grp["t_sec"], grp["Blue_stamina"], label="Blue")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Overall Stamina"); ax.set_title("Stamina Over Time")
    ax.grid(True, alpha=.25); ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Recent events (10s window around scrub time)")
    window = (df["t"] <= st.session_state.t) & (df["t"] >= st.session_state.t - 10)
    recent = df[window & (df["event"].notna()) & (df["event"]!="rest")]
    counts = recent.groupby(["actor","event"]).size().reset_index(name="count").sort_values("count", ascending=False)
    st.dataframe(counts, use_container_width=True, height=260)

# -------- Data & Export --------
with tab_data:
    st.dataframe(df.tail(400), use_container_width=True, height=360)
    st.download_button("Download timeline CSV", df.to_csv(index=False).encode(),
                       "sim_fight_timeline.csv", "text/csv")
    preset = {
        "rounds": rounds, "round_length_sec": round_sec, "rest_length_sec": rest_sec, "hz": hz,
        "impact_costs": {"head": head_imp, "torso": torso_imp, "arm": arm_imp, "leg": leg_imp},
        "recovery": {"active_per_sec": base_rec, "rest_per_sec": rest_rec},
    }
    st.download_button("Download preset JSON", json.dumps(preset, indent=2).encode(),
                       "hud_preset.json", "application/json")
