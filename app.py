import os
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


# ============ Page setup ============
st.set_page_config(page_title="Fight Health & Fatigue HUD", layout="wide")
st.markdown("""
<style>
/* cleaner, modern look */
section.main > div { padding-top: 1rem; }
.block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
.smallcaps { letter-spacing: .08em; text-transform: uppercase; font-size: .8rem; opacity: .75; }
.kpi { font-size: 1.2rem; font-weight: 700; }
hr { margin: 0.6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ============ Constants ============
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
DEFAULT_BASE_RATES = {
    "jab": 0.25, "cross": 0.10, "hook": 0.08, "uppercut": 0.05,
    "advance": 0.40, "retreat": 0.25, "circle": 0.30, "clinch": 0.03
}
P_LAND = {"jab": 0.35, "cross": 0.30, "hook": 0.28, "uppercut": 0.25}
P_TARGET = {"head": 0.6, "torso": 0.3, "left_arm": 0.05, "right_arm": 0.03, "legs": 0.02}

DEFAULT_ZONE_WEIGHTS = {"head": 0.20, "left_arm": 0.15, "right_arm": 0.20, "torso": 0.25, "legs": 0.20}

ASSETS_RED = "assets/red_silhouette.png"
ASSETS_BLUE = "assets/blue_silhouette.png"

# ===== Utility: ensure placeholder silhouettes if you haven’t provided art =====
def ensure_placeholder(path: str, tint=(22,22,22), accent=(85,85,85), w=520, h=900):
    if os.path.exists(path):
        return
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(img)
    # Simple boxer shape
    d.ellipse((210, 70, 310, 170), fill=tint)                      # head
    d.rounded_rectangle((190, 170, 330, 540), 40, fill=tint)       # torso/hips upper
    d.polygon([(190,540),(330,540),(360,660),(160,660)], fill=tint) # hips/shorts
    d.rounded_rectangle((170, 660, 230, 850), 25, fill=tint)       # left leg
    d.rounded_rectangle((290, 660, 350, 850), 25, fill=tint)       # right leg
    d.rounded_rectangle((120, 240, 190, 490), 30, fill=tint)       # left arm
    d.rounded_rectangle((330, 240, 400, 490), 30, fill=tint)       # right arm
    d.ellipse((90, 350, 180, 440), fill=tint)                      # left glove
    d.ellipse((340, 350, 430, 440), fill=tint)                     # right glove

    # Zone markers (accent) so overlay makes visual sense even without heat
    d.ellipse((230, 90, 290, 150), fill=accent)                    # head
    d.rounded_rectangle((210, 260, 310, 500), 25, fill=accent)     # torso
    d.rounded_rectangle((130, 250, 180, 450), 20, fill=accent)     # L arm
    d.rounded_rectangle((340, 250, 390, 450), 20, fill=accent)     # R arm
    d.polygon([(195, 660),(325,660),(340,760),(180,760)], fill=accent)  # legs
    img.save(path)

os.makedirs("assets", exist_ok=True)
ensure_placeholder(ASSETS_RED)
ensure_placeholder(ASSETS_BLUE)

# ===== Model pieces =====
def overall_stamina(fatigue_dict, weights):
    weighted_fatigue = sum(weights[z] * fatigue_dict[z] for z in ZONES)
    return max(0.0, 100.0 - weighted_fatigue)

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
    fighters = ["Red", "Blue"]
    rows = []
    fatigue = {f: {z: 0.0 for z in ZONES} for f in fighters}
    t_global = 0.0

    targets = list(P_TARGET.keys())
    p_targets = list(P_TARGET.values())

    for r in range(1, rounds + 1):
        # Active phase
        for _ in range(int(round_sec * hz)):
            t_global += dt
            for f in fighters:
                opp = "Blue" if f == "Red" else "Red"
                ov_stam = overall_stamina(fatigue[f], weights)
                ov_fatigue_norm = (100.0 - ov_stam) / 100.0
                rate_scale = max(0.2, 1.0 - 0.5 * ov_fatigue_norm) * round_intensity[r-1]

                events = []
                for ev, lam in base_rates.items():
                    # Poisson-ish Bernoulli draw per dt
                    lam_t = lam * rate_scale * dt
                    if np.random.rand() < lam_t:
                        events.append(ev)

                for ev in events:
                    info = action_costs[ev]
                    fatigue[f][info["zone"]] = min(100.0, fatigue[f][info["zone"]] + info["cost"])

                    if ev in ["jab","cross","hook","uppercut"]:
                        if np.random.rand() < P_LAND[ev]:
                            tgt = np.random.choice(targets, p=p_targets)
                            force = float(np.clip(np.random.normal(0.6, 0.2), 0.05, 1.0))
                            impact = impact_costs[tgt] * force
                            fatigue[opp][tgt] = min(100.0, fatigue[opp][tgt] + impact)
                            landed, target, fval = 1, tgt, round(force, 3)
                        else:
                            landed, target, fval = 0, None, None
                    else:
                        landed, target, fval = None, None, None

                    rows.append({
                        "t": t_global, "round": r, "phase": "active", "actor": f, "opponent": opp,
                        "event": ev, "landed": landed, "target": target, "force": fval,
                        **{f"Red_{z}_fatigue": fatigue["Red"][z] for z in ZONES},
                        **{f"Blue_{z}_fatigue": fatigue["Blue"][z] for z in ZONES},
                        "Red_stamina": overall_stamina(fatigue["Red"], weights),
                        "Blue_stamina": overall_stamina(fatigue["Blue"], weights)
                    })

                # continuous recovery during active
                for z in ZONES:
                    fatigue[f][z] = max(0.0, fatigue[f][z] - base_recovery[z]*dt)

        # Rest phase
        if r < rounds:
            for i in range(int(rest_sec * hz)):
                t_global += dt
                for f in fighters:
                    for z in ZONES:
                        fatigue[f][z] = max(0.0, fatigue[f][z] - rest_recovery[z]*dt)
                # Sample a few points so the CSV doesn't explode
                if i % hz == 0:
                    rows.append({
                        "t": t_global, "round": r, "phase": "rest", "actor": None, "opponent": None,
                        "event": "rest", "landed": None, "target": None, "force": None,
                        **{f"Red_{z}_fatigue": fatigue["Red"][z] for z in ZONES},
                        **{f"Blue_{z}_fatigue": fatigue["Blue"][z] for z in ZONES},
                        "Red_stamina": overall_stamina(fatigue["Red"], weights),
                        "Blue_stamina": overall_stamina(fatigue["Blue"], weights)
                    })

    return pd.DataFrame(rows)

# ===== Color mapping: fatigue (0-100) -> RGB green->yellow->red
def fatigue_to_rgb(v: float):
    v = float(np.clip(v, 0.0, 100.0)) / 100.0
    r = min(1.0, 2.0 * v)
    g = min(1.0, 2.0 * (1.0 - v))
    b = 0.0
    return int(r*255), int(g*255), int(b*255)

# ===== Heat overlay drawing on top of silhouette image
def draw_heat_overlay(img: Image.Image, fatigue_dict: dict, alpha=120) -> Image.Image:
    """
    Draws 5-zone heat overlay on the silhouette. Zones are defined as relative boxes to image size.
    """
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)

    # Define relative zones (x0,y0,x1,y1) in 0..1 coords (tuned to placeholder proportions)
    boxes = {
        "head":      (0.44, 0.07, 0.56, 0.17),
        "torso":     (0.42, 0.29, 0.58, 0.56),
        "left_arm":  (0.23, 0.27, 0.35, 0.54),
        "right_arm": (0.65, 0.27, 0.77, 0.54),
        "legs":      (0.40, 0.62, 0.60, 0.84),
    }

    # Draws rounded rectangles or ellipses per zone
    for z, (x0, y0, x1, y1) in boxes.items():
        color = fatigue_to_rgb(fatigue_dict[z]) + (alpha,)
        X0, Y0, X1, Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
        if z == "head":
            d.ellipse((X0, Y0, X1, Y1), fill=color)
        else:
            radius = max(8, int((X1-X0)*0.2))
            d.rounded_rectangle((X0, Y0, X1, Y1), radius=radius, fill=color)

    combined = Image.alpha_composite(img.convert("RGBA"), overlay)
    return combined

# ===== Sidebar controls =====
st.sidebar.header("Simulation Controls")
rounds = st.sidebar.slider("Rounds", 1, 12, 5)
round_sec = st.sidebar.slider("Round length (sec)", 60, 240, 180, 10)
rest_sec  = st.sidebar.slider("Rest length (sec)", 0, 120, 60, 5)
hz        = st.sidebar.slider("Sampling rate (Hz)", 1, 20, 5)

st.sidebar.subheader("Recovery rates")
base_rec  = st.sidebar.slider("Active recovery (/s)", 0.00, 0.20, 0.06, 0.005)
rest_rec  = st.sidebar.slider("Rest recovery (/s)",   0.00, 0.30, 0.12, 0.005)

st.sidebar.subheader("Impact multipliers")
head_imp  = st.sidebar.slider("Head impact",  0.1, 2.0, 1.2, 0.05)
torso_imp = st.sidebar.slider("Torso impact", 0.1, 2.0, 0.9, 0.05)
arm_imp   = st.sidebar.slider("Arm impact",   0.1, 2.0, 0.6, 0.05)
leg_imp   = st.sidebar.slider("Leg impact",   0.1, 2.0, 0.5, 0.05)

st.sidebar.subheader("Zone weights (overall stamina)")
w_head = st.sidebar.slider("Head",      0.00, 0.50, 0.20, 0.01)
w_larm = st.sidebar.slider("Left arm",  0.00, 0.50, 0.15, 0.01)
w_rarm = st.sidebar.slider("Right arm", 0.00, 0.50, 0.20, 0.01)
w_torso= st.sidebar.slider("Torso",     0.00, 0.50, 0.25, 0.01)
w_legs = st.sidebar.slider("Legs",      0.00, 0.50, 0.20, 0.01)
weights_sum = w_head + w_larm + w_rarm + w_torso + w_legs
st.sidebar.caption(f"Weights sum: **{weights_sum:.2f}** (target ~1.00)")

# ===== Run / cache sim =====
@st.cache_data(show_spinner=False)
def run_sim_cached(params):
    base_recovery = {z: params['base_rec'] for z in ZONES}
    rest_recovery = {z: params['rest_rec'] for z in ZONES}
    impact_costs = {
        "head": params['head_imp'],
        "torso": params['torso_imp'],
        "left_arm": params['arm_imp'],
        "right_arm": params['arm_imp'],
        "legs": params['leg_imp']
    }
    weights = {
        "head": params['w_head'], "left_arm": params['w_larm'],
        "right_arm": params['w_rarm'], "torso": params['w_torso'], "legs": params['w_legs']
    }
    df = simulate(
        rounds=params['rounds'], round_sec=params['round_sec'], rest_sec=params['rest_sec'], hz=params['hz'],
        action_costs=DEFAULT_ACTION_COSTS, impact_costs=impact_costs,
        base_recovery=base_recovery, rest_recovery=rest_recovery,
        zone_weights=weights
    )
    return df

params = dict(
    rounds=rounds, round_sec=round_sec, rest_sec=rest_sec, hz=hz,
    base_rec=base_rec, rest_rec=rest_rec,
    head_imp=head_imp, torso_imp=torso_imp, arm_imp=arm_imp, leg_imp=leg_imp,
    w_head=w_head, w_larm=w_larm, w_rarm=w_rarm, w_torso=w_torso, w_legs=w_legs
)
df = run_sim_cached(params)
st.success("Simulation generated")

# ===== Playback state =====
if "t" not in st.session_state:
    st.session_state.t = float(df["t"].min())

# ===== Header row: KPIs + playback =====
k1, k2, k3 = st.columns([1.2, 1.2, 2])
with k1:
    st.markdown('<div class="smallcaps">Red — Stamina</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{df.iloc[-1]["Red_stamina"]:.1f}</div>', unsafe_allow_html=True)
    st.progress(int(df.iloc[-1]["Red_stamina"]))
with k2:
    st.markdown('<div class="smallcaps">Blue — Stamina</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{df.iloc[-1]["Blue_stamina"]:.1f}</div>', unsafe_allow_html=True)
    st.progress(int(df.iloc[-1]["Blue_stamina"]))
with k3:
    st.markdown("**Playback**")
    st.session_state.t = st.slider("Scrub time (s)", float(df["t"].min()), float(df["t"].max()), st.session_state.t, 0.2)

# ===== Helper to grab closest frame =====
def get_frame_at(t):
    idx = (df["t"] - t).abs().argmin()
    return df.iloc[int(idx)]

frame = get_frame_at(st.session_state.t)
red_state  = {z: float(frame[f"Red_{z}_fatigue"]) for z in ZONES}
blue_state = {z: float(frame[f"Blue_{z}_fatigue"]) for z in ZONES}

# ===== Middle: Body images with overlays + quick zone readouts =====
left, right = st.columns(2)

def render_fighter(panel, title, image_path, state_dict, badge_color):
    panel.markdown(f"### {title}")
    try:
        base = Image.open(image_path).convert("RGBA")
    except Exception:
        # Fallback: generate one if missing
        ensure_placeholder(image_path)
        base = Image.open(image_path).convert("RGBA")

    overlayed = draw_heat_overlay(base, state_dict, alpha=120)
    panel.image(overlayed, use_column_width=True)

    # tiny zone readout
    zc1, zc2, zc3 = panel.columns(3)
    zc1.metric("Head",      f"{state_dict['head']:.1f}")
    zc2.metric("Torso",     f"{state_dict['torso']:.1f}")
    zc3.metric("Legs",      f"{state_dict['legs']:.1f}")
    zc4, zc5 = panel.columns(2)
    zc4.metric("Left arm",  f"{state_dict['left_arm']:.1f}")
    zc5.metric("Right arm", f"{state_dict['right_arm']:.1f}")

with left:
    render_fighter(left, "🟥 Red — Body Fatigue Map", ASSETS_RED, red_state, "#e11d48")
with right:
    render_fighter(right, "🟦 Blue — Body Fatigue Map", ASSETS_BLUE, blue_state, "#2563eb")

# ===== Bottom: Trends + Recent events =====
st.divider()
st.subheader("Trends & Event Feed")

c1, c2 = st.columns([2, 1])

with c1:
    d2 = df.copy()
    d2["t_sec"] = d2["t"].round().astype(int)
    grp = d2.groupby("t_sec").agg({"Red_stamina":"mean","Blue_stamina":"mean"}).reset_index()

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(grp["t_sec"], grp["Red_stamina"], label="Red")
    ax.plot(grp["t_sec"], grp["Blue_stamina"], label="Blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Overall Stamina")
    ax.set_title("Stamina Over Time")
    ax.grid(True, alpha=.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with c2:
    window = (df["t"] <= st.session_state.t) & (df["t"] >= st.session_state.t - 10)
    recent = df[window & (df["event"].notna()) & (df["event"] != "rest")]
    counts = (recent.groupby(["actor","event"])
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False))
    st.write("**Recent 10s — Event Counts**")
    st.dataframe(counts, use_container_width=True, height=260)

# ===== Data & Exports =====
with st.expander("Data & Exports", expanded=False):
    st.dataframe(df.tail(200), use_container_width=True, height=300)
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download timeline CSV", csv_bytes, "sim_fight_timeline.csv", "text/csv")

    preset = {
        "rounds": rounds, "round_length_sec": round_sec, "rest_length_sec": rest_sec, "hz": hz,
        "impact_costs": {"head": head_imp, "torso": torso_imp, "arm": arm_imp, "leg": leg_imp},
        "recovery": {"active_per_sec": base_rec, "rest_per_sec": rest_rec},
        "zone_weights": {"head": w_head, "left_arm": w_larm, "right_arm": w_rarm, "torso": w_torso, "legs": w_legs}
    }
    st.download_button("Download preset JSON", json.dumps(preset, indent=2).encode(),
                       "hud_preset.json", "application/json")

st.caption("Tip: Replace assets/red_silhouette.png and assets/blue_silhouette.png with your brand art (transparent PNGs). The heat overlay adapts automatically.")

