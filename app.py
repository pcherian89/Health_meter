# app.py — Stakeholder-stable HUD with two-tank fatigue (acute + residual) + demo boosts
import json
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt


st.set_page_config(page_title="Fight Health & Fatigue HUD (Prototype)", layout="wide")
st.title("Fight Health & Fatigue HUD (Prototype)")

# ------------------ constants ------------------
ZONES = ["head", "left_arm", "right_arm", "torso", "legs"]

# Effort costs on the actor (shoulders/legs/etc)
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

# Impact multipliers on the recipient
DEFAULT_IMPACT_COSTS = {"head": 1.2, "torso": 0.9, "left_arm": 0.6, "right_arm": 0.6, "legs": 0.5}

# Baseline action rates (per second) before scaling by fatigue/round/demo boost
DEFAULT_BASE_RATES = {
    "jab": 0.25, "cross": 0.10, "hook": 0.08, "uppercut": 0.05,
    "advance": 0.40, "retreat": 0.25, "circle": 0.30, "clinch": 0.03
}

# Land probabilities & target distribution
P_LAND = {"jab": 0.35, "cross": 0.30, "hook": 0.28, "uppercut": 0.25}
P_TARGET = {"head": 0.6, "torso": 0.3, "left_arm": 0.05, "right_arm": 0.03, "legs": 0.02}

# Overall stamina weights
ZONE_WEIGHTS = {"head": 0.20, "left_arm": 0.15, "right_arm": 0.20, "torso": 0.25, "legs": 0.20}

# ------------------ two-tank model (logical persistence) ------------------
# Load is split into: Acute (fast) + Residual (slow). We decay with half-lives.
ALPHA_IMPACT = 0.7   # fraction into Acute; (1-ALPHA_IMPACT) into Residual

# Half-lives (seconds) — tune these to taste
HL_ACUTE_ACTIVE = 45.0
HL_ACUTE_REST   = 20.0
HL_RESID_ACTIVE = 240.0
HL_RESID_REST   = 120.0

def half_life_decay(value, dt, half_life):
    if half_life <= 0:
        return max(0.0, value)
    return value * np.exp(-dt * np.log(2.0) / half_life)

def stamina_from_fatigue(fatigue_dict):
    weighted = sum(ZONE_WEIGHTS[z] * fatigue_dict[z] for z in ZONES)
    return max(0.0, 100.0 - weighted)

def round_intensity_factor(r: int) -> float:
    # Slightly decreasing intensity across rounds (clamped)
    return max(0.70, 1.10 - 0.03 * (r - 1))

# ------------------ simulator ------------------
def simulate(
    rounds=5, round_sec=180, rest_sec=60, hz=5,
    action_costs=None, impact_costs=None, base_rates=None,
    seed=42, rate_boost=1.0, impact_boost=1.0
):
    np.random.seed(seed)
    action_costs = action_costs or DEFAULT_ACTION_COSTS
    impact_costs = impact_costs or DEFAULT_IMPACT_COSTS
    base_rates   = base_rates or DEFAULT_BASE_RATES

    dt = 1.0 / hz
    fighters = ["Red", "Blue"]

    # two-tank state
    acute = {f: {z: 0.0 for z in ZONES} for f in fighters}
    resid = {f: {z: 0.0 for z in ZONES} for f in fighters}

    def zval(f, z):  # displayed zone value
        return min(100.0, acute[f][z] + resid[f][z])

    rows = []
    t = 0.0

    # ---- Pre-frame at t=0 so sliders show exact 100.0 at start ----
    rows.append({
        "t": 0.0, "round": 1, "phase": "pre", "actor": None, "opponent": None,
        "event": None, "landed": None, "target": None, "force": None,
        **{f"Red_{zz}_fatigue": zval('Red', zz) for zz in ZONES},
        **{f"Blue_{zz}_fatigue": zval('Blue', zz) for zz in ZONES},
        "Red_stamina": stamina_from_fatigue({z: zval('Red', z) for z in ZONES}),
        "Blue_stamina": stamina_from_fatigue({z: zval('Blue', z) for z in ZONES}),
    })

    targets, p_targets = list(P_TARGET.keys()), list(P_TARGET.values())

    for r in range(1, rounds + 1):
        intensity = round_intensity_factor(r)

        # -------- Active phase --------
        for _ in range(int(round_sec * hz)):
            t += dt
            for f in fighters:
                opp = "Blue" if f == "Red" else "Red"

                # fatigue → lower work rate
                ov = stamina_from_fatigue({z: zval(f, z) for z in ZONES})
                rate_scale = max(0.25, 1 - 0.5 * ((100 - ov) / 100)) * intensity

                # stochastic actions
                events = []
                for ev, lam in base_rates.items():
                    if np.random.rand() < (lam * rate_scale * rate_boost) * dt:
                        events.append(ev)

                for ev in events:
                    # Actor effort cost (slightly coupled to activity to make them tire)
                    info = action_costs[ev]
                    z = info["zone"]
                    cost = info["cost"] * (0.85 + 0.15 * rate_boost)
                    acute[f][z] = min(100.0, acute[f][z] + ALPHA_IMPACT * cost)
                    resid[f][z] = min(100.0, resid[f][z] + (1 - ALPHA_IMPACT) * cost)

                    landed, target, force = None, None, None
                    # If it's a punch and it lands, recipient gets impact on the hit zone
                    if ev in ["jab", "cross", "hook", "uppercut"]:
                        if np.random.rand() < P_LAND[ev]:
                            target = np.random.choice(targets, p=p_targets)
                            force = float(np.clip(np.random.normal(0.6, 0.2), 0.05, 1.0))
                            impact = (impact_costs[target] * impact_boost) * force
                            acute[opp][target] = min(100.0, acute[opp][target] + ALPHA_IMPACT * impact)
                            resid[opp][target] = min(100.0, resid[opp][target] + (1 - ALPHA_IMPACT) * impact)
                            landed = 1
                        else:
                            landed = 0

                    rows.append({
                        "t": t, "round": r, "phase": "active", "actor": f, "opponent": opp,
                        "event": ev, "landed": landed, "target": target, "force": force,
                        **{f"Red_{zz}_fatigue": zval('Red', zz) for zz in ZONES},
                        **{f"Blue_{zz}_fatigue": zval('Blue', zz) for zz in ZONES},
                        "Red_stamina": stamina_from_fatigue({z: zval('Red', z) for z in ZONES}),
                        "Blue_stamina": stamina_from_fatigue({z: zval('Blue', z) for z in ZONES}),
                    })

                # continuous decay (half-life) during active
                for z in ZONES:
                    acute[f][z] = half_life_decay(acute[f][z], dt, HL_ACUTE_ACTIVE)
                    resid[f][z] = half_life_decay(resid[f][z], dt, HL_RESID_ACTIVE)

        # -------- Rest phase (between rounds) --------
        if r < rounds:
            for i in range(int(rest_sec * hz)):
                t += dt
                for f in fighters:
                    for z in ZONES:
                        acute[f][z] = half_life_decay(acute[f][z], dt, HL_ACUTE_REST)
                        resid[f][z] = half_life_decay(resid[f][z], dt, HL_RESID_REST)
                if i % hz == 0:  # sparse logging
                    rows.append({
                        "t": t, "round": r, "phase": "rest", "actor": None, "opponent": None,
                        "event": "rest", "landed": None, "target": None, "force": None,
                        **{f"Red_{zz}_fatigue": zval('Red', zz) for zz in ZONES},
                        **{f"Blue_{zz}_fatigue": zval('Blue', zz) for zz in ZONES},
                        "Red_stamina": stamina_from_fatigue({z: zval('Red', z) for z in ZONES}),
                        "Blue_stamina": stamina_from_fatigue({z: zval('Blue', z) for z in ZONES}),
                    })

    return pd.DataFrame(rows)

# ------------------ visuals ------------------
def color_from_fatigue(v: float, intensity: float = 1.0):
    # 0→green, 100→red; intensity boosts perceived level (display only)
    v = float(np.clip(v * intensity, 0, 100)) / 100.0
    r = min(1.0, 2.0 * v)
    g = min(1.0, 2.0 * (1.0 - v))
    return (r, g, 0.0)

def body_heat_panel(ax, title: str, fatigue_dict: dict, intensity: float):
    ax.set_title(title, fontsize=12, pad=8)
    zones = ["head", "torso", "left_arm", "right_arm", "legs"]
    vals = [float(fatigue_dict[z]) for z in zones]
    colors = [color_from_fatigue(v, intensity) for v in vals]
    y = np.arange(len(zones))
    ax.barh(y, vals, height=0.55, color=colors)
    ax.set_yticks(y, zones)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    for i, v in enumerate(vals):
        ax.text(min(98, v + 1), i, f"{v:.1f}", va="center", fontsize=10)

def current_frame(df: pd.DataFrame, t: float) -> pd.Series:
    return df.iloc[int((df["t"] - t).abs().argmin())]

# ------------------ sidebar ------------------
st.sidebar.header("Simulation Controls")
seed      = st.sidebar.number_input("Random seed (same settings + same seed = same fight)", value=7, step=1)
rounds    = st.sidebar.slider("Rounds", 1, 20, 5)
round_sec = st.sidebar.slider("Round length (sec)", 60, 240, 180, 10)
rest_sec  = st.sidebar.slider("Rest length (sec)", 0, 120, 60, 5)
hz        = st.sidebar.slider("Sampling rate (Hz)", 1, 20, 5)

st.sidebar.subheader("Impact multipliers")
head_imp  = st.sidebar.slider("Head impact",  0.1, 2.5, 1.2, 0.05)
torso_imp = st.sidebar.slider("Torso impact", 0.1, 2.5, 0.9, 0.05)
arm_imp   = st.sidebar.slider("Arm impact",   0.1, 2.5, 0.6, 0.05)
leg_imp   = st.sidebar.slider("Leg impact",   0.1, 2.5, 0.5, 0.05)

st.sidebar.subheader("Display")
intensity = st.sidebar.slider("Heat intensity (visual)", 0.5, 3.0, 1.6, 0.1)

st.sidebar.subheader("Demo boost")
rate_boost   = st.sidebar.slider("Action frequency ×", 0.5, 3.0, 1.5, 0.1)
impact_boost = st.sidebar.slider("Impact strength ×",  0.5, 3.0, 1.3, 0.1)

st.sidebar.subheader("Use CSV instead of simulator (optional)")
uploaded = st.sidebar.file_uploader("Upload timeline CSV", type=["csv"])

# ------------------ data source ------------------
@st.cache_data(show_spinner=False)
def _simulate_cached(_seed, _rounds, _round_sec, _rest_sec, _hz,
                     _head, _torso, _arm, _leg, _rate_boost, _impact_boost):
    impact = {"head": _head, "torso": _torso, "left_arm": _arm, "right_arm": _arm, "legs": _leg}
    return simulate(_rounds, _round_sec, _rest_sec, _hz,
                    DEFAULT_ACTION_COSTS, impact, DEFAULT_BASE_RATES,
                    seed=_seed, rate_boost=_rate_boost, impact_boost=_impact_boost)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = _simulate_cached(seed, rounds, round_sec, rest_sec, hz,
                          head_imp, torso_imp, arm_imp, leg_imp,
                          rate_boost, impact_boost)

# ------------------ playback (current-frame logic) ------------------
if "t" not in st.session_state:
    st.session_state.t = 0.0

st.session_state.t = st.slider(
    "Scrub time (s)",
    0.0,                                # start at exact zero
    float(df["t"].max()),
    st.session_state.t, 0.2
)

frame = current_frame(df, st.session_state.t)

# ------------------ header KPIs from CURRENT frame ------------------
k1, k2 = st.columns(2)
with k1:
    st.subheader("Red — Stamina")
    st.metric(label="", value=f"{frame['Red_stamina']:.1f}")
    st.progress(int(frame["Red_stamina"]))
with k2:
    st.subheader("Blue — Stamina")
    st.metric(label="", value=f"{frame['Blue_stamina']:.1f}")
    st.progress(int(frame["Blue_stamina"]))

# ------------------ body-part heat panels ------------------
colL, colR = st.columns(2)
red_state  = {z: float(frame[f"Red_{z}_fatigue"]) for z in ZONES}
blue_state = {z: float(frame[f"Blue_{z}_fatigue"]) for z in ZONES}

with colL:
    st.subheader("Red — Body Heat")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    body_heat_panel(ax, "", red_state, intensity)
    st.pyplot(fig, clear_figure=True)

with colR:
    st.subheader("Blue — Body Heat")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    body_heat_panel(ax, "", blue_state, intensity)
    st.pyplot(fig, clear_figure=True)

# ------------------ trend chart ------------------
st.markdown("---")
st.subheader("Stamina Over Time")
d2 = df.copy(); d2["t_sec"] = d2["t"].round().astype(int)
grp = d2.groupby("t_sec").agg({"Red_stamina":"mean","Blue_stamina":"mean"}).reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(grp["t_sec"], grp["Red_stamina"], label="Red")
ax.plot(grp["t_sec"], grp["Blue_stamina"], label="Blue")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Overall Stamina"); ax.grid(True, alpha=.25); ax.legend()
st.pyplot(fig, clear_figure=True)

# ------------------ export ------------------
with st.expander("Data & Export"):
    st.dataframe(df.tail(200), use_container_width=True, height=320)
    st.download_button("Download timeline CSV", df.to_csv(index=False).encode(),
                       "sim_fight_timeline.csv", "text/csv")
    preset = {
        "seed": seed,
        "rounds": rounds, "round_length_sec": round_sec, "rest_length_sec": rest_sec, "hz": hz,
        "impact_costs": {"head": head_imp, "torso": torso_imp, "arm": arm_imp, "leg": leg_imp},
        "demo_boost": {"rate_x": rate_boost, "impact_x": impact_boost},
        "model": {
            "two_tank": True, "alpha_impact": ALPHA_IMPACT,
            "half_life_sec": {
                "acute_active": HL_ACUTE_ACTIVE, "acute_rest": HL_ACUTE_REST,
                "resid_active": HL_RESID_ACTIVE, "resid_rest": HL_RESID_REST
            }
        }
    }
    st.download_button("Download preset JSON", json.dumps(preset, indent=2).encode(),
                       "hud_preset.json", "application/json")

