# app.py — minimal, reliable HUD
import time
import json
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

st.set_page_config(page_title="Fight Health & Fatigue HUD", layout="wide")
st.title("Fight Health & Fatigue HUD (Prototype)")

# ----------------- constants -----------------
ZONES = ["head","left_arm","right_arm","torso","legs"]
DEFAULT_ACTION_COSTS = {
    "jab": {"zone":"right_arm","cost":0.20},
    "cross": {"zone":"right_arm","cost":0.35},
    "hook": {"zone":"left_arm","cost":0.40},
    "uppercut":{"zone":"right_arm","cost":0.45},
    "advance":{"zone":"legs","cost":0.04},
    "retreat":{"zone":"legs","cost":0.03},
    "circle":{"zone":"legs","cost":0.035},
    "clinch":{"zone":"torso","cost":0.25},
}
DEFAULT_IMPACT_COSTS = {"head":1.2,"torso":0.9,"left_arm":0.6,"right_arm":0.6,"legs":0.5}
DEFAULT_BASE_RECOVERY = {z:0.06 for z in ZONES}
DEFAULT_REST_RECOVERY = {z:0.12 for z in ZONES}
DEFAULT_ROUND_INTENSITY = [1.10,1.00,0.95,0.90,0.85]
DEFAULT_BASE_RATES = {"jab":0.25,"cross":0.10,"hook":0.08,"uppercut":0.05,"advance":0.40,"retreat":0.25,"circle":0.30,"clinch":0.03}
P_LAND = {"jab":0.35,"cross":0.30,"hook":0.28,"uppercut":0.25}
P_TARGET = {"head":0.6,"torso":0.3,"left_arm":0.05,"right_arm":0.03,"legs":0.02}
ZONE_WEIGHTS = {"head":0.20,"left_arm":0.15,"right_arm":0.20,"torso":0.25,"legs":0.20}

# ----------------- helpers -----------------
def draw_placeholder(w=360, h=720):
    """neutral human-ish placeholder silhouette"""
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(img)
    base = (25,25,25)
    d.ellipse((150, 30, 210, 90), fill=base)                        # head
    d.rounded_rectangle((135, 90, 225, 420), 36, fill=base)         # torso
    d.rounded_rectangle((90,  150, 135, 380), 26, fill=base)        # left arm
    d.rounded_rectangle((225, 150, 270, 380), 26, fill=base)        # right arm
    d.polygon([(135,420),(225,420),(250,490),(110,490)], fill=base) # hips
    d.rounded_rectangle((125, 490, 165, 690), 22, fill=base)        # left leg
    d.rounded_rectangle((195, 490, 235, 690), 22, fill=base)        # right leg
    return img

def fatigue_to_rgb(v: float, amplify: float = 1.0):
    v = float(np.clip(v * amplify, 0.0, 100.0)) / 100.0
    r = min(1.0, 2.0*v)
    g = min(1.0, 2.0*(1.0 - v))
    return int(r*255), int(g*255), 0

def draw_heat(img: Image.Image, fatigue: dict, alpha=128, amplify=1.6) -> Image.Image:
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    boxes = {
        "head":      (0.42, 0.05, 0.58, 0.13),
        "torso":     (0.38, 0.13, 0.62, 0.58),
        "left_arm":  (0.25, 0.20, 0.37, 0.55),
        "right_arm": (0.63, 0.20, 0.75, 0.55),
        "legs":      (0.39, 0.60, 0.61, 0.95),
    }
    for z, (x0,y0,x1,y1) in boxes.items():
        X0, Y0, X1, Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
        color = fatigue_to_rgb(fatigue[z], amplify) + (alpha,)
        if z=="head":
            d.ellipse((X0,Y0,X1,Y1), fill=color)
        else:
            d.rounded_rectangle((X0,Y0,X1,Y1), radius=max(8,int((X1-X0)*0.2)), fill=color)
    return Image.alpha_composite(img.convert("RGBA"), overlay)

def stamina(fatigue_dict):
    wf = sum(ZONE_WEIGHTS[z]*fatigue_dict[z] for z in ZONES)
    return max(0.0, 100.0 - wf)

def simulate(rounds=5, round_sec=180, rest_sec=60, hz=5,
             action_costs=None, impact_costs=None,
             base_recovery=None, rest_recovery=None,
             base_rates=None, round_intensity=None, seed=42):
    np.random.seed(seed)
    action_costs   = action_costs   or DEFAULT_ACTION_COSTS
    impact_costs   = impact_costs   or DEFAULT_IMPACT_COSTS
    base_recovery  = base_recovery  or DEFAULT_BASE_RECOVERY
    rest_recovery  = rest_recovery  or DEFAULT_REST_RECOVERY
    base_rates     = base_rates     or DEFAULT_BASE_RATES
    round_intensity= round_intensity or DEFAULT_ROUND_INTENSITY

    dt = 1.0/hz
    fighters = ["Red","Blue"]
    fatigue = {f:{z:0.0 for z in ZONES} for f in fighters}
    rows = []; t = 0.0
    targets = list(P_TARGET.keys()); probs = list(P_TARGET.values())

    for r in range(1, rounds+1):
        # active
        for _ in range(int(round_sec*hz)):
            t += dt
            for f in fighters:
                opp = "Blue" if f=="Red" else "Red"
                ov = stamina(fatigue[f])
                rate_scale = max(0.25, 1 - 0.5*((100-ov)/100)) * round_intensity[r-1]

                events = []
                for ev, lam in base_rates.items():
                    if np.random.rand() < lam*rate_scale*dt:
                        events.append(ev)

                for ev in events:
                    info = action_costs[ev]
                    fatigue[f][info["zone"]] = min(100.0, fatigue[f][info["zone"]] + info["cost"])
                    landed, target, force = None, None, None
                    if ev in ["jab","cross","hook","uppercut"]:
                        if np.random.rand() < P_LAND[ev]:
                            target = np.random.choice(targets, p=probs)
                            force  = float(np.clip(np.random.normal(0.6,0.2), 0.05, 1.0))
                            fatigue[opp][target] = min(100.0, fatigue[opp][target] + impact_costs[target]*force)
                            landed = 1
                        else:
                            landed = 0
                    rows.append({
                        "t":t,"round":r,"phase":"active","actor":f,"opponent":opp,"event":ev,
                        "landed":landed,"target":target,"force":force,
                        **{f"Red_{z}_fatigue":fatigue["Red"][z] for z in ZONES},
                        **{f"Blue_{z}_fatigue":fatigue["Blue"][z] for z in ZONES},
                        "Red_stamina": stamina(fatigue["Red"]),
                        "Blue_stamina": stamina(fatigue["Blue"]),
                    })
                # recover during active
                for z in ZONES:
                    fatigue[f][z] = max(0.0, fatigue[f][z] - base_recovery[z]*dt)
        # rest
        if r < rounds:
            for i in range(int(rest_sec*hz)):
                t += dt
                for f in fighters:
                    for z in ZONES:
                        fatigue[f][z] = max(0.0, fatigue[f][z] - rest_recovery[z]*dt)
                if i % hz == 0:
                    rows.append({
                        "t":t,"round":r,"phase":"rest","actor":None,"opponent":None,"event":"rest",
                        "landed":None,"target":None,"force":None,
                        **{f"Red_{z}_fatigue":fatigue["Red"][z] for z in ZONES},
                        **{f"Blue_{z}_fatigue":fatigue["Blue"][z] for z in ZONES},
                        "Red_stamina": stamina(fatigue["Red"]),
                        "Blue_stamina": stamina(fatigue["Blue"]),
                    })
    return pd.DataFrame(rows)

# ----------------- sidebar -----------------
st.sidebar.header("Simulation Controls")
rounds    = st.sidebar.slider("Rounds", 1, 12, 5)
round_sec = st.sidebar.slider("Round length (sec)", 60, 240, 180, 10)
rest_sec  = st.sidebar.slider("Rest length (sec)", 0, 120, 60, 5)
hz        = st.sidebar.slider("Sampling rate (Hz)", 1, 20, 5)
st.sidebar.subheader("Recovery")
base_rec  = st.sidebar.slider("Active recovery (/s)", 0.00, 0.20, 0.06, 0.005)
rest_rec  = st.sidebar.slider("Rest recovery (/s)", 0.00, 0.30, 0.12, 0.005)
st.sidebar.subheader("Impact multipliers")
head_imp  = st.sidebar.slider("Head impact", 0.1, 2.0, 1.2, 0.05)
torso_imp = st.sidebar.slider("Torso impact", 0.1, 2.0, 0.9, 0.05)
arm_imp   = st.sidebar.slider("Arm impact", 0.1, 2.0, 0.6, 0.05)
leg_imp   = st.sidebar.slider("Leg impact", 0.1, 2.0, 0.5, 0.05)
st.sidebar.subheader("Display")
intensity = st.sidebar.slider("Body heat intensity", 0.5, 3.0, 1.8, 0.1)
img_width = st.sidebar.slider("Image width (px)", 260, 420, 300, 10)

# ----------------- data -----------------
@st.cache_data(show_spinner=False)
def _run(params):
    impact_costs = {"head":params["head"],"torso":params["torso"],"left_arm":params["arm"],"right_arm":params["arm"],"legs":params["leg"]}
    base_recov   = {z: params["base_rec"] for z in ZONES}
    rest_recov   = {z: params["rest_rec"] for z in ZONES}
    return simulate(params["rounds"], params["round_sec"], params["rest_sec"], params["hz"],
                    DEFAULT_ACTION_COSTS, impact_costs, base_recov, rest_recov)

params = dict(rounds=rounds, round_sec=round_sec, rest_sec=rest_sec, hz=hz,
              head=head_imp, torso=torso_imp, arm=arm_imp, leg=leg_imp,
              base_rec=base_rec, rest_rec=rest_rec)
df = _run(params)

# ----------------- playback (scrub only, no play to avoid jitter) -----------------
if "t" not in st.session_state:
    st.session_state.t = float(df["t"].min())

scrub = st.slider("Scrub time (s)", float(df["t"].min()), float(df["t"].max()), st.session_state.t, 0.2)
st.session_state.t = scrub

# ----------------- header metrics -----------------
m1, m2 = st.columns(2)
with m1:
    st.metric("Red — Stamina", f"{df.iloc[-1]['Red_stamina']:.1f}")
    st.progress(int(df.iloc[-1]['Red_stamina']))
with m2:
    st.metric("Blue — Stamina", f"{df.iloc[-1]['Blue_stamina']:.1f}")
    st.progress(int(df.iloc[-1]['Blue_stamina']))

# ----------------- frame + images -----------------
def frame_at(t): return df.iloc[int((df["t"]-t).abs().argmin())]
f = frame_at(st.session_state.t)
red  = {z: float(f[f"Red_{z}_fatigue"]) for z in ZONES}
blue = {z: float(f[f"Blue_{z}_fatigue"]) for z in ZONES}

colL, colR = st.columns(2)
base_img = draw_placeholder()

with colL:
    st.subheader("Red — Body Fatigue Map")
    st.image(draw_heat(base_img, red, amplify=intensity), width=img_width)
    c1,c2,c3 = st.columns(3); c1.metric("Head", f"{red['head']:.1f}"); c2.metric("Torso", f"{red['torso']:.1f}"); c3.metric("Legs", f"{red['legs']:.1f}")
    c4,c5 = st.columns(2); c4.metric("Left arm", f"{red['left_arm']:.1f}"); c5.metric("Right arm", f"{red['right_arm']:.1f}")

with colR:
    st.subheader("Blue — Body Fatigue Map")
    st.image(draw_heat(base_img, blue, amplify=intensity), width=img_width)
    c1,c2,c3 = st.columns(3); c1.metric("Head", f"{blue['head']:.1f}"); c2.metric("Torso", f"{blue['torso']:.1f}"); c3.metric("Legs", f"{blue['legs']:.1f}")
    c4,c5 = st.columns(2); c4.metric("Left arm", f"{blue['left_arm']:.1f}"); c5.metric("Right arm", f"{blue['right_arm']:.1f}")

# ----------------- analytics -----------------
st.markdown("---")
st.subheader("Stamina Over Time")
d2 = df.copy(); d2["t_sec"] = d2["t"].round().astype(int)
grp = d2.groupby("t_sec").agg({"Red_stamina":"mean","Blue_stamina":"mean"}).reset_index()
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(grp["t_sec"], grp["Red_stamina"], label="Red")
ax.plot(grp["t_sec"], grp["Blue_stamina"], label="Blue")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Overall Stamina"); ax.grid(True, alpha=.25); ax.legend()
st.pyplot(fig, clear_figure=True)

# ----------------- exports -----------------
with st.expander("Data & Export"):
    st.dataframe(df.tail(200), use_container_width=True, height=320)
    st.download_button("Download timeline CSV", df.to_csv(index=False).encode(), "sim_fight_timeline.csv", "text/csv")
    preset = {
        "rounds": rounds, "round_length_sec": round_sec, "rest_length_sec": rest_sec, "hz": hz,
        "impact_costs": {"head": head_imp, "torso": torso_imp, "arm": arm_imp, "leg": leg_imp},
        "recovery": {"active_per_sec": base_rec, "rest_per_sec": rest_rec},
    }
    st.download_button("Download preset JSON", json.dumps(preset, indent=2).encode(), "hud_preset.json", "application/json")
