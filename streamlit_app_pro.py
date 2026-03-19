
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CFL Play Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

TEAM_COLORS = {
    "BC": {"bg": "#F97316", "fg": "#FFFFFF"},
    "CGY": {"bg": "#D32F2F", "fg": "#FFFFFF"},
    "EDM": {"bg": "#F4C542", "fg": "#111111"},
    "HAM": {"bg": "#111111", "fg": "#F4C542"},
    "MTL": {"bg": "#0B1F4D", "fg": "#FFFFFF"},
    "OTT": {"bg": "#C62828", "fg": "#FFFFFF"},
    "SSK": {"bg": "#2E7D32", "fg": "#FFFFFF"},
    "TOR": {"bg": "#1976D2", "fg": "#FFFFFF"},
    "WPG": {"bg": "#003F87", "fg": "#FFFFFF"},
}

PRESETS = {
    "WPG trailing early, own 30, 1st & 10": {
        "team": "WPG", "quarter": 1, "minutes": 4, "seconds": 0,
        "down": 1, "ytg": 10.0, "field_side": "Own", "ball_on": 30.0, "score_diff": -10
    },
    "WPG 2nd & medium in own territory while leading": {
        "team": "WPG", "quarter": 3, "minutes": 9, "seconds": 15,
        "down": 2, "ytg": 6.0, "field_side": "Own", "ball_on": 42.0, "score_diff": 7
    },
    "MTL red-zone 1st down while trailing": {
        "team": "MTL", "quarter": 4, "minutes": 6, "seconds": 40,
        "down": 1, "ytg": 8.0, "field_side": "Opp", "ball_on": 14.0, "score_diff": -6
    },
    "BC late-half midfield 2nd & long": {
        "team": "BC", "quarter": 2, "minutes": 1, "seconds": 42,
        "down": 2, "ytg": 11.0, "field_side": "Own", "ball_on": 54.0, "score_diff": 3
    },
    "TOR short-yardage in opponent territory": {
        "team": "TOR", "quarter": 2, "minutes": 7, "seconds": 5,
        "down": 2, "ytg": 2.0, "field_side": "Opp", "ball_on": 34.0, "score_diff": 0
    },
}

@st.cache_resource
def load_model():
    with open(BASE / "cfl_play_predictor_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_team_lookup():
    return pd.read_csv(BASE / "cfl_team_bucket_lookup.csv")

@st.cache_data
def load_comparables():
    return pd.read_csv(BASE / "cfl_historical_comparables.csv")

@st.cache_data
def load_metrics():
    with open(BASE / "cfl_model_metrics.json", "r") as f:
        return json.load(f)

MODEL = load_model()
TEAM_LOOKUP = load_team_lookup()
COMPS = load_comparables()
METRICS = load_metrics()

TEAMS = sorted(TEAM_LOOKUP["possession_team"].dropna().astype(str).unique().tolist())

def distance_bucket(x):
    if x <= 3:
        return "Short (1-3)"
    if x <= 7:
        return "Medium (4-7)"
    return "Long (8+)"

def field_bucket(x):
    if x <= 20:
        return "Red Zone"
    if x <= 40:
        return "Opponent Territory"
    if x <= 60:
        return "Midfield"
    if x <= 80:
        return "Own Territory"
    return "Own Deep"

def score_bucket(x):
    if x <= -8:
        return "Trailing 8+"
    if x <= -1:
        return "Trailing 1-7"
    if x == 0:
        return "Tied"
    if x <= 7:
        return "Leading 1-7"
    return "Leading 8+"

def time_bucket(sec_half):
    if sec_half <= 120:
        return "2-min"
    if sec_half <= 600:
        return "Late Half"
    return "Early Half"

def half_seconds_from_quarter_clock(quarter, mins, secs):
    rem = int(mins) * 60 + int(secs)
    if int(quarter) in (1, 3):
        return rem + 900
    return rem

def yards_to_endzone_from_ball_on(side, yardline):
    yard = float(yardline)
    return 110 - yard if side == "Own" else yard

def get_bucket_lookup(team, down, ytg, yte, score_diff, sec_half):
    tb = distance_bucket(ytg)
    fb = field_bucket(yte)
    sb = score_bucket(score_diff)
    tmb = time_bucket(sec_half)
    subset = TEAM_LOOKUP[
        (TEAM_LOOKUP["possession_team"] == team) &
        (TEAM_LOOKUP["down"] == down) &
        (TEAM_LOOKUP["distance_bucket"] == tb) &
        (TEAM_LOOKUP["field_bucket"] == fb) &
        (TEAM_LOOKUP["score_bucket"] == sb) &
        (TEAM_LOOKUP["time_bucket"] == tmb)
    ].copy()
    if subset.empty:
        return None
    return subset.sort_values("plays", ascending=False).iloc[0].to_dict()

def get_comparables(team, quarter, down, ytg, yte, sec_half, score_diff, top_n=12):
    d = COMPS.copy()
    d = d[(d["possession_team"] == team) & (d["down"] == down)]
    if d.empty:
        return d
    d["distance_score"] = (
        ((d["quarter"] - quarter) / 1.0) ** 2 +
        ((d["yards_to_go"] - ytg) / 4.0) ** 2 +
        ((d["yards_to_endzone"] - yte) / 15.0) ** 2 +
        ((d["seconds_in_half_remaining"] - sec_half) / 240.0) ** 2 +
        ((d["score_diff_offense"] - score_diff) / 7.0) ** 2
    )
    d = d.sort_values(["distance_score", "cfl_game_id", "play_id"]).head(top_n).copy()
    d["called_play"] = np.where(d["called_pass"] == 1, "Pass", "Run")
    return d[[
        "cfl_game_id", "play_id", "called_play", "play_result", "description",
        "quarter", "yards_to_go", "yards_to_endzone", "seconds_in_half_remaining",
        "score_diff_offense", "distance_score"
    ]]

def set_preset(preset_name):
    p = PRESETS[preset_name]
    for k, v in p.items():
        st.session_state[k] = v

for k, v in {
    "team": "WPG", "quarter": 1, "minutes": 4, "seconds": 0, "down": 1,
    "ytg": 10.0, "field_side": "Own", "ball_on": 30.0, "score_diff": -10
}.items():
    st.session_state.setdefault(k, v)

team = st.session_state["team"]
colors = TEAM_COLORS.get(team, {"bg": "#1f2937", "fg": "#ffffff"})

st.markdown(f"""
<style>
.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
}}
.main-banner {{
    background: linear-gradient(135deg, {colors['bg']} 0%, #0f172a 100%);
    color: {colors['fg']};
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}}
.small-pill {{
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    margin-right: 0.4rem;
    font-size: 0.9rem;
}}
div[data-testid="stMetric"] {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 10px 14px;
    border-radius: 16px;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-banner">
    <div style="display:flex; justify-content:space-between; gap:1rem; align-items:center; flex-wrap:wrap;">
        <div>
            <div style="font-size:1.7rem; font-weight:700;">CFL Play Predictor</div>
            <div style="opacity:0.92; margin-top:0.25rem;">A situational model that quantifies play-calling tendencies using historical CFL data.</div>
        </div>
        <div>
            <span class="small-pill">Model: {METRICS['model_type']}</span>
            <span class="small-pill">2024 Data</span>
            <span class="small-pill">V.5.20260310</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Game Inputs")
    preset = st.selectbox("Quick preset", ["Custom"] + list(PRESETS.keys()))
    if preset != "Custom":
        if st.button("Load preset", use_container_width=True):
            set_preset(preset)

    st.selectbox("Offense team", TEAMS, key="team")
    st.selectbox("Quarter", [1, 2, 3, 4], key="quarter")
    st.number_input("Minutes left in quarter", min_value=0, max_value=15, step=1, key="minutes")
    st.number_input("Seconds left in quarter", min_value=0, max_value=59, step=1, key="seconds")
    st.selectbox("Down", [1, 2, 3], key="down")
    st.number_input("Yards to go", min_value=0.0, max_value=50.0, step=1.0, key="ytg")
    st.selectbox("Ball side", ["Own", "Opp"], key="field_side")
    st.number_input("Ball on", min_value=1.0, max_value=55.0, step=1.0, key="ball_on")
    st.number_input("Score diff (offense perspective)", min_value=-60, max_value=60, step=1, key="score_diff")
    st.caption("Negative = offense trailing")

team = st.session_state["team"]
quarter = st.session_state["quarter"]
minutes = st.session_state["minutes"]
seconds = st.session_state["seconds"]
down = st.session_state["down"]
ytg = float(st.session_state["ytg"])
field_side = st.session_state["field_side"]
ball_on = float(st.session_state["ball_on"])
score_diff = int(st.session_state["score_diff"])

yte = yards_to_endzone_from_ball_on(field_side, ball_on)
sec_half = half_seconds_from_quarter_clock(quarter, minutes, seconds)

x = pd.DataFrame([{
    "possession_team": team,
    "quarter": quarter,
    "down": down,
    "yards_to_go": ytg,
    "yards_to_endzone": yte,
    "seconds_in_half_remaining": sec_half,
    "score_diff_offense": score_diff
}])

pass_prob = float(MODEL.predict_proba(x)[0, 1])
run_prob = 1 - pass_prob

lookup = get_bucket_lookup(team, down, ytg, yte, score_diff, sec_half)
comparables = get_comparables(team, quarter, down, ytg, yte, sec_half, score_diff)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Pass probability", f"{pass_prob:.1%}")
m2.metric("Run probability", f"{run_prob:.1%}")
m3.metric("Model accuracy", f"{METRICS['accuracy_at_0_5_threshold']:.1%}")
m4.metric("Bucket sample", f"{int(lookup['plays']) if lookup else 0}")

left, right = st.columns([1.08, 0.92])

with left:
    st.subheader("Situation snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.info(f"**Distance**\n\n{distance_bucket(ytg)}")
    s2.info(f"**Field**\n\n{field_bucket(yte)}")
    s3.info(f"**Score**\n\n{score_bucket(score_diff)}")
    s4.info(f"**Time**\n\n{time_bucket(sec_half)}")

    st.subheader("Historical tendency vs league")
    if lookup:
        a, b, c = st.columns(3)
        a.metric("Team bucket pass rate", f"{lookup['pass_prob_hist']:.1%}")
        b.metric("League bucket pass rate", f"{lookup['league_pass_prob_hist']:.1%}")
        c.metric("Delta vs league", f"{lookup['pass_prob_delta_vs_league']:+.1%}")
        st.caption(f"Sample size: {int(lookup['plays'])} team plays in this bucket, {int(lookup['league_plays'])} league plays.")
    else:
        st.warning("No exact historical bucket match found. Model output still works; treat the historical layer with caution.")

    st.subheader("Defensive read")
    if pass_prob >= 0.70:
        st.success("Strong pass tendency. Pressure package and quick-game alerts are justified.")
    elif pass_prob >= 0.58:
        st.info("Moderate pass lean. Guard against dropback while staying honest to draw/screen risk.")
    elif run_prob >= 0.70:
        st.success("Strong run tendency. Consider box count, edge fit, and motion discipline.")
    elif run_prob >= 0.58:
        st.info("Moderate run lean. Balanced front with run-first eyes.")
    else:
        st.info("Mixed tendency. Use comparables and bucket history to avoid overcommitting.")

    st.subheader("Coaching notes")
    notes = [
        f"{team} is in a {distance_bucket(ytg).lower()} distance spot.",
        f"Field state is {field_bucket(yte).lower()}, with score state {score_bucket(score_diff).lower()}.",
        "This tool is strongest as a between-snaps decision support layer, not a replacement for film or personnel tags."
    ]
    for n in notes:
        st.write(f"- {n}")

with right:
    st.subheader("Live scenario summary")
    st.markdown(f"""
**Team:** {team}  
**Game state:** Q{quarter}, {minutes}:{str(seconds).zfill(2)} left  
**Situation:** {down} & {ytg:g} at {field_side.lower()} {ball_on:g}  
**Score diff (offense):** {score_diff:+d}  
**Derived yards_to_endzone:** {yte:.1f}  
**Derived seconds_in_half_remaining:** {sec_half}
""")

    st.subheader("Closest historical comparables")
    if comparables.empty:
        st.warning("No comparables found for this team/down combination.")
    else:
        view = comparables.copy()
        view["distance_score"] = view["distance_score"].round(3)
        st.dataframe(view, use_container_width=True, height=460)

st.divider()

with st.expander("Model details"):
    st.markdown(f"""
- **Target:** run vs pass  
- **Model:** {METRICS['model_type']}  
- **Features:** {", ".join(METRICS['features_used'])}  
- **Rows modeled:** {METRICS['overall_rows_modeled']:,}  
- **Test accuracy:** {METRICS['accuracy_at_0_5_threshold']:.1%}  
- **Log loss:** {METRICS['log_loss']:.4f}  
- **Target definition:** {METRICS['target_definition']}
""")
