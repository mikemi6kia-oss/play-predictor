import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CFL Play Frequency Outliers",
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

@st.cache_data
def load_scouting_report():
    try:
        df = pd.read_excel("CFL_EXEC_SCOUTING_REPORT.xlsx")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading scouting report: {e}")
        return pd.DataFrame()

MODEL = load_model()
TEAM_LOOKUP = load_team_lookup()
COMPS = load_comparables()
METRICS = load_metrics()
SCOUTING = load_scouting_report()

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

def get_comparables(team, quarter, down, ytg, yte, sec_half, score_diff, top_n=10):
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

def compute_model_delta(team, quarter, minutes, seconds, down, ytg, field_side, ball_on, score_diff):
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

    model_pass_prob = float(MODEL.predict_proba(x)[0, 1])
    lookup = get_bucket_lookup(team, down, ytg, yte, score_diff, sec_half)

    if lookup is not None:
        league_pass_rate = float(lookup["league_pass_prob_hist"])
        team_plays = int(lookup["plays"])
        league_plays = int(lookup["league_plays"])
        model_delta_vs_league = model_pass_prob - league_pass_rate
    else:
        league_pass_rate = None
        team_plays = 0
        league_plays = 0
        model_delta_vs_league = None

    return {
        "model_pass_prob": model_pass_prob,
        "league_pass_rate": league_pass_rate,
        "model_delta_vs_league": model_delta_vs_league,
        "team_plays": team_plays,
        "league_plays": league_plays,
        "lookup": lookup,
        "yte": yte,
        "sec_half": sec_half,
    }

@st.cache_data(show_spinner=False)
def scan_high_confidence_tendencies():
    rows = []

    quarters = [1, 2, 3, 4]
    downs = [1, 2]
    minutes_list = [12, 9, 6, 3]
    seconds_list = [0]
    ytg_values = [2.0, 5.0, 10.0]
    field_sides = ["Own", "Opp"]
    ball_on_values = [15.0, 30.0, 45.0]
    score_diffs = [-10, -3, 0, 3, 10]

    for team in TEAMS:
        for q in quarters:
            for d in downs:
                for mins in minutes_list:
                    for secs in seconds_list:
                        for ytg in ytg_values:
                            for side in field_sides:
                                for ball in ball_on_values:
                                    for score in score_diffs:
                                        try:
                                            info = compute_model_delta(
                                                team=team,
                                                quarter=q,
                                                minutes=mins,
                                                seconds=secs,
                                                down=d,
                                                ytg=ytg,
                                                field_side=side,
                                                ball_on=ball,
                                                score_diff=score,
                                            )

                                            delta = info["model_delta_vs_league"]
                                            if delta is None:
                                                continue

                                            if (
                                                abs(delta) >= 0.20 and
                                                info["team_plays"] >= 10 and
                                                info["league_plays"] >= 20
                                            ):
                                                tendency = "Pass-heavy" if delta > 0 else "Run-heavy"
                                                rows.append({
                                                    "team": team,
                                                    "quarter": q,
                                                    "minutes": mins,
                                                    "seconds": secs,
                                                    "down": d,
                                                    "yards_to_go": ytg,
                                                    "field_side": side,
                                                    "ball_on": ball,
                                                    "score_diff": score,
                                                    "model_pass_prob": info["model_pass_prob"],
                                                    "league_pass_rate": info["league_pass_rate"],
                                                    "delta_vs_league": delta,
                                                    "tendency": tendency,
                                                    "team_plays": info["team_plays"],
                                                    "league_plays": info["league_plays"],
                                                })
                                        except Exception:
                                            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["abs_delta"] = df["delta_vs_league"].abs()
    df = df.sort_values(["team", "abs_delta"], ascending=[True, False]).reset_index(drop=True)
    df["rank_within_team"] = df.groupby("team").cumcount() + 1
    df["scenario_label"] = df.apply(
        lambda r: (
            f'{r["team"]} | {r["tendency"]} | Δ {r["delta_vs_league"]:+.1%} | '
            f'Q{int(r["quarter"])} {int(r["minutes"])}:{int(r["seconds"]):02d} | '
            f'{int(r["down"])} & {int(r["yards_to_go"])} | '
            f'{r["field_side"]} {int(r["ball_on"])} | '
            f'{int(r["score_diff"]):+d} | '
            f'N={int(r["team_plays"])}/{int(r["league_plays"])}'
        ),
        axis=1
    )
    return df

# Session defaults
for k, v in {
    "team": "WPG",
    "quarter": 1,
    "minutes": 4,
    "seconds": 0,
    "down": 1,
    "ytg": 10.0,
    "field_side": "Own",
    "ball_on": 30.0,
    "score_diff": -10
}.items():
    st.session_state.setdefault(k, v)

team = st.session_state["team"]
colors = TEAM_COLORS.get(team, {"bg": "#1f2937", "fg": "#ffffff"})

st.markdown(f"""
<style>
.block-container {{
    padding-top: 1.0rem;
    padding-bottom: 1.0rem;
}}
.main-banner {{
    background: linear-gradient(135deg, {colors['bg']} 0%, #0f172a 100%);
    color: {colors['fg']};
    padding: 1.1rem 1.3rem;
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
            <div style="font-size:1.7rem; font-weight:700;">CFL Play Frequency Outliers</div>
            <div style="opacity:0.92; margin-top:0.25rem;">
                A data-driven model used to anticipate offensive tendencies.
            </div>
        </div>
        <div>
            <span class="small-pill">Model: {METRICS['model_type']}</span>
            <span class="small-pill">2024 CFL Data</span>
            <span class="small-pill">v10.20250323</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Game Inputs")
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

# Current inputs
team = st.session_state["team"]
quarter = st.session_state["quarter"]
minutes = st.session_state["minutes"]
seconds = st.session_state["seconds"]
down = st.session_state["down"]
ytg = float(st.session_state["ytg"])
field_side = st.session_state["field_side"]
ball_on = float(st.session_state["ball_on"])
score_diff = int(st.session_state["score_diff"])

# Model + lookup
model_info = compute_model_delta(team, quarter, minutes, seconds, down, ytg, field_side, ball_on, score_diff)
lookup = model_info["lookup"]
comparables = get_comparables(
    team=team,
    quarter=quarter,
    down=down,
    ytg=ytg,
    yte=model_info["yte"],
    sec_half=model_info["sec_half"],
    score_diff=score_diff
)

pass_prob = model_info["model_pass_prob"]
run_prob = 1 - pass_prob

m1, m2, m3, m4 = st.columns(4)
m1.metric("Pass probability", f"{pass_prob:.1%}")
m2.metric("Run probability", f"{run_prob:.1%}")
m3.metric("Model accuracy", f"{METRICS['accuracy_at_0_5_threshold']:.1%}")
m4.metric("Bucket sample", f"{int(lookup['plays']) if lookup else 0}")

left, right = st.columns([1.02, 0.98])

with left:
    st.subheader("Situation snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.info(f"**Distance**\n\n{distance_bucket(ytg)}")
    s2.info(f"**Field**\n\n{field_bucket(model_info['yte'])}")
    s3.info(f"**Score**\n\n{score_bucket(score_diff)}")
    s4.info(f"**Time**\n\n{time_bucket(model_info['sec_half'])}")

    st.subheader("Model vs league")
    if lookup is not None and model_info["league_pass_rate"] is not None:
        a, b, c = st.columns(3)
        a.metric("Model pass probability", f"{model_info['model_pass_prob']:.1%}")
        b.metric("League bucket pass rate", f"{model_info['league_pass_rate']:.1%}")
        c.metric("Delta vs league", f"{model_info['model_delta_vs_league']:+.1%}")
        st.caption(
            f"Sample size: {int(lookup['plays'])} team plays in this bucket, "
            f"{int(lookup['league_plays'])} league plays."
        )
    else:
        st.warning("No exact historical bucket match found. Model output still works, but no league bucket baseline is available.")

    st.subheader("High tendency alerts")
    delta = model_info["model_delta_vs_league"]

    if delta is None:
        st.info("No league comparison alert available for this exact situation bucket.")
    elif delta >= 0.20:
        st.warning(f"Pass-heavy outlier alert: model is {delta:+.1%} vs league average in this scenario.")
    elif delta <= -0.20:
        st.warning(f"Run-heavy outlier alert: model is {delta:+.1%} vs league average in this scenario.")
    else:
        st.info("No major outlier alert. Current scenario is within ±20% of league average.")

with right:
    st.subheader("Live scenario summary")
    st.markdown(f"""
**Team:** {team}  
**Game state:** Q{quarter}, {minutes}:{str(seconds).zfill(2)} left  
**Situation:** {down} & {ytg:g} at {field_side.lower()} {ball_on:g}  
**Score diff (offense):** {score_diff:+d}  
**Derived yards_to_endzone:** {model_info['yte']:.1f}  
**Derived seconds_in_half_remaining:** {model_info['sec_half']}
""")

    st.subheader("Closest historical comparables")
    if comparables.empty:
        st.warning("No comparables found for this team/down combination.")
    else:
        view = comparables.copy()
        view["distance_score"] = view["distance_score"].round(3)
        st.dataframe(view, use_container_width=True, height=430)

st.divider()

tab1, tab2 = st.tabs(["Top 5 Model Tendencies", "Scouting Insights"])

with tab1:
    st.subheader("Top 5 high-confidence tendencies per team")

    if st.button("Generate top 5 tendencies"):
        scan_df = scan_high_confidence_tendencies()

        if scan_df.empty:
            st.info("No high-confidence model tendencies found with the current thresholds.")
        else:
            top5_df = scan_df.groupby("team", group_keys=False).head(5).copy()

            display_df = top5_df[[
                "team",
                "rank_within_team",
                "tendency",
                "delta_vs_league",
                "model_pass_prob",
                "league_pass_rate",
                "quarter",
                "minutes",
                "seconds",
                "down",
                "yards_to_go",
                "field_side",
                "ball_on",
                "score_diff",
                "team_plays",
                "league_plays",
                "scenario_label",
            ]].copy()

            display_df["delta_vs_league"] = display_df["delta_vs_league"].map(lambda x: f"{x:+.1%}")
            display_df["model_pass_prob"] = display_df["model_pass_prob"].map(lambda x: f"{x:.1%}")
            display_df["league_pass_rate"] = display_df["league_pass_rate"].map(lambda x: f"{x:.1%}")
            display_df["clock"] = display_df.apply(lambda r: f'{int(r["minutes"])}:{int(r["seconds"]):02d}', axis=1)

            st.dataframe(
                display_df[[
                    "team",
                    "rank_within_team",
                    "tendency",
                    "delta_vs_league",
                    "model_pass_prob",
                    "league_pass_rate",
                    "quarter",
                    "clock",
                    "down",
                    "yards_to_go",
                    "field_side",
                    "ball_on",
                    "score_diff",
                    "team_plays",
                    "league_plays",
                    "scenario_label",
                ]],
                use_container_width=True,
                height=520
            )

            csv_data = top5_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download top 5 tendencies per team (CSV)",
                data=csv_data,
                file_name="cfl_top_5_tendencies_per_team.csv",
                mime="text/csv"
            )

with tab2:
    st.subheader("Scouting insights")
    if SCOUTING.empty:
        st.info("Scouting report file not found. Add CFL_EXEC_SCOUTING_REPORT.xlsx to the app folder.")
    else:
        scouting_team_col = "possession_team" if "possession_team" in SCOUTING.columns else "team"
        scouting_team = st.selectbox("Select team for scouting report", sorted(SCOUTING[scouting_team_col].dropna().unique()))

        team_report = SCOUTING[SCOUTING[scouting_team_col] == scouting_team].copy()

        if team_report.empty:
            st.info("No scouting rows found for this team.")
        else:
            display_cols = [c for c in [
                "quarter",
                "down",
                "field_zone",
                "ytg_bucket",
                "score_bucket",
                "plays",
                "pass_rate",
                "td_rate",
                "league_pass_rate",
                "league_td_rate",
                "pass_delta",
                "td_delta",
                "alert",
                "rank",
            ] if c in team_report.columns]

            pretty = team_report.copy()

            for col in ["pass_rate", "td_rate", "league_pass_rate", "league_td_rate", "pass_delta", "td_delta"]:
                if col in pretty.columns:
                    pretty[col] = pretty[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

            st.dataframe(pretty[display_cols], use_container_width=True, height=320)

            if "play_examples" in team_report.columns:
                st.markdown("### Referenced play descriptions")
                for _, row in team_report.iterrows():
                    title = f'Rank {int(row["rank"])} | Q{int(row["quarter"])} | Down {int(row["down"])} | {row["field_zone"]} | {row["ytg_bucket"]}'
                    with st.expander(title):
                        examples = str(row["play_examples"]).split(" || ")
                        for i, ex in enumerate(examples, start=1):
                            st.write(f"{i}. {ex}")

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
