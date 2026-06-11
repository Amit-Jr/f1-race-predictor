import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Barcelona 2026 — Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── F1 Design System ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=Barlow+Condensed:wght@400;600;700;800;900&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
    color: #f0f0f0 !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1400px !important; }

/* Hide default streamlit decorations */
#MainMenu, footer, header { visibility: hidden; }

/* ── HERO ── */
.f1-hero {
    position: relative;
    width: 100%;
    padding: 3.5rem 0 2rem;
    overflow: hidden;
    margin-bottom: 2.5rem;
}

.f1-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, #e10600 0%, #8b0000 30%, #0a0a0a 70%);
    opacity: 0.15;
    z-index: 0;
}

.f1-hero-inner {
    position: relative;
    z-index: 1;
}

.f1-eyebrow {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #e10600;
    margin-bottom: 0.75rem;
}

.f1-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5.5rem);
    font-weight: 900;
    line-height: 0.92;
    letter-spacing: -0.01em;
    text-transform: uppercase;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.f1-title span {
    color: #e10600;
}

.f1-subtitle {
    font-size: 0.95rem;
    color: #888;
    font-weight: 400;
    margin-top: 1rem;
    letter-spacing: 0.02em;
}

/* Red rule line */
.f1-rule {
    height: 3px;
    background: linear-gradient(90deg, #e10600 0%, #ff4d4d 40%, transparent 100%);
    margin: 1.5rem 0 2rem;
    border: none;
}

/* ── STAT CARDS ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
}

.stat-card {
    flex: 1;
    min-width: 160px;
    background: #111;
    border: 1px solid #1e1e1e;
    border-top: 2px solid #e10600;
    padding: 1.25rem 1.5rem;
    border-radius: 2px;
    position: relative;
    overflow: hidden;
}

.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, #e10600 0%, transparent 70%);
    opacity: 0.06;
}

.stat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.5rem;
    font-family: 'Barlow Condensed', sans-serif;
}

.stat-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}

.stat-unit {
    font-size: 0.75rem;
    color: #e10600;
    font-weight: 600;
    margin-top: 0.25rem;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #fff;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e1e;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-header::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.4rem;
    background: #e10600;
    border-radius: 1px;
}

/* ── PODIUM ── */
.podium-wrapper {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 0.75rem;
    padding: 2rem 1rem 0;
    margin-bottom: 2rem;
}

.podium-slot {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

.podium-driver {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #fff;
    text-align: center;
}

.podium-team {
    font-size: 0.65rem;
    color: #888;
    text-align: center;
}

.podium-block {
    width: 120px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 2px 2px 0 0;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 2.5rem;
    color: #0a0a0a;
    padding: 1rem 0 0.5rem;
}

.podium-p1 { background: #FFD700; height: 140px; }
.podium-p2 { background: #C0C0C0; height: 110px; }
.podium-p3 { background: #cd7f32; height: 85px; }

/* ── RESULT TABLE ── */
.result-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

.result-table th {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid #1e1e1e;
}

.result-table td {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #111;
    color: #ddd;
    vertical-align: middle;
}

.result-table tr:hover td { background: #111; }

.pos-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #fff;
}

.pos-p1 { color: #FFD700 !important; }
.pos-p2 { color: #C0C0C0 !important; }
.pos-p3 { color: #cd7f32 !important; }

.driver-name {
    font-weight: 600;
    color: #fff;
    font-size: 0.9rem;
}

.team-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    background: #1a1a1a;
    color: #888;
    border: 1px solid #222;
}

.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.score-bar {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #e10600, #ff4d4d);
}

.score-num {
    font-size: 0.75rem;
    color: #666;
    min-width: 30px;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0;
    border-bottom: 1px solid #1e1e1e;
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #666 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 0 !important;
    transition: all 0.15s ease !important;
}

.stTabs [aria-selected="true"] {
    color: #fff !important;
    border-bottom: 2px solid #e10600 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding: 1.5rem 0 0 !important;
    background: transparent !important;
}

/* ── PLOTLY containers ── */
.js-plotly-plot { border-radius: 2px; }

/* ── METRIC OVERRIDE ── */
[data-testid="metric-container"] {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-top: 2px solid #e10600 !important;
    padding: 1rem !important;
    border-radius: 2px !important;
}

[data-testid="stMetricLabel"] { color: #666 !important; font-size: 0.7rem !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'Barlow Condensed', sans-serif !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #e10600; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Team colours ──────────────────────────────────────────────
TEAM_COLORS = {
    "Mercedes":                        "#00D2BE",
    "Ferrari":                         "#DC0000",
    "McLaren Mercedes":                "#FF8000",
    "Red Bull Racing Red Bull Ford":   "#3671C6",
    "Alpine Mercedes":                 "#FF87BC",
    "Williams Mercedes":               "#64C4FF",
    "Aston Martin Honda":              "#358C75",
    "Haas Ferrari":                    "#B6BABD",
    "Racing Bulls Red Bull Ford":      "#6692FF",
    "Audi":                            "#C6C6C6",
    "Cadillac Ferrari":                "#E8002D",
    "Racing Bulls":                    "#6692FF",
}

# ── Data & model (cached) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_predict():
    BASE = Path(__file__).parent / "Data"

    RACE_FILES = {
        2021: f'{BASE}/formula1_2021season_raceResults.csv',
        2022: f'{BASE}/Formula1_2022season_raceResults.csv',
        2023: f'{BASE}/Formula1_2023season_raceResults.csv',
        2024: f'{BASE}/Formula1_2024season_raceResults.csv',
        2025: f'{BASE}/Formula1_2025Season_RaceResults.csv',
        2026: f'{BASE}/Formula1_2026Season_RaceResults.csv',
    }
    QUAL_FILES = {
        2022: f'{BASE}/Formula1_2022season_qualifyingResults.csv',
        2023: f'{BASE}/Formula1_2023season_qualifyingResults.csv',
        2024: f'{BASE}/Formula1_2024season_qualifyingResults.csv',
        2025: f'{BASE}/Formula1_2025Season_QualifyingResults.csv',
        2026: f'{BASE}/Formula1_2026Season_QualifyingResults.csv',
    }

    def cp(pos):
        try: return int(pos)
        except: return 20

    frames = []
    for yr, path in RACE_FILES.items():
        df = pd.read_csv(path); df['Season'] = yr
        df['Position_Clean'] = df['Position'].apply(cp)
        frames.append(df)
    all_races = pd.concat(frames, ignore_index=True)

    qframes = []
    for yr, path in QUAL_FILES.items():
        df = pd.read_csv(path); df['Season'] = yr
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(20)
        qframes.append(df)
    all_qual = pd.concat(qframes, ignore_index=True)

    sprint = pd.read_csv(f'{BASE}/Formula1_2026Season_SprintResults.csv')
    barcelona = all_races[all_races['Track'] == 'Spain'].copy()
    drivers_2026 = all_races[all_races['Season'] == 2026]['Driver'].unique()

    # Feature engineering
    feat_rows = []
    for driver in drivers_2026:
        r = {'Driver': driver}
        bh = barcelona[barcelona['Driver'] == driver]
        r['barcelona_avg_finish']  = bh['Position_Clean'].mean() if len(bh) else 15.0
        r['barcelona_appearances'] = len(bh)
        r['barcelona_best_finish'] = bh['Position_Clean'].min() if len(bh) else 20

        f25 = all_races[(all_races['Driver']==driver)&(all_races['Season']==2025)]
        r['avg_finish_2025'] = f25['Position_Clean'].mean() if len(f25) else 15.0

        f26 = all_races[(all_races['Driver']==driver)&(all_races['Season']==2026)]
        r['avg_finish_2026'] = f26['Position_Clean'].mean() if len(f26) else 15.0
        r['races_2026'] = len(f26)

        sp = sprint[sprint['Driver']==driver]
        r['sprint_points_2026']  = sp['Points'].sum() if len(sp) else 0
        r['sprint_avg_position'] = sp['Position'].apply(cp).mean() if len(sp) else 15.0

        dr = all_races[(all_races['Driver']==driver)&(all_races['Season'].isin([2024,2025,2026]))].copy()
        dr['gd'] = dr['Starting Grid'] - dr['Position_Clean']
        r['avg_grid_delta'] = dr['gd'].mean() if len(dr) else 0.0

        adr = all_races[(all_races['Driver']==driver)&(all_races['Season'].isin([2024,2025,2026]))]
        r['dnf_rate'] = (adr['Position']=='NC').sum()/len(adr) if len(adr) else 0.1

        tm = all_races[(all_races['Driver']==driver)&(all_races['Season']==2026)]['Team'].mode()
        if len(tm):
            team = tm.iloc[0]
            tr = all_races[(all_races['Team']==team)&(all_races['Season']==2026)]
            r['team_avg_finish_2026'] = tr['Position_Clean'].mean()
            r['Team'] = team
        else:
            r['team_avg_finish_2026'] = 15.0; r['Team'] = 'Unknown'

        qr = all_qual[(all_qual['Driver']==driver)&(all_qual['Season'].isin([2025,2026]))]
        r['avg_qual_position'] = qr['Position'].mean() if len(qr) else 15.0
        feat_rows.append(r)

    features_df = pd.DataFrame(feat_rows)

    # Training
    train_rows = []
    for _, row in barcelona.iterrows():
        season, driver, actual = row['Season'], row['Driver'], row['Position_Clean']
        bh = barcelona[(barcelona['Driver']==driver)&(barcelona['Season']<season)]
        ps = all_races[(all_races['Driver']==driver)&(all_races['Season']==season-1)]
        pq = all_qual[(all_qual['Driver']==driver)&(all_qual['Season']==season-1)]
        cs = all_races[(all_races['Driver']==driver)&(all_races['Season']==season)&(all_races['Track']!='Spain')]
        t = {
            'actual_position':         actual,
            'barcelona_avg_finish':    bh['Position_Clean'].mean() if len(bh) else 15.0,
            'barcelona_appearances':   len(bh),
            'barcelona_best_finish':   bh['Position_Clean'].min() if len(bh) else 20,
            'avg_finish_prior_season': ps['Position_Clean'].mean() if len(ps) else 15.0,
            'avg_qual_position':       pq['Position'].mean() if len(pq) else 15.0,
            'current_season_avg':      cs['Position_Clean'].mean() if len(cs) else 15.0,
            'avg_grid_delta':          (cs['Starting Grid']-cs['Position_Clean']).mean() if len(cs) else 0.0,
            'dnf_rate':                (ps['Position']=='NC').sum()/len(ps) if len(ps) else 0.1,
            'team_avg_finish':         cs['Position_Clean'].mean() if len(cs) else 15.0,
        }
        train_rows.append(t)

    train_df = pd.DataFrame(train_rows)
    FCOLS = ['barcelona_avg_finish','barcelona_appearances','barcelona_best_finish',
             'avg_finish_prior_season','avg_qual_position','current_season_avg',
             'avg_grid_delta','dnf_rate','team_avg_finish']

    X = train_df[FCOLS].fillna(15.0)
    y = train_df['actual_position']

    xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    rf  = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)

    cv_xgb = cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_rf  = cross_val_score(rf,  X, y, cv=5, scoring='neg_mean_absolute_error')

    xgb.fit(X, y); rf.fit(X, y)

    pcols = ['barcelona_avg_finish','barcelona_appearances','barcelona_best_finish',
             'avg_finish_2025','avg_qual_position','avg_finish_2026',
             'avg_grid_delta','dnf_rate','team_avg_finish_2026']
    Xp = features_df[pcols].fillna(15.0)
    Xp.columns = FCOLS

    features_df['predicted_position'] = (xgb.predict(Xp) + rf.predict(Xp)) / 2
    results = features_df.sort_values('predicted_position').reset_index(drop=True)
    results['Rank'] = results.index + 1

    feat_imp = pd.Series(xgb.feature_importances_, index=FCOLS).sort_values(ascending=False)

    return results, barcelona, feat_imp, -cv_xgb.mean(), -cv_rf.mean(), all_races, train_df

# ── Load ───────────────────────────────────────────────────────
with st.spinner(""):
    results, barcelona, feat_imp, mae_xgb, mae_rf, all_races, train_df = load_and_predict()

# ── HERO ───────────────────────────────────────────────────────
st.markdown("""
<div class="f1-hero">
  <div class="f1-hero-inner">
    <div class="f1-eyebrow">🏁 2026 Season · Round 7</div>
    <div class="f1-title">Barcelona<br><span>Catalunya</span></div>
    <div class="f1-subtitle">Circuit de Barcelona-Catalunya · 14 June 2026 · 66 Laps · 307.236 km<br>ML Prediction Model — XGBoost + Random Forest Ensemble</div>
  </div>
</div>
<hr class="f1-rule">
""", unsafe_allow_html=True)

# ── STAT CARDS ─────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-label">Drivers</div>
    <div class="stat-value">22</div>
    <div class="stat-unit">2026 Grid</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Training Races</div>
    <div class="stat-value">99</div>
    <div class="stat-unit">2021–2025 Barcelona</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">XGBoost MAE</div>
    <div class="stat-value">{:.2f}</div>
    <div class="stat-unit">positions (5-fold CV)</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Random Forest MAE</div>
    <div class="stat-value">{:.2f}</div>
    <div class="stat-unit">positions (5-fold CV)</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Lap Record</div>
    <div class="stat-value">1:16</div>
    <div class="stat-unit">Verstappen · 2023</div>
  </div>
</div>
""".format(mae_xgb, mae_rf), unsafe_allow_html=True)

# ── PODIUM ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Predicted Podium</div>', unsafe_allow_html=True)

p1 = results.iloc[0]; p2 = results.iloc[1]; p3 = results.iloc[2]

st.markdown(f"""
<div class="podium-wrapper">
  <div class="podium-slot">
    <div class="podium-driver">{p2['Driver']}</div>
    <div class="podium-team">{p2['Team']}</div>
    <div class="podium-block podium-p2">2</div>
  </div>
  <div class="podium-slot">
    <div class="podium-driver">{p1['Driver']}</div>
    <div class="podium-team">{p1['Team']}</div>
    <div class="podium-block podium-p1">1</div>
  </div>
  <div class="podium-slot">
    <div class="podium-driver">{p3['Driver']}</div>
    <div class="podium-team">{p3['Team']}</div>
    <div class="podium-block podium-p3">3</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Full Grid", "Analytics", "Driver History", "Model Info"])

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#aaa'),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor='#1a1a1a', zerolinecolor='#222', tickfont=dict(color='#666')),
    yaxis=dict(gridcolor='#1a1a1a', zerolinecolor='#222', tickfont=dict(color='#666')),
)

def plotly_layout(**overrides):
    layout = {**PLOTLY_LAYOUT}
    for axis in ('xaxis', 'yaxis'):
        if axis in overrides:
            layout[axis] = {**layout.get(axis, {}), **overrides.pop(axis)}
    layout.update(overrides)
    return layout

# ── TAB 1: Full Grid ──────────────────────────────────────────
with tab1:
    pos_colors = {1:'#FFD700', 2:'#C0C0C0', 3:'#cd7f32'}
    max_score = results['predicted_position'].max()

    rows_html = ""
    for _, row in results.iterrows():
        rank = int(row['Rank'])
        pc = pos_colors.get(rank, '#fff')
        pcc = f'pos-p{rank}' if rank <= 3 else ''
        team = row['Team']
        tc = TEAM_COLORS.get(team, '#888')
        bar_w = max(4, int((1 - row['predicted_position']/max_score) * 120))
        sc = row['predicted_position']
        sp = int(row['sprint_points_2026'])
        bcn = f"{row['barcelona_avg_finish']:.1f}" if row['barcelona_appearances'] > 0 else "—"
        form = f"{row['avg_finish_2026']:.1f}"

        rows_html += f"""
        <tr>
          <td><span class="pos-num {pcc}" style="color:{pc}">{rank}</span></td>
          <td>
            <div class="driver-name">{row['Driver']}</div>
          </td>
          <td><span class="team-tag" style="border-color:{tc}40;color:{tc}">{team.split()[0]}</span></td>
          <td>
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:{bar_w}px"></div>
              <span class="score-num">{sc:.2f}</span>
            </div>
          </td>
          <td style="color:#aaa;font-size:0.8rem">{bcn}</td>
          <td style="color:#aaa;font-size:0.8rem">{form}</td>
          <td style="color:{'#e10600' if sp > 0 else '#444'};font-weight:600;font-size:0.85rem">{sp}</td>
        </tr>"""

    st.markdown(f"""
    <table class="result-table">
      <thead>
        <tr>
          <th>POS</th><th>Driver</th><th>Team</th>
          <th>Confidence Score</th><th>BCN Avg</th>
          <th>2026 Form</th><th>Sprint Pts</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ── TAB 2: Analytics ──────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header" style="font-size:1rem">Feature Importance</div>', unsafe_allow_html=True)
        feat_labels = {
            'barcelona_avg_finish': 'Barcelona Avg Finish',
            'barcelona_appearances': 'Barcelona Experience',
            'barcelona_best_finish': 'Barcelona Best',
            'avg_finish_prior_season': 'Prior Season Form',
            'avg_qual_position': 'Qualifying Avg',
            'current_season_avg': '2026 Season Form',
            'avg_grid_delta': 'Grid Delta',
            'dnf_rate': 'DNF Rate',
            'team_avg_finish': 'Team Performance',
        }
        fi = feat_imp.reset_index()
        fi.columns = ['Feature','Importance']
        fi['Label'] = fi['Feature'].map(feat_labels)
        fi = fi.sort_values('Importance')

        fig_imp = go.Figure(go.Bar(
            x=fi['Importance'], y=fi['Label'],
            orientation='h',
            marker=dict(
                color=fi['Importance'],
                colorscale=[[0,'#3a0000'],[0.5,'#8b0000'],[1,'#e10600']],
                line=dict(width=0)
            ),
            text=[f"{v:.3f}" for v in fi['Importance']],
            textposition='outside',
            textfont=dict(color='#666', size=10),
        ))
        fig_imp.update_layout(**plotly_layout(
            height=340,
            xaxis=dict(showgrid=False, showticklabels=False, zerolinecolor='#1a1a1a'),
            yaxis=dict(gridcolor='#111', tickfont=dict(color='#aaa', size=11)),
        ))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header" style="font-size:1rem">2026 Form vs Prediction</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        for _, row in results.iterrows():
            tc = TEAM_COLORS.get(row['Team'], '#888')
            fig_sc.add_trace(go.Scatter(
                x=[row['avg_finish_2026']],
                y=[row['predicted_position']],
                mode='markers+text',
                marker=dict(size=10 + row['sprint_points_2026']*0.4,
                            color=tc, line=dict(color='#0a0a0a', width=1.5)),
                text=[row['Driver'].split()[-1]],
                textposition='top center',
                textfont=dict(size=9, color='#888'),
                showlegend=False,
                hovertemplate=f"<b>{row['Driver']}</b><br>2026 Avg: {row['avg_finish_2026']:.1f}<br>Predicted: {row['predicted_position']:.2f}<extra></extra>"
            ))
        fig_sc.update_layout(**plotly_layout(
            height=340,
            xaxis=dict(title='2026 Avg Finish (lower=better)', gridcolor='#111', tickfont=dict(color='#666')),
            yaxis=dict(title='Predicted Position', gridcolor='#111', tickfont=dict(color='#666')),
        ))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Sprint points bar
    st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1rem">2026 Sprint Points</div>', unsafe_allow_html=True)
    sp_data = results[['Driver','Team','sprint_points_2026']].sort_values('sprint_points_2026', ascending=True)
    sp_data = sp_data[sp_data['sprint_points_2026'] > 0]

    fig_sp = go.Figure(go.Bar(
        x=sp_data['sprint_points_2026'],
        y=sp_data['Driver'],
        orientation='h',
        marker=dict(
            color=[TEAM_COLORS.get(t,'#888') for t in sp_data['Team']],
            line=dict(width=0)
        ),
        text=sp_data['sprint_points_2026'],
        textposition='outside',
        textfont=dict(color='#888', size=10),
    ))
    fig_sp.update_layout(**plotly_layout(
        height=280,
        xaxis=dict(showgrid=False, showticklabels=False, zerolinecolor='#111'),
        yaxis=dict(gridcolor='#111', tickfont=dict(color='#ccc', size=11)),
    ))
    st.plotly_chart(fig_sp, use_container_width=True)

# ── TAB 3: Driver History ─────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header" style="font-size:1rem">Barcelona Historical Results (Top 10 Predicted)</div>', unsafe_allow_html=True)

    top10_drivers = results.head(10)['Driver'].tolist()
    fig_hist = go.Figure()

    for driver in top10_drivers:
        d = barcelona[barcelona['Driver']==driver].sort_values('Season')
        if len(d) == 0: continue
        team = results[results['Driver']==driver]['Team'].values[0]
        tc = TEAM_COLORS.get(team, '#888')
        fig_hist.add_trace(go.Scatter(
            x=d['Season'], y=d['Position_Clean'],
            mode='lines+markers',
            name=driver.split()[-1],
            line=dict(color=tc, width=2),
            marker=dict(size=8, color=tc, line=dict(color='#0a0a0a', width=2)),
            hovertemplate=f"<b>{driver}</b><br>%{{x}}: P%{{y}}<extra></extra>"
        ))

    fig_hist.update_layout(**plotly_layout(
        height=380,
        yaxis=dict(autorange='reversed', title='Finishing Position',
                   gridcolor='#111', tickfont=dict(color='#666'),
                   dtick=2),
        xaxis=dict(title='Season', tickvals=[2021,2022,2023,2024,2025],
                   gridcolor='#111', tickfont=dict(color='#666')),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#aaa', size=10),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        hovermode='x unified',
    ))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Per-driver breakdown
    st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1rem">Driver Breakdown</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select driver", results['Driver'].tolist(),
                            format_func=lambda x: f"P{int(results[results['Driver']==x]['Rank'].values[0])}  {x}")

    dr = results[results['Driver']==selected].iloc[0]
    team = dr['Team']
    tc = TEAM_COLORS.get(team, '#888')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Position", f"P{int(dr['Rank'])}")
    c2.metric("BCN Avg Finish", f"P{dr['barcelona_avg_finish']:.1f}" if dr['barcelona_appearances'] else "No data")
    c3.metric("2026 Avg Finish", f"P{dr['avg_finish_2026']:.1f}")
    c4.metric("Sprint Points", int(dr['sprint_points_2026']))

# ── TAB 4: Model Info ─────────────────────────────────────────
with tab4:
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="section-header" style="font-size:1rem">Model Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="background:#111;border:1px solid #1e1e1e;border-left:3px solid #e10600;padding:1.25rem;border-radius:2px;font-size:0.875rem;line-height:1.8;color:#aaa">
<b style="color:#fff">Training Data</b><br>
2021–2025 Barcelona GP results · 99 driver-race entries<br><br>
<b style="color:#fff">Models</b><br>
① XGBoost Regressor — n_estimators=200, max_depth=3, lr=0.05<br>
② Random Forest — n_estimators=200, max_depth=4<br>
③ Final: Ensemble average of ① + ②<br><br>
<b style="color:#fff">Evaluation</b><br>
5-Fold Cross Validation on historical Barcelona data<br>
Metric: Mean Absolute Error (positions)
</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
<div style="margin-top:1rem;background:#111;border:1px solid #1e1e1e;border-left:3px solid #FFD700;padding:1.25rem;border-radius:2px;font-size:0.875rem;line-height:1.8;color:#aaa">
<b style="color:#FFD700">CV Results</b><br>
XGBoost MAE: <b style="color:#fff">{mae_xgb:.2f}</b> positions<br>
Random Forest MAE: <b style="color:#fff">{mae_rf:.2f}</b> positions<br>
Ensemble: Best of both worlds
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header" style="font-size:1rem">Features Used</div>', unsafe_allow_html=True)
        features_info = [
            ("Barcelona Avg Finish", "Historical avg at this exact track (2021–2025)"),
            ("Barcelona Appearances", "Experience level at Circuit de Catalunya"),
            ("Barcelona Best Finish", "Career best at Barcelona"),
            ("Prior Season Form", "Full 2025 season average finish position"),
            ("Qualifying Avg", "Avg qualifying position 2025–2026"),
            ("2026 Season Form", "Current 2026 avg finish (6 races)"),
            ("Grid Delta", "Positions gained/lost from grid vs finish"),
            ("DNF Rate", "Reliability risk — DNF% in 2024–2026"),
            ("Team Performance", "Constructor's avg finish in 2026"),
        ]
        for name, desc in features_info:
            st.markdown(f"""
<div style="padding:0.6rem 0;border-bottom:1px solid #111;display:flex;gap:0.75rem;align-items:flex-start">
  <span style="color:#e10600;font-size:0.6rem;margin-top:0.35rem">▶</span>
  <div>
    <div style="font-size:0.8rem;font-weight:600;color:#fff">{name}</div>
    <div style="font-size:0.72rem;color:#666">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="margin-top:1rem;background:#111;border:1px solid #1e1e1e;border-left:3px solid #888;padding:1.25rem;border-radius:2px;font-size:0.8rem;line-height:1.8;color:#666">
<b style="color:#aaa">⚠ Limitations</b><br>
· Only 5 Barcelona data points per driver<br>
· 2026 regulation changes reduce historical relevance<br>
· No lap time, tire strategy, or weather data<br>
· New drivers default to P15 (no Barcelona history)
</div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1.5rem;border-top:1px solid #1a1a1a;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem">
  <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;
              letter-spacing:0.2em;text-transform:uppercase;color:#333">
    F1 Barcelona 2026 · ML Race Predictor · Built by Amit
  </div>
  <div style="font-size:0.7rem;color:#333">
    XGBoost + Random Forest Ensemble · 5-Fold CV Validated
  </div>
</div>
""", unsafe_allow_html=True)
