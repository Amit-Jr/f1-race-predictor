# ============================================================
# F1 2026 Barcelona Grand Prix - Race Prediction Model
# Author: Amit | ML Intern Project
# Date: June 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ============================================================
# SECTION 1: LOAD & CLEAN DATA
# ============================================================

BASE = '/mnt/user-data/uploads'

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

def clean_position(pos):
    """Convert position to numeric. NC/DQ/DNF → 20 (last place proxy)"""
    try:
        return int(pos)
    except:
        return 20

def load_all_races():
    frames = []
    for yr, path in RACE_FILES.items():
        df = pd.read_csv(path)
        df['Season'] = yr
        df['Position_Clean'] = df['Position'].apply(clean_position)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_all_qualifying():
    frames = []
    for yr, path in QUAL_FILES.items():
        df = pd.read_csv(path)
        df['Season'] = yr
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(20)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

print("Loading data...")
all_races = load_all_races()
all_qual = load_all_qualifying()
sprint_2026 = pd.read_csv(f'{BASE}/Formula1_2026Season_SprintResults.csv')

print(f"Total race rows loaded: {len(all_races)}")
print(f"Seasons covered: {sorted(all_races['Season'].unique())}")
print(f"Tracks in dataset: {all_races['Track'].nunique()}")

# ============================================================
# SECTION 2: EXTRACT BARCELONA DATA
# ============================================================

# Barcelona is labeled 'Spain' in all seasons present
barcelona_races = all_races[all_races['Track'] == 'Spain'].copy()
print(f"\nBarcelona race rows found: {len(barcelona_races)}")
print(f"Barcelona seasons: {sorted(barcelona_races['Season'].unique())}")
print(barcelona_races[['Season','Driver','Position_Clean','Team']].sort_values(['Season','Position_Clean']))

# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================

# --- 2026 drivers list (the ones we need predictions for) ---
drivers_2026 = all_races[all_races['Season'] == 2026]['Driver'].unique()
print(f"\n2026 drivers: {len(drivers_2026)}")

feature_rows = []

for driver in drivers_2026:

    row = {'Driver': driver}

    # --- Feature 1: Barcelona historical avg finish ---
    bcn_hist = barcelona_races[barcelona_races['Driver'] == driver]
    row['barcelona_avg_finish'] = bcn_hist['Position_Clean'].mean() if len(bcn_hist) > 0 else 15.0
    row['barcelona_appearances'] = len(bcn_hist)
    row['barcelona_best_finish'] = bcn_hist['Position_Clean'].min() if len(bcn_hist) > 0 else 20

    # --- Feature 2: Overall recent form (2025 full season avg) ---
    form_2025 = all_races[(all_races['Driver'] == driver) & (all_races['Season'] == 2025)]
    row['avg_finish_2025'] = form_2025['Position_Clean'].mean() if len(form_2025) > 0 else 15.0

    # --- Feature 3: 2026 season form so far ---
    form_2026 = all_races[(all_races['Driver'] == driver) & (all_races['Season'] == 2026)]
    row['avg_finish_2026'] = form_2026['Position_Clean'].mean() if len(form_2026) > 0 else 15.0
    row['races_2026'] = len(form_2026)

    # --- Feature 4: Sprint points 2026 (current pace indicator) ---
    sp = sprint_2026[sprint_2026['Driver'] == driver]
    row['sprint_points_2026'] = sp['Points'].sum() if len(sp) > 0 else 0
    row['sprint_avg_position'] = sp['Position'].apply(clean_position).mean() if len(sp) > 0 else 15.0

    # --- Feature 5: Grid-to-finish delta (qualifying → race performance) ---
    # Positive = gains positions, Negative = loses positions
    driver_races = all_races[
        (all_races['Driver'] == driver) &
        (all_races['Season'].isin([2024, 2025, 2026]))
    ].copy()
    driver_races['grid_delta'] = driver_races['Starting Grid'] - driver_races['Position_Clean']
    row['avg_grid_delta'] = driver_races['grid_delta'].mean() if len(driver_races) > 0 else 0.0

    # --- Feature 6: DNF rate (reliability) ---
    all_driver_races = all_races[
        (all_races['Driver'] == driver) &
        (all_races['Season'].isin([2024, 2025, 2026]))
    ]
    dnf_count = (all_driver_races['Position'] == 'NC').sum()
    row['dnf_rate'] = dnf_count / len(all_driver_races) if len(all_driver_races) > 0 else 0.1

    # --- Feature 7: Team performance (2026 avg finish for team) ---
    driver_team_2026 = all_races[
        (all_races['Driver'] == driver) & (all_races['Season'] == 2026)
    ]['Team'].mode()
    if len(driver_team_2026) > 0:
        team = driver_team_2026.iloc[0]
        team_races = all_races[(all_races['Team'] == team) & (all_races['Season'] == 2026)]
        row['team_avg_finish_2026'] = team_races['Position_Clean'].mean()
        row['Team'] = team
    else:
        row['team_avg_finish_2026'] = 15.0
        row['Team'] = 'Unknown'

    # --- Feature 8: Qualifying performance (avg grid position 2025-2026) ---
    qual_recent = all_qual[
        (all_qual['Driver'] == driver) &
        (all_qual['Season'].isin([2025, 2026]))
    ]
    row['avg_qual_position'] = qual_recent['Position'].mean() if len(qual_recent) > 0 else 15.0

    feature_rows.append(row)

features_df = pd.DataFrame(feature_rows)
print(f"\nFeature matrix shape: {features_df.shape}")
print(features_df[['Driver','barcelona_avg_finish','avg_finish_2026','sprint_points_2026','avg_qual_position']].to_string())

# ============================================================
# SECTION 4: BUILD TRAINING DATA
# ============================================================
# Train on historical Barcelona results (2021-2025)
# For each Barcelona race, build features using data available BEFORE that race

train_rows = []

for _, race_row in barcelona_races.iterrows():
    season = race_row['Season']
    driver = race_row['Driver']
    actual_position = race_row['Position_Clean']

    # Historical Barcelona before this season
    bcn_before = barcelona_races[
        (barcelona_races['Driver'] == driver) &
        (barcelona_races['Season'] < season)
    ]

    # Recent form: races in prior season
    prior_season_races = all_races[
        (all_races['Driver'] == driver) &
        (all_races['Season'] == season - 1)
    ]

    # Qualifying avg in prior season
    prior_qual = all_qual[
        (all_qual['Driver'] == driver) &
        (all_qual['Season'] == season - 1)
    ]

    # Current season form before Barcelona
    # Barcelona is race ~6-8 in season, so use races before it
    current_season_before = all_races[
        (all_races['Driver'] == driver) &
        (all_races['Season'] == season)
    ]
    # Approximate: exclude the Barcelona row itself
    current_season_before = current_season_before[current_season_before['Track'] != 'Spain']

    t = {
        'Driver': driver,
        'Season': season,
        'actual_position': actual_position,
        'barcelona_avg_finish': bcn_before['Position_Clean'].mean() if len(bcn_before) > 0 else 15.0,
        'barcelona_appearances': len(bcn_before),
        'barcelona_best_finish': bcn_before['Position_Clean'].min() if len(bcn_before) > 0 else 20,
        'avg_finish_prior_season': prior_season_races['Position_Clean'].mean() if len(prior_season_races) > 0 else 15.0,
        'avg_qual_position': prior_qual['Position'].mean() if len(prior_qual) > 0 else 15.0,
        'current_season_avg': current_season_before['Position_Clean'].mean() if len(current_season_before) > 0 else 15.0,
        'avg_grid_delta': (
            (current_season_before['Starting Grid'] - current_season_before['Position_Clean']).mean()
            if len(current_season_before) > 0 else 0.0
        ),
        'dnf_rate': (
            (prior_season_races['Position'] == 'NC').sum() / len(prior_season_races)
            if len(prior_season_races) > 0 else 0.1
        ),
    }

    # Team avg in current season before Barcelona
    team_rows = current_season_before.copy()
    t['team_avg_finish'] = team_rows['Position_Clean'].mean() if len(team_rows) > 0 else 15.0

    train_rows.append(t)

train_df = pd.DataFrame(train_rows)
print(f"\nTraining set shape: {train_df.shape}")
print(f"Training rows per season:\n{train_df['Season'].value_counts().sort_index()}")

# ============================================================
# SECTION 5: TRAIN MODEL
# ============================================================

FEATURE_COLS = [
    'barcelona_avg_finish',
    'barcelona_appearances',
    'barcelona_best_finish',
    'avg_finish_prior_season',
    'avg_qual_position',
    'current_season_avg',
    'avg_grid_delta',
    'dnf_rate',
    'team_avg_finish',
]

X_train = train_df[FEATURE_COLS].fillna(15.0)
y_train = train_df['actual_position']

print(f"\nTraining samples: {len(X_train)}")
print(f"Features: {FEATURE_COLS}")

# XGBoost
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# Cross validation
cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"\nXGBoost CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} positions")

# Random Forest for comparison
rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"Random Forest CV MAE: {-cv_rf.mean():.2f} ± {cv_rf.std():.2f} positions")

# Train final model on all data
xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ============================================================
# SECTION 6: PREDICT BARCELONA 2026
# ============================================================

# Build prediction features for 2026 drivers
predict_feature_cols = [
    'barcelona_avg_finish',
    'barcelona_appearances',
    'barcelona_best_finish',
    'avg_finish_2025',       # maps to avg_finish_prior_season
    'avg_qual_position',
    'avg_finish_2026',       # maps to current_season_avg
    'avg_grid_delta',
    'dnf_rate',
    'team_avg_finish_2026',  # maps to team_avg_finish
]

X_pred = features_df[predict_feature_cols].fillna(15.0)
X_pred.columns = FEATURE_COLS  # rename to match training

xgb_preds = xgb.predict(X_pred)
rf_preds = rf.predict(X_pred)

# Ensemble: average of both models
ensemble_preds = (xgb_preds + rf_preds) / 2

features_df['xgb_pred'] = xgb_preds
features_df['rf_pred'] = rf_preds
features_df['predicted_position'] = ensemble_preds

# Rank drivers by predicted position
results = features_df[['Driver', 'Team', 'predicted_position', 'xgb_pred', 'rf_pred',
                         'barcelona_avg_finish', 'avg_finish_2026', 'sprint_points_2026']].copy()
results = results.sort_values('predicted_position').reset_index(drop=True)
results['Predicted_Rank'] = results.index + 1

print("\n" + "="*60)
print("BARCELONA 2026 GRAND PRIX - PREDICTED FINISHING ORDER")
print("="*60)
for _, row in results.iterrows():
    print(f"P{int(row['Predicted_Rank']):2d}  {row['Driver']:<25}  {row['Team']:<35}  score: {row['predicted_position']:.2f}")

# ============================================================
# SECTION 7: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('F1 2026 Barcelona GP — Race Prediction Model', fontsize=16, fontweight='bold', y=1.01)

# --- Plot 1: Predicted Finishing Order ---
ax1 = axes[0, 0]
top10 = results.head(10)
colors = ['gold', 'silver', '#cd7f32'] + ['#4a90d9'] * 7
bars = ax1.barh(range(len(top10)), top10['predicted_position'], color=colors, edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(top10)))
ax1.set_yticklabels([f"P{i+1}  {d}" for i, d in enumerate(top10['Driver'])], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Predicted Position Score (lower = better)')
ax1.set_title('Top 10 Predicted Finishers')
ax1.axvline(x=top10['predicted_position'].mean(), color='red', linestyle='--', alpha=0.5, label='Average')
ax1.grid(axis='x', alpha=0.3)

# --- Plot 2: Feature Importance ---
ax2 = axes[0, 1]
feat_imp = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=ax2, color='#4a90d9', edgecolor='white')
ax2.set_title('Feature Importance (XGBoost)')
ax2.set_xlabel('Importance Score')
ax2.grid(axis='x', alpha=0.3)

# --- Plot 3: 2026 Form vs Prediction ---
ax3 = axes[1, 0]
scatter_data = results.dropna(subset=['avg_finish_2026'])
sc = ax3.scatter(scatter_data['avg_finish_2026'], scatter_data['predicted_position'],
                  c=scatter_data['sprint_points_2026'], cmap='RdYlGn', s=100, edgecolors='gray', linewidth=0.5)
for _, row in scatter_data.iterrows():
    ax3.annotate(row['Driver'].split()[-1], (row['avg_finish_2026'], row['predicted_position']),
                 fontsize=7, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')
plt.colorbar(sc, ax=ax3, label='Sprint Points 2026')
ax3.set_xlabel('2026 Avg Finish Position (lower = better)')
ax3.set_ylabel('Predicted Barcelona Position')
ax3.set_title('2026 Form vs Predicted Barcelona Finish\n(Color = Sprint Points)')
ax3.grid(alpha=0.3)

# --- Plot 4: Barcelona Historical Performance (top drivers) ---
ax4 = axes[1, 1]
top_drivers = results.head(8)['Driver'].tolist()
bcn_hist_plot = barcelona_races[barcelona_races['Driver'].isin(top_drivers)].copy()
for driver in top_drivers:
    d = bcn_hist_plot[bcn_hist_plot['Driver'] == driver].sort_values('Season')
    if len(d) > 0:
        ax4.plot(d['Season'], d['Position_Clean'], marker='o', label=driver.split()[-1], linewidth=1.5, markersize=5)
ax4.invert_yaxis()
ax4.set_xlabel('Season')
ax4.set_ylabel('Finishing Position')
ax4.set_title('Barcelona Historical Results (Top 8 Predicted)')
ax4.legend(fontsize=7, ncol=2)
ax4.grid(alpha=0.3)
ax4.set_xticks([2021, 2022, 2023, 2024, 2025])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/barcelona_2026_prediction.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved.")

# ============================================================
# SECTION 8: MODEL EVALUATION SUMMARY
# ============================================================

print("\n" + "="*60)
print("MODEL EVALUATION SUMMARY")
print("="*60)
print(f"Training data: 2021–2025 Barcelona races")
print(f"Training samples: {len(X_train)} driver-race entries")
print(f"XGBoost CV MAE : {-cv_scores.mean():.2f} positions (±{cv_scores.std():.2f})")
print(f"Random Forest MAE: {-cv_rf.mean():.2f} positions (±{cv_rf.std():.2f})")
print(f"Final model: Ensemble (avg of XGB + RF)")
print(f"\nTop 3 most important features:")
top_feats = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(3)
for feat, imp in top_feats.items():
    print(f"  {feat}: {imp:.3f}")

print("\nLimitations:")
print("  - Only 5 Barcelona data points per driver (one per year)")
print("  - 2026 regulation changes reduce historical relevance")
print("  - No lap time, tire strategy, or weather data included")
print("  - New 2026 drivers (Antonelli, Lindblad) have no Barcelona history")

# Save predictions CSV
results[['Predicted_Rank','Driver','Team','predicted_position','barcelona_avg_finish','avg_finish_2026','sprint_points_2026']].to_csv(
    '/mnt/user-data/outputs/barcelona_2026_predictions.csv', index=False
)
print("\nPredictions CSV saved.")
