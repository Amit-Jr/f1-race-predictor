# F1 Race Predictor — Barcelona 2026

Predicts finishing order for the 2026 Spanish Grand Prix using 
5 years of historical F1 data and an XGBoost + Random Forest ensemble.

## Demo
[screenshot of the streamlit dashboard]

## How it works
- Pulls race results from 2021–2026 seasons
- Engineers features: Barcelona track history, current season form, 
  qualifying pace, sprint points, team performance, DNF rate
- Trains on 99 Barcelona race entries (2021–2025)
- Predicts 2026 finishing order using ensemble model

## Results
| Model | CV MAE |
|---|---|
| XGBoost | 3.91 positions |
| Random Forest | 3.65 positions |
| Ensemble | Best of both |

## Stack
Python · XGBoost · Scikit-learn · Streamlit · Plotly · Pandas

## Run locally
pip install -r requirements.txt
streamlit run f1_dashboard.py

## Data
Historical F1 race results 2021–2026 (Kaggle dataset)
Sprint results, qualifying results, season standings

## Limitations
- 5 Barcelona data points per driver — small sample
- No lap time, tire strategy, or weather data
- 2026 regulation changes reduce historical relevance
