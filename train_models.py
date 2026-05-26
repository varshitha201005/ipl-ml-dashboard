import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os

matches    = pd.read_csv('data/matches_clean.csv')
deliveries = pd.read_csv('data/deliveries_clean.csv')

matches = matches[matches['winner'] != 'No Result'].copy()
print(f"✅ Training data: {len(matches)} matches")
os.makedirs('models', exist_ok=True)

# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
print("\n🔧 Engineering features...")

team_wins  = matches['winner'].value_counts()
team_total = pd.Series(matches['team1'].tolist() + matches['team2'].tolist()).value_counts()
team_winrate = (team_wins / team_total).fillna(0)

matches['team1_winrate'] = matches['team1'].map(team_winrate).fillna(0)
matches['team2_winrate'] = matches['team2'].map(team_winrate).fillna(0)

def get_h2h(row):
    t1, t2 = row['team1'], row['team2']
    h2h = matches[
        ((matches['team1'] == t1) & (matches['team2'] == t2)) |
        ((matches['team1'] == t2) & (matches['team2'] == t1))
    ]
    if len(h2h) == 0:
        return 0.5
    return len(h2h[h2h['winner'] == t1]) / len(h2h)

matches['h2h_winrate'] = matches.apply(get_h2h, axis=1)

venue_team_wins = matches.groupby(['venue', 'winner']).size().reset_index(name='wins')
venue_total     = matches.groupby('venue').size().reset_index(name='total')
venue_stats     = venue_team_wins.merge(venue_total, on='venue')
venue_stats['venue_winrate'] = venue_stats['wins'] / venue_stats['total']

def get_venue_winrate(row):
    v = venue_stats[
        (venue_stats['venue'] == row['venue']) &
        (venue_stats['winner'] == row['team1'])
    ]
    return v['venue_winrate'].values[0] if len(v) > 0 else 0.5

matches['team1_venue_winrate'] = matches.apply(get_venue_winrate, axis=1)

matches_sorted = matches.sort_values('date').reset_index(drop=True)

def get_recent_form(team, current_idx, n=5):
    past   = matches_sorted.iloc[:current_idx]
    past   = past[(past['team1'] == team) | (past['team2'] == team)]
    if len(past) == 0:
        return 0.5
    recent = past.tail(n)
    return len(recent[recent['winner'] == team]) / len(recent)

print("  Computing recent form (takes ~1 min)...")
matches_sorted['team1_form'] = [
    get_recent_form(matches_sorted.iloc[i]['team1'], i)
    for i in range(len(matches_sorted))
]
matches_sorted['team2_form'] = [
    get_recent_form(matches_sorted.iloc[i]['team2'], i)
    for i in range(len(matches_sorted))
]
matches = matches_sorted.copy()
print("  ✅ Features done!")

# ══════════════════════════════════════════════════════════
# MODEL 1 — Match Winner (Random Forest)
# ══════════════════════════════════════════════════════════
print("\n🤖 Training Model 1 — Match Winner Predictor...")

features_match = [
    'team1', 'team2', 'venue', 'toss_winner', 'toss_decision',
    'team1_winrate', 'team2_winrate', 'h2h_winrate',
    'team1_venue_winrate', 'team1_form', 'team2_form'
]
target_match = 'winner'

df_match   = matches[features_match + [target_match]].dropna()
encoders   = {}
df_encoded = df_match.copy()

for col in ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', target_match]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col]   = le

X = df_encoded[features_match]
y = df_encoded[target_match]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)

acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"  ✅ Accuracy : {acc * 100:.2f}%")

joblib.dump(rf_model, 'models/match_predictor.pkl')
joblib.dump(encoders, 'models/encoders.pkl')

meta = {
    'teams'  : sorted(matches['team1'].unique().tolist()),
    'venues' : sorted(matches['venue'].unique().tolist()),
}
joblib.dump(meta, 'models/meta.pkl')
print("  ✅ Saved match_predictor.pkl + encoders.pkl + meta.pkl")

# ══════════════════════════════════════════════════════════
# MODEL 2 — Score Predictor (Gradient Boosting)
# ══════════════════════════════════════════════════════════
print("\n🤖 Training Model 2 — Score Predictor...")

ball_df = deliveries.copy()
ball_df = ball_df.merge(
    matches[['match_id', 'venue', 'season']],
    on='match_id', how='left'
)

innings_agg = ball_df.groupby(['match_id', 'inning']).agg(
    total_runs    = ('total_runs', 'sum'),
    total_wickets = ('is_wicket',  'sum'),
    venue         = ('venue',         'first'),
    batting_team  = ('batting_team',  'first'),
    season        = ('season',        'first')
).reset_index()

first_innings  = innings_agg[innings_agg['inning'] == 1].copy()
score_encoders = {}

for col in ['venue', 'batting_team']:
    le = LabelEncoder()
    first_innings[col] = le.fit_transform(first_innings[col].astype(str))
    score_encoders[col] = le

features_score = ['venue', 'batting_team', 'total_wickets', 'season']
target_score   = 'total_runs'

df_s = first_innings[features_score + [target_score]].dropna()
X_s  = df_s[features_score]
y_s  = df_s[target_score]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb_model.fit(X_train_s, y_train_s)

mae = mean_absolute_error(y_test_s, gb_model.predict(X_test_s))
print(f"  ✅ MAE : {mae:.2f} runs")

joblib.dump(gb_model,       'models/score_predictor.pkl')
joblib.dump(score_encoders, 'models/score_encoders.pkl')
print("  ✅ Saved score_predictor.pkl + score_encoders.pkl")

# ══════════════════════════════════════════════════════════
# MODEL 3 — Toss Impact (Logistic Regression)
# ══════════════════════════════════════════════════════════
print("\n🤖 Training Model 3 — Toss Impact Analyzer...")

features_toss = ['toss_decision_enc', 'season', 'team1_winrate', 'team2_winrate']
target_toss   = 'toss_match_winner'

df_toss = matches[features_toss + [target_toss]].dropna()
X_t     = df_toss[features_toss]
y_t     = df_toss[target_toss]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_t, y_train_t)

acc_t = accuracy_score(y_test_t, log_model.predict(X_test_t))
print(f"  ✅ Accuracy : {acc_t * 100:.2f}%")

joblib.dump(log_model, 'models/toss_impact.pkl')
print("  ✅ Saved toss_impact.pkl")

print("\n" + "=" * 50)
print("✅ All 3 Models Trained & Saved!")
print("=" * 50)
for f in os.listdir('models'):
    print(f"  → {f}")