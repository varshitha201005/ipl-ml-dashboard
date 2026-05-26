import pandas as pd
import numpy as np

matches    = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

print("🔄 Cleaning data...")

# ── 1. Rename matches id column ────────────────────────────
matches.rename(columns={'id': 'match_id'}, inplace=True)

# ── 2. Fix Missing Values ──────────────────────────────────
matches['winner']        = matches['winner'].fillna('No Result')
matches['result_margin'] = matches['result_margin'].fillna(0)
matches['city']          = matches['city'].fillna('Unknown')

# ── 3. Standardize Team Names ──────────────────────────────
team_name_map = {
    'Delhi Daredevils'            : 'Delhi Capitals',
    'Kings XI Punjab'             : 'Punjab Kings',
    'Royal Challengers Bangalore' : 'Royal Challengers Bengaluru',
    'Rising Pune Supergiants'     : 'Rising Pune Supergiant',
    'Pune Warriors'               : 'Pune Warriors India',
}

for col in ['team1', 'team2', 'toss_winner', 'winner']:
    matches[col] = matches[col].replace(team_name_map)

for col in ['batting_team', 'bowling_team']:
    deliveries[col] = deliveries[col].replace(team_name_map)

# ── 4. Fix Data Types ──────────────────────────────────────
def fix_season(s):
    s = str(s)
    if '/' in s:
        return int(s.split('/')[0]) + 1
    return int(s)

matches['season'] = matches['season'].apply(fix_season)
matches['date']   = pd.to_datetime(matches['date'])

# ── 5. Derive First Innings Score from deliveries ──────────
first_innings = deliveries[deliveries['inning'] == 1].groupby('match_id').agg(
    first_innings_score   = ('total_runs', 'sum'),
    first_innings_wickets = ('is_wicket', 'sum'),
).reset_index()

matches = matches.merge(first_innings, on='match_id', how='left')
matches['first_innings_score']   = matches['first_innings_score'].fillna(0).astype(int)
matches['first_innings_wickets'] = matches['first_innings_wickets'].fillna(0).astype(int)

# ── 6. Add stage column from match_type ───────────────────
matches['stage'] = matches['match_type'].fillna('League')

# ── 7. Engineer ML Features ───────────────────────────────
# Toss winner = match winner?
matches['toss_match_winner'] = (
    matches['toss_winner'] == matches['winner']
).astype(int)

# Batting first team
matches['batting_first_team'] = matches.apply(
    lambda r: r['team2'] if r['toss_decision'] == 'field'
    else r['toss_winner'], axis=1
)

# Batting first won?
matches['batting_first_won'] = (
    matches['batting_first_team'] == matches['winner']
).astype(int)

# Run rate first innings
matches['run_rate_first'] = (
    matches['first_innings_score'] / 20
).round(2)

# Toss decision encoded
matches['toss_decision_enc'] = (
    matches['toss_decision'] == 'bat'
).astype(int)

# ── 8. Clean Deliveries ────────────────────────────────────
deliveries['extras_type']      = deliveries['extras_type'].fillna('none')
deliveries['dismissal_kind']   = deliveries['dismissal_kind'].fillna('none')
deliveries['player_dismissed'] = deliveries['player_dismissed'].fillna('none')
deliveries['fielder']          = deliveries['fielder'].fillna('none')
deliveries['batsman_runs']     = pd.to_numeric(deliveries['batsman_runs'], errors='coerce').fillna(0)
deliveries['is_wicket']        = pd.to_numeric(deliveries['is_wicket'], errors='coerce').fillna(0)

# ── 9. Save Cleaned Files ──────────────────────────────────
matches.to_csv('data/matches_clean.csv', index=False)
deliveries.to_csv('data/deliveries_clean.csv', index=False)

print("✅ matches_clean.csv saved!")
print("✅ deliveries_clean.csv saved!")

print(f"\n📊 Verification:")
print(f"  Matches shape     : {matches.shape}")
print(f"  Deliveries shape  : {deliveries.shape}")
print(f"  Seasons           : {sorted(matches['season'].unique())}")
print(f"  Teams             : {sorted(set(matches['team1'].unique()))}")
print(f"  Toss winner won   : {matches['toss_match_winner'].value_counts().to_dict()}")
print(f"  Finals found      : {len(matches[matches['stage'] == 'Final'])}")
print("\n✅ Data Cleaning Complete!")