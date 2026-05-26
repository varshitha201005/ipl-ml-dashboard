import pandas as pd
import numpy as np

# ── Load Data ──────────────────────────────────────────────
matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

print("=" * 60)
print("MATCHES DATASET")
print("=" * 60)

# Basic info
print(f"\n✅ Total Matches     : {len(matches)}")
print(f"✅ Seasons Available : {sorted(matches['season'].unique())}")
print(f"✅ Total Teams       : {matches['team1'].nunique()}")
print(f"✅ Total Venues      : {matches['venue'].nunique()}")

# Missing values
print("\n📌 Missing Values in matches.csv:")
print(matches.isnull().sum()[matches.isnull().sum() > 0])

print("\n" + "=" * 60)
print("DELIVERIES DATASET")
print("=" * 60)

print(f"\n✅ Total Deliveries  : {len(deliveries)}")
print(f"✅ Total Players     : {deliveries['striker'].nunique()}")
print(f"✅ Total Bowlers     : {deliveries['bowler'].nunique()}")

# Missing values
print("\n📌 Missing Values in deliveries.csv:")
print(deliveries.isnull().sum()[deliveries.isnull().sum() > 0])

print("\n" + "=" * 60)
print("TEAMS LIST")
print("=" * 60)
teams = sorted(set(matches['team1'].unique()) | set(matches['team2'].unique()))
for t in teams:
    print(f"  → {t}")