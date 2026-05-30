import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ══════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ══════════════════════════════════════════════════════════
matches    = pd.read_csv('data/matches_clean.csv')
deliveries = pd.read_csv('data/deliveries_clean.csv')

rf_model       = joblib.load('models/match_predictor.pkl')
encoders       = joblib.load('models/encoders.pkl')
meta           = joblib.load('models/meta.pkl')
gb_model       = joblib.load('models/score_predictor.pkl')
score_encoders = joblib.load('models/score_encoders.pkl')
log_model      = joblib.load('models/toss_impact.pkl')

# ── Ensure numeric types on raw DataFrames at startup ──────
deliveries['batsman_runs'] = pd.to_numeric(deliveries['batsman_runs'], errors='coerce').fillna(0)
deliveries['is_wicket']    = pd.to_numeric(deliveries['is_wicket'],    errors='coerce').fillna(0)

# ── Pre-compute heavy joins at startup ─────────────────────
DEL_VENUE   = deliveries.merge(matches[['match_id','venue']], on='match_id', how='left')
DEL_SEASON  = deliveries.merge(matches[['match_id','season']], on='match_id', how='left')

# Pre-compute career rankings data
ORANGE_CAP = DEL_SEASON.groupby(['season','batter'])['batsman_runs'].sum().reset_index()
ORANGE_CAP = ORANGE_CAP.dropna(subset=['season'])
ORANGE_CAP = ORANGE_CAP.loc[ORANGE_CAP.groupby('season')['batsman_runs'].idxmax()].reset_index(drop=True)
ORANGE_CAP.columns = ['Season', 'Player', 'Runs']
ORANGE_CAP = ORANGE_CAP.sort_values('Season')

PURPLE_CAP = DEL_SEASON[DEL_SEASON['is_wicket']==1].groupby(['season','bowler']).size().reset_index()
PURPLE_CAP.columns = ['Season', 'Player', 'Wickets']
PURPLE_CAP = PURPLE_CAP.dropna(subset=['Season'])
PURPLE_CAP = PURPLE_CAP.loc[PURPLE_CAP.groupby('Season')['Wickets'].idxmax()].reset_index(drop=True)
PURPLE_CAP = PURPLE_CAP.sort_values('Season') 

def get_phase(over):
    try:
        over = float(over)
        if over <= 6:    return 'Powerplay (1-6)'
        elif over <= 15: return 'Middle (7-15)'
        else:            return 'Death (16-20)'
    except (TypeError, ValueError):
        return 'Middle (7-15)'

DEL_VENUE['phase'] = DEL_VENUE['over'].apply(get_phase)

ACTIVE_TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]
ALL_TEAMS  = sorted(meta['teams'])
VENUES     = sorted(meta['venues'])
PLAYERS    = sorted(deliveries['batter'].unique().tolist())

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
GOLD = '#f59e0b'
DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#9ca3af', family='Inter'),
    xaxis=dict(gridcolor='#1e2a3a', linecolor='#1e2a3a'),
    yaxis=dict(gridcolor='#1e2a3a', linecolor='#1e2a3a'),
    margin=dict(l=40, r=20, t=40, b=40),
)

def stat_card(icon, value, label, sub):
    return html.Div([
        html.Div(icon,  style={'fontSize': '1.8rem', 'marginBottom': '10px'}),
        html.Div(value, style={'fontSize': '1.8rem', 'fontWeight': '700', 'color': GOLD, 'lineHeight': '1'}),
        html.Div(label, style={'color': '#6b7280', 'fontSize': '0.75rem', 'marginTop': '6px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Div(sub,   style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginTop': '6px'}),
    ], style={
        'background': 'linear-gradient(135deg, #0d1117, #161b27)',
        'border': '1px solid #1e2a3a', 'borderRadius': '12px',
        'padding': '24px', 'marginBottom': '16px',
    })

def info_row(label, value):
    return html.Div([
        html.Span(label, style={'color': '#6b7280', 'fontSize': '0.8rem', 'width': '130px', 'display': 'inline-block'}),
        html.Span(value, style={'color': '#e5e7eb', 'fontSize': '0.8rem', 'fontWeight': '600'}),
    ], style={'marginBottom': '10px'})

def chart_card(children):
    return html.Div(children, style={
        'background': '#0d1117', 'border': '1px solid #1e2a3a',
        'borderRadius': '12px', 'padding': '16px', 'marginBottom': '24px'
    })

def page_header(title, subtitle):
    return html.Div([
        html.H3(title,    style={'color': '#fff', 'fontWeight': '700', 'marginBottom': '4px'}),
        html.P(subtitle,  style={'color': '#6b7280', 'marginBottom': '28px'}),
    ])

def nav_link(icon, label, href):
    return dcc.Link(href=href, style={'textDecoration': 'none'}, children=
        html.Div([
            html.Span(icon,  style={'marginRight': '12px', 'fontSize': '1rem'}),
            html.Span(label, style={'color': '#9ca3af', 'fontSize': '0.875rem', 'fontWeight': '500'}),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'padding': '12px 20px',
            'borderLeft': '3px solid transparent', 'cursor': 'pointer', 'transition': 'all 0.2s ease'
        })
    )

def section_label(text):
    return html.P(text, style={
        'color': '#4b5563', 'fontSize': '0.65rem', 'letterSpacing': '2px',
        'padding': '16px 20px 8px', 'textTransform': 'uppercase', 'marginBottom': '0'
    })

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
sidebar = html.Div([
    # Close button — only visible on mobile
    html.Button("✕", id="close-sidebar", n_clicks=0, style={
        'position': 'absolute', 'top': '12px', 'right': '12px',
        'background': 'none', 'border': 'none', 'color': '#9ca3af',
        'fontSize': '1.2rem', 'cursor': 'pointer',
        'zIndex': '1001'
    }, className="mobile-close-btn"),
    html.Div([
        html.H2("🏏 IPL", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '1.4rem'}),
        html.P("ML Dashboard • 2008–2024", style={'color': '#6b7280', 'fontSize': '0.75rem', 'marginTop': '4px'}),
    ], style={'padding': '24px 20px', 'borderBottom': '1px solid #1e2a3a', 'textAlign': 'center'}),

    html.Div([
        section_label("OVERVIEW"),
        nav_link("🏠", "Home Dashboard",       "/"),
        nav_link("🏆", "Team Intelligence",    "/team-intelligence"),
        nav_link("🏟️", "Phase & Pitch Stats",  "/phase-intelligence"),
        nav_link("📊", "Career Rankings",      "/career-rankings"),
        nav_link("⚔️", "Player Comparison",    "/player-comparison"),

        section_label("ML MODELS"),
        nav_link("🔮", "Match Predictor",      "/match-predictor"),
        nav_link("🏏", "Score Predictor",      "/score-predictor"),
        nav_link("📈", "Performance Predictor","/performance-predictor"),
        nav_link("🔍", "Similar Player Finder","/similar-player"),
        nav_link("🪙", "Toss Analysis",        "/toss-analysis"),
    ]),

    html.Div([
        html.P(f"Matches: {len(matches):,}", style={'color': '#6b7280', 'fontSize': '0.75rem', 'marginBottom': '4px'}),
        html.P(f"Players: {len(PLAYERS):,}", style={'color': '#6b7280', 'fontSize': '0.75rem', 'marginBottom': '4px'}),
        html.P("Data: IPL 2008–2024",         style={'color': '#6b7280', 'fontSize': '0.75rem'}),
    ], style={'padding': '20px', 'borderTop': '1px solid #1e2a3a', 'marginTop': '40px'}),

], id='sidebar', style={
    'position': 'fixed', 'left': '0', 'top': '0',
    'height': '100vh', 'width': '240px',
    'background': 'linear-gradient(180deg, #0d1117 0%, #161b27 100%)',
    'borderRight': '1px solid #1e2a3a', 'zIndex': '1000', 'overflowY': 'auto'
})

# ══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
def dashboard_layout():
    total_matches = len(matches)
    total_seasons = matches['season'].nunique()
    total_teams   = len(set(matches['team1'].unique()) | set(matches['team2'].unique()))
    total_players = deliveries['batter'].nunique()

    winners = matches[matches['match_type'] == 'Final'][['season','winner']].drop_duplicates('season').sort_values('season')

    top_runs = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
    top_runs.columns = ['Player', 'Runs']
    top_runs = top_runs.sort_values('Runs', ascending=False).head(10)

    top_wickets = deliveries[deliveries['is_wicket'] == 1].groupby('bowler').size().reset_index()
    top_wickets.columns = ['Player', 'Wickets']
    top_wickets = top_wickets.sort_values('Wickets', ascending=False).head(10)

    team_wins = matches[matches['winner'] != 'No Result']['winner'].value_counts().reset_index()
    team_wins.columns = ['Team', 'Wins']

    matches_per_season = matches.groupby('season').size().reset_index()
    matches_per_season.columns = ['Season', 'Matches']

    fig_runs = px.bar(top_runs, x='Runs', y='Player', orientation='h',
        title='🏆 Top 10 Run Scorers (All Time)',
        color='Runs', color_continuous_scale=[[0,'#1e2a3a'],[1,GOLD]])
    fig_runs.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_runs.update_yaxes(categoryorder='total ascending')

    fig_wickets = px.bar(top_wickets, x='Wickets', y='Player', orientation='h',
        title='🎯 Top 10 Wicket Takers (All Time)',
        color='Wickets', color_continuous_scale=[[0,'#1e2a3a'],[1,'#ef4444']])
    fig_wickets.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_wickets.update_yaxes(categoryorder='total ascending')

    fig_teams = px.bar(team_wins.head(10), x='Team', y='Wins',
        title='🏟️ Most Wins by Team (All Time)',
        color='Wins', color_continuous_scale=[[0,'#1e2a3a'],[1,GOLD]])
    fig_teams.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_teams.update_xaxes(tickangle=-30)

    fig_season = px.line(matches_per_season, x='Season', y='Matches',
        title='📅 Matches Per Season', markers=True)
    fig_season.update_traces(line_color=GOLD, marker_color=GOLD)
    fig_season.update_layout(**DARK, title_font_color='#fff')

    return html.Div([
        page_header("Dashboard", "IPL Statistics Overview • 2008–2024"),
        dbc.Row([
            dbc.Col(stat_card("🏏", str(total_matches), "Total Matches",  "All seasons combined"), md=3),
            dbc.Col(stat_card("📅", str(total_seasons), "Seasons",        "2008 to 2024"),          md=3),
            dbc.Col(stat_card("🏟️", str(total_teams),  "Teams",          "All franchises"),         md=3),
            dbc.Col(stat_card("👤", str(total_players), "Players",        "Unique batters"),         md=3),
        ], className="mb-4"),

        html.Div([
            html.P("🏆 Season Champions", style={'color': '#fff', 'fontWeight': '600', 'fontSize': '1rem', 'marginBottom': '16px'}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Span(str(int(row['season'])), style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.85rem', 'width': '50px', 'display': 'inline-block'}),
                    html.Span(row['winner'], style={'color': '#e5e7eb', 'fontSize': '0.85rem'}),
                ], style={'padding': '8px 12px', 'borderBottom': '1px solid #1e2a3a'}), md=4)
                for _, row in winners.iterrows()
            ]),
        ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '12px', 'padding': '20px', 'marginBottom': '24px'}),

        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_runs,    config={'displayModeBar': False})), md=6),
            dbc.Col(chart_card(dcc.Graph(figure=fig_wickets, config={'displayModeBar': False})), md=6),
        ]),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_teams,  config={'displayModeBar': False})), md=7),
            dbc.Col(chart_card(dcc.Graph(figure=fig_season, config={'displayModeBar': False})), md=5),
        ]),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 2 — TEAM INTELLIGENCE
# ══════════════════════════════════════════════════════════
def team_intelligence_layout():
    valid = matches[matches['winner'] != 'No Result']

    # Win % per team
    wins   = valid['winner'].value_counts()
    played = pd.Series(valid['team1'].tolist() + valid['team2'].tolist()).value_counts()
    win_pct = (wins / played * 100).round(1).reset_index()
    win_pct.columns = ['Team', 'Win%']
    win_pct = win_pct[win_pct['Team'].isin(ACTIVE_TEAMS)].sort_values('Win%', ascending=False)

    # Season wise wins
    season_wins = valid.groupby(['season','winner']).size().reset_index(name='Wins')
    season_wins = season_wins[season_wins['winner'].isin(ACTIVE_TEAMS)]

    # Titles won
    finals = matches[matches['match_type'] == 'Final']
    titles = finals['winner'].value_counts().reset_index()
    titles.columns = ['Team', 'Titles']

    fig_winpct = px.bar(win_pct, x='Team', y='Win%',
        title='🏆 Win Percentage by Team (Active Teams)',
        color='Win%', color_continuous_scale=[[0,'#1e2a3a'],[1,GOLD]])
    fig_winpct.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_winpct.update_xaxes(tickangle=-30)

    fig_titles = px.bar(titles, x='Team', y='Titles',
        title='🥇 IPL Titles Won',
        color='Titles', color_continuous_scale=[[0,'#1e2a3a'],[1,GOLD]])
    fig_titles.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_titles.update_xaxes(tickangle=-30)

    fig_season_wins = px.line(season_wins, x='season', y='Wins', color='winner',
        title='📈 Season-wise Wins per Team')
    fig_season_wins.update_layout(**DARK, title_font_color='#fff')

    return html.Div([
        page_header("Team Intelligence", "Win rates, titles and season-wise performance"),

        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_winpct,      config={'displayModeBar': False})), md=7),
            dbc.Col(chart_card(dcc.Graph(figure=fig_titles,      config={'displayModeBar': False})), md=5),
        ]),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_season_wins, config={'displayModeBar': False}))),
        ]),

        # Head to head section
        html.Div([
            html.P("⚔️ Head to Head", style={'color': '#fff', 'fontWeight': '600', 'fontSize': '1rem', 'marginBottom': '20px'}),
            dbc.Row([
                dbc.Col([
                    html.Label("Team 1", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                    dcc.Dropdown(id='h2h-team1',
                        options=[{'label': t, 'value': t} for t in ACTIVE_TEAMS],
                        value=ACTIVE_TEAMS[0], clearable=False,
                        style={'color': '#000000'}),
                ], md=5),
                dbc.Col(html.Div("VS", style={'color': GOLD, 'fontWeight': '800', 'fontSize': '1.5rem', 'textAlign': 'center', 'paddingTop': '28px'}), md=2),
                dbc.Col([
                    html.Label("Team 2", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                    dcc.Dropdown(id='h2h-team2',
                        options=[{'label': t, 'value': t} for t in ACTIVE_TEAMS],
                        value=ACTIVE_TEAMS[1], clearable=False,
                        style={'color': '#000000'}),
                ], md=5),
            ], className="mb-3"),
            html.Div(id='h2h-result'),
        ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '12px', 'padding': '24px', 'marginBottom': '24px'}),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 3 — PHASE & PITCH INTELLIGENCE
# ══════════════════════════════════════════════════════════
def phase_intelligence_layout():
    del_m = DEL_VENUE.dropna(subset=['venue', 'phase']).copy()

    # Runs per phase
    phase_runs = del_m.groupby('phase')['batsman_runs'].sum().reset_index()
    phase_runs.columns = ['Phase', 'Runs']
    phase_order = ['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']
    phase_runs['Phase'] = pd.Categorical(phase_runs['Phase'], categories=phase_order, ordered=True)
    phase_runs = phase_runs.sort_values('Phase')

    # Wickets per phase
    phase_wkts = del_m[del_m['is_wicket'] == 1].groupby('phase').size().reset_index()
    phase_wkts.columns = ['Phase', 'Wickets']
    phase_wkts['Phase'] = pd.Categorical(phase_wkts['Phase'], categories=phase_order, ordered=True)
    phase_wkts = phase_wkts.sort_values('Phase')

    # ── Pitch Stats ────────────────────────────────────────
    # Avg runs per venue
    venue_runs = del_m.groupby('venue')['batsman_runs'].sum().reset_index()
    venue_innings = del_m.groupby('venue')['match_id'].nunique().reset_index()
    venue_stats = venue_runs.merge(venue_innings, on='venue')
    venue_stats.columns = ['Venue', 'Total_Runs', 'Matches']
    venue_stats['Avg_Runs'] = (venue_stats['Total_Runs'] / venue_stats['Matches']).round(1)
    venue_stats = venue_stats.sort_values('Avg_Runs', ascending=False).head(15)

    # Wickets per venue
    venue_wkts = del_m[del_m['is_wicket'] == 1].groupby('venue').size().reset_index()
    venue_wkts.columns = ['venue', 'Wickets']
    venue_wkts2 = venue_wkts.merge(venue_innings, on='venue')
    venue_wkts2.columns = ['Venue', 'Wickets', 'Matches']
    venue_wkts2['Avg_Wickets'] = (venue_wkts2['Wickets'] / venue_wkts2['Matches']).round(2)
    venue_wkts2 = venue_wkts2.sort_values('Avg_Wickets', ascending=False).head(15)

    # For top_venues reference keep lowercase venue column available
    top_venues = venue_wkts.head(10)['venue'].tolist()

    # Pace vs Spin per venue
    # Identify bowler type — simple heuristic: spinners bowl slower variations
    spin_keywords = ['jadeja', 'ashwin', 'chahal', 'kuldeep', 'rashid', 'narine', 'mishra', 'tahir', 'imad', 'piyush', 'harbhajan', 'ojha']
    del_m = del_m.copy()
    del_m['bowler_lower'] = del_m['bowler'].fillna('').str.lower()
    del_m['bowler_type']  = del_m['bowler_lower'].apply(
        lambda x: 'Spin' if x and any(k in x for k in spin_keywords) else 'Pace'
    )

    venue_bowler = del_m[del_m['is_wicket'] == 1].groupby(['venue','bowler_type']).size().reset_index()
    venue_bowler.columns = ['Venue','Bowler Type','Wickets']
    venue_bowler = venue_bowler[venue_bowler['Venue'].isin(top_venues)]

    # Charts
    fig_phase_runs = px.bar(phase_runs, x='Phase', y='Runs',
        title='🏏 Total Runs by Match Phase',
        color='Phase', color_discrete_map={
            'Powerplay (1-6)': GOLD,
            'Middle (7-15)': '#3b82f6',
            'Death (16-20)': '#ef4444'
        })
    fig_phase_runs.update_layout(**DARK, title_font_color='#fff', showlegend=False)

    fig_phase_wkts = px.bar(phase_wkts, x='Phase', y='Wickets',
        title='🎯 Wickets by Match Phase',
        color='Phase', color_discrete_map={
            'Powerplay (1-6)': GOLD,
            'Middle (7-15)': '#3b82f6',
            'Death (16-20)': '#ef4444'
        })
    fig_phase_wkts.update_layout(**DARK, title_font_color='#fff', showlegend=False)

    fig_venue_runs = px.bar(venue_stats, x='Venue', y='Avg_Runs',
        title='🏟️ Average Runs Scored per Venue (Batting Friendly)',
        color='Avg_Runs', color_continuous_scale=[[0,'#1e2a3a'],[1,GOLD]])
    fig_venue_runs.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_venue_runs.update_xaxes(tickangle=-45)

    fig_venue_wkts = px.bar(venue_wkts2, x='Venue', y='Avg_Wickets',
        title='🎳 Average Wickets per Venue (Bowling Friendly)',
        color='Avg_Wickets', color_continuous_scale=[[0,'#1e2a3a'],[1,'#ef4444']])
    fig_venue_wkts.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_venue_wkts.update_xaxes(tickangle=-45)

    fig_pace_spin = px.bar(venue_bowler, x='Venue', y='Wickets', color='Bowler Type',
        title='🌀 Pace vs Spin Wickets at Top Venues',
        color_discrete_map={'Spin': GOLD, 'Pace': '#3b82f6'},
        barmode='group')
    fig_pace_spin.update_layout(**DARK, title_font_color='#fff')
    fig_pace_spin.update_xaxes(tickangle=-45)

    return html.Div([
        page_header("Phase & Pitch Intelligence", "Match phase analysis + venue and pitch statistics"),

        html.P("📊 Match Phase Analysis", style={'color': '#fff', 'fontWeight': '600', 'fontSize': '1rem', 'marginBottom': '16px'}),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_phase_runs,  config={'displayModeBar': False})), md=6),
            dbc.Col(chart_card(dcc.Graph(figure=fig_phase_wkts,  config={'displayModeBar': False})), md=6),
        ]),

        html.P("🏟️ Pitch & Venue Statistics", style={'color': '#fff', 'fontWeight': '600', 'fontSize': '1rem', 'marginBottom': '16px'}),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_venue_runs,  config={'displayModeBar': False}))),
        ]),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_venue_wkts,  config={'displayModeBar': False}))),
        ]),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_pace_spin,   config={'displayModeBar': False}))),
        ]),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 4 — CAREER RANKINGS
# ══════════════════════════════════════════════════════════
def career_rankings_layout():

    orange_cap = ORANGE_CAP
    purple_cap = PURPLE_CAP

    fig_runs = px.bar(top_runs, x='Runs', y='Player', orientation='h',
        title='🏏 All Time Top 15 Run Scorers',
        color='Runs', color_continuous_scale=[[0,'#1e2a3a'],[1,GOLD]])
    fig_runs.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_runs.update_yaxes(categoryorder='total ascending')

    fig_wkts = px.bar(top_wkts, x='Wickets', y='Player', orientation='h',
        title='🎯 All Time Top 15 Wicket Takers',
        color='Wickets', color_continuous_scale=[[0,'#1e2a3a'],[1,'#ef4444']])
    fig_wkts.update_layout(**DARK, title_font_color='#fff', coloraxis_showscale=False)
    fig_wkts.update_yaxes(categoryorder='total ascending')

    # Default player for timeline
    default_player = 'V Kohli' if 'V Kohli' in PLAYERS else PLAYERS[0]

    return html.Div([
        page_header("Career Rankings", "All time leaderboards, Orange Cap & Purple Cap winners"),

        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_runs, config={'displayModeBar': False})), md=6),
            dbc.Col(chart_card(dcc.Graph(figure=fig_wkts, config={'displayModeBar': False})), md=6),
        ]),

        # Orange & Purple Cap Tables
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("🟠 Orange Cap Winners (Season)", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '1rem', 'marginBottom': '16px'}),
                    html.Div([
                        html.Div([
                            html.Span(str(int(row['Season'])), style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.85rem', 'width': '50px', 'display': 'inline-block'}),
                            html.Span(row['Player'], style={'color': '#e5e7eb', 'fontSize': '0.85rem', 'width': '200px', 'display': 'inline-block'}),
                            html.Span(f"{int(row['Runs'])} runs", style={'color': '#9ca3af', 'fontSize': '0.8rem'}),
                        ], style={'padding': '8px 0', 'borderBottom': '1px solid #1e2a3a'})
                        for _, row in orange_cap.iterrows()
                    ]),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '12px', 'padding': '20px'}),
            ], md=6),

            dbc.Col([
                html.Div([
                    html.P("🟣 Purple Cap Winners (Season)", style={'color': '#a855f7', 'fontWeight': '700', 'fontSize': '1rem', 'marginBottom': '16px'}),
                    html.Div([
                        html.Div([
                            html.Span(str(int(row['Season'])), style={'color': '#a855f7', 'fontWeight': '700', 'fontSize': '0.85rem', 'width': '50px', 'display': 'inline-block'}),
                            html.Span(row['Player'], style={'color': '#e5e7eb', 'fontSize': '0.85rem', 'width': '200px', 'display': 'inline-block'}),
                            html.Span(f"{int(row['Wickets'])} wkts", style={'color': '#9ca3af', 'fontSize': '0.8rem'}),
                        ], style={'padding': '8px 0', 'borderBottom': '1px solid #1e2a3a'})
                        for _, row in purple_cap.iterrows()
                    ]),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '12px', 'padding': '20px'}),
            ], md=6),
        ], className="mb-4"),

        # ── Player Career Timeline ─────────────────────────────
        html.Div([
            html.P("🗓️ Player Career Timeline", style={
                'color': '#fff', 'fontWeight': '700', 'fontSize': '1rem', 'marginBottom': '6px'
            }),
            html.P("Select a player to see which team they played for in each season",
                style={'color': '#6b7280', 'fontSize': '0.8rem', 'marginBottom': '16px'}),
            dcc.Dropdown(
                id='career-player',
                options=[{'label': p, 'value': p} for p in sorted(PLAYERS)],
                value=default_player,
                placeholder='Search player name...',
                style={'color': '#000000', 'marginBottom': '16px'},
                clearable=False,
            ),
            html.Div(id='career-timeline-result'),
        ], style={
            'background': '#0d1117', 'border': '1px solid #1e2a3a',
            'borderRadius': '12px', 'padding': '24px', 'marginTop': '8px'
        }),
    ])
# ══════════════════════════════════════════════════════════
# PAGE 5 — PLAYER COMPARISON
# ══════════════════════════════════════════════════════════
def player_comparison_layout():
    top_players = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
    top_players = top_players.sort_values('batsman_runs', ascending=False).head(100)['batter'].tolist()

    return html.Div([
        page_header("Player Comparison", "Compare two players side by side across all statistics"),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("🏏 Player 1", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                    dcc.Dropdown(id='compare-p1',
                        options=[{'label': p, 'value': p} for p in top_players],
                        value=top_players[0], clearable=False,
                        style={'color': '#000000'}),
                ], md=5),
                dbc.Col(html.Div("VS", style={'color': GOLD, 'fontWeight': '800', 'fontSize': '1.5rem', 'textAlign': 'center', 'paddingTop': '28px'}), md=2),
                dbc.Col([
                    html.Label("🏏 Player 2", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                    dcc.Dropdown(id='compare-p2',
                        options=[{'label': p, 'value': p} for p in top_players],
                        value=top_players[1], clearable=False,
                        style={'color': '#000000'}),
                ], md=5),
            ], className="mb-3"),
        ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '12px', 'padding': '24px', 'marginBottom': '24px'}),

        html.Div(id='comparison-result'),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 6 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════
def match_predictor_layout():
    return html.Div([
        page_header("Match Winner Predictor", "Random Forest Model • Trained on 1090 IPL matches"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("🔮 Predict Match Winner", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '1.2rem', 'marginBottom': '6px'}),
                    html.P("Select two teams, venue and toss details", style={'color': '#6b7280', 'fontSize': '0.85rem', 'marginBottom': '24px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("🏠 Team 1", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='team1', options=[{'label': t, 'value': t} for t in ACTIVE_TEAMS],
                                value=ACTIVE_TEAMS[0], clearable=False,
                                style={'color': '#000000'}),
                        ], md=6),
                        dbc.Col([
                            html.Label("✈️ Team 2", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='team2', options=[{'label': t, 'value': t} for t in ACTIVE_TEAMS],
                                value=ACTIVE_TEAMS[1], clearable=False,
                                style={'color': '#000000'}),
                        ], md=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("🏟️ Venue", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='venue', options=[{'label': v, 'value': v} for v in VENUES],
                                value=VENUES[0], clearable=False,
                                style={'color': '#000000'}),
                        ], md=12),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("🪙 Toss Winner", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='toss-winner', options=[{'label': t, 'value': t} for t in ACTIVE_TEAMS],
                                value=ACTIVE_TEAMS[0], clearable=False,
                                style={'color': '#000000'}),
                        ], md=6),
                        dbc.Col([
                            html.Label("🏏 Toss Decision", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='toss-decision',
                                options=[{'label': 'Bat First', 'value': 'bat'}, {'label': 'Field First', 'value': 'field'}],
                                value='bat', clearable=False,
                                style={'color': '#000000'}),
                        ], md=6),
                    ], className="mb-3"),
                    html.Button("🔮 Predict Winner", id='predict-btn', n_clicks=0, style={
                        'background': f'linear-gradient(135deg, {GOLD}, #d97706)', 'border': 'none',
                        'color': '#000', 'fontWeight': '700', 'padding': '12px 32px',
                        'borderRadius': '8px', 'width': '100%', 'marginTop': '8px',
                        'fontSize': '0.95rem', 'cursor': 'pointer'
                    }),
                    html.Div(id='prediction-result'),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '32px'}),
            ], md=7),
            dbc.Col([
                html.Div([
                    html.P("📊 Model Info", style={'color': '#fff', 'fontWeight': '600', 'marginBottom': '16px'}),
                    info_row("Algorithm",  "Random Forest"),
                    info_row("Trees",      "200 estimators"),
                    info_row("Accuracy",   "52.75%"),
                    info_row("Training",   "1090 matches"),
                    info_row("Features",   "11 features"),
                    html.Hr(style={'borderColor': '#1e2a3a', 'margin': '16px 0'}),
                    *[html.P(f"→ {f}", style={'color': '#6b7280', 'fontSize': '0.78rem', 'marginBottom': '4px'})
                      for f in ['Team 1 & Team 2','Venue','Toss Winner','Toss Decision',
                                'Team Win Rates','Head-to-Head','Venue Win Rate','Recent Form']],
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '24px'}),
            ], md=5),
        ]),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 7 — SCORE PREDICTOR
# ══════════════════════════════════════════════════════════
def score_predictor_layout():
    return html.Div([
        page_header("Score Predictor", "Gradient Boosting Model • Predicts first innings total"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("🏏 Predict First Innings Score", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '1.2rem', 'marginBottom': '6px'}),
                    html.P("Select batting team, venue and wickets to predict final score", style={'color': '#6b7280', 'fontSize': '0.85rem', 'marginBottom': '24px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("🏏 Batting Team", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='batting-team',
                                options=[{'label': t, 'value': t} for t in ACTIVE_TEAMS],
                                value=ACTIVE_TEAMS[0], clearable=False,
                                style={'color': '#000000'}),
                        ], md=12),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("🏟️ Venue", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='score-venue',
                                options=[{'label': v, 'value': v} for v in VENUES],
                                value=VENUES[0], clearable=False,
                                style={'color': '#000000'}),
                        ], md=12),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("❌ Wickets Fallen", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Slider(id='wickets', min=0, max=9, step=1, value=3,
                                marks={i: {'label': str(i), 'style': {'color': '#9ca3af'}} for i in range(10)},
                                tooltip={"placement": "bottom", "always_visible": True}),
                        ], md=12),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("📅 Season", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                            dcc.Dropdown(id='score-season',
                                options=[{'label': str(s), 'value': s} for s in range(2008, 2025)],
                                value=2024, clearable=False,
                                style={'color': '#000000'}),
                        ], md=12),
                    ], className="mb-3"),
                    html.Button("🏏 Predict Score", id='score-btn', n_clicks=0, style={
                        'background': f'linear-gradient(135deg, {GOLD}, #d97706)', 'border': 'none',
                        'color': '#000', 'fontWeight': '700', 'padding': '12px 32px',
                        'borderRadius': '8px', 'width': '100%', 'marginTop': '8px',
                        'fontSize': '0.95rem', 'cursor': 'pointer'
                    }),
                    html.Div(id='score-result'),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '32px'}),
            ], md=7),
            dbc.Col([
                html.Div([
                    html.P("📊 Model Info", style={'color': '#fff', 'fontWeight': '600', 'marginBottom': '16px'}),
                    info_row("Algorithm",  "Gradient Boosting"),
                    info_row("Estimators", "200"),
                    info_row("MAE",        "~22 runs"),
                    info_row("Training",   "1090 innings"),
                    info_row("Output",     "Score range ±15"),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '24px'}),
            ], md=5),
        ]),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 8 — PERFORMANCE PREDICTOR (ML)
# ══════════════════════════════════════════════════════════
def performance_predictor_layout():
    top_players = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
    top_players = top_players.sort_values('batsman_runs', ascending=False).head(50)['batter'].tolist()

    return html.Div([
        page_header("Performance Predictor", "ML model predicts a player's next season run tally"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("📈 Predict Next Season Performance", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '1.2rem', 'marginBottom': '6px'}),
                    html.P("Select a player to forecast their next season runs based on career trend", style={'color': '#6b7280', 'fontSize': '0.85rem', 'marginBottom': '24px'}),
                    html.Label("👤 Select Player", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                    dcc.Dropdown(id='perf-player',
                        options=[{'label': p, 'value': p} for p in top_players],
                        value=top_players[0], clearable=False,
                        style={'color': '#000000', 'marginBottom': '16px'}),
                    html.Button("📈 Predict Performance", id='perf-btn', n_clicks=0, style={
                        'background': f'linear-gradient(135deg, {GOLD}, #d97706)', 'border': 'none',
                        'color': '#000', 'fontWeight': '700', 'padding': '12px 32px',
                        'borderRadius': '8px', 'width': '100%', 'fontSize': '0.95rem', 'cursor': 'pointer'
                    }),
                    html.Div(id='perf-result'),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '32px'}),
            ], md=7),
            dbc.Col([
                html.Div([
                    html.P("📊 Model Info", style={'color': '#fff', 'fontWeight': '600', 'marginBottom': '16px'}),
                    info_row("Algorithm",  "Linear Regression"),
                    info_row("Features",   "Season-wise runs"),
                    info_row("Method",     "Career trend + weighted avg"),
                    info_row("Output",     "Predicted runs range"),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '24px'}),
            ], md=5),
        ]),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 9 — SIMILAR PLAYER FINDER (ML)
# ══════════════════════════════════════════════════════════
def similar_player_layout():
    top_players = deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
    top_players = top_players.sort_values('batsman_runs', ascending=False).head(100)['batter'].tolist()

    return html.Div([
        page_header("Similar Player Finder", "KMeans Clustering ML model finds players with similar profiles"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("🔍 Find Similar Players", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '1.2rem', 'marginBottom': '6px'}),
                    html.P("Select a player to find others with similar batting profiles using clustering", style={'color': '#6b7280', 'fontSize': '0.85rem', 'marginBottom': '24px'}),
                    html.Label("👤 Select Player", style={'color': '#9ca3af', 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                    dcc.Dropdown(id='similar-player',
                        options=[{'label': p, 'value': p} for p in top_players],
                        value=top_players[0], clearable=False,
                        style={'color': '#000000', 'marginBottom': '16px'}),
                    html.Button("🔍 Find Similar Players", id='similar-btn', n_clicks=0, style={
                        'background': f'linear-gradient(135deg, {GOLD}, #d97706)', 'border': 'none',
                        'color': '#000', 'fontWeight': '700', 'padding': '12px 32px',
                        'borderRadius': '8px', 'width': '100%', 'fontSize': '0.95rem', 'cursor': 'pointer'
                    }),
                    html.Div(id='similar-result'),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '32px'}),
            ], md=7),
            dbc.Col([
                html.Div([
                    html.P("📊 Model Info", style={'color': '#fff', 'fontWeight': '600', 'marginBottom': '16px'}),
                    info_row("Algorithm",  "KMeans Clustering"),
                    info_row("Clusters",   "8 player archetypes"),
                    info_row("Features",   "Runs, SR, Avg, 4s, 6s"),
                    info_row("Output",     "5 most similar players"),
                ], style={'background': '#0d1117', 'border': '1px solid #1e2a3a', 'borderRadius': '16px', 'padding': '24px'}),
            ], md=5),
        ]),
    ])

# ══════════════════════════════════════════════════════════
# PAGE 10 — TOSS ANALYSIS
# ══════════════════════════════════════════════════════════
def toss_analysis_layout():
    toss_dec = matches['toss_decision'].value_counts().reset_index()
    toss_dec.columns = ['Decision', 'Count']

    toss_impact = matches[matches['winner'] != 'No Result']['toss_match_winner'].value_counts().reset_index()
    toss_impact.columns = ['Toss Won Match', 'Count']
    toss_impact['Toss Won Match'] = toss_impact['Toss Won Match'].map({1: 'Toss Winner Won', 0: 'Toss Winner Lost'})

    toss_season = matches.groupby(['season','toss_decision']).size().reset_index()
    toss_season.columns = ['Season','Decision','Count']

    bat_first = matches[matches['winner'] != 'No Result']['batting_first_won'].value_counts()
    chase_win = bat_first.get(0, 0)
    bat_win   = bat_first.get(1, 0)
    total     = bat_win + chase_win

    fig_toss_pie = go.Figure(data=[go.Pie(
        labels=toss_impact['Toss Won Match'], values=toss_impact['Count'],
        hole=0.6, marker_colors=[GOLD, '#1e2a3a'],
    )])
    fig_toss_pie.update_layout(**DARK, title='🪙 Does Toss Winner Win the Match?', title_font_color='#fff')

    fig_dec = px.bar(toss_dec, x='Decision', y='Count',
        title='🏏 Toss Decision Preference',
        color='Decision', color_discrete_map={'bat': GOLD, 'field': '#3b82f6'})
    fig_dec.update_layout(**DARK, title_font_color='#fff', showlegend=False)

    fig_season_toss = px.bar(toss_season, x='Season', y='Count', color='Decision',
        title='📅 Toss Decision Trend by Season',
        color_discrete_map={'bat': GOLD, 'field': '#3b82f6'}, barmode='group')
    fig_season_toss.update_layout(**DARK, title_font_color='#fff')

    return html.Div([
        page_header("Toss Impact Analysis", "Does the toss actually matter? Let the data decide."),
        dbc.Row([
            dbc.Col(stat_card("🪙", f"{round(554/1090*100,1)}%", "Toss Winners Won",  "Out of 1090 matches"), md=3),
            dbc.Col(stat_card("🏏", f"{round(bat_win/total*100,1)}%",   "Batting First Won", "First innings"), md=3),
            dbc.Col(stat_card("🎯", f"{round(chase_win/total*100,1)}%", "Chasing Won",        "Chase favored"), md=3),
            dbc.Col(stat_card("📊", "61.47%", "Toss Model Accuracy", "Logistic Regression"), md=3),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_toss_pie,    config={'displayModeBar': False})), md=5),
            dbc.Col(chart_card(dcc.Graph(figure=fig_dec,         config={'displayModeBar': False})), md=7),
        ]),
        dbc.Row([
            dbc.Col(chart_card(dcc.Graph(figure=fig_season_toss, config={'displayModeBar': False}))),
        ]),
        html.Div([
            html.P("🤖 ML Insight", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '1rem', 'marginBottom': '8px'}),
            html.P("The Logistic Regression model achieved 61.47% accuracy — better than random (50%). "
                   "While toss has some influence, team quality and player form remain far stronger predictors of match outcomes.",
                   style={'color': '#9ca3af', 'fontSize': '0.875rem', 'lineHeight': '1.6'}),
        ], style={'background': '#0d1117', 'border': f'1px solid rgba(245,158,11,0.3)', 'borderRadius': '12px', 'padding': '24px'}),
    ])

# ══════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "IPL ML Dashboard"
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Overlay for mobile — clicking it closes the sidebar
    html.Div(id='sidebar-overlay', n_clicks=0, style={
        'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0',
        'width': '100vw', 'height': '100vh', 'background': 'rgba(0,0,0,0.5)',
        'zIndex': '999'
    }),
    sidebar,
    html.Div([
        html.Div([
            # Hamburger button — only visible on mobile
            html.Button("☰", id="open-sidebar", n_clicks=0, style={
                'background': 'none', 'border': 'none', 'color': '#fff',
                'fontSize': '1.4rem', 'cursor': 'pointer', 'marginRight': '12px',
                'display': 'none'
            }, className="hamburger-btn"),
            html.Span("IPL Statistics & ML Dashboard", style={'color': '#fff', 'fontWeight': '600', 'fontSize': '1rem'}),
            html.Span("2008 – 2024", style={
                'background': f'linear-gradient(135deg, {GOLD}, #d97706)',
                'color': '#000', 'padding': '4px 12px', 'borderRadius': '20px',
                'fontSize': '0.75rem', 'fontWeight': '700'
            }),
        ], style={
            'background': 'linear-gradient(90deg, #0d1117, #161b27)',
            'borderBottom': '1px solid #1e2a3a', 'padding': '16px 32px',
            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
            'position': 'sticky', 'top': '0', 'zIndex': '998'
        }),
        html.Div(id='page-content', style={'padding': '32px'}),
    ], id='main-content', style={'marginLeft': '240px', 'minHeight': '100vh', 'background': '#0a0e1a'}),
], style={'fontFamily': 'Inter, sans-serif'})

# ══════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════

# ── Sidebar toggle (works on both mobile and desktop) ──────
@app.callback(
    Output('sidebar', 'style'),
    Output('sidebar-overlay', 'style'),
    Output('main-content', 'style'),
    Input('open-sidebar', 'n_clicks'),
    Input('close-sidebar', 'n_clicks'),
    Input('sidebar-overlay', 'n_clicks'),
    Input('url', 'pathname'),
    State('sidebar', 'style'),
    prevent_initial_call=True
)
def toggle_sidebar(open_clicks, close_clicks, overlay_clicks, pathname, current_style):
    from dash import ctx
    trigger = ctx.triggered_id

    sidebar_base = {
        'position': 'fixed', 'top': '0', 'height': '100vh', 'width': '240px',
        'background': 'linear-gradient(180deg, #0d1117 0%, #161b27 100%)',
        'borderRight': '1px solid #1e2a3a', 'zIndex': '1000', 'overflowY': 'auto',
        'transition': 'left 0.3s ease'
    }
    overlay_hidden = {'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0',
                      'width': '100vw', 'height': '100vh', 'background': 'rgba(0,0,0,0.5)', 'zIndex': '999'}
    overlay_shown  = {**overlay_hidden, 'display': 'block'}

    sidebar_open   = {**sidebar_base, 'left': '0'}
    sidebar_closed = {**sidebar_base, 'left': '-240px'}
    content_with_sidebar    = {'marginLeft': '240px', 'minHeight': '100vh', 'background': '#0a0e1a', 'transition': 'margin-left 0.3s ease'}
    content_without_sidebar = {'marginLeft': '0',     'minHeight': '100vh', 'background': '#0a0e1a', 'transition': 'margin-left 0.3s ease'}

    if trigger == 'open-sidebar':
        return sidebar_open, overlay_shown, content_without_sidebar
    else:
        # close on: close button, overlay click, or page navigation
        return sidebar_closed, overlay_hidden, content_without_sidebar

@app.callback(Output('page-content','children'), Input('url','pathname'))
def display_page(pathname):
    if pathname == '/team-intelligence':      return team_intelligence_layout()
    elif pathname == '/phase-intelligence':   return phase_intelligence_layout()
    elif pathname == '/career-rankings':      return career_rankings_layout()
    elif pathname == '/player-comparison':    return player_comparison_layout()
    elif pathname == '/match-predictor':      return match_predictor_layout()
    elif pathname == '/score-predictor':      return score_predictor_layout()
    elif pathname == '/performance-predictor':return performance_predictor_layout()
    elif pathname == '/similar-player':       return similar_player_layout()
    elif pathname == '/toss-analysis':        return toss_analysis_layout()
    else:                                     return dashboard_layout()

# ── Head to Head ───────────────────────────────────────────
@app.callback(Output('h2h-result','children'),
    Input('h2h-team1','value'), Input('h2h-team2','value'))
def h2h_result(t1, t2):
    h2h = matches[
        ((matches['team1']==t1)&(matches['team2']==t2)) |
        ((matches['team1']==t2)&(matches['team2']==t1))
    ]
    h2h = h2h[h2h['winner'] != 'No Result']
    if len(h2h) == 0:
        return html.P("No matches found between these teams.", style={'color':'#6b7280'})

    t1_wins = len(h2h[h2h['winner']==t1])
    t2_wins = len(h2h[h2h['winner']==t2])
    total   = len(h2h)

    return dbc.Row([
        dbc.Col(html.Div([
            html.P(t1,      style={'color': GOLD,     'fontWeight': '700', 'fontSize': '1rem', 'textAlign': 'center'}),
            html.P(str(t1_wins), style={'color': '#fff', 'fontWeight': '800', 'fontSize': '2.5rem', 'textAlign': 'center', 'lineHeight': '1'}),
            html.P("Wins",  style={'color': '#6b7280', 'fontSize': '0.8rem', 'textAlign': 'center'}),
        ], style={'background': 'rgba(245,158,11,0.08)', 'borderRadius': '12px', 'padding': '20px'}), md=4),

        dbc.Col(html.Div([
            html.P("Total",  style={'color': '#9ca3af', 'fontWeight': '600', 'fontSize': '0.85rem', 'textAlign': 'center'}),
            html.P(str(total), style={'color': '#fff', 'fontWeight': '800', 'fontSize': '2.5rem', 'textAlign': 'center', 'lineHeight': '1'}),
            html.P("Matches", style={'color': '#6b7280', 'fontSize': '0.8rem', 'textAlign': 'center'}),
        ], style={'background': '#161b27', 'borderRadius': '12px', 'padding': '20px'}), md=4),

        dbc.Col(html.Div([
            html.P(t2,      style={'color': '#3b82f6', 'fontWeight': '700', 'fontSize': '1rem', 'textAlign': 'center'}),
            html.P(str(t2_wins), style={'color': '#fff', 'fontWeight': '800', 'fontSize': '2.5rem', 'textAlign': 'center', 'lineHeight': '1'}),
            html.P("Wins",  style={'color': '#6b7280', 'fontSize': '0.8rem', 'textAlign': 'center'}),
        ], style={'background': 'rgba(59,130,246,0.08)', 'borderRadius': '12px', 'padding': '20px'}), md=4),
    ])

# ── Match Predictor ────────────────────────────────────────
@app.callback(Output('prediction-result','children'),
    Input('predict-btn','n_clicks'),
    State('team1','value'), State('team2','value'),
    State('venue','value'), State('toss-winner','value'),
    State('toss-decision','value'), prevent_initial_call=True)
def predict_match(n, team1, team2, venue, toss_winner, toss_decision):
    try:
        team_wins  = matches[matches['winner']!='No Result']['winner'].value_counts()
        team_total = pd.Series(matches['team1'].tolist()+matches['team2'].tolist()).value_counts()
        winrate    = (team_wins/team_total).fillna(0)
        t1_wr = winrate.get(team1, 0.5)
        t2_wr = winrate.get(team2, 0.5)

        h2h = matches[
            ((matches['team1']==team1)&(matches['team2']==team2))|
            ((matches['team1']==team2)&(matches['team2']==team1))
        ]
        h2h_wr = len(h2h[h2h['winner']==team1])/len(h2h) if len(h2h)>0 else 0.5
        vm = matches[matches['venue']==venue]
        venue_wr = len(vm[vm['winner']==team1])/len(vm) if len(vm)>0 else 0.5

        def safe_enc(enc, val):
            return enc.transform([val])[0] if val in enc.classes_ else 0

        row = pd.DataFrame([{
            'team1': safe_enc(encoders['team1'],team1),
            'team2': safe_enc(encoders['team2'],team2),
            'venue': safe_enc(encoders['venue'],venue),
            'toss_winner': safe_enc(encoders['toss_winner'],toss_winner),
            'toss_decision': safe_enc(encoders['toss_decision'],toss_decision),
            'team1_winrate': t1_wr, 'team2_winrate': t2_wr,
            'h2h_winrate': h2h_wr, 'team1_venue_winrate': venue_wr,
            'team1_form': 0.5, 'team2_form': 0.5,
        }])

        proba   = rf_model.predict_proba(row)[0]
        classes = encoders['winner'].classes_
        pred    = classes[np.argmax(proba)]
        conf    = proba[np.argmax(proba)]*100
        t1_prob = proba[list(classes).index(team1)]*100 if team1 in classes else 50
        t2_prob = proba[list(classes).index(team2)]*100 if team2 in classes else 50

        return html.Div([
            html.P("🏆 Predicted Winner", style={'color':'#9ca3af','fontSize':'0.85rem','marginBottom':'8px'}),
            html.P(pred, style={'color':GOLD,'fontSize':'2rem','fontWeight':'800','marginBottom':'4px'}),
            html.P(f"Confidence: {conf:.1f}%", style={'color':'#6b7280','fontSize':'0.85rem','marginBottom':'20px'}),
            html.Div([
                html.Div([html.Span(team1,style={'color':'#e5e7eb','fontSize':'0.8rem'}),html.Span(f"{t1_prob:.1f}%",style={'color':GOLD,'fontSize':'0.8rem','fontWeight':'700'})],style={'display':'flex','justifyContent':'space-between','marginBottom':'6px'}),
                html.Div(html.Div(style={'width':f'{t1_prob:.1f}%','height':'8px','background':f'linear-gradient(90deg,{GOLD},#d97706)','borderRadius':'4px'}),style={'background':'#1e2a3a','borderRadius':'4px','marginBottom':'12px'}),
                html.Div([html.Span(team2,style={'color':'#e5e7eb','fontSize':'0.8rem'}),html.Span(f"{t2_prob:.1f}%",style={'color':'#3b82f6','fontSize':'0.8rem','fontWeight':'700'})],style={'display':'flex','justifyContent':'space-between','marginBottom':'6px'}),
                html.Div(html.Div(style={'width':f'{t2_prob:.1f}%','height':'8px','background':'linear-gradient(90deg,#3b82f6,#2563eb)','borderRadius':'4px'}),style={'background':'#1e2a3a','borderRadius':'4px'}),
            ]),
        ], style={'background':'rgba(245,158,11,0.08)','border':'1px solid rgba(245,158,11,0.3)','borderRadius':'12px','padding':'24px','marginTop':'20px','textAlign':'center'})
    except Exception as e:
        return html.Div(f"⚠️ Error: {str(e)}", style={'color':'#ef4444','marginTop':'16px'})

# ── Score Predictor ────────────────────────────────────────
@app.callback(Output('score-result','children'),
    Input('score-btn','n_clicks'),
    State('batting-team','value'), State('score-venue','value'),
    State('wickets','value'), State('score-season','value'),
    prevent_initial_call=True)
def predict_score(n, batting_team, venue, wickets, season):
    try:
        def safe_enc(enc, val):
            return enc.transform([val])[0] if val in enc.classes_ else 0

        row = pd.DataFrame([{
            'venue': safe_enc(score_encoders['venue'],venue),
            'batting_team': safe_enc(score_encoders['batting_team'],batting_team),
            'total_wickets': wickets, 'season': season,
        }])
        pred = gb_model.predict(row)[0]
        low  = max(0, int(pred)-15)
        high = int(pred)+15

        return html.Div([
            html.P("🏏 Predicted Score", style={'color':'#9ca3af','fontSize':'0.85rem','marginBottom':'8px'}),
            html.P(str(int(pred)), style={'color':GOLD,'fontSize':'3rem','fontWeight':'800','lineHeight':'1'}),
            html.P(f"Range: {low} – {high}", style={'color':'#6b7280','fontSize':'0.9rem','marginTop':'8px'}),
            html.Hr(style={'borderColor':'#1e2a3a','margin':'16px 0'}),
            html.P(f"📍 {venue}", style={'color':'#9ca3af','fontSize':'0.8rem'}),
            html.P(f"🏏 {batting_team}  •  ❌ {wickets} wkts  •  📅 {season}", style={'color':'#9ca3af','fontSize':'0.8rem','marginTop':'4px'}),
        ], style={'background':'rgba(245,158,11,0.08)','border':'1px solid rgba(245,158,11,0.3)','borderRadius':'12px','padding':'24px','marginTop':'20px','textAlign':'center'})
    except Exception as e:
        return html.Div(f"⚠️ Error: {str(e)}", style={'color':'#ef4444','marginTop':'16px'})

# ── Player Comparison ──────────────────────────────────────
@app.callback(Output('comparison-result','children'),
    Input('compare-p1','value'), Input('compare-p2','value'))
def compare_players(p1, p2):
    if p1 == p2:
        return html.P("⚠️ Please select two different players.",
            style={'color': GOLD, 'marginTop': '16px', 'fontSize': '0.9rem'})
    
    del_m = DEL_SEASON.copy()

    def get_stats(player):
        bat   = del_m[del_m['batter']==player]
        runs  = int(bat['batsman_runs'].sum())
        balls = len(bat)
        fours = int((bat['batsman_runs']==4).sum())
        sixes = int((bat['batsman_runs']==6).sum())
        inns  = bat['match_id'].nunique()
        avg   = round(runs/max(inns,1),1)
        sr    = round(runs/max(balls,1)*100,1)
        return {'Runs':runs,'Innings':inns,'Avg':avg,'SR':sr,'4s':fours,'6s':sixes}

    s1 = get_stats(p1)
    s2 = get_stats(p2)

    p1_season = del_m[del_m['batter']==p1].groupby('season')['batsman_runs'].sum().reset_index()
    p2_season = del_m[del_m['batter']==p2].groupby('season')['batsman_runs'].sum().reset_index()
    p1_season['Player'] = p1
    p2_season['Player'] = p2
    combined = pd.concat([p1_season, p2_season])
    combined.columns = ['Season','Runs','Player']

    fig = px.line(combined, x='Season', y='Runs', color='Player',
        title='📈 Season-wise Runs Comparison',
        color_discrete_map={p1: GOLD, p2: '#3b82f6'},
        markers=True)
    fig.update_layout(**DARK, title_font_color='#fff')

    stats_keys = ['Runs','Innings','Avg','SR','4s','6s']
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.P(p1, style={'color':GOLD,'fontWeight':'700','fontSize':'1rem','textAlign':'center','marginBottom':'16px'}),
                *[html.Div([
                    html.Span(k, style={'color':'#6b7280','fontSize':'0.8rem','width':'80px','display':'inline-block'}),
                    html.Span(str(s1[k]), style={'color':'#fff','fontWeight':'700','fontSize':'0.9rem'}),
                ], style={'marginBottom':'8px'}) for k in stats_keys],
            ], style={'background':'rgba(245,158,11,0.08)','border':'1px solid rgba(245,158,11,0.3)','borderRadius':'12px','padding':'20px'}), md=5),

            dbc.Col(html.Div("VS", style={'color':GOLD,'fontWeight':'800','fontSize':'1.5rem','textAlign':'center','paddingTop':'40px'}), md=2),

            dbc.Col(html.Div([
                html.P(p2, style={'color':'#3b82f6','fontWeight':'700','fontSize':'1rem','textAlign':'center','marginBottom':'16px'}),
                *[html.Div([
                    html.Span(k, style={'color':'#6b7280','fontSize':'0.8rem','width':'80px','display':'inline-block'}),
                    html.Span(str(s2[k]), style={'color':'#fff','fontWeight':'700','fontSize':'0.9rem'}),
                ], style={'marginBottom':'8px'}) for k in stats_keys],
            ], style={'background':'rgba(59,130,246,0.08)','border':'1px solid rgba(59,130,246,0.3)','borderRadius':'12px','padding':'20px'}), md=5),
        ], className="mb-4"),
        chart_card(dcc.Graph(figure=fig, config={'displayModeBar':False})),
    ])
# ── Performance Predictor ──────────────────────────────────
@app.callback(Output('perf-result','children'),
    Input('perf-btn','n_clicks'),
    State('perf-player','value'), prevent_initial_call=True)
def predict_performance(n, player):
    try:
        from sklearn.linear_model import LinearRegression
        del_m    = deliveries.merge(matches[['match_id','season']], on='match_id', how='left')
        p_data   = del_m[del_m['batter']==player].groupby('season')['batsman_runs'].sum().reset_index()
        p_data.columns = ['Season','Runs']

        if len(p_data) < 2:
            return html.P("Not enough data for this player.", style={'color':'#6b7280','marginTop':'16px'})

        X = p_data['Season'].values.reshape(-1,1)
        y = p_data['Runs'].values
        model = LinearRegression()
        model.fit(X,y)
        next_season = matches['season'].max() + 1
        pred = max(0, int(model.predict([[next_season]])[0]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_data['Season'], y=p_data['Runs'],
            mode='lines+markers', name='Actual', line=dict(color=GOLD)))
        fig.add_trace(go.Scatter(x=[next_season], y=[pred],
            mode='markers', name=f'Predicted {next_season}',
            marker=dict(color='#ef4444', size=12, symbol='star')))
        fig.update_layout(**DARK, title=f'📈 {player} — Career Trend & Forecast', title_font_color='#fff')

        return html.Div([
            html.Div([
                html.P(f"Predicted Runs in {next_season}", style={'color':'#9ca3af','fontSize':'0.85rem','marginBottom':'8px'}),
                html.P(str(pred), style={'color':GOLD,'fontSize':'3rem','fontWeight':'800','lineHeight':'1'}),
                html.P(f"Range: {max(0,pred-150)} – {pred+150}", style={'color':'#6b7280','fontSize':'0.85rem','marginTop':'8px'}),
            ], style={'background':'rgba(245,158,11,0.08)','border':'1px solid rgba(245,158,11,0.3)','borderRadius':'12px','padding':'24px','marginTop':'20px','textAlign':'center','marginBottom':'16px'}),
            chart_card(dcc.Graph(figure=fig, config={'displayModeBar':False})),
        ])
    except Exception as e:
        return html.Div(f"⚠️ Error: {str(e)}", style={'color':'#ef4444','marginTop':'16px'})

# ── Similar Player Finder ──────────────────────────────────
@app.callback(Output('similar-result','children'),
    Input('similar-btn','n_clicks'),
    State('similar-player','value'), prevent_initial_call=True)
def find_similar(n, player):
    try:
        del_m = deliveries.merge(matches[['match_id','season']], on='match_id', how='left')
        stats = del_m.groupby('batter').agg(
            Runs   = ('batsman_runs','sum'),
            Balls  = ('batsman_runs','count'),
            Fours  = ('batsman_runs', lambda x: (x==4).sum()),
            Sixes  = ('batsman_runs', lambda x: (x==6).sum()),
            Innings= ('match_id','nunique'),
        ).reset_index()
        stats['SR']  = (stats['Runs']/stats['Balls']*100).round(1)
        stats['Avg'] = (stats['Runs']/stats['Innings']).round(1)
        stats = stats[stats['Runs'] >= 200].copy()

        features = ['Runs','SR','Avg','Fours','Sixes']
        X = stats[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        stats['Cluster'] = kmeans.fit_predict(X_scaled)

        if player not in stats['batter'].values:
            return html.P("Player not found in dataset.", style={'color':'#6b7280','marginTop':'16px'})

        player_cluster = stats[stats['batter']==player]['Cluster'].values[0]
        similar = stats[
            (stats['Cluster']==player_cluster) &
            (stats['batter']!=player)
        ].sort_values('Runs', ascending=False).head(5)

        player_stats = stats[stats['batter']==player].iloc[0]

        return html.Div([
            html.P(f"🎯 Players similar to {player}", style={'color':'#fff','fontWeight':'600','fontSize':'1rem','marginTop':'20px','marginBottom':'16px'}),
            html.Div([
                html.Div([
                    html.Span(f"👤 {row['batter']}", style={'color':GOLD,'fontWeight':'600','fontSize':'0.9rem','display':'block','marginBottom':'4px'}),
                    html.Span(f"Runs: {int(row['Runs'])}  •  SR: {row['SR']}  •  Avg: {row['Avg']}  •  6s: {int(row['Sixes'])}", style={'color':'#9ca3af','fontSize':'0.8rem'}),
                ], style={'background':'#161b27','border':'1px solid #1e2a3a','borderRadius':'8px','padding':'16px','marginBottom':'8px'})
                for _, row in similar.iterrows()
            ]),
        ])
    except Exception as e:
        return html.Div(f"⚠️ Error: {str(e)}", style={'color':'#ef4444','marginTop':'16px'})

# ══════════════════════════════════════════════════════════
# ── Player Career Timeline Callback ───────────────────────
@app.callback(
    Output('career-timeline-result', 'children'),
    Input('career-player', 'value')
)
def player_career_timeline(player):
    if not player:
        return html.P("Select a player above.", style={'color': '#6b7280'})
    try:
        del_m = deliveries.merge(
            matches[['match_id', 'season', 'team1', 'team2']],
            on='match_id', how='left'
        )

        player_data = del_m[del_m['batter'] == player].copy()

        if len(player_data) == 0:
            return html.P(f"No data found for {player}.", style={'color': '#6b7280'})

        season_stats = player_data.groupby('season').agg(
            Runs    = ('batsman_runs', 'sum'),
            Balls   = ('batsman_runs', 'count'),
            Matches = ('match_id',     'nunique'),
            Team    = ('batting_team', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')
        ).reset_index()

        season_stats['SR']  = (season_stats['Runs'] / season_stats['Balls'] * 100).round(1)
        season_stats['Avg'] = (season_stats['Runs'] / season_stats['Matches']).round(1)
        season_stats = season_stats.sort_values('season')

        total_runs     = int(season_stats['Runs'].sum())
        total_matches  = int(season_stats['Matches'].sum())
        seasons_played = len(season_stats)
        teams_played   = season_stats['Team'].nunique()

        fig = px.bar(season_stats, x='season', y='Runs',
            color='Team',
            title=f'🏏 {player} — Runs Per Season by Team',
            text='Runs')
        fig.update_traces(textposition='outside', textfont_color='#9ca3af')
        fig.update_layout(**DARK, title_font_color='#fff')
        fig.update_xaxes(type='category', title='Season')
        fig.update_yaxes(title='Runs')

        return html.Div([
            # Summary stat cards
            dbc.Row([
                dbc.Col(stat_card("🏏", str(total_runs),    "Career Runs",   "All seasons combined"), md=3),
                dbc.Col(stat_card("🎮", str(total_matches), "Matches",       "IPL career"),            md=3),
                dbc.Col(stat_card("📅", str(seasons_played),"Seasons",       "Seasons played"),        md=3),
                dbc.Col(stat_card("🏟️", str(teams_played),  "Teams",         "Franchises represented"),md=3),
            ], className="mb-4"),

            # Season by season table
            html.Div([
                # Table header
                html.Div([
                    html.Span("Season",  style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.8rem', 'width': '80px',  'display': 'inline-block'}),
                    html.Span("Team",    style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.8rem', 'width': '220px', 'display': 'inline-block'}),
                    html.Span("Matches", style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.8rem', 'width': '80px',  'display': 'inline-block'}),
                    html.Span("Runs",    style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.8rem', 'width': '80px',  'display': 'inline-block'}),
                    html.Span("Avg",     style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.8rem', 'width': '80px',  'display': 'inline-block'}),
                    html.Span("SR",      style={'color': GOLD, 'fontWeight': '700', 'fontSize': '0.8rem', 'width': '70px',  'display': 'inline-block'}),
                ], style={'padding': '10px 0', 'borderBottom': f'2px solid {GOLD}', 'marginBottom': '8px'}),

                # Table rows
                *[html.Div([
                    html.Span(str(int(row['season'])),  style={'color': '#e5e7eb', 'fontSize': '0.85rem', 'fontWeight': '600', 'width': '80px',  'display': 'inline-block'}),
                    html.Span(row['Team'],               style={'color': GOLD,     'fontSize': '0.85rem',                      'width': '220px', 'display': 'inline-block'}),
                    html.Span(str(int(row['Matches'])),  style={'color': '#9ca3af','fontSize': '0.85rem',                      'width': '80px',  'display': 'inline-block'}),
                    html.Span(str(int(row['Runs'])),     style={'color': '#fff',   'fontSize': '0.85rem', 'fontWeight': '700', 'width': '80px',  'display': 'inline-block'}),
                    html.Span(str(row['Avg']),           style={'color': '#9ca3af','fontSize': '0.85rem',                      'width': '80px',  'display': 'inline-block'}),
                    html.Span(str(row['SR']),            style={'color': '#9ca3af','fontSize': '0.85rem',                      'width': '70px',  'display': 'inline-block'}),
                ], style={'padding': '10px 0', 'borderBottom': '1px solid #1e2a3a'})
                for _, row in season_stats.iterrows()],

            ], style={'overflowX': 'auto', 'marginBottom': '24px'}),

            # Bar chart
            chart_card(dcc.Graph(figure=fig, config={'displayModeBar': False})),
        ])

    except Exception as e:
        return html.Div(f"⚠️ Error: {str(e)}", style={'color': '#ef4444', 'marginTop': '16px'})

if __name__ == '__main__':
    app.run(debug=True)