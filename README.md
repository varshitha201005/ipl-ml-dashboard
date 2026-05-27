---
title: IPL ML Dashboard
emoji: 🏏
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
---
# 🏏 IPL Statistics & ML Dashboard

> **Analyze. Predict. Dominate.**

A full-stack IPL Cricket Analytics Dashboard that helps fans, analysts and enthusiasts explore 16 seasons of IPL data and predict match outcomes using Machine Learning.

---

## 🚀 Live Demo

🌐 **Web App:** [Click here to open IPL ML Dashboard](https://huggingface.co/spaces/Varshitha20/ipl-ml-dashboard)

---

## 📌 Features

- 🏠 **Home Dashboard** — KPI cards, top scorers, wicket takers, season champions
- 🏆 **Team Intelligence** — Win rates, titles, season-wise performance, head-to-head stats
- 🏟️ **Phase & Pitch Stats** — Powerplay/Middle/Death analysis, venue stats, pace vs spin breakdown
- 📊 **Career Rankings** — Top 15 run scorers, wicket takers, Orange & Purple cap winners per season
- ⚔️ **Player Comparison** — Side-by-side player stats with season-wise runs chart
- 🔮 **Match Predictor** — ML model predicts match winner with win probabilities
- 🏏 **Score Predictor** — Predicts first innings total based on team, venue & wickets
- 📈 **Performance Predictor** — Forecasts a player's next season run tally using career trend
- 🔍 **Similar Player Finder** — Finds batters with similar profiles using KMeans clustering
- 🪙 **Toss Analysis** — Toss impact stats and logistic regression model insights

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Plotly Dash, Dash Bootstrap Components |
| Backend | Python, Flask (via Dash) |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly Express, Plotly Graph Objects |
| Deployment | Docker, Hugging Face Spaces |
| Version Control | Git & GitHub |

---

## 🤖 ML Models Used

| Model | Algorithm | Purpose |
|---|---|---|
| Match Predictor | Random Forest | Predict match winner |
| Score Predictor | Gradient Boosting | Predict first innings total |
| Performance Predictor | Linear Regression | Forecast next season runs |
| Similar Player Finder | KMeans Clustering | Find similar batting profiles |
| Toss Impact | Logistic Regression | Analyze toss effect on outcome |

---

## 📊 Dataset

- **Source:** IPL ball-by-ball and match data (2008–2024)
- **Matches:** 1,095 total across 16 seasons
- **Teams:** 15 franchises
- **Players:** 673 unique batters

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.11+
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/varshitha201005/ipl-ml-dashboard.git

# Navigate to project folder
cd ipl-ml-dashboard

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open your browser at `http://127.0.0.1:8050`

---

## 📂 Project Structure

```
ipl-ml-dashboard/
├── app.py                   # Main Dash application (1100+ lines)
├── clean_data.py            # Data preprocessing pipeline
├── train_models.py          # ML model training scripts
├── eda.py                   # Exploratory data analysis
├── Dockerfile               # Docker config for Hugging Face Spaces
├── requirements.txt         # Python dependencies
├── assets/
│   ├── dropdown.css         # Dark theme + mobile responsive CSS
│   └── style.css            # Global styles
├── data/
│   ├── matches_clean.csv    # Processed match data
│   └── deliveries_clean.csv # Processed ball-by-ball data
└── models/
    ├── match_predictor.pkl
    ├── score_predictor.pkl
    ├── encoders.pkl
    ├── score_encoders.pkl
    ├── meta.pkl
    └── toss_impact.pkl
```

---

## 👩‍💻 Developer

**Varshitha Sharigudam** — B.Tech Data Science Student, Mahatma Gandhi Institute Of Technology
📧 varshithasharigudam@gmail.com
💼 [LinkedIn](https://www.linkedin.com/in/varshitha-sharigudam-8b36722b8)
🐙 [GitHub](https://github.com/varshitha201005)
🌐 [Live Project](https://huggingface.co/spaces/Varshitha20/ipl-ml-dashboard)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Plotly Dash](https://dash.plotly.com) — Web framework
- [Scikit-learn](https://scikit-learn.org) — Machine learning
- [Hugging Face](https://huggingface.co) — Deployment platform
- [Kaggle](https://kaggle.com) — Dataset source