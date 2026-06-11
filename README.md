# 🏎️ F1 360° Analytics

> A full end-to-end Formula 1 data analytics pipeline — from raw telemetry collection to machine learning race predictions and an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit&logoColor=white)

---

## Overview

F1 360° Analytics covers the **entire 2023 Formula 1 World Championship** — all 22 races, 22 drivers, and 24,422 laps — using real telemetry data from the FastF1 API. The project is structured as a progressive pipeline:

1. **Data collection** — automated fetching and caching of the full season
2. **Exploratory analysis** — driver performance, consistency metrics, head-to-head battles
3. **Strategy analysis** — pit stop patterns, 1-stop vs 2-stop strategies, fastest crews
4. **Telemetry visualization** — speed, throttle & braking traces on actual circuit layouts
5. **ML race prediction** — XGBoost model that predicts finishing positions
6. **Interactive dashboard** — a Streamlit app bringing everything together

---

## Project Structure

```
f1-360-analytics/
│
├── notebooks/
│   ├── 01_data_collection.ipynb       # Fetch all 22 races via FastF1
│   ├── 02_driver_analysis.ipynb       # Lap times, consistency, comparisons
│   ├── 03_pit_stop_strategy.ipynb     # Pit stop analysis & strategy breakdowns
│   ├── 04_telemetry_analysis.ipynb    # Speed maps & telemetry traces
│   └── 05_race_predictor_ml.ipynb     # XGBoost race outcome predictor
│
├── data/
│   └── processed/                     # Cleaned CSVs (generated locally)
│
├── dashboard/
│   └── app.py                         # Streamlit interactive dashboard
│
├── requirements.txt
└── README.md
```

---

## Notebooks

### 01 — Data Collection
Fetches all 22 races from the 2023 season using the FastF1 API. Outputs a clean dataset of **24,422 laps × 34 columns** saved to `data/processed/f1_2023_full_season.csv`.

> ⚠️ First run downloads ~500 MB of telemetry data. Subsequent runs use a local cache and complete in 2–3 minutes.

### 02 — Driver Performance Analysis
- Average and best lap times per driver across the full season
- Consistency analysis using standard deviation of lap times
- Head-to-head comparisons (Verstappen vs Hamilton vs Leclerc, and more)

### 03 — Pit Stop Strategy
- Total pit stops per driver across the season
- Races ranked by strategic complexity (most pit stops)
- Per-driver breakdown of 1-stop, 2-stop, and 3-stop strategies

### 04 — Telemetry Analysis
- Speed, throttle, and brake traces comparing any two drivers on any circuit
- Color-coded speed maps plotted on actual circuit GPS coordinates
- Pinpoints exactly where drivers gain or lose time on track

### 05 — Race Predictor (ML)
- Feature engineering from lap data: avg speed, consistency score, pit stop count, etc.
- XGBoost Regressor trained on 2023 season data
- Evaluated with Mean Absolute Error on held-out races
- Generates hypothetical championship standings from model predictions

---

## Key Findings

- **Max Verstappen** recorded the lowest average lap time across the season (89.05 s among full-season drivers)
- **Zandvoort (Dutch GP)** had the highest pit stop count (100+) — the circuit is exceptionally hard on tires
- **2-stop strategy** was the dominant meta in 2023, adopted by almost every driver in the majority of races
- The ML model placed Verstappen at **P2 overall** when averaging across the full season (his poor early-race starts suppressed the prediction)
- **Monza telemetry** shows cars exceeding 320 km/h on the main straight with some of the heaviest braking zones on the calendar at the chicanes

---

## Dashboard

The Streamlit dashboard has four interactive pages:

| Page | What you can explore |
|------|----------------------|
| 🏠 Overview | Season KPIs, average lap time chart, race calendar |
| 👤 Driver Analysis | Multi-driver comparison, stats table |
| 🔄 Pit Stop Strategy | Charts and per-driver race strategy breakdown |
| 🗺️ Telemetry | Live driver-vs-driver speed, throttle, and brake comparison |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| FastF1 | Official F1 telemetry & timing data |
| Pandas / NumPy | Data processing & feature engineering |
| Matplotlib / Seaborn / Plotly | Visualizations |
| Scikit-learn | ML preprocessing & evaluation |
| XGBoost | Race outcome prediction |
| Streamlit | Interactive web dashboard |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Diyasadineni/f1-360-analytics.git
cd f1-360-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**macOS users:** XGBoost requires OpenMP. Install it first:

```bash
brew install libomp
```

### 3. Collect the data

Open `notebooks/01_data_collection.ipynb` and run all cells. This downloads and caches the full 2023 season locally.

### 4. Run the notebooks in order

```
01 → 02 → 03 → 04 → 05
```

### 5. Launch the dashboard

```bash
python3 -m streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501** 🌐

---

## Dataset

Raw telemetry is not included in this repository due to its size (~500 MB). The collection notebook downloads and caches everything automatically via the FastF1 API.

| Attribute | Value |
|-----------|-------|
| Season | 2023 Formula 1 World Championship |
| Races | 22 Grand Prix events |
| Drivers | 22 |
| Total laps | 24,422 |
| Features per lap | 34 |

---

## Contributing

Pull requests are welcome. Ideas for extending the project:

- Add 2024 season data for year-over-year comparisons
- Improve the ML model with qualifying times and weather features
- Add sprint race and qualifying session analysis
- Deploy the Streamlit dashboard to Streamlit Cloud or Hugging Face Spaces

## Acknowledgements

- [FastF1](https://theoehrly.github.io/Fast-F1/) — the backbone of this project's data layer
- [Ergast API](http://ergast.com/mrd/) — historical F1 results and standings
- Formula 1 — for making telemetry data publicly accessible

---

*Built with ❤️ and lots of ☕ by [Diyasadineni](https://github.com/Diyasadineni)*
