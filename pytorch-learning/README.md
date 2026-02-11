# NBA PPG Predictor – Next 3 Games (Actual Schedule)

Predicts **next 3 games PPG** using the actual schedule: opponent, home/away, and all available features.

## Features

- **2026 data** – Seasons through 2026-27 (`force_refresh=True` to refetch)
- **Actual schedule** – Next 3 games from NBA schedule (opponent, date, home/away)
- **Full feature set** – MIN, PTS, FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PLUS_MINUS, FANTASY_PTS, FG2M, FG2A
- **Opponent** – Team encoded for each game
- **Home/away** – 1 = home, 0 = away

## Setup

```bash
uv sync
```

## Train

```bash
uv run python main.py
```

First run fetches game logs (including 2025-26). Set `force_refresh=True` in `main.py` to refetch and include latest 2026 games.

## Predict

```bash
uv run python predict.py "LeBron James"
uv run python predict.py Jokic
uv run python predict.py Jokic --json
```

Output includes:
- Next 3 games with actual opponent, date, home/away
- Expected PPG and confidence intervals per game
- Feature breakdown
- Recent game history (2pt/3pt/ft)

## Model

- Input: Rolling stats (last 10 games) + position + opponent + home/away
- Output: PTS for that game
- Called 3 times for next 3 games with different opponent/home inputs
