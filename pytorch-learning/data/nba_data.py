"""
NBA game-by-game data with all features: opponent, home/away, point types, etc.
Fetches actual schedule for next 3 games.
"""

import logging
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from nba_api.stats.endpoints import (
        leaguegamelog,
        scheduleleaguev2,
    )
except ImportError:
    raise ImportError("Please install nba_api: uv add nba_api")

logger = logging.getLogger(__name__)

# Seasons through 2026
DEFAULT_SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2019, 2027)]

# All numeric columns from LeagueGameLog
STAT_COLS = [
    "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK",
    "TOV", "PF", "PLUS_MINUS", "FANTASY_PTS",
]
# Point breakdown: 2PT = 2*(FGM-FG3M), 3PT = 3*FG3M, FT = FTM
# We use FGM, FG3M, FTM as proxy (model learns the weights)

# NBA team abbreviations for encoding
TEAM_ABBREVS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]
TEAM_TO_IDX = {t: i for i, t in enumerate(TEAM_ABBREVS)}


def _normalize_name(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _parse_matchup(matchup: str, team_abbrev: str) -> tuple[str, int]:
    """Parse MATCHUP 'DEN vs. HOU' or 'DEN @ HOU'. Returns (opponent_abbrev, home: 1=home, 0=away)."""
    matchup = str(matchup).strip()
    team_abbrev = str(team_abbrev).strip().upper()
    parts = re.split(r"\s+vs\.\s+|\s+@\s+", matchup, maxsplit=1)
    if len(parts) != 2:
        return "NYK", 1
    left, right = parts[0].strip().upper(), parts[1].strip().upper()
    if left == team_abbrev:
        opponent = right
        home = 1 if "vs." in matchup.lower() else 0
    else:
        opponent = left
        home = 0 if "vs." in matchup.lower() else 1
    return opponent, home


def fetch_league_game_logs(
    seasons: list[str] | None = None,
    cache_path: Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch all player game logs. Set force_refresh=True to refetch and include 2026 data."""
    seasons = seasons or DEFAULT_SEASONS
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists() and not force_refresh:
        logger.info("Loading cached game logs from %s", cache_path)
        df = pd.read_csv(cache_path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    all_logs = []
    for season in seasons:
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
            )
            data = log.get_data_frames()[0]
            if not data.empty:
                all_logs.append(data)
                logger.info("Fetched %s: %d games", season, len(data))
        except Exception as e:
            logger.warning("Failed %s: %s", season, e)
        time.sleep(0.8)

    if not all_logs:
        raise RuntimeError("No game log data fetched.")

    df = pd.concat(all_logs, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent, home/away, point-type breakdown."""
    df = df.copy()
    opps, homes = [], []
    for _, row in df.iterrows():
        opp, home = _parse_matchup(row["MATCHUP"], row["TEAM_ABBREVIATION"])
        opps.append(opp)
        homes.append(home)
    df["OPPONENT"] = opps
    df["HOME_AWAY"] = homes
    df["OPPONENT_IDX"] = df["OPPONENT"].map(lambda x: TEAM_TO_IDX.get(x.upper(), 0))
    # Point breakdown
    df["FG2M"] = (df["FGM"] - df["FG3M"]).clip(0)
    df["FG2A"] = (df["FGA"] - df["FG3A"]).clip(0)
    return df


def infer_position_from_stats(game_logs: pd.DataFrame) -> dict[int, int]:
    agg = (
        game_logs.groupby("PLAYER_ID")
        .agg({"PTS": "mean", "REB": "mean", "AST": "mean", "BLK": "mean"})
        .reset_index()
    )
    results = {}
    for _, row in agg.iterrows():
        pid = int(row["PLAYER_ID"])
        ast, reb, blk = row["AST"], row["REB"], row["BLK"]
        if ast >= 4 and ast > reb * 0.8:
            results[pid] = 0
        elif blk >= 0.8 or reb >= 8:
            results[pid] = 2
        else:
            results[pid] = 1
    return results


def build_rolling_features(
    game_logs: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Build rolling stats + opponent + home for each game."""
    df = _add_derived_features(game_logs)
    positions = infer_position_from_stats(df)

    stat_cols = [c for c in STAT_COLS + ["FG2M", "FG2A"] if c in df.columns]
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    roll_cols = {}
    for c in stat_cols:
        roll_cols[f"ROLL_{c}"] = df.groupby("PLAYER_ID")[c].transform(
            lambda x: x.shift(1).rolling(window, min_periods=window).mean()
        )

    out = df.copy()
    for k, v in roll_cols.items():
        out[k] = v

    out["POSITION"] = out["PLAYER_ID"].map(positions).fillna(1).astype(int)

    # Opponent and home for the *next* game (the one we're predicting)
    out["OPPONENT_IDX"] = df.groupby("PLAYER_ID")["OPPONENT_IDX"].shift(-1)
    out["HOME_AWAY"] = df.groupby("PLAYER_ID")["HOME_AWAY"].shift(-1)
    out["TARGET_PTS"] = df.groupby("PLAYER_ID")["PTS"].shift(-1)
    keep = (
        ["PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "GAME_ID", "SEASON_ID", "TEAM_ABBREVIATION", "MATCHUP"]
        + ["POSITION", "OPPONENT_IDX", "HOME_AWAY"]
        + [c for c in out.columns if c.startswith("ROLL_")]
        + ["TARGET_PTS"]
    )
    keep = [c for c in keep if c in out.columns]
    out = out[keep].dropna(subset=["TARGET_PTS", "OPPONENT_IDX", "HOME_AWAY"])
    return out.dropna(subset=[c for c in keep if c.startswith("ROLL_")])


def prepare_game_dataset(
    game_logs: pd.DataFrame,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    X: [rolling_stats, position, opponent_idx, home_away] for each game
    y: PTS for that game
    """
    rolling = build_rolling_features(game_logs, window=window)

    roll_cols = [c for c in rolling.columns if c.startswith("ROLL_")]
    meta_cols = ["POSITION", "OPPONENT_IDX", "HOME_AWAY"]
    feature_cols = roll_cols + meta_cols

    X = rolling[feature_cols].values.astype(np.float32)
    y = rolling["TARGET_PTS"].values.astype(np.float32).reshape(-1, 1)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6

    meta = {
        "feature_cols": feature_cols,
        "mean": X_mean,
        "std": X_std,
        "window": window,
        "team_abbrevs": TEAM_ABBREVS,
    }
    return X, y, meta


def fetch_player_next_3_games(
    player_team: str,
    season: str,
    from_date: datetime | str | None = None,
    cache_path: Path | str | None = None,
) -> list[dict]:
    """
    Fetch actual next 3 games for a team from schedule.
    Returns [{"opponent": "HOU", "home": True, "date": "..."}, ...]
    """
    from_date = from_date or datetime.now()
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date.replace("Z", "+00:00"))

    cache_path = Path(cache_path) if cache_path else Path("data/cache/schedule_2025_26.csv")

    if cache_path.exists():
        sched = pd.read_csv(cache_path)
    else:
        try:
            s = scheduleleaguev2.ScheduleLeagueV2(league_id="00", season=season)
            dfs = s.get_data_frames()
            sched = dfs[0] if dfs and not dfs[0].empty else pd.DataFrame()
            if not sched.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                sched.to_csv(cache_path, index=False)
        except Exception as e:
            logger.warning("Could not fetch schedule: %s", e)
            return []

    if sched.empty or "gameDate" not in sched.columns:
        return []

    sched["gameDate"] = pd.to_datetime(sched["gameDate"])
    sched = sched[sched["gameDate"] >= pd.Timestamp(from_date)]
    sched = sched.sort_values("gameDate")
    team_abbrev = player_team.upper()
    if len(team_abbrev) > 3:
        team_abbrev = {"TRAIL BLAZERS": "POR", "76ERS": "PHI", "NUGGETS": "DEN"}.get(team_abbrev, team_abbrev[:3])

    games = []
    for _, row in sched.iterrows():
        home = str(row.get("homeTeam_teamTricode", "")).upper()
        away = str(row.get("awayTeam_teamTricode", "")).upper()
        if home == team_abbrev:
            games.append({
                "opponent": away,
                "home": True,
                "date": row["gameDate"].strftime("%Y-%m-%d") if hasattr(row["gameDate"], "strftime") else str(row["gameDate"])[:10],
            })
        elif away == team_abbrev:
            games.append({
                "opponent": home,
                "home": False,
                "date": row["gameDate"].strftime("%Y-%m-%d") if hasattr(row["gameDate"], "strftime") else str(row["gameDate"])[:10],
            })
        if len(games) >= 3:
            break

    return games[:3]


def get_current_season() -> str:
    """Return current NBA season string, e.g. 2025-26."""
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[-2:]}"
    return f"{now.year - 1}-{str(now.year)[-2:]}"


def get_player_for_prediction(
    game_logs: pd.DataFrame,
    player_id: int,
    meta: dict,
    next_3_opponents: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, str] | None:
    """
    Build feature vectors for prediction.
    If next_3_opponents provided, returns 3 vectors (one per game).
    Else returns 1 vector (generic next game).
    Returns (X_vectors, recent_games_df, player_name) or None.
    """
    group = game_logs[game_logs["PLAYER_ID"] == player_id].sort_values("GAME_DATE", ascending=False)
    window = meta.get("window", 10)
    if len(group) < window:
        return None

    latest = group.iloc[:window].sort_values("GAME_DATE")
    positions = infer_position_from_stats(game_logs)
    pos = positions.get(player_id, 1)
    team_abbrev = str(group.iloc[0]["TEAM_ABBREVIATION"]).upper()

    roll_cols = [c for c in meta["feature_cols"] if c.startswith("ROLL_")]
    base_vec = np.zeros(len(roll_cols), dtype=np.float32)
    for i, col in enumerate(roll_cols):
        stat = col.replace("ROLL_", "")
        if stat in latest.columns:
            base_vec[i] = latest[stat].mean()
    base_vec = np.nan_to_num(base_vec, nan=0.0)

    if next_3_opponents and len(next_3_opponents) >= 3:
        vectors = []
        for g in next_3_opponents[:3]:
            opp_idx = TEAM_TO_IDX.get(g["opponent"].upper(), 0)
            home = 1 if g["home"] else 0
            vec = np.concatenate([base_vec, [pos, opp_idx, home]]).astype(np.float32)
            vectors.append(vec)
        X = np.stack(vectors, axis=0)
    else:
        vec = np.concatenate([base_vec, [pos, 0, 1]]).astype(np.float32)
        X = np.expand_dims(vec, axis=0)

    return X, group, str(group.iloc[0]["PLAYER_NAME"])


def search_players(
    game_logs: pd.DataFrame,
    query: str,
) -> list[tuple[int, str]]:
    q = _normalize_name(query.lower())
    names = game_logs.groupby(["PLAYER_ID", "PLAYER_NAME"]).size().reset_index()
    matches = []
    for _, row in names.iterrows():
        if q in _normalize_name(str(row["PLAYER_NAME"]).lower()):
            matches.append((int(row["PLAYER_ID"]), str(row["PLAYER_NAME"])))
    return matches
