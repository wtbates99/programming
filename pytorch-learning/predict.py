"""
Predict next 3 games PPG for a player. Uses actual schedule (opponent, home/away).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.nba_data import (
    fetch_league_game_logs,
    fetch_player_next_3_games,
    get_current_season,
    get_player_for_prediction,
    search_players,
)
from models.player_predictor import NextGamePPGPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_model_and_meta(output_dir: Path) -> tuple[NextGamePPGPredictor, dict]:
    output_dir = Path(output_dir)
    with open(output_dir / "meta.json") as f:
        meta = json.load(f)
    meta["mean"] = np.array(meta["mean"], dtype=np.float32)
    meta["std"] = np.array(meta["std"], dtype=np.float32)

    model = NextGamePPGPredictor(
        input_dim=len(meta["feature_cols"]),
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    )
    model.load_state_dict(torch.load(output_dir / "model.pt", map_location="cpu"))
    model.eval()

    return model, meta


def predict_next_3_games(
    player_query: str,
    model_dir: Path | str = "output",
    data_cache: Path | str | None = None,
    n_mc_samples: int = 50,
) -> dict | None:
    """
    Predict next 3 games PPG using actual schedule (opponent, home/away).
    """
    model_dir = Path(model_dir)
    if not (model_dir / "model.pt").exists():
        raise FileNotFoundError(f"No model at {model_dir}. Run training first.")

    data_cache = Path(data_cache) if data_cache else Path("data/cache/league_game_logs.csv")
    if not data_cache.exists():
        raise FileNotFoundError(f"No data at {data_cache}. Run training first.")

    game_logs = fetch_league_game_logs(cache_path=data_cache)
    model, meta = load_model_and_meta(model_dir)

    matches = search_players(game_logs, player_query)
    if not matches:
        return None

    query_norm = player_query.strip().lower()
    exact = [(pid, name) for pid, name in matches if name.lower() == query_norm]
    player_id, player_name = exact[0] if exact else matches[0]

    player_games = game_logs[game_logs["PLAYER_ID"] == player_id]
    if player_games.empty:
        return None
    team_abbrev = str(player_games.iloc[0]["TEAM_ABBREVIATION"])
    season = get_current_season()

    cache_schedule = Path("data/cache") / f"schedule_{season.replace('-', '_')}.csv"
    next_3 = fetch_player_next_3_games(team_abbrev, season, cache_path=cache_schedule)

    result = get_player_for_prediction(game_logs, player_id, meta, next_3_opponents=next_3 if next_3 else None)
    if result is None:
        return None

    X_batch, games_df, _ = result
    X_norm = (X_batch - meta["mean"]) / meta["std"]
    X_t = torch.from_numpy(X_norm).float()

    predictions = []
    for i in range(X_t.shape[0]):
        x = X_t[i : i + 1]
        mean, std = model.predict_with_uncertainty(x, n_samples=n_mc_samples)
        m, s = mean.squeeze().item(), std.squeeze().item()
        low_25 = m - 0.674 * s
        high_75 = m + 0.674 * s
        low_10 = m - 1.282 * s
        high_90 = m + 1.282 * s

        game_info = next_3[i] if i < len(next_3) else {"opponent": "?", "home": True, "date": "?"}
        predictions.append({
            "game": f"Game {i + 1}",
            "opponent": game_info.get("opponent", "?"),
            "home": game_info.get("home", True),
            "date": game_info.get("date", "?"),
            "expected_ppg": round(m, 1),
            "std": round(s, 2),
            "interval_25_75": [round(low_25, 1), round(high_75, 1)],
            "interval_10_90": [round(low_10, 1), round(high_90, 1)],
        })

    feature_cols = meta["feature_cols"]
    vec = X_batch[0]
    contributions = []
    for i, col in enumerate(feature_cols):
        val = vec[i]
        norm_val = (vec[i] - meta["mean"][i]) / (meta["std"][i] + 1e-6)
        contributions.append({"feature": col, "value": round(float(val), 2), "normalized": round(float(norm_val), 2)})
    contributions.sort(key=lambda x: abs(x["normalized"]), reverse=True)

    recent = games_df.head(15)[["GAME_DATE", "MATCHUP", "PTS", "MIN", "FGM", "FG3M", "FTM"]].copy()
    recent["GAME_DATE"] = pd.to_datetime(recent["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    history = recent.to_dict(orient="records")

    return {
        "player_name": player_name,
        "player_id": int(player_id),
        "team": team_abbrev,
        "season": season,
        "predictions": predictions,
        "feature_breakdown": contributions[:10],
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict next 3 games PPG. Uses actual schedule (opponent, home/away)."
    )
    parser.add_argument("player", nargs="?", help="Player name (e.g. 'LeBron James', 'Jokic')")
    parser.add_argument("--model-dir", default="output", help="Path to trained model")
    parser.add_argument("--data-cache", default=None, help="Path to cached game logs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.player:
        parser.print_help()
        print("\nExamples:")
        print("  uv run python predict.py 'LeBron James'")
        print("  uv run python predict.py Jokic")
        return

    try:
        out = predict_next_3_games(
            args.player,
            model_dir=args.model_dir,
            data_cache=args.data_cache,
        )
        if out is None:
            logger.warning('No player found matching "%s"', args.player)
            return

        if args.json:
            def _to_native(obj):
                if isinstance(obj, dict):
                    return {k: _to_native(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_native(x) for x in obj]
                if hasattr(obj, "item"):
                    return obj.item()
                return obj

            print(json.dumps(_to_native(out), indent=2), flush=True)
            return

        print("\n" + "=" * 60)
        print(f"  {out['player_name']} ({out['team']}) - {out['season']}")
        print("=" * 60)

        print("\n  NEXT 3 GAMES (actual schedule)")
        print("-" * 50)
        for p in out["predictions"]:
            loc = "vs" if p["home"] else "@"
            i25, i75 = p["interval_25_75"]
            i10, i90 = p["interval_10_90"]
            print(f"  {p['date']} | {loc} {p['opponent']}")
            print(f"    Expected: {p['expected_ppg']} ppg (±{p['std']:.1f})")
            print(f"    50%: {i25:.1f}-{i75:.1f}  |  80%: {i10:.1f}-{i90:.1f}")

        print("\n  TOP FEATURES")
        print("-" * 40)
        for c in out["feature_breakdown"]:
            print(f"    {c['feature']}: {c['value']}")

        print("\n  RECENT GAMES")
        print("-" * 40)
        for h in out["history"][:12]:
            print(f"    {h.get('GAME_DATE','')} | {h.get('MATCHUP','')} | {h.get('PTS',0)} pts | {h.get('FGM',0)}/{h.get('FG3M',0)}/{h.get('FTM',0)} 2pt/3pt/ft")

        print()

    except FileNotFoundError as e:
        logger.error("%s", e)
    except Exception as e:
        logger.exception("Error: %s", e)


if __name__ == "__main__":
    main()
