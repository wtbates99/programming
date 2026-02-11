"""
NBA PPG Predictor - Game-by-game with opponent, home/away, all features.
Predicts next 3 games using actual schedule.
"""

import logging

from train import train

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    print("=" * 60)
    print("NBA PPG Predictor - Training")
    print("=" * 60)

    model, history, meta = train(
        epochs=80,
        batch_size=128,
        window=10,
        cache_path="data/cache/league_game_logs.csv",
        output_dir="output",
        force_refresh=True,  # Set True to refetch and include latest 2026 games
    )

    print("\nTraining complete!")
    print(f"Final loss: {history[-1]:.4f}")
    print("\nPredict next 3 games (actual schedule):")
    print("  uv run python predict.py 'LeBron James'")
    print("  uv run python predict.py Jokic")


if __name__ == "__main__":
    main()
