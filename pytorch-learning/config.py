"""
Configuration for NBA PPG Predictor.
"""

from pathlib import Path

# Data
CACHE_DIR = Path("data/cache")
GAME_LOGS_CACHE = CACHE_DIR / "league_game_logs.csv"

# Seasons through 2026-27
DEFAULT_SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2019, 2027)]

# Model
OUTPUT_DIR = Path("output")
MIN_GAMES_PER_SEASON = 15

# Training
EPOCHS = 150
BATCH_SIZE = 32 
LR = 1e-3
HIDDEN_DIMS = [256, 128, 64, 32]
DROPOUT = 0.20
