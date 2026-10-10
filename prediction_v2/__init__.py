"""
Prediction Engine v2 — short-horizon, snapshot-based, SHADOW MODE.

Separate from Ranking Engine v1 (ranking/, engine_runs/): own universe
(v2_universe_members), own runs and predictions (db/models/prediction.py),
own versions. Nothing here reads-and-rewrites a v1 table or changes v1
scoring. Outputs are experimental and are not exposed as recommendations
(api/routes/predictions.py: administrators only unless explicitly enabled).

Modules:
  calendar    trading-day arithmetic (weekends + administrator holiday list)
  features    short-horizon features from daily bars, cutoff-safe
  rules       versioned, transparent baseline setups -> UP/DOWN/NEUTRAL/NO_CALL
  service     prediction runs (TODAY_PREOPEN / TODAY_CONFIRMED / TOMORROW_EOD)
  outcomes    1/3/5-session outcome evaluation (idempotent)
  exits       shadow exit-state machine
  performance aggregation of evaluated outcomes
  backtest    chronological backtest with splits, embargo, costs, baselines
  universe    v2 universe membership (independent of v1)
"""
ENGINE_VERSION = "prediction-v2.0-shadow"
