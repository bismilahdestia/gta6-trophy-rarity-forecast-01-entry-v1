# GTA VI Trophy Rarity Forecast — Exophase earn rates, one week after launch

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bismilahdestia/gta6-trophy-rarity-forecast-01-entry-v1/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bismilahdestia/gta6-trophy-rarity-forecast-01-entry-v1/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

A forecaster for the [make.fun GTA VI Trophy Rarity Forecast bounty](https://make.fun/c/gta6-trophy-rarity-forecast-01): for every PlayStation 5 trophy of Grand Theft Auto VI, it forecasts the earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who will have earned it, one week after launch — as five quantiles per trophy, from the house's reads of the trophy page up to the deadline.

## How it works

`main.py` (Python 3.13, standard library only) reads one case file from `/input`, fits a per-trophy model on the logit scale over the case's history of reads, and writes `/output/<id>.out` with five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. The model accounts for how Exophase refreshes its counts hours apart rather than continuously, and for the player base growing between reads: newly tracked players hold fewer trophies, so rates move at refreshes, not smoothly.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs on the same case write the same forecast. Runs in well under the 8-second per-case cap.

## Layout

- `main.py` — the forecaster (standard library only, no network)
- `LICENSE` — MIT
