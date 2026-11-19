"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case file per run from the input directory and writes, for every
trophy, five quantiles (5/25/50/75/95%) of its Exophase earn rate at the
case's target. Python 3.13, standard library only, no network, deterministic.

Model (per trophy, on the logit scale)
---------------------------------------
Exophase does not refresh continuously: the page sits unchanged for hours,
then jumps. At a refresh the tracked-player count jumps and almost every
rate FALLS, because the newly tracked players hold fewer trophies than the
early crowd. So a rate moves at refreshes, not smoothly, and the forecast
has to model the refresh, not extrapolate a flat line:

1. Players at target. Fit log(players) against log(hours since the
   first read) over the case's distinct reads (least squares, exponent
   shrunk toward 1.2 and clamped to [0.2, 2.0]); with no refresh in the
   history yet, expect one refresh jump inside the horizon instead
   (the page sits flat for hours, then the tracked count multiplies
   several-fold). Read the fit at the target to get N_target.
2. Newcomer rate. At every refresh (a jump in players tracked) the rate
   the newly added players hold is implied by the counts:
   rho = (N_new * r_new - N_old * r_old) / (N_new - N_old). Take the
   player-weighted median over the case's refreshes, shrunk toward a
   tier prior (newcomers hold a fraction of the current rate: rarer
   trophies much less). With no refresh in the history, use the prior.
3. Progression. Holders grow two ways: newcomers arrive holding the
   trophy at rho, and players who do not hold it yet earn it at a daily
   hazard that depends on tier and on what the trophy asks for (story /
   chapter trophies are earned fast, collections and grinds slowly,
   the platinum slowest). Existing non-holders earn over the full
   horizon; newcomers on average over half of it.
4. Median. Holders at target over players at target, soft-capped by a
   per-tier ceiling (a week in, the platinum cannot plausibly be common,
   while an early story bronze can be).
5. Spread. Logit sigma combines a floor for refresh/model uncertainty
   (0.75), a random-walk term (0.5 * sqrt(days)), a term for how far the
   median moved from the last rate, and one for how much the player base
   grows; clamped to [0.70, 1.60]. The practice horizon is short but
   straddles a refresh, so the floor -- not the random walk -- is what
   keeps the band honest there.

Run by the grader as `python3 -P -s /bot/main.py`; outside the grader,
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input / /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZS = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Per-tier assumptions. CEILING is a soft cap on the median rate (%)
# a week after launch; NEWCOMER_FRAC is the share of the current rate
# that newly tracked players are assumed to hold when no refresh has
# been observed yet; HAZARD is the daily rate at which a player who
# does not hold the trophy earns it.
# The newcomer fractions are set by the one refresh the house has
# actually seen: at 04:30Z the tracked count jumped 638 -> 3,868 while
# total trophies earned went 943 -> 4,276, so the 3,230 newly tracked
# players held 3,333 trophies, 1.03 each against the early crowd's
# 1.48 -- about 0.70 of the early rate on average, and far less for
# the rarest trophies, which is what the tier spread below encodes
# (story trophies sit above their tier's fraction: newcomers played
# the same opening hours, they just synced later).
TIER = {
    "platinum": {"ceiling": 2.0, "newcomer_frac": 0.10, "hazard": 0.003},
    "gold": {"ceiling": 10.0, "newcomer_frac": 0.35, "hazard": 0.012},
    "silver": {"ceiling": 30.0, "newcomer_frac": 0.45, "hazard": 0.025},
    "bronze": {"ceiling": 75.0, "newcomer_frac": 0.60, "hazard": 0.045},
}
DEFAULT_TIER = {"ceiling": 30.0, "newcomer_frac": 0.45, "hazard": 0.025}
# Fallback last rates (%) when a case carries no history at all.
TIER_PRIOR_RATE = {"platinum": 0.5, "gold": 2.0, "silver": 5.0, "bronze": 8.0}

# Story trophies are earned far faster than grinds; collections slowest.
STORY_WORDS = ("chapter", "story", "complete the", "finish the")
GRIND_WORDS = ("collect", "find all", "all ", "every ", "100 ", "win 10", "win 20")


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours_since_start(when, start):
    """Hours since the case's first read, plus one: the first reads
    predate the release-day midnight the catalogue uses as its origin,
    so an offset from the first read keeps log-time defined and
    positive for every read, including the earliest."""
    return 1.0 + max(0.0, (when - start).total_seconds() / 3600.0)


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def _round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def _story_factor(description):
    text = (description or "").lower()
    if any(w in text for w in GRIND_WORDS):
        return 0.35
    if any(w in text for w in STORY_WORDS):
        return 2.5
    return 1.0


def _newcomer_factor(description):
    """Newcomers are likelier to hold a story trophy (they played the
    same opening hours, they just synced later) and less likely to hold
    a grind trophy."""
    text = (description or "").lower()
    if any(w in text for w in GRIND_WORDS):
        return 0.8
    if any(w in text for w in STORY_WORDS):
        return 1.25
    return 1.0


def _distinct_reads(history):
    """Collapse consecutive reads with identical counts: they carry no
    new information about how rates move, only about the flat stretches."""
    out = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if out and out[-1][0] == key:
            continue
        out.append((key, read))
    return [read for _, read in out]


def _fit_players_exponent(reads, start):
    """Least-squares exponent k of N ~ (hours since the first read)^k
    over the distinct reads; None when the history shows fewer than
    three distinct player counts."""
    pts = []
    for read in reads:
        n = read.get("players_tracked")
        if n and n > 0:
            pts.append((math.log(_hours_since_start(_parse_time(read["at"]), start)),
                        math.log(n)))
    counts = {p[1] for p in pts}
    if len(pts) < 3 or len(counts) < 3:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0:
        return None
    return num / den


def _newcomer_rate(reads, idx, last_rate, tier, description):
    """Player-weighted median of the newcomer rate implied by each
    refresh jump, shrunk toward the tier prior (pseudo-count of 200
    players). Falls back to the prior when no refresh was observed."""
    frac = min(0.95, tier["newcomer_frac"] * _newcomer_factor(description))
    prior = frac * last_rate
    obs = []  # (rho, weight)
    for a, b in zip(reads, reads[1:]):
        na, nb = a.get("players_tracked"), b.get("players_tracked")
        ra = (a.get("rates") or [None] * (idx + 1))[idx]
        rb = (b.get("rates") or [None] * (idx + 1))[idx]
        if not na or not nb or nb <= na or ra is None or rb is None:
            continue
        rho = (nb * rb - na * ra) / (nb - na)
        obs.append((_clamp_pct(rho), float(nb - na)))
    if not obs:
        return prior
    obs.sort()
    total = sum(w for _, w in obs)
    acc = 0.0
    median = obs[-1][0]
    for rho, w in obs:
        acc += w
        if acc >= total / 2.0:
            median = rho
            break
    return (total * median + 200.0 * prior) / (total + 200.0)


def _forecast_one(trophy, idx, reads, last, horizon_minutes, target_dt, start):
    tier = TIER.get((trophy.get("tier") or "").lower(), DEFAULT_TIER)
    rates = []
    for read in reads:
        rs = read.get("rates") or []
        rates.append(rs[idx] if idx < len(rs) else None)
    known = [r for r in rates if r is not None]
    last_rate = known[-1] if known else TIER_PRIOR_RATE.get(
        (trophy.get("tier") or "").lower(), 8.0)
    # NOTE: `last` is the case's actual last read (not the de-duplicated
    # list's tail): collapsing the flat launch-night reads must not move
    # the "now" the forecast grows from.
    last_n = last.get("players_tracked") or 638
    days = horizon_minutes / 1440.0

    # A trophy literally everyone holds stays held: newcomers pick it
    # up immediately too, so only a thin lower tail is honest.
    if last_rate >= 99.5:
        centre = _logit(99.5)
        sigma = 0.40
        return [_round4(_clamp_pct(_inv_logit(centre + z * sigma))) for z in ZS]

    # --- players at target -------------------------------------------
    # No refresh observed yet (the history is flat): the page sits
    # unchanged for hours and then a refresh multiplies the tracked
    # count several-fold, so expect one such jump inside the horizon,
    # growing likelier the longer the horizon is. With refreshes in
    # the history, fit the growth instead (below).
    k = _fit_players_exponent(reads, start)
    if k is None:
        hours = horizon_minutes / 60.0
        # The one refresh on record multiplied the tracked count ~6x,
        # so a jump of that order is the prior when none has been seen.
        mult = 1.0 + 9.0 * (1.0 - math.exp(-hours / 2.0))
        n_target = last_n * mult
    else:
        k = 0.7 * k + 0.3 * 1.2  # shrink the fitted exponent toward 1.2
        k = min(2.0, max(0.2, k))
        t_last = _hours_since_start(_parse_time(last["at"]), start)
        t_target = _hours_since_start(target_dt, start)
        n_target = last_n * (t_target / t_last) ** k
    n_target = min(max(n_target, last_n), 2_000_000.0)

    # --- newcomers and progression ------------------------------------
    rho = _newcomer_rate(reads, idx, last_rate, tier, trophy.get("description"))
    hazard = tier["hazard"] * _story_factor(trophy.get("description"))
    holders = last_n * last_rate / 100.0
    non_holders = last_n - holders
    newcomers = n_target - last_n
    gains = non_holders * (1.0 - math.exp(-hazard * days))
    gains += newcomers * (1.0 - rho / 100.0) * (1.0 - math.exp(-hazard * days / 2.0))
    holders_t = holders + newcomers * rho / 100.0 + gains
    raw = 100.0 * holders_t / n_target
    ceiling = tier["ceiling"]
    median = ceiling * (1.0 - math.exp(-raw / ceiling))  # soft cap

    # --- spread ---------------------------------------------------------
    shift = abs(_logit(median) - _logit(last_rate))
    growth = math.log10(n_target / last_n) if n_target > last_n else 0.0
    var = 0.75 ** 2 + (0.5 * math.sqrt(days)) ** 2
    var += (0.20 * shift) ** 2 + (0.30 * growth) ** 2
    sigma = min(1.60, max(0.70, math.sqrt(var)))

    centre = _logit(median)
    return [_round4(_clamp_pct(_inv_logit(centre + z * sigma))) for z in ZS]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    reads = _distinct_reads(history)
    last = history[-1]  # the case's real last read, not the de-duplicated tail
    start = _parse_time(history[0]["at"])
    target_dt = _parse_time(case["target"])
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append({"id": trophy["id"],
                     "q": _forecast_one(trophy, i, reads, last,
                                        case["horizon_minutes"], target_dt, start)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
