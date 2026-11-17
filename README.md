[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/shiryuu156/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/shiryuu156/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCV Forecast — make.fun bounty entry

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers hour by
hour (hours 2–24), given hour 1's print.

## Approach: scale-adjusted comparable-shape forecaster

Launch-day Twitch viewership follows a strong systematic shape (rolling
midnight-local unlock + global day/night cycle): a dip in early UTC hours, a
rise through the US day, a peak in US prime time (hours 19–21 UTC), then a
decline. The persistence baseline ignores this shape entirely.

1. **Shape**: for each comparable launch, compute per-hour log ratios of
   viewers to hour 1. Console-only launches (rolling midnight unlock, no
   simultaneous PC unlock) match GTA VI's pattern best and are the primary
   shape source.
2. **Scale**: the peak ratio falls as hour 1 grows (correlation −0.85 on the
   console set). Each hour's log ratio is regressed on (log hour-1 −
   reference), slope clamped to [−0.6, 0.1], so a huge hour-1 print (as GTA VI
   will likely have) correctly predicts a smaller multiple.
3. **Spreads**: quantile spreads from residual dispersion across comparables
   after the scale regression, with a floor (0.15 log units) and a small
   horizon term for shape uncertainty.

## Validation (leave-one-out, pinball loss on log scale)

| | This model | Persistence baseline |
|---|---|---|
| Console-only comparables (n=8) | **0.173** | 0.323 |
| All comparables (n=37) | **0.700** | 0.930 |

Wins 7/8 console launches and 27/37 overall — a 46% improvement on the
most relevant set.

## Files

- `main.py` — the entry (Python 3.13, standard library only, deterministic,
  ~0.2s per case, well under the 8s cap)
- `LICENSE` — MIT (from the kit starter)

## Run locally

```sh
MAKEFUN_INPUT_DIR=../gta6-twitch-ccu-forecast-01-kit/input \
  MAKEFUN_OUTPUT_DIR=out python3 main.py
```
