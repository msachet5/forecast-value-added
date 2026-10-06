# Override analysis

Planners adjust statistical forecasts. Field research across supply chain companies (Fildes, Goodwin, Lawrence and Nikolopoulos, 2009) found three regularities:

1. Large adjustments tend to improve accuracy; small ones mostly add noise.
2. Upward adjustments are wrong more often than downward ones: optimism leaks into the number.
3. Many adjustments are made to forecasts that were already good enough.

`override_summary` tests these on your data. Each override is tagged by direction (up or down) and relative size (< 10%, 10 to 25%, 25 to 50%, ≥ 50%), and each group gets a win rate, WAPE before and after, FVA and total error units added.

```bash
fva overrides forecasts.csv --baseline statistical --final planner --who planner
```

## Turning it into policy

* If small adjustments lose, set a minimum adjustment threshold (for example "do not adjust by less than 10%").
* If upward adjustments lose, require a written reason for increases, or review them separately.
* Use `worst_overriders` for coaching and for routing adjustment rights to the people and segments where judgment pays. It is not a blame tool.
