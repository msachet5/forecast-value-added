# Forecast value added

Forecast value added, introduced by Michael Gilliland at SAS, measures the change in forecast error caused by each step of a forecasting process. The reference point is a naive forecast: something that costs nothing to produce.

**FVA of a step = error of the previous step − error of this step.**

Positive FVA: the step improved the forecast. Negative FVA: the organisation spent effort making the forecast worse. Zero: the step added cost without value.

## The stairstep

| Step | WAPE | FVA vs previous | FVA vs naive |
|---|---:|---:|---:|
| naive | 62.5% | | |
| statistical | 51.0% | +11.5 pp | +11.5 pp |
| planner override | 51.1% | −0.1 pp | +11.4 pp |
| consensus | 52.1% | −1.0 pp | +10.4 pp |

Read it top to bottom. Each row should be lower than the one above. In this example the statistical model earns its keep, the planner review breaks even and the consensus step (a sales uplift) destroys a point of accuracy.

## Choosing the benchmark

* **Naive** (last value) is the classic FVA benchmark.
* **Seasonal naive** (same period last season) is a stronger benchmark for seasonal data and the one most statistical models should clearly beat.
* Use both; `fva report` does.

## Item level, not just total

A process can add value in aggregate while destroying it on half the items. `item_fva` and `fva_scorecard` show the share of series, and the share of volume, where each step is better or worse.

## What to do with the result

1. Steps with negative FVA across several reviews are candidates for removal or redesign, not for more effort.
2. Steps with positive FVA on some segments only (for example A items, or promoted items) should be targeted at those segments.
3. Report FVA every cycle, not once; it is a process metric.
