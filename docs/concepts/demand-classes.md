# Demand classification

## Syntetos-Boylan quadrants

Two numbers describe a demand pattern:

* **ADI**, the average inter-demand interval: how many periods, on average, per period with demand.
* **CV²**, the squared coefficient of variation of the non-zero demand sizes: how much the size varies when demand happens.

| | CV² < 0.49 | CV² ≥ 0.49 |
|---|---|---|
| **ADI < 1.32** | smooth | erratic |
| **ADI ≥ 1.32** | intermittent | lumpy |

The cut-offs come from Syntetos, Boylan and Croston (2005), who derived them from where Croston's method (with the SBA correction) outperforms exponential smoothing.

## Why it matters

* **Method choice.** Exponential smoothing for smooth and erratic; Croston SBA or TSB for intermittent and lumpy.
* **Accuracy targets.** A 30% WAPE is poor for a smooth fast mover and excellent for a lumpy spare part. Set targets per class.
* **Metric choice.** Item-level WAPE misleads on intermittent series; prefer MASE or aggregate before judging.

## ABC-XYZ

ABC ranks series by value (units, revenue, or units times price). XYZ ranks them by the coefficient of variation of period demand, zeros included. Together they give the familiar 3×3 policy grid: AX items deserve automation and tight targets; CZ items deserve simple rules and generous safety stock or make-to-order.

## Options

```python
demand_profile(df, adi_method="intervals")  # mean gap between demands, as in the paper
abc_xyz(df, price=prices, a_share=0.7)  # value by price, custom A line
```
