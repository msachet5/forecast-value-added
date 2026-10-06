# Error metrics

| Metric | Formula | Use it for |
|---|---|---|
| WAPE | Σ\|F − A\| / ΣA | The headline accuracy number planners recognise. Volume weighted. |
| Accuracy | max(0, 1 − WAPE) | The same, in the "forecast accuracy %" form many dashboards show. |
| Bias | Σ(F − A) / ΣA | Direction of error. Positive = over-forecast, which drives excess inventory. |
| MAE, RMSE | mean \|e\|, √mean e² | Unit-level error; RMSE feeds safety stock. |
| MASE | MAE / in-sample MAE of seasonal naive | Comparing across items of different size; robust for intermittent items. Below 1 beats seasonal naive on history. |

## Tracking signal

Bias over a whole period can hide a recent drift. A tracking signal monitors it period by period.

* **Trigg (default):** smoothed error / smoothed absolute error, α = 0.1. It lives between −1 and 1. The library flags values outside the 95% band expected for unbiased errors, which is 1.96 × √(α / (2 − α)) / √(2/π) ≈ 0.56 at α = 0.1.
* **RSFE / MAD:** the textbook running sum of errors divided by mean absolute deviation, flagged at ±4. Over long windows unbiased forecasts drift past ±4 by chance, so use `window=` to limit it.

Positive values mean the forecast is persistently too low.
