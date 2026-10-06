# forecast-value-added

Did your forecasting process, and your overrides, beat a naive forecast?

`forecast-value-added` is the planner layer on top of open-source forecasting. It classifies demand, backtests honest benchmarks, builds the forecast value added (FVA) stairstep, finds biased series and tells you which judgmental overrides add value and which destroy it.

```bash
pip install forecast-value-added
fva demo
```

Open `fva_demo/report.html`. [See a live demo report](demo/report.html).

## Who it is for

* **Demand planners and S&OP leads** who need one page that shows where the process adds value.
* **Forecasting data scientists** who want planner-grade evaluation on top of statsforecast, sktime or their own models.
* **Analytics consultants** who audit a client's forecasting process.

## Next

* [Quick start](quickstart.md): your own CSV in five minutes.
* [Forecast value added](concepts/fva.md): the idea and how to read the stairstep.
* [Demand classification](concepts/demand-classes.md): why lumpy items need a different target.
