# Command line

```text
fva demo                       synthetic data, full report
fva report SALES               HTML report and Parquet tables
fva classify SALES             demand class and ABC-XYZ per series
fva backtest SALES             rolling-origin backtest of benchmark models
fva overrides FORECASTS        override analysis
fva safety-stock EVALUATION    safety stock and reorder point
fva generate                   write a synthetic sales file
```

Common options: `--id-col`, `--date-col`, `--target-col`, `--fill-gaps 1w`, `--season-length 52`, `--h 4`, `--windows 6`.

Run `fva <command> --help` for the full list.
