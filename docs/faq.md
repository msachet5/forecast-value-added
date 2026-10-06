# FAQ

**Is a naive forecast really a fair benchmark?** It is the cheapest possible forecast, which is the point: any step that costs effort must beat it. Seasonal naive is the tougher benchmark for seasonal data, and the report shows both.

**My statistical model loses to seasonal naive on many items. Is the library wrong?** Possibly not. Published forecasting competitions show simple benchmarks are hard to beat on noisy, intermittent retail data. Check the demand class breakdown: losses usually concentrate in intermittent and lumpy items, where WAPE favours flat or zero forecasts.

**Can I use daily data?** Yes. Set `--season-length 7` (weekly cycle) and `--h` to your decision horizon in days.

**Does it forecast?** Only the benchmarks needed to measure value added. Bring forecasts from any tool, or use the statsforecast adapter.

**Is my data sent anywhere?** No. Everything runs locally.
