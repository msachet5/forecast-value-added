# Issues to open on day one

Create these as GitHub issues labelled `good first issue` before launch. Each is small, self-contained and teaches a contributor the codebase.

1. **Add `mean_absolute_scaled_error` per demand class to the HTML report.** The data is in `stairstep(..., metric="mase")`; add a toggle or a second table in `templates/report.html.j2`.
2. **Add the ADIDA benchmark** (aggregate, forecast with SES, disaggregate) to `models.py` with a textbook test.
3. **Support monthly data in `generate_sales`** with `freq="1mo"` and `season_length=12`; add a test that classifies the generated patterns correctly.
4. **Add an `--lag` option to `fva report`** that keeps only forecasts made `lag` periods before each target date (needs a `cutoff` column).
5. **Write a docs page "FVA in Excel"** showing how to read the CSV tables (`--format csv`) with a pivot table.
