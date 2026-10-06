# Power BI

```bash
fva report sales.csv --forecasts forecasts.csv --format html,parquet --out fva
```

writes `fva/tables/*.parquet`. In Power BI Desktop: **Get Data → Folder**, point at `fva/tables`, and combine, or load each Parquet file.

| Table | Grain | Typical visual |
|---|---|---|
| stairstep | step | Bar chart of WAPE by step; card for FVA |
| stairstep_by_class, stairstep_by_abc | segment × step | Matrix with conditional formatting on FVA |
| items | series | Table of worst items; scatter of FVA vs volume |
| classes | series | Slicers for demand class and ABC-XYZ |
| tracking | series | Bias alert list |
| override_summary | direction × size | Matrix of win rate and FVA |
| evaluation | series × period | Line chart of actual vs each step |

Relationships: `items[unique_id]`, `tracking[unique_id]` and `evaluation[unique_id]` to `classes[unique_id]` (many to one).

Suggested measures:

```DAX
WAPE = DIVIDE ( SUMX ( evaluation, ABS ( evaluation[final] - evaluation[y] ) ), SUM ( evaluation[y] ) )
Bias % = DIVIDE ( SUM ( evaluation[final] ) - SUM ( evaluation[y] ), SUM ( evaluation[y] ) )
```
