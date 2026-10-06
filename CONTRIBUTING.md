# Contributing

Thanks for helping planners measure their forecasts honestly.

## Ground rules

1. **Public or synthetic data only.** Never commit company data, even anonymised. Use `generate_sales` or the M5 dataset in tests and examples.
2. **Every metric needs a textbook test.** If you add or change a calculation, add a test with a hand-checkable example and cite the source in the docstring.
3. **Planner language first.** Function names, report labels and docs should read naturally to a demand planner, not only to a data scientist.

## Setup

```bash
git clone https://github.com/msachet5/forecast-value-added
cd forecast-value-added
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Pull requests

- One change per pull request, with a test.
- Update `CHANGELOG.md` under an "Unreleased" heading.
- CI must pass on Python 3.10 to 3.13.

## Reporting a calculation error

A wrong number in an FVA report can drive a wrong business decision, so calculation bugs are the top priority. Open an issue with the "calculation error" template, a minimal input and the value you expected (with its source). Fixes are logged in the changelog's corrections section.

## Support policy

This is maintained in spare time. Issues are triaged once a month; calculation errors are looked at within a week.
