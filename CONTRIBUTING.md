# Contributing

Issues and pull requests are welcome, in particular bug reports, ports to newer PyPOTS releases and new repair methods.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
ruff check src tests scripts
pytest
```

## Guidelines

- Keep changes that alter results separate from refactors, and say in the pull request which numbers move.
- New repair methods go in `src/repair4twin/imputers.py` and must be added to `IMPUTERS` and `repair()`.
- `tests/test_paper_numbers.py` must keep passing against `results/all_results.csv`. Regenerated results belong in a new file, not over the released one.
