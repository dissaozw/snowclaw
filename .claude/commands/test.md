# /test — Run the test suite

Run the project's test suite using pytest. Analyze any failures and suggest fixes.

## Instructions

1. Run `pytest` from the project root with verbose output:
   ```
   pytest -v $ARGUMENTS
   ```
   If no arguments are provided, run all tests. Common arguments:
   - A specific file path (e.g. `packages/biomechanics/tests/test_metrics.py`)
   - `-k <pattern>` to filter by test name
   - `--cov` to include coverage reporting

2. If all tests pass, report a short summary (total passed, time taken).

3. If any tests fail:
   - Read the failing test file and the source file it covers.
   - Identify the root cause of each failure.
   - Propose a concrete fix (show the diff) and ask the user whether to apply it.

4. Never silently skip or ignore test failures.
