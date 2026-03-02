# /lint — Lint and format Python code

Run ruff to check and optionally fix Python code quality issues.

## Instructions

1. First, check if ruff is installed:
   ```
   python -m ruff version
   ```
   If not installed, install it: `pip install ruff`.

2. Run the linter in check mode:
   ```
   ruff check packages/ $ARGUMENTS
   ```

3. Run the formatter in check mode:
   ```
   ruff format --check packages/
   ```

4. Report the results:
   - If clean: confirm no issues found.
   - If issues exist: group them by category (imports, unused variables, style, etc.) and show the count.

5. Ask the user if they'd like auto-fixes applied. If yes:
   ```
   ruff check --fix packages/
   ruff format packages/
   ```

6. After auto-fix, re-run the linter to confirm no remaining issues and report any that require manual intervention.
