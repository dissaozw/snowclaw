# /review — Code review

Perform a thorough code review on the specified files or the current staged/unstaged changes.

## Instructions

1. Determine what to review:
   - If `$ARGUMENTS` specifies file paths, review those files.
   - If no arguments, review the current uncommitted changes (`git diff` and `git diff --cached`).
   - If there are no uncommitted changes, ask the user what to review.

2. Read all relevant files and analyze them against this checklist:

   **Correctness**
   - Logic errors, off-by-one mistakes, unhandled edge cases
   - Incorrect use of NumPy broadcasting or array operations
   - Missing or wrong type annotations

   **Security**
   - Input validation at system boundaries
   - No hardcoded secrets or credentials
   - Safe handling of file paths and external data

   **Testing**
   - Are new functions covered by tests?
   - Are edge cases and boundary conditions tested?
   - Do tests follow the project pattern (`packages/<name>/tests/test_*.py`)?

   **Style & Conventions**
   - Follows project conventions from AGENTS.md (Y-up coords, degrees, Pydantic v2)
   - Docstrings on public functions
   - Consistent naming (snake_case for Python)

   **Performance**
   - Unnecessary copies of large arrays
   - Vectorizable loops left as Python iteration
   - Redundant computation that could be cached

3. Present findings grouped by severity:
   - **Must fix** — bugs, security issues, incorrect behavior
   - **Should fix** — missing tests, unclear naming, style violations
   - **Consider** — performance suggestions, optional improvements

4. For each finding, show the exact location (file:line) and a suggested fix.
