# /simplify — Simplify complex code

Analyze code for unnecessary complexity and suggest simplifications.

## Instructions

1. Determine the target:
   - If `$ARGUMENTS` specifies file paths, analyze those files.
   - If no arguments, ask the user which files or functions to simplify.

2. Read the target files and identify:

   **Unnecessary complexity**
   - Functions longer than ~40 lines that can be decomposed
   - Nested conditionals deeper than 3 levels
   - Redundant variable assignments or intermediate values
   - Over-engineered abstractions for simple operations

   **NumPy/SciPy-specific simplifications**
   - Python loops that can be replaced with vectorized operations
   - Manual implementations of functions available in NumPy/SciPy
   - Unnecessary `.copy()` calls or array conversions

   **Dead code**
   - Unreachable branches
   - Unused imports, variables, or functions
   - Commented-out code blocks

   **Readability wins**
   - Complex expressions that benefit from named intermediates
   - Boolean expressions that can be simplified
   - Opportunities to use Python builtins (any, all, zip, enumerate)

3. For each suggestion:
   - Show the current code snippet (file:line range)
   - Show the simplified version
   - Explain why the simplification is safe (preserves behavior)

4. Ask the user which simplifications to apply before making changes.

5. After applying changes, run `pytest -v` to confirm nothing broke.
