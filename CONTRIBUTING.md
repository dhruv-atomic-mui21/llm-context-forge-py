# Contributing to LLM Context Forge

First off, thanks for taking the time to contribute!

## Development Setup

1. Fork and clone the repository.
2. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
3. (Optional but recommended) Run tests to ensure everything works locally:
   ```bash
   pytest tests/
   ```

## Workflow

- Create a new branch for your feature or bugfix.
- Make your changes.
- Write tests for your changes.
- Ensure all tests pass (`pytest tests/`).
- Code must be formatted with `ruff` and pass type checking with `mypy`.
  ```bash
  ruff check llm_context_forge/
  mypy llm_context_forge/
  ```
- Submit a Pull Request targeting the `main` branch.

## Code Style

- We use `ruff` to enforce standard Python formatting.
- Type hints are required for all new functions and classes.
- Use docstrings for all public classes, methods, and functions. 

## Releasing (Maintainers Only)

Our release process is automated via GitHub Actions. Merging a tag (e.g. `v1.0.0`) will automatically build and publish to PyPI.
