project_name := "ai_readiness"

_default:
    @ just --list

# Run recipes for MR approval
pre-mr: format lint test

# Formats Code
format:
    uv run ruff check --select I --fix src examples
    uv run ruff format src examples

# Lints Code
lint *options:
    uv run ruff check src examples {{ options }}

# Tests code
[group("Dev")]
test *options:
    uv run pytest -s tests/ {{ options }}

# Build docs
build-docs:
    uv run mkdocs serve