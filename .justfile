project_name := "ai_readiness"

_default:
    @ just --list

# Run recipes for MR approval
pre-mr: format lint

# Formats Code
format:
    uv run ruff check --select I --fix src
    uv run ruff format src

# Lints Code
lint *options:
    uv run ruff check src {{ options }}

# Build docs
build-docs:
    uv run mkdocs serve