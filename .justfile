project_name := "ai_readiness"

_default:
    @ just --list

# Run recipes for MR approval
pre-mr: format lint

# Formats Code
format:
    poetry run ruff check --select I --fix src
    poetry run ruff format src

# Lints Code
lint *options:
    poetry run ruff check src {{ options }}
