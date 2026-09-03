# Installation

```bash
pip install omnicast
```

`LSTMForecaster` depends on PyTorch, which is kept out of the base install because
it is a large, optional dependency:

```bash
pip install omnicast[torch]
```

## Local development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --extra dev            # core + test/lint tooling
uv sync --extra dev --extra torch   # also install LSTMForecaster's dependency
uv sync --extra docs           # sphinx + theme, to build this site
```

Run the test suite and linter:

```bash
uv run pytest
uv run ruff check .
```

Build these docs:

```bash
uv run sphinx-build -b html docs docs/_build/html
```
