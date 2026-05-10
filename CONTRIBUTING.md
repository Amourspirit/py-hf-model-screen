# Contributing

Thanks for your interest in contributing to hf-model-screen.

## Development setup

1. Install uv.
2. Create a local `.env` from `.env.example` and set `HF_TOKEN` if you plan to run live model calls.
3. Install dependencies:

```bash
uv sync
```

## Running tests

Run the full test suite before opening a pull request:

```bash
uv run pytest
```

## Configuration notes

- Shared defaults belong in `project-config/default_config.yaml`.
- Local machine-specific configs belong in `project-config-local/*.yaml` (ignored by git).

## Pull requests

1. Fork the repository and create a feature branch.
2. Keep changes focused and include tests when behavior changes.
3. Update docs/config examples when adding or changing user-facing behavior.
4. Open a pull request with a clear description of what changed and why.
