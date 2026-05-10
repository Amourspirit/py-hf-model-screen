# hf-model-screen

Hugging Face model screening harness for private local repositories.

## What it does

- Reads selected files from a local repository checkout
- Builds a compact repo context block
- Runs fixed prompts against one or more Hugging Face chat models
- Captures latency, usage, and heuristic quality metrics
- Writes JSON, CSV, and Markdown reports

## Configuration files

- Shared defaults: `project-config/example_config.yaml`
- Local configs (git-ignored): `project-config-local/*.yaml`

Config precedence is:

1. CLI overrides
2. User config file
3. Shared defaults in `project-config/example_config.yaml`

If you pass `--config qwen_config`, the script resolves it to:

- `project-config-local/qwen_config.yaml`

If you pass `--config /abs/path/custom.yaml`, that file is used directly.

## Environment setup

1. Copy `.env.example` to `.env`
2. Set `HF_TOKEN` in `.env`
3. Install dependencies with `uv sync`

## Direct script usage

```bash
uv run python ./scripts/hf_repo_model_screen.py \
	--repo /path/to/project
```

```bash
uv run python ./scripts/hf_repo_model_screen.py \
	--repo /path/to/project \
	--config qwen_config \
	--passes 2 \
	--max-context-chars 36000 \
	--max-file-chars 3500 \
	--max-tokens 900 \
	--save-context
```

Print merged runtime config:

```bash
uv run python ./scripts/hf_repo_model_screen.py \
	--repo /path/to/project \
	--config qwen_config \
	--print-effective-config
```

Build and save context only (no API call):

```bash
uv run python ./scripts/hf_repo_model_screen.py \
	--repo /path/to/project \
	--config qwen_config \
	--dry-run-context
```

## Makefile usage

Default config (`project-config/example_config.yaml`):

```bash
make screen-default REPO=/path/to/project
```

Named local config (`project-config-local/qwen_config.yaml`):

```bash
make screen REPO=/path/to/project CONFIG=qwen_config
```

Explicit config path:

```bash
make screen-path REPO=/path/to/project CONFIG_PATH=/abs/path/custom.yaml
```

Optional output dir override:

```bash
make screen REPO=/path/to/project CONFIG=qwen_config OUTDIR=./outputs
```

Context-only dry run:

```bash
make screen-context-dry REPO=/path/to/project CONFIG=qwen_config
```

## Testing

Run tests with:

```bash
uv run pytest
```

Tests mock OpenAI calls and do not hit external APIs.
