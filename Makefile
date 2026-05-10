PYTHON_RUN := uv run python ./scripts/hf_repo_model_screen.py
CONFIG ?= default_config
CONFIG_PATH ?=
REPO ?=
OUTDIR ?=
EXTRA_ARGS ?=

.PHONY: screen screen-default screen-path screen-context-dry help

help:
	@echo "Usage:"
	@echo "  make screen REPO=/path/to/repo [CONFIG=qwen_config] [OUTDIR=./outputs]"
	@echo "  make screen-path REPO=/path/to/repo CONFIG_PATH=/abs/path/config.yaml"
	@echo "  make screen-default REPO=/path/to/repo"
	@echo "  make screen-context-dry REPO=/path/to/repo [CONFIG=qwen_config]"

screen:
	@if [ -z "$(REPO)" ]; then echo "REPO is required"; exit 1; fi
	@$(PYTHON_RUN) \
		--repo "$(REPO)" \
		--config "$(CONFIG)" \
		$(if $(OUTDIR),--outdir "$(OUTDIR)",) \
		$(EXTRA_ARGS)

screen-default:
	@if [ -z "$(REPO)" ]; then echo "REPO is required"; exit 1; fi
	@$(PYTHON_RUN) \
		--repo "$(REPO)" \
		$(if $(OUTDIR),--outdir "$(OUTDIR)",) \
		$(EXTRA_ARGS)

screen-path:
	@if [ -z "$(REPO)" ]; then echo "REPO is required"; exit 1; fi
	@if [ -z "$(CONFIG_PATH)" ]; then echo "CONFIG_PATH is required"; exit 1; fi
	@if [ ! -f "$(CONFIG_PATH)" ]; then echo "Config file not found: $(CONFIG_PATH)"; exit 1; fi
	@$(PYTHON_RUN) \
		--repo "$(REPO)" \
		--config "$(CONFIG_PATH)" \
		$(if $(OUTDIR),--outdir "$(OUTDIR)",) \
		$(EXTRA_ARGS)

screen-context-dry:
	@if [ -z "$(REPO)" ]; then echo "REPO is required"; exit 1; fi
	@$(PYTHON_RUN) \
		--repo "$(REPO)" \
		--config "$(CONFIG)" \
		$(if $(OUTDIR),--outdir "$(OUTDIR)",) \
		--dry-run-context \
		$(EXTRA_ARGS)
