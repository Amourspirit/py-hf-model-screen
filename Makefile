PYTHON_RUN := uv run python ./scripts/hf_repo_model_screen.py
CONFIG ?= ./project-config/default_config.yaml
REPO ?=
OUTDIR ?=
NAME ?=
EXTRA_ARGS ?=

.PHONY: screen screen-context-dry help

help:
	@echo "Usage:"
	@echo "  make screen [CONFIG=config_name_or_path] [REPO=/path/to/repo] [NAME=short_name] [OUTDIR=./outputs]"
	@echo "  make screen-context-dry [CONFIG=config_name_or_path] [REPO=/path/to/repo] [OUTDIR=./outputs]"
	@echo ""
	@echo "Notes:"
	@echo "  - CONFIG defaults to ./project-config/default_config.yaml"
	@echo "  - CONFIG can be a config name (resolved to project-config-local/) or an absolute path"
	@echo "  - REPO overrides the repo path from config (optional)"
	@echo "  - NAME sets --short-name for output filenames (optional)"

screen:
	@$(PYTHON_RUN) \
		--config "$(CONFIG)" \
		$(if $(REPO),--repo "$(REPO)",) \
		$(if $(NAME),--short-name "$(NAME)",) \
		$(if $(OUTDIR),--outdir "$(OUTDIR)",) \
		$(EXTRA_ARGS)

screen-context-dry:
	@$(PYTHON_RUN) \
		--config "$(CONFIG)" \
		$(if $(REPO),--repo "$(REPO)",) \
		$(if $(OUTDIR),--outdir "$(OUTDIR)",) \
		--dry-run-context \
		$(EXTRA_ARGS)
