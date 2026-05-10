from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "hf_repo_model_screen.py"
)
SPEC = spec_from_file_location("hf_repo_model_screen", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load hf_repo_model_screen module for tests")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

LOCAL_CONFIG_DIR = MODULE.LOCAL_CONFIG_DIR
build_runtime_config = MODULE.build_runtime_config
merge_dicts = MODULE.merge_dicts
resolve_config_path = MODULE.resolve_config_path
run_one = MODULE.run_one
runtime_config_to_dict = MODULE.runtime_config_to_dict
PromptConfig = MODULE.PromptConfig
RuntimeConfig = MODULE.RuntimeConfig


def test_resolve_config_path_name_mode() -> None:
    resolved = resolve_config_path("qwen_config")
    assert resolved == (LOCAL_CONFIG_DIR / "qwen_config.yaml").resolve()


def test_resolve_config_path_explicit_file(tmp_path) -> None:
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("model_screen: {}", encoding="utf-8")
    resolved = resolve_config_path(str(cfg))
    assert resolved == cfg.resolve()


def test_resolve_config_path_name_mode_with_yaml_suffix() -> None:
    resolved = resolve_config_path("qwen_config.yaml")
    assert resolved == (LOCAL_CONFIG_DIR / "qwen_config.yaml").resolve()


def test_merge_dicts_replaces_lists() -> None:
    defaults = {
        "config": {"models": ["a", "b"], "include_files": ["README.md"]},
        "params": {"passes": 2},
    }
    overrides = {
        "config": {"models": ["only-this"]},
        "params": {"passes": 4},
    }

    merged = merge_dicts(defaults, overrides)

    assert merged["config"]["models"] == ["only-this"]
    assert merged["config"]["include_files"] == ["README.md"]
    assert merged["params"]["passes"] == 4


def test_build_runtime_config_supports_params_alias_prams() -> None:
    section = {
        "config": {
            "models": ["m1"],
            "include_files": ["README.md"],
            "include_globs": ["docs/**/*.md"],
            "prompts": [{"id": "repo_state", "title": "Repo", "prompt": "summarize"}],
        },
        "prams": {
            "repo": "/tmp/repo",
            "output_dir": "./outputs",
            "passes": 3,
            "max_context_chars": 1200,
            "max_file_chars": 500,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 10,
            "save_context": True,
        },
    }

    config = build_runtime_config(section)

    assert config.repo == "/tmp/repo"
    assert config.passes == 3
    assert config.max_context_chars == 1200
    assert config.max_file_chars == 500
    assert config.max_tokens == 200
    assert config.temperature == 0.1
    assert config.top_p == 0.9
    assert config.top_k == 10
    assert config.save_context is True
    assert len(config.prompts) == 1


def test_build_runtime_config_rejects_invalid_prompt_shape() -> None:
    section = {
        "config": {
            "models": ["m1"],
            "include_files": [],
            "include_globs": [],
            "prompts": [{"id": "missing-title-and-prompt"}],
        },
        "params": {"repo": "/tmp/repo"},
    }

    with pytest.raises(RuntimeError, match="id, title, and prompt"):
        build_runtime_config(section)


def test_run_one_uses_client_and_handles_success() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )

    response_obj, output, elapsed, error = run_one(
        client=client,
        model="m1",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
        provider=None,
    )

    assert response_obj is response
    assert output == "ok"
    assert elapsed >= 0
    assert error is None


def test_run_one_handles_exceptions_without_raising() -> None:
    def _boom(**_):
        raise RuntimeError("boom")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_boom))
    )

    response_obj, output, elapsed, error = run_one(
        client=client,
        model="m1",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
        provider=None,
    )

    assert response_obj is None
    assert output == ""
    assert elapsed >= 0
    assert "RuntimeError" in (error or "")


def test_runtime_config_to_dict_serializes_prompt_objects() -> None:
    runtime = RuntimeConfig(
        repo="/tmp/repo",
        output_dir="./outputs",
        models=["m1"],
        include_files=["README.md"],
        include_globs=["docs/**/*.md"],
        prompts=[PromptConfig(id="repo_state", title="Repo", prompt="Summarize repo")],
        passes=2,
        max_context_chars=36000,
        max_file_chars=3500,
        max_tokens=900,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
        provider=None,
        save_context=True,
        base_url="https://router.huggingface.co/v1",
    )

    serialized = runtime_config_to_dict(runtime)

    assert serialized["repo"] == "/tmp/repo"
    assert serialized["models"] == ["m1"]
    assert serialized["prompts"] == [
        {"id": "repo_state", "title": "Repo", "prompt": "Summarize repo"}
    ]
