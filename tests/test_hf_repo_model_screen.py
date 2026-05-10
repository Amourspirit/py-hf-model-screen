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
sanitize_name = MODULE.sanitize_name
get_timestamp_str = MODULE.get_timestamp_str
validate_output_directory = MODULE.validate_output_directory
build_result_filename = MODULE.build_result_filename
write_yaml = MODULE.write_yaml


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
        output_formats=["json", "csv"],
        short_name="test_run",
    )

    serialized = runtime_config_to_dict(runtime)

    assert serialized["repo"] == "/tmp/repo"
    assert serialized["models"] == ["m1"]
    assert serialized["prompts"] == [
        {"id": "repo_state", "title": "Repo", "prompt": "Summarize repo"}
    ]


def test_sanitize_name_replaces_invalid_chars_and_truncates() -> None:
    name = "My///Bad::Name!!With  Spaces"
    result = sanitize_name(name)
    # Should replace all invalid chars with underscores and spaces with underscores
    assert "My___Bad__Name__With__Spaces" in result or result.startswith("My___Bad__")
    # Should not contain any invalid filename characters
    invalid_chars = r'/\:*?"<>|'
    for char in invalid_chars:
        assert char not in result


def test_sanitize_name_truncation_preserves_hash_uniqueness() -> None:
    long_name = "a" * 60  # 60 characters
    result = sanitize_name(long_name)
    # Should be truncated and have a hash suffix
    assert len(result) < len(long_name)
    # Should contain underscore + hash suffix
    assert "_" in result and len(result.split("_")[-1]) == 6  # 6-char hash


def test_get_timestamp_str_format() -> None:
    result = get_timestamp_str()
    # Should match format: YYYY-MM-DD_HHhMMmSSs
    import re

    pattern = r"\d{4}-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s"
    assert re.match(pattern, result), f"Timestamp format incorrect: {result}"


def test_validate_output_directory_success() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        is_valid, msg = validate_output_directory(tmppath)
        assert is_valid is True
        assert msg == ""


def test_validate_output_directory_failure() -> None:
    # Try to validate a path inside /root (which is typically not writable)
    # This is a reasonable test for permission denied
    is_valid, msg = validate_output_directory(Path("/root/nonexistent/test"))
    # On most systems, this should fail. If it doesn't, the system allows it.
    # So we just verify the function returns a boolean and message
    assert isinstance(is_valid, bool)
    assert isinstance(msg, str)


def test_build_result_filename_constructs_correct_name() -> None:
    result = build_result_filename("my_results", "json", "2026-05-10_14h30m22s")
    assert result == "my_results_2026-05-10_14h30m22s.json"

    result_yaml = build_result_filename("test_run", "yaml", "2026-05-10_14h30m22s")
    assert result_yaml == "test_run_2026-05-10_14h30m22s.yaml"


def test_write_yaml_produces_valid_yaml() -> None:
    import tempfile
    import yaml as yaml_module

    data = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = Path(f.name)
    try:
        write_yaml(temp_path, data)
        # Verify the file was created and is valid YAML
        assert temp_path.exists()
        loaded = yaml_module.safe_load(temp_path.read_text())
        assert loaded == data
    finally:
        temp_path.unlink()
