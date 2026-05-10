#!/usr/bin/env python3
"""
Hugging Face model screening harness for private local repos.

What it does
------------
- Reads selected files from a local repository checkout
- Builds a compact repo context block
- Runs fixed prompts against one or more Hugging Face chat models
- Captures latency, usage, and heuristic quality metrics
- Writes JSON, CSV, and Markdown reports

Tested design target
--------------------
- Hugging Face OpenAI-compatible router: https://router.huggingface.co/v1
- Chat completions only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "project-config" / "default_config.yaml"
LOCAL_CONFIG_DIR = PROJECT_ROOT / "project-config-local"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class PromptConfig:
    id: str
    title: str
    prompt: str


@dataclass
class RuntimeConfig:
    repo: str | None
    output_dir: str
    models: List[str]
    include_files: List[str]
    include_globs: List[str]
    prompts: List[PromptConfig]
    passes: int
    max_context_chars: int
    max_file_chars: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    provider: str | None
    save_context: bool
    base_url: str
    output_formats: List[str]
    short_name: str


@dataclass
class RunMetrics:
    model: str
    prompt_id: str
    prompt_title: str
    pass_index: int
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    output_chars: int
    output_lines: int
    bullet_count: int
    starts_with_plan_phrase: bool
    contains_plan_phrase_anywhere: bool
    contains_direct_answer_cue: bool
    contains_tests_reference: bool
    contains_docs_reference: bool
    contains_config_reference: bool
    contains_risk_reference: bool
    contains_file_reference: bool
    contains_repo_hallucination_hint: bool
    contains_markdown_code_fence: bool
    non_ascii_ratio: float
    repeated_fragment_score: float
    finish_reason: str | None
    error: str | None


def get_with_alias(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def sanitize_name(name: str) -> str:
    """
    Sanitize a name for use as a filename.
    - Replace invalid filename chars with underscores
    - Convert spaces to underscores
    - Truncate to 50 chars and append MD5 hash to preserve uniqueness
    """
    invalid_chars = r'/\:*?"<>|'
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")
    sanitized = sanitized.replace(" ", "_")

    if len(sanitized) > 50:
        name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
        sanitized = sanitized[:50] + "_" + name_hash
    return sanitized


def get_timestamp_str() -> str:
    """Return current timestamp in human-readable format: YYYY-MM-DD_HHhMMmSSs"""
    return datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")


def validate_output_directory(outdir: Path) -> tuple[bool, str]:
    """
    Validate that output directory is writable.
    Returns (True, "") on success, (False, error_msg) on failure.
    """
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        test_file = outdir / ".hf_model_screen_write_test"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        return True, ""
    except Exception as exc:
        return False, f"Output directory not writable: {exc}"


def write_yaml(path: Path, data: Any) -> None:
    """Write data to YAML file."""
    path.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")


def build_result_filename(short_name: str, format_ext: str, timestamp: str) -> str:
    """
    Build timestamped result filename.
    Returns: {short_name}_{timestamp}.{ext}
    """
    return f"{short_name}_{timestamp}.{format_ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen Hugging Face chat models against local repo context."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a YAML config file, or config name that resolves to "
            "project-config-local/<name>.yaml"
        ),
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Absolute path to your local repository checkout",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for reports (overrides YAML output_dir)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model IDs to test (overrides YAML models)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="How many times to run each prompt per model",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Max total chars of repo context",
    )
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=None,
        help="Max chars per included file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="max_tokens for completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        dest="top_p",
        default=None,
        help="Sampling top_p",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        dest="top_k",
        default=None,
        help="Provider-specific top_k via extra_body",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token; otherwise reads HF_TOKEN env var",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider hint, e.g. hf-inference",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Extra repo file to include, relative path",
    )
    parser.add_argument(
        "--extra-glob",
        action="append",
        default=[],
        help="Extra glob to include, relative pattern",
    )
    parser.add_argument(
        "--save-context",
        action="store_true",
        default=None,
        help="Save assembled repo context to disk",
    )
    parser.add_argument(
        "--no-save-context",
        action="store_false",
        dest="save_context",
        help="Do not save assembled repo context",
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print resolved runtime configuration as JSON and continue",
    )
    parser.add_argument(
        "--dry-run-context",
        action="store_true",
        help=(
            "Build repo context and write repo_context.txt, then exit without "
            "calling the API"
        ),
    )
    parser.add_argument(
        "--output-formats",
        nargs="*",
        default=None,
        help=(
            "Output formats to generate (default: json, csv, markdown, yaml). "
            "Space-separated list, e.g., json yaml csv"
        ),
    )
    parser.add_argument(
        "--short-name",
        default=None,
        help=(
            "Short name for output filenames (will be sanitized). "
            "Example: qwen_screening. Defaults to 'results'"
        ),
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str | None) -> Path:
    if not config_arg:
        return DEFAULT_CONFIG_PATH

    raw = config_arg.strip()
    input_path = Path(raw).expanduser()
    looks_like_path = input_path.is_absolute() or raw.startswith(".") or "/" in raw
    if looks_like_path:
        return input_path.resolve()

    name = raw if raw.endswith((".yaml", ".yml")) else f"{raw}.yaml"
    return (LOCAL_CONFIG_DIR / name).resolve()


def read_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Config file not found: {path}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Unable to parse YAML config at {path}: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Config root must be a mapping in {path}")
    return loaded


def load_model_screen_section(path: Path) -> Dict[str, Any]:
    raw = read_yaml_file(path)
    section = raw.get("model_screen", raw)
    if not isinstance(section, dict):
        raise RuntimeError(
            f"Config file {path} must have a 'model_screen' mapping at top-level"
        )
    return section


def merge_dicts(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_prompts(raw_prompts: Any) -> List[PromptConfig]:
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise RuntimeError(
            "Config must define a non-empty list at model_screen.config.prompts"
        )

    prompts: List[PromptConfig] = []
    for idx, entry in enumerate(raw_prompts, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Prompt at index {idx} must be a mapping")

        prompt_id = entry.get("id")
        title = entry.get("title")
        prompt = entry.get("prompt")
        if not prompt_id or not title or not prompt:
            raise RuntimeError(
                f"Prompt at index {idx} must define non-empty id, title, and prompt"
            )
        prompts.append(
            PromptConfig(id=str(prompt_id), title=str(title), prompt=str(prompt))
        )

    return prompts


def build_runtime_config(section: Dict[str, Any]) -> RuntimeConfig:
    cfg = section.get("config", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("model_screen.config must be a mapping")

    params = section.get("params", section.get("prams", {}))
    if not isinstance(params, dict):
        raise RuntimeError("model_screen.params must be a mapping")

    models = get_with_alias(cfg, "models", default=[])
    include_files = get_with_alias(cfg, "include_files", "include-files", default=[])
    include_globs = get_with_alias(cfg, "include_globs", "include-globs", default=[])
    prompts = validate_prompts(get_with_alias(cfg, "prompts", default=[]))

    if not isinstance(models, list):
        raise RuntimeError("model_screen.config.models must be a list")
    if not isinstance(include_files, list):
        raise RuntimeError("model_screen.config.include_files must be a list")
    if not isinstance(include_globs, list):
        raise RuntimeError("model_screen.config.include_globs must be a list")

    repo = get_with_alias(params, "repo", default=None)
    output_dir = get_with_alias(params, "output_dir", "output-dir", default="./outputs")
    passes = int(get_with_alias(params, "passes", default=2))
    max_context_chars = int(
        get_with_alias(params, "max_context_chars", "max-context-chars", default=36000)
    )
    max_file_chars = int(
        get_with_alias(params, "max_file_chars", "max-file-chars", default=3500)
    )
    max_tokens = int(get_with_alias(params, "max_tokens", "max-tokens", default=900))
    temperature = float(get_with_alias(params, "temperature", default=0.2))
    top_p = float(get_with_alias(params, "top_p", "top-p", default=0.95))
    top_k = int(get_with_alias(params, "top_k", "top-k", default=20))
    provider = get_with_alias(params, "provider", default=None)
    save_context = bool(
        get_with_alias(params, "save_context", "save-context", default=False)
    )
    base_url = str(
        get_with_alias(
            params,
            "base_url",
            "base-url",
            default="https://router.huggingface.co/v1",
        )
    )
    output_formats = get_with_alias(
        params,
        "output_formats",
        "output-formats",
        default=["json", "csv", "markdown", "yaml"],
    )
    if not isinstance(output_formats, list):
        output_formats = ["json", "csv", "markdown", "yaml"]
    short_name = str(
        get_with_alias(params, "short_name", "short-name", default="results")
    )

    return RuntimeConfig(
        repo=str(repo) if repo is not None else None,
        output_dir=str(output_dir),
        models=[str(model) for model in models],
        include_files=[str(path) for path in include_files],
        include_globs=[str(pattern) for pattern in include_globs],
        prompts=prompts,
        passes=passes,
        max_context_chars=max_context_chars,
        max_file_chars=max_file_chars,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        provider=str(provider) if provider is not None else None,
        save_context=save_context,
        base_url=base_url,
        output_formats=[str(fmt) for fmt in output_formats],
        short_name=short_name,
    )


def apply_cli_overrides(
    config: RuntimeConfig, args: argparse.Namespace
) -> RuntimeConfig:
    if args.repo:
        config.repo = args.repo
    if args.outdir:
        config.output_dir = args.outdir
    if args.models is not None and len(args.models) > 0:
        config.models = args.models
    if args.passes is not None:
        config.passes = args.passes
    if args.max_context_chars is not None:
        config.max_context_chars = args.max_context_chars
    if args.max_file_chars is not None:
        config.max_file_chars = args.max_file_chars
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.provider is not None:
        config.provider = args.provider
    if args.save_context is not None:
        config.save_context = args.save_context

    if args.extra_file:
        config.include_files = [*config.include_files, *args.extra_file]
    if args.extra_glob:
        config.include_globs = [*config.include_globs, *args.extra_glob]
    if args.output_formats is not None and len(args.output_formats) > 0:
        config.output_formats = args.output_formats
    if args.short_name is not None:
        config.short_name = args.short_name
    return config


def resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token

    raise RuntimeError("No Hugging Face token found. Set HF_TOKEN or pass --token.")


def safe_read_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.strip()
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]..."
        return text
    except Exception as exc:
        return f"[error reading {path}: {exc}]"


def gather_repo_files(
    repo_root: Path,
    include_files: List[str],
    include_globs: List[str],
) -> List[Path]:
    found: List[Path] = []

    for rel in include_files:
        p = repo_root / rel
        if p.exists() and p.is_file():
            found.append(p)

    for pattern in include_globs:
        for p in repo_root.glob(pattern):
            if p.is_file():
                found.append(p)

    unique = []
    seen = set()
    for p in sorted(found):
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def render_repo_tree(repo_root: Path, max_entries: int = 200) -> str:
    """
    Lightweight top-level-ish tree. Avoids huge recursive dumps.
    """
    lines = []
    count = 0

    def add_line(line: str):
        nonlocal count
        if count < max_entries:
            lines.append(line)
            count += 1

    top = sorted(repo_root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for item in top:
        if item.name.startswith(".git"):
            continue
        if item.is_dir():
            add_line(f"{item.name}/")
            try:
                children = sorted(
                    item.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
                )
                for child in children[:20]:
                    suffix = "/" if child.is_dir() else ""
                    add_line(f"  {child.name}{suffix}")
            except Exception:
                pass
        else:
            add_line(item.name)

    if count >= max_entries:
        lines.append("...[tree truncated]...")

    return "\n".join(lines)


def build_repo_context(
    repo_root: Path,
    files: List[Path],
    max_context_chars: int,
    max_file_chars: int,
) -> str:
    parts: List[str] = []

    parts.append("# Repository Tree\n")
    parts.append(render_repo_tree(repo_root))
    parts.append("\n")

    for path in files:
        rel = path.relative_to(repo_root)
        text = safe_read_text(path, max_file_chars)
        section = f"# File: {rel}\n{text}\n"
        parts.append(section)

    context = "\n".join(parts)
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n...[context truncated]..."
    return context


def build_messages(repo_context: str, prompt_text: str) -> List[Dict[str, str]]:
    system = (
        "You are evaluating a private software repository for workflow fit.\n"
        "Stay grounded in the provided repository context.\n"
        "Do not invent files, services, or recent activity not supported by the context.\n"
        "Do not narrate your plan.\n"
        "Answer directly.\n"
    )

    user = f"{prompt_text}\n\nRepository context:\n\n{repo_context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


PLAN_PATTERNS = [
    r"\bi'll check\b",
    r"\bi will check\b",
    r"\blet me check\b",
    r"\bi'll gather\b",
    r"\blet me gather\b",
    r"\bi'll explore\b",
    r"\blet me explore\b",
    r"\bi'll look at\b",
    r"\blet me look at\b",
    r"\bi need to\b",
]

DIRECT_ANSWER_PATTERNS = [
    r"\bbased on the repository context\b",
    r"\bhere(?:'s| is)\b",
    r"\bthe current state\b",
    r"\bsummary\b",
]

FILE_REF_PATTERNS = [
    r"\bREADME\.md\b",
    r"\bAGENTS\.md\b",
    r"\bMakefile\b",
    r"\bdocker-compose\.yml\b",
    r"\b[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+\.py\b",
]

HALLUCINATION_HINTS = [
    r"\bgit status\b",
    r"\brecent commits\b",
    r"\bI checked\b",
    r"\bI found\b",
]


def count_bullets(text: str) -> int:
    return sum(
        1 for line in text.splitlines() if re.match(r"^\s*(?:[-*•]|\d+\.)\s+", line)
    )


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / max(1, len(text))


def repeated_fragment_score(text: str) -> float:
    """
    Cheap corruption/repetition heuristic.
    Higher = more suspicious.
    """
    if not text:
        return 0.0
    windows = [text[i : i + 8] for i in range(0, max(0, len(text) - 7), 4)]
    if not windows:
        return 0.0
    freq: Dict[str, int] = {}
    for w in windows:
        freq[w] = freq.get(w, 0) + 1
    repeated = sum(v for v in freq.values() if v > 2)
    return repeated / len(windows)


def has_any_pattern(text: str, patterns: List[str], flags: int = re.IGNORECASE) -> bool:
    return any(re.search(p, text, flags=flags) for p in patterns)


def analyze_output(
    model: str,
    prompt_id: str,
    prompt_title: str,
    pass_index: int,
    elapsed_seconds: float,
    response_obj: Any | None,
    output_text: str,
    error: str | None,
) -> RunMetrics:
    usage = getattr(response_obj, "usage", None)
    choices = getattr(response_obj, "choices", None)

    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    finish_reason = None
    if choices and len(choices) > 0:
        finish_reason = getattr(choices[0], "finish_reason", None)

    stripped = output_text.strip()
    starts_with_plan_phrase = has_any_pattern(stripped[:200], PLAN_PATTERNS)
    contains_plan_phrase_anywhere = has_any_pattern(stripped, PLAN_PATTERNS)
    contains_direct_answer_cue = has_any_pattern(stripped, DIRECT_ANSWER_PATTERNS)
    contains_tests_reference = bool(
        re.search(r"\btest(?:s|ing)?\b|\bpytest\b|\bvalidation\b", stripped, re.I)
    )
    contains_docs_reference = bool(
        re.search(r"\bdocs?\b|\breadme\b|\bdocumentation\b", stripped, re.I)
    )
    contains_config_reference = bool(
        re.search(r"\bconfig\b|\b\.env\b|\byml\b|\byaml\b", stripped, re.I)
    )
    contains_risk_reference = bool(
        re.search(r"\brisk\b|\bunsafe\b|\bif done incorrectly\b", stripped, re.I)
    )
    contains_file_reference = has_any_pattern(stripped, FILE_REF_PATTERNS)
    contains_repo_hallucination_hint = has_any_pattern(stripped, HALLUCINATION_HINTS)
    contains_markdown_code_fence = "```" in stripped

    return RunMetrics(
        model=model,
        prompt_id=prompt_id,
        prompt_title=prompt_title,
        pass_index=pass_index,
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_chars=len(output_text),
        output_lines=len(output_text.splitlines()),
        bullet_count=count_bullets(output_text),
        starts_with_plan_phrase=starts_with_plan_phrase,
        contains_plan_phrase_anywhere=contains_plan_phrase_anywhere,
        contains_direct_answer_cue=contains_direct_answer_cue,
        contains_tests_reference=contains_tests_reference,
        contains_docs_reference=contains_docs_reference,
        contains_config_reference=contains_config_reference,
        contains_risk_reference=contains_risk_reference,
        contains_file_reference=contains_file_reference,
        contains_repo_hallucination_hint=contains_repo_hallucination_hint,
        contains_markdown_code_fence=contains_markdown_code_fence,
        non_ascii_ratio=round(non_ascii_ratio(output_text), 4),
        repeated_fragment_score=round(repeated_fragment_score(output_text), 4),
        finish_reason=finish_reason,
        error=error,
    )


def run_one(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    provider: str | None,
) -> Tuple[Any | None, str, float, str | None]:
    start = time.perf_counter()
    try:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "top_k": top_k,
            },
        }

        response = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - start
        content = response.choices[0].message.content if response.choices else ""
        return response, content or "", elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return None, "", elapsed, f"{type(exc).__name__}: {exc}"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(metrics_rows: List[RunMetrics]) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[RunMetrics]] = {}
    for row in metrics_rows:
        by_model.setdefault(row.model, []).append(row)

    summary = []
    for model, rows in by_model.items():
        n = len(rows)
        avg_elapsed = sum(r.elapsed_seconds for r in rows) / n
        avg_prompt_tokens = sum(r.prompt_tokens or 0 for r in rows) / n if n else 0.0
        avg_completion_tokens = (
            sum(r.completion_tokens or 0 for r in rows) / n if n else 0.0
        )

        directness_penalty = sum(
            1
            for r in rows
            if r.starts_with_plan_phrase or r.contains_plan_phrase_anywhere
        )
        corruption_penalty = sum(
            1
            for r in rows
            if r.repeated_fragment_score > 0.20 or r.non_ascii_ratio > 0.10
        )
        structure_bonus = sum(1 for r in rows if r.bullet_count >= 4)
        test_bonus = sum(1 for r in rows if r.contains_tests_reference)
        docs_bonus = sum(1 for r in rows if r.contains_docs_reference)
        config_bonus = sum(1 for r in rows if r.contains_config_reference)
        risk_bonus = sum(1 for r in rows if r.contains_risk_reference)
        file_bonus = sum(1 for r in rows if r.contains_file_reference)

        composite = (
            50
            + structure_bonus * 5
            + test_bonus * 4
            + docs_bonus * 3
            + config_bonus * 3
            + risk_bonus * 3
            + file_bonus * 4
            - directness_penalty * 8
            - corruption_penalty * 12
        )
        composite = max(0, min(100, composite))

        summary.append(
            {
                "model": model,
                "runs": n,
                "avg_elapsed_seconds": round(avg_elapsed, 2),
                "avg_prompt_tokens": round(avg_prompt_tokens, 1),
                "avg_completion_tokens": round(avg_completion_tokens, 1),
                "directness_penalty_count": directness_penalty,
                "corruption_penalty_count": corruption_penalty,
                "structure_bonus_count": structure_bonus,
                "tests_bonus_count": test_bonus,
                "docs_bonus_count": docs_bonus,
                "config_bonus_count": config_bonus,
                "risk_bonus_count": risk_bonus,
                "file_bonus_count": file_bonus,
                "heuristic_composite_score_100": composite,
            }
        )

    summary.sort(
        key=lambda x: (-x["heuristic_composite_score_100"], x["avg_elapsed_seconds"])
    )
    return summary


def write_markdown_report(
    path: Path,
    repo_root: Path,
    models: List[str],
    metrics_rows: List[RunMetrics],
    raw_outputs: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
) -> None:
    lines = []
    lines.append("# Hugging Face Model Screening Report\n")
    lines.append(f"- Repo: `{repo_root}`")
    lines.append(f"- Models tested: {', '.join(f'`{m}`' for m in models)}")
    lines.append("")

    lines.append("## Summary Ranking\n")
    for row in summary_rows:
        lines.append(
            f"- **{row['model']}** - score {row['heuristic_composite_score_100']}/100, "
            f"avg elapsed {row['avg_elapsed_seconds']}s, "
            f"avg prompt tokens {row['avg_prompt_tokens']}, "
            f"avg completion tokens {row['avg_completion_tokens']}"
        )
    lines.append("")

    lines.append("## Per-run Metrics\n")
    for m in metrics_rows:
        lines.append(
            f"- **{m.model} / {m.prompt_id} / pass {m.pass_index}** - "
            f"{m.elapsed_seconds:.2f}s, prompt={m.prompt_tokens}, completion={m.completion_tokens}, "
            f"bullets={m.bullet_count}, plan_start={m.starts_with_plan_phrase}, "
            f"tests={m.contains_tests_reference}, docs={m.contains_docs_reference}, "
            f"config={m.contains_config_reference}, risk={m.contains_risk_reference}, "
            f"files={m.contains_file_reference}, repeated={m.repeated_fragment_score}, "
            f"non_ascii_ratio={m.non_ascii_ratio}, finish={m.finish_reason}, error={m.error}"
        )
    lines.append("")

    lines.append("## Raw Outputs\n")
    for item in raw_outputs:
        lines.append(
            f"### {item['model']} / {item['prompt_id']} / pass {item['pass_index']}\n"
        )
        lines.append("```text")
        lines.append(item["output"])
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def runtime_config_to_dict(config: RuntimeConfig) -> Dict[str, Any]:
    return {
        "repo": config.repo,
        "output_dir": config.output_dir,
        "models": config.models,
        "include_files": config.include_files,
        "include_globs": config.include_globs,
        "prompts": [asdict(prompt) for prompt in config.prompts],
        "passes": config.passes,
        "max_context_chars": config.max_context_chars,
        "max_file_chars": config.max_file_chars,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "provider": config.provider,
        "save_context": config.save_context,
        "base_url": config.base_url,
        "output_formats": config.output_formats,
        "short_name": config.short_name,
    }


def build_effective_runtime_config(
    args: argparse.Namespace,
) -> tuple[RuntimeConfig, Path]:
    defaults_section = load_model_screen_section(DEFAULT_CONFIG_PATH)
    config_source = DEFAULT_CONFIG_PATH

    if args.config:
        user_config_path = resolve_config_path(args.config)
        user_section = load_model_screen_section(user_config_path)
        merged_section = merge_dicts(defaults_section, user_section)
        config_source = user_config_path
    else:
        merged_section = defaults_section

    runtime = build_runtime_config(merged_section)
    runtime = apply_cli_overrides(runtime, args)

    if not runtime.repo:
        raise RuntimeError(
            "Missing repository path. Set model_screen.params.repo in config or pass --repo."
        )
    if not runtime.models:
        raise RuntimeError("No models configured. Set model_screen.config.models.")

    return runtime, config_source


def main() -> int:
    args = parse_args()

    try:
        runtime, config_source = build_effective_runtime_config(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    repo_root = Path(runtime.repo).expanduser().resolve()
    outdir = Path(runtime.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Validate output directory writability before expensive operations
    is_valid, validation_msg = validate_output_directory(outdir)
    if not is_valid:
        print(validation_msg, file=sys.stderr)
        return 2

    if not repo_root.exists() or not repo_root.is_dir():
        print(
            f"Repo path does not exist or is not a directory: {repo_root}",
            file=sys.stderr,
        )
        return 2

    repo_files = gather_repo_files(
        repo_root, runtime.include_files, runtime.include_globs
    )
    repo_context = build_repo_context(
        repo_root=repo_root,
        files=repo_files,
        max_context_chars=runtime.max_context_chars,
        max_file_chars=runtime.max_file_chars,
    )

    if args.print_effective_config:
        print(
            json.dumps(
                {
                    "config_source": str(config_source),
                    "runtime": runtime_config_to_dict(runtime),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    if args.dry_run_context:
        context_path = outdir / "repo_context.txt"
        context_path.write_text(repo_context, encoding="utf-8")
        print(f"Dry run complete. Context written to: {context_path}")
        return 0

    if runtime.save_context:
        (outdir / "repo_context.txt").write_text(repo_context, encoding="utf-8")

    token = resolve_token(args.token)

    client = OpenAI(
        base_url=runtime.base_url,
        api_key=token,
    )

    metrics_rows: List[RunMetrics] = []
    raw_outputs: List[Dict[str, Any]] = []

    for model in runtime.models:
        for prompt_cfg in runtime.prompts:
            messages = build_messages(repo_context, prompt_cfg.prompt)
            for pass_index in range(1, runtime.passes + 1):
                print(
                    f"Running model={model} prompt={prompt_cfg.id} pass={pass_index} ..."
                )
                response, output, elapsed, error = run_one(
                    client=client,
                    model=model,
                    messages=messages,
                    max_tokens=runtime.max_tokens,
                    temperature=runtime.temperature,
                    top_p=runtime.top_p,
                    top_k=runtime.top_k,
                    provider=runtime.provider,
                )

                metrics = analyze_output(
                    model=model,
                    prompt_id=prompt_cfg.id,
                    prompt_title=prompt_cfg.title,
                    pass_index=pass_index,
                    elapsed_seconds=elapsed,
                    response_obj=response,
                    output_text=output,
                    error=error,
                )
                metrics_rows.append(metrics)
                raw_outputs.append(
                    {
                        "model": model,
                        "prompt_id": prompt_cfg.id,
                        "prompt_title": prompt_cfg.title,
                        "pass_index": pass_index,
                        "output": output,
                        "error": error,
                    }
                )

    summary_rows = summarize_scores(metrics_rows)

    metrics_dicts = [asdict(m) for m in metrics_rows]
    # Prepare report data
    report_data = {
        "repo": str(repo_root),
        "models": runtime.models,
        "config_source": str(config_source),
        "config": runtime_config_to_dict(runtime),
        "summary": summary_rows,
        "metrics": metrics_dicts,
        "outputs": raw_outputs,
    }

    # Generate timestamped filenames
    timestamp = get_timestamp_str()
    safe_name = sanitize_name(runtime.short_name)
    write_dir = outdir
    fallback_used = False

    # Try to write reports to primary output directory, fallback to temp if needed
    try:
        if "json" in runtime.output_formats:
            write_json(
                write_dir / build_result_filename(safe_name, "json", timestamp),
                report_data,
            )
        if "csv" in runtime.output_formats:
            write_csv(
                write_dir / build_result_filename(safe_name, "csv", timestamp),
                metrics_dicts,
            )
        if "markdown" in runtime.output_formats:
            write_markdown_report(
                write_dir / build_result_filename(safe_name, "md", timestamp),
                repo_root=repo_root,
                models=runtime.models,
                metrics_rows=metrics_rows,
                raw_outputs=raw_outputs,
                summary_rows=summary_rows,
            )
        if "yaml" in runtime.output_formats:
            write_yaml(
                write_dir / build_result_filename(safe_name, "yaml", timestamp),
                report_data,
            )
    except (OSError, IOError, PermissionError) as exc:
        # Fallback to temp directory
        fallback_used = True
        write_dir = Path(tempfile.gettempdir()) / "hf-model-screen"
        write_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[WARNING] Primary output directory unavailable: {exc}",
            file=sys.stderr,
        )
        print(
            f"[WARNING] Attempting fallback write to: {write_dir}",
            file=sys.stderr,
        )

        try:
            if "json" in runtime.output_formats:
                write_json(
                    write_dir / build_result_filename(safe_name, "json", timestamp),
                    report_data,
                )
            if "csv" in runtime.output_formats:
                write_csv(
                    write_dir / build_result_filename(safe_name, "csv", timestamp),
                    metrics_dicts,
                )
            if "markdown" in runtime.output_formats:
                write_markdown_report(
                    write_dir / build_result_filename(safe_name, "md", timestamp),
                    repo_root=repo_root,
                    models=runtime.models,
                    metrics_rows=metrics_rows,
                    raw_outputs=raw_outputs,
                    summary_rows=summary_rows,
                )
            if "yaml" in runtime.output_formats:
                write_yaml(
                    write_dir / build_result_filename(safe_name, "yaml", timestamp),
                    report_data,
                )
        except Exception as fallback_exc:
            print(f"[ERROR] Fallback write failed: {fallback_exc}", file=sys.stderr)
            return 2

    print(f"\nDone. Reports written to: {write_dir}")
    for fmt in runtime.output_formats:
        ext = "md" if fmt == "markdown" else fmt
        filename = build_result_filename(safe_name, ext, timestamp)
        print(f"- {write_dir / filename}")

    if fallback_used:
        print(
            "\n[WARNING] Primary output directory was unavailable; reports saved to fallback location above."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
