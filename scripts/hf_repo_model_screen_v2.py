#!/usr/bin/env python3
"""Round-two Hugging Face model screening harness.

This variant keeps configuration external and adds roadmap-fidelity metrics
for repos like llm-spiral-project where the best model must align to the
canonical roadmap, architecture docs, runbooks, and PR slice conventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "project-config" / "default_config.yaml"
LOCAL_CONFIG_DIR = PROJECT_ROOT / "project-config-local"
DEFAULT_OUTPUT_FORMATS = ["json", "csv", "markdown", "yaml"]

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
    models: list[str]
    include_files: list[str]
    include_globs: list[str]
    prompts: list[PromptConfig]
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
    output_formats: list[str]
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
    contains_roadmap_reference: bool
    contains_architecture_reference: bool
    contains_runbook_reference: bool
    mentions_active_workstream: bool
    mentions_planned_or_deferred_state: bool
    mentions_next_pr_or_slice: bool
    mentions_canonical_doc_path: bool
    contains_generic_cleanup_drift: bool
    contains_repo_hallucination_hint: bool
    contains_markdown_code_fence: bool
    non_ascii_ratio: float
    repeated_fragment_score: float
    finish_reason: str | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-two HF model screening harness")
    parser.add_argument("--config", default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--passes", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--max-file-chars", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, dest="top_p", default=None)
    parser.add_argument("--top-k", type=int, dest="top_k", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--extra-file", action="append", default=[])
    parser.add_argument("--extra-glob", action="append", default=[])
    parser.add_argument("--save-context", action="store_true", default=None)
    parser.add_argument("--no-save-context", action="store_false", dest="save_context")
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--dry-run-context", action="store_true")
    parser.add_argument("--output-formats", nargs="*", default=None)
    parser.add_argument("--short-name", default=None)
    return parser.parse_args()


def get_with_alias(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


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


def read_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Config root must be a mapping in {path}")
    return loaded


def load_model_screen_section(path: Path) -> dict[str, Any]:
    raw = read_yaml_file(path)
    section = raw.get("model_screen", raw)
    if not isinstance(section, dict):
        raise RuntimeError(f"Config file {path} must have a 'model_screen' mapping at top-level")
    return section


def merge_dicts(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_prompts(raw_prompts: Any) -> list[PromptConfig]:
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise RuntimeError("Config must define a non-empty list at model_screen.config.prompts")
    prompts: list[PromptConfig] = []
    for idx, entry in enumerate(raw_prompts, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Prompt at index {idx} must be a mapping")
        pid = entry.get("id")
        title = entry.get("title")
        prompt = entry.get("prompt")
        if not pid or not title or not prompt:
            raise RuntimeError(f"Prompt at index {idx} must define non-empty id, title, and prompt")
        prompts.append(PromptConfig(id=str(pid), title=str(title), prompt=str(prompt)))
    return prompts


def build_runtime_config(section: dict[str, Any]) -> RuntimeConfig:
    cfg = section.get("config", {})
    params = section.get("params", section.get("prams", {}))
    if not isinstance(cfg, dict) or not isinstance(params, dict):
        raise RuntimeError("model_screen.config and model_screen.params must be mappings")
    output_formats = get_with_alias(params, "output_formats", "output-formats", default=DEFAULT_OUTPUT_FORMATS)
    if not isinstance(output_formats, list) or not output_formats:
        output_formats = DEFAULT_OUTPUT_FORMATS.copy()
    return RuntimeConfig(
        repo=(str(get_with_alias(params, "repo", default=None)) if get_with_alias(params, "repo", default=None) is not None else None),
        output_dir=str(get_with_alias(params, "output_dir", "output-dir", default="./outputs")),
        models=[str(m) for m in get_with_alias(cfg, "models", default=[])],
        include_files=[str(p) for p in get_with_alias(cfg, "include_files", "include-files", default=[])],
        include_globs=[str(p) for p in get_with_alias(cfg, "include_globs", "include-globs", default=[])],
        prompts=validate_prompts(get_with_alias(cfg, "prompts", default=[])),
        passes=int(get_with_alias(params, "passes", default=2)),
        max_context_chars=int(get_with_alias(params, "max_context_chars", "max-context-chars", default=36000)),
        max_file_chars=int(get_with_alias(params, "max_file_chars", "max-file-chars", default=3500)),
        max_tokens=int(get_with_alias(params, "max_tokens", "max-tokens", default=900)),
        temperature=float(get_with_alias(params, "temperature", default=0.2)),
        top_p=float(get_with_alias(params, "top_p", "top-p", default=0.95)),
        top_k=int(get_with_alias(params, "top_k", "top-k", default=20)),
        provider=(str(get_with_alias(params, "provider", default=None)) if get_with_alias(params, "provider", default=None) is not None else None),
        save_context=bool(get_with_alias(params, "save_context", "save-context", default=False)),
        base_url=str(get_with_alias(params, "base_url", "base-url", default="https://router.huggingface.co/v1")),
        output_formats=[str(fmt) for fmt in output_formats],
        short_name=str(get_with_alias(params, "short_name", "short-name", default="results")),
    )


def apply_cli_overrides(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    if args.repo:
        config.repo = args.repo
    if args.outdir:
        config.output_dir = args.outdir
    if args.models:
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
    if args.output_formats:
        config.output_formats = [str(fmt) for fmt in args.output_formats]
    if args.short_name is not None:
        config.short_name = args.short_name
    return config


def sanitize_name(name: str) -> str:
    invalid_chars = r'/\\:*?"<>|'
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")
    sanitized = sanitized.replace(" ", "_")
    if len(sanitized) > 50:
        name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
        sanitized = sanitized[:50] + "_" + name_hash
    return sanitized


def get_timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")


def validate_output_directory(outdir: Path) -> tuple[bool, str]:
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        test_file = outdir / ".hf_model_screen_write_test"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        return True, ""
    except Exception as exc:
        return False, f"Output directory not writable: {exc}"


def build_result_filename(short_name: str, ext: str, timestamp: str) -> str:
    return f"{short_name}_{timestamp}.{ext}"


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
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]..."
        return text
    except Exception as exc:
        return f"[error reading {path}: {exc}]"


def gather_repo_files(repo_root: Path, include_files: list[str], include_globs: list[str]) -> list[Path]:
    found: list[Path] = []
    for rel in include_files:
        p = repo_root / rel
        if p.exists() and p.is_file():
            found.append(p)
    for pattern in include_globs:
        for p in repo_root.glob(pattern):
            if p.is_file():
                found.append(p)
    unique: list[Path] = []
    seen: set[str] = set()
    for p in sorted(found):
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def render_repo_tree(repo_root: Path, max_entries: int = 200) -> str:
    lines: list[str] = []
    count = 0
    def add_line(line: str) -> None:
        nonlocal count
        if count < max_entries:
            lines.append(line)
            count += 1
    for item in sorted(repo_root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if item.name.startswith(".git"):
            continue
        if item.is_dir():
            add_line(f"{item.name}/")
            try:
                children = sorted(item.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
                for child in children[:20]:
                    add_line(f"  {child.name}{'/' if child.is_dir() else ''}")
            except Exception:
                pass
        else:
            add_line(item.name)
    if count >= max_entries:
        lines.append("...[tree truncated]...")
    return "\n".join(lines)


def build_repo_context(repo_root: Path, files: list[Path], max_context_chars: int, max_file_chars: int) -> str:
    parts = ["# Repository Tree\n", render_repo_tree(repo_root), "\n"]
    for path in files:
        rel = path.relative_to(repo_root)
        parts.append(f"# File: {rel}\n{safe_read_text(path, max_file_chars)}\n")
    context = "\n".join(parts)
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n...[context truncated]..."
    return context


def build_messages(repo_context: str, prompt_text: str) -> list[dict[str, str]]:
    system = (
        "You are evaluating a private software repository for workflow fit.\n"
        "Stay grounded in the provided repository context.\n"
        "Prefer documented roadmap and architecture evidence over generic cleanup suggestions.\n"
        "Do not invent files, services, recent activity, or roadmap state not supported by the context.\n"
        "Do not narrate your plan.\n"
        "Answer directly.\n"
    )
    user = f"{prompt_text}\n\nRepository context:\n\n{repo_context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


PLAN_PATTERNS = [r"\bi'll check\b", r"\bi will check\b", r"\blet me check\b", r"\bi'll gather\b", r"\blet me gather\b", r"\bi'll explore\b", r"\blet me explore\b", r"\bi'll look at\b", r"\blet me look at\b", r"\bi need to\b"]
DIRECT_ANSWER_PATTERNS = [r"\bbased on the repository context\b", r"\bhere(?:'s| is)\b", r"\bthe current state\b", r"\bsummary\b", r"\brecommended pr slice\b"]
FILE_REF_PATTERNS = [r"\bREADME\.md\b", r"\bAGENTS\.md\b", r"\bMakefile\b", r"\bdocker-compose\.yml\b", r"\b[a-zA-Z0-9_./\-]+\.(?:py|md|ya?ml|toml)\b"]
HALLUCINATION_HINTS = [r"\bgit status\b", r"\brecent commits\b", r"\bI checked\b", r"\bI found\b"]
ROADMAP_PATTERNS = [r"\bstack-roadmap\.md\b", r"\broadmap\b", r"\bfeature slice tracker\b", r"\bnext recommended pr\b", r"\bpr next-\d+\b"]
ARCHITECTURE_PATTERNS = [r"\bstack-architecture\.md\b", r"\brag-architecture\.md\b", r"\barchitecture\b"]
RUNBOOK_PATTERNS = [r"\brunbook\b", r"\bhybrid-retrieval-runtime-validation\.md\b", r"\bimage-retrieval-runtime-validation\.md\b"]
ACTIVE_WORKSTREAM_PATTERNS = [r"\bactive\b", r"\bplanned\b", r"\bdeferred\b", r"\bimplemented baseline\b", r"\bcurrent active architecture direction\b", r"\bworkstream\b"]
NEXT_SLICE_PATTERNS = [r"\bnext recommended\b", r"\bnext pr\b", r"\bsmallest safe pull request\b", r"\bpart \d+\b", r"\bslice\b"]
CANONICAL_DOC_PATTERNS = [r"\bdocument/roadmaps/stack-roadmap\.md\b", r"\bdocument/architecture/stack-architecture\.md\b", r"\bdocument/architecture/rag-architecture\.md\b", r"\bdocument/structure\.md\b", r"\bdocument/runbooks/[a-zA-Z0-9._/-]+\.md\b"]
GENERIC_CLEANUP_PATTERNS = [r"\bupdate default model\b", r"\bupdate .*\.env\.example\b", r"\bconfiguration-only change\b", r"\bgeneric cleanup\b"]


def count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*(?:[-*•]|\d+\.)\s+", line))


def non_ascii_ratio(text: str) -> float:
    return 0.0 if not text else sum(1 for ch in text if ord(ch) > 127) / max(1, len(text))


def repeated_fragment_score(text: str) -> float:
    if not text:
        return 0.0
    windows = [text[i:i+8] for i in range(0, max(0, len(text) - 7), 4)]
    if not windows:
        return 0.0
    freq: dict[str, int] = {}
    for window in windows:
        freq[window] = freq.get(window, 0) + 1
    repeated = sum(count for count in freq.values() if count > 2)
    return repeated / len(windows)


def has_any_pattern(text: str, patterns: list[str], flags: int = re.IGNORECASE) -> bool:
    return any(re.search(pattern, text, flags=flags) for pattern in patterns)


def analyze_output(model: str, prompt_id: str, prompt_title: str, pass_index: int, elapsed_seconds: float, response_obj: Any | None, output_text: str, error: str | None) -> RunMetrics:
    usage = getattr(response_obj, "usage", None)
    choices = getattr(response_obj, "choices", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    stripped = output_text.strip()
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
        starts_with_plan_phrase=has_any_pattern(stripped[:200], PLAN_PATTERNS),
        contains_plan_phrase_anywhere=has_any_pattern(stripped, PLAN_PATTERNS),
        contains_direct_answer_cue=has_any_pattern(stripped, DIRECT_ANSWER_PATTERNS),
        contains_tests_reference=bool(re.search(r"\btest(?:s|ing)?\b|\bpytest\b|\bvalidation\b", stripped, re.I)),
        contains_docs_reference=bool(re.search(r"\bdocs?\b|\breadme\b|\bdocumentation\b", stripped, re.I)),
        contains_config_reference=bool(re.search(r"\bconfig\b|\b\.env\b|\byml\b|\byaml\b", stripped, re.I)),
        contains_risk_reference=bool(re.search(r"\brisk\b|\bunsafe\b|\bif done incorrectly\b", stripped, re.I)),
        contains_file_reference=has_any_pattern(stripped, FILE_REF_PATTERNS),
        contains_roadmap_reference=has_any_pattern(stripped, ROADMAP_PATTERNS),
        contains_architecture_reference=has_any_pattern(stripped, ARCHITECTURE_PATTERNS),
        contains_runbook_reference=has_any_pattern(stripped, RUNBOOK_PATTERNS),
        mentions_active_workstream=has_any_pattern(stripped, ACTIVE_WORKSTREAM_PATTERNS),
        mentions_planned_or_deferred_state=bool(re.search(r"\bplanned\b|\bdeferred\b|\bimplemented\b|\bactive\b", stripped, re.I)),
        mentions_next_pr_or_slice=has_any_pattern(stripped, NEXT_SLICE_PATTERNS),
        mentions_canonical_doc_path=has_any_pattern(stripped, CANONICAL_DOC_PATTERNS),
        contains_generic_cleanup_drift=has_any_pattern(stripped, GENERIC_CLEANUP_PATTERNS),
        contains_repo_hallucination_hint=has_any_pattern(stripped, HALLUCINATION_HINTS),
        contains_markdown_code_fence="```" in stripped,
        non_ascii_ratio=round(non_ascii_ratio(output_text), 4),
        repeated_fragment_score=round(repeated_fragment_score(output_text), 4),
        finish_reason=finish_reason,
        error=error,
    )


def run_one(client: OpenAI, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float, top_p: float, top_k: int, provider: str | None) -> tuple[Any | None, str, float, str | None]:
    start = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "extra_body": {"top_k": top_k}}
        response = client.chat.completions.create(**kwargs)
        return response, (response.choices[0].message.content if response.choices else "") or "", time.perf_counter() - start, None
    except Exception as exc:
        return None, "", time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(metrics_rows: list[RunMetrics]) -> list[dict[str, Any]]:
    by_model: dict[str, list[RunMetrics]] = {}
    for row in metrics_rows:
        by_model.setdefault(row.model, []).append(row)
    summary: list[dict[str, Any]] = []
    for model, rows in by_model.items():
        n = len(rows)
        avg_elapsed = sum(r.elapsed_seconds for r in rows) / n if n else 0.0
        avg_prompt_tokens = sum(r.prompt_tokens or 0 for r in rows) / n if n else 0.0
        avg_completion_tokens = sum(r.completion_tokens or 0 for r in rows) / n if n else 0.0
        directness_penalty = sum(1 for r in rows if r.starts_with_plan_phrase or r.contains_plan_phrase_anywhere)
        corruption_penalty = sum(1 for r in rows if r.repeated_fragment_score > 0.20 or r.non_ascii_ratio > 0.10)
        generic_cleanup_penalty = sum(1 for r in rows if r.contains_generic_cleanup_drift)
        structure_bonus = sum(1 for r in rows if r.bullet_count >= 4)
        test_bonus = sum(1 for r in rows if r.contains_tests_reference)
        docs_bonus = sum(1 for r in rows if r.contains_docs_reference)
        config_bonus = sum(1 for r in rows if r.contains_config_reference)
        risk_bonus = sum(1 for r in rows if r.contains_risk_reference)
        file_bonus = sum(1 for r in rows if r.contains_file_reference)
        roadmap_bonus = sum(1 for r in rows if r.contains_roadmap_reference)
        architecture_bonus = sum(1 for r in rows if r.contains_architecture_reference)
        runbook_bonus = sum(1 for r in rows if r.contains_runbook_reference)
        active_workstream_bonus = sum(1 for r in rows if r.mentions_active_workstream)
        next_slice_bonus = sum(1 for r in rows if r.mentions_next_pr_or_slice)
        canonical_doc_bonus = sum(1 for r in rows if r.mentions_canonical_doc_path)
        composite = max(0, min(100, 50 + structure_bonus*5 + test_bonus*4 + docs_bonus*3 + config_bonus*3 + risk_bonus*3 + file_bonus*4 + roadmap_bonus*6 + architecture_bonus*4 + runbook_bonus*3 + active_workstream_bonus*5 + next_slice_bonus*6 + canonical_doc_bonus*5 - directness_penalty*8 - corruption_penalty*12 - generic_cleanup_penalty*8))
        summary.append({
            "model": model,
            "runs": n,
            "avg_elapsed_seconds": round(avg_elapsed, 2),
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "avg_completion_tokens": round(avg_completion_tokens, 1),
            "directness_penalty_count": directness_penalty,
            "corruption_penalty_count": corruption_penalty,
            "generic_cleanup_penalty_count": generic_cleanup_penalty,
            "structure_bonus_count": structure_bonus,
            "tests_bonus_count": test_bonus,
            "docs_bonus_count": docs_bonus,
            "config_bonus_count": config_bonus,
            "risk_bonus_count": risk_bonus,
            "file_bonus_count": file_bonus,
            "roadmap_bonus_count": roadmap_bonus,
            "architecture_bonus_count": architecture_bonus,
            "runbook_bonus_count": runbook_bonus,
            "active_workstream_bonus_count": active_workstream_bonus,
            "next_slice_bonus_count": next_slice_bonus,
            "canonical_doc_bonus_count": canonical_doc_bonus,
            "heuristic_composite_score_100": composite,
        })
    summary.sort(key=lambda x: (-x["heuristic_composite_score_100"], x["avg_elapsed_seconds"]))
    return summary


def write_markdown_report(path: Path, repo_root: Path, models: list[str], metrics_rows: list[RunMetrics], raw_outputs: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Hugging Face Model Screening Report\n", f"- Repo: `{repo_root}`", f"- Models tested: {', '.join(f'`{m}`' for m in models)}", "", "## Summary Ranking\n"]
    for row in summary_rows:
        lines.append(f"- **{row['model']}** — score {row['heuristic_composite_score_100']}/100, avg elapsed {row['avg_elapsed_seconds']}s, avg prompt tokens {row['avg_prompt_tokens']}, avg completion tokens {row['avg_completion_tokens']}")
    lines.append("")
    lines.append("## Per-run Metrics\n")
    for m in metrics_rows:
        lines.append(f"- **{m.model} / {m.prompt_id} / pass {m.pass_index}** — {m.elapsed_seconds:.2f}s, prompt={m.prompt_tokens}, completion={m.completion_tokens}, bullets={m.bullet_count}, roadmap={m.contains_roadmap_reference}, architecture={m.contains_architecture_reference}, runbook={m.contains_runbook_reference}, next_slice={m.mentions_next_pr_or_slice}, canonical_doc={m.mentions_canonical_doc_path}, cleanup_drift={m.contains_generic_cleanup_drift}, finish={m.finish_reason}, error={m.error}")
    lines.append("")
    lines.append("## Raw Outputs\n")
    for item in raw_outputs:
        lines.append(f"### {item['model']} / {item['prompt_id']} / pass {item['pass_index']}\n")
        lines.append("```text")
        lines.append(item.get("output", ""))
        lines.append("```")
        if item.get("error"):
            lines.append(f"Error: {item['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def runtime_config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    data = asdict(config)
    data["prompts"] = [asdict(prompt) for prompt in config.prompts]
    return data


def write_outputs(outdir: Path, short_name: str, timestamp: str, result_data: dict[str, Any], metrics_rows: list[RunMetrics], repo_root: Path, models: list[str], raw_outputs: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output_formats: list[str]) -> None:
    metrics_dicts = [asdict(row) for row in metrics_rows]
    format_set = {fmt.lower() for fmt in output_formats}
    if "json" in format_set:
        write_json(outdir / build_result_filename(short_name, "json", timestamp), result_data)
    if "yaml" in format_set:
        write_yaml(outdir / build_result_filename(short_name, "yaml", timestamp), result_data)
    if "csv" in format_set:
        write_csv(outdir / build_result_filename(short_name, "csv", timestamp), metrics_dicts)
    if "markdown" in format_set or "md" in format_set:
        write_markdown_report(outdir / build_result_filename(short_name, "md", timestamp), repo_root, models, metrics_rows, raw_outputs, summary_rows)


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    defaults_section = load_model_screen_section(DEFAULT_CONFIG_PATH)
    merged_section = defaults_section if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve() else merge_dicts(defaults_section, load_model_screen_section(config_path))
    config = apply_cli_overrides(build_runtime_config(merged_section), args)
    if args.print_effective_config:
        print(json.dumps(runtime_config_to_dict(config), indent=2))
    if not config.repo:
        raise RuntimeError("A repo path is required via config or --repo")
    repo_root = Path(config.repo).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise RuntimeError(f"Repo path does not exist or is not a directory: {repo_root}")
    outdir = Path(config.output_dir).expanduser().resolve()
    ok, error_msg = validate_output_directory(outdir)
    if not ok:
        raise RuntimeError(error_msg)
    repo_files = gather_repo_files(repo_root, config.include_files, config.include_globs)
    repo_context = build_repo_context(repo_root, repo_files, config.max_context_chars, config.max_file_chars)
    if config.save_context or args.dry_run_context:
        (outdir / "repo_context.txt").write_text(repo_context, encoding="utf-8")
    if args.dry_run_context:
        print(f"Wrote repo context to {(outdir / 'repo_context.txt')}")
        return 0
    client = OpenAI(base_url=config.base_url, api_key=resolve_token(args.token))
    metrics_rows: list[RunMetrics] = []
    raw_outputs: list[dict[str, Any]] = []
    for model in config.models:
        for prompt_cfg in config.prompts:
            messages = build_messages(repo_context, prompt_cfg.prompt)
            for pass_index in range(1, config.passes + 1):
                print(f"Running model={model} prompt={prompt_cfg.id} pass={pass_index} ...")
                response, output, elapsed, error = run_one(client, model, messages, config.max_tokens, config.temperature, config.top_p, config.top_k, config.provider)
                metrics = analyze_output(model, prompt_cfg.id, prompt_cfg.title, pass_index, elapsed, response, output, error)
                metrics_rows.append(metrics)
                raw_outputs.append({"model": model, "prompt_id": prompt_cfg.id, "prompt_title": prompt_cfg.title, "pass_index": pass_index, "output": output, "error": error})
    summary_rows = summarize_scores(metrics_rows)
    result_data = {"repo": str(repo_root), "models": config.models, "config_source": str(config_path), "config": runtime_config_to_dict(config), "summary": summary_rows, "metrics": [asdict(row) for row in metrics_rows], "outputs": raw_outputs}
    timestamp = get_timestamp_str()
    short_name = sanitize_name(config.short_name or "results")
    write_outputs(outdir, short_name, timestamp, result_data, metrics_rows, repo_root, config.models, raw_outputs, summary_rows, config.output_formats)
    print(f"Done. Wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
