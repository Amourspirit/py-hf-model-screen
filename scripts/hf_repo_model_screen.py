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
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


DEFAULT_MODELS = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3.6-35B-A3B",
]

DEFAULT_INCLUDE_FILES = [
    "README.md",
    "AGENTS.md",
    "document/README.md",
    "document/roadmaps/stack-roadmap.md",
    "document/architecture/stack-architecture.md",
    "document/architecture/rag-architecture.md",
    "document/structure.md",
    "document/runbooks/hybrid-retrieval-runtime-validation.md",
    "document/runbooks/image-retrieval-runtime-validation.md",
    ".env.example",
    "docker-compose.yml",
    "Makefile",
    "Project-Spiral.code-workspace",
    "project-config/ingest/profiles.yml",
    "project-config/ingest/sources.yml",
    "codebase/service-tier/router-service/README.md",
    "codebase/service-tier/retrieval-service/README.md",
    "codebase/service-tier/ingest-service/README.md",
    "codebase/service-tier/inference-service/README.md",
    "codebase/service-tier/vision-embedding-service/README.md",
    "codebase/service-tier/logging-service/README.md",
]

DEFAULT_INCLUDE_GLOBS = [
    "project-config/retrieval/*.yml",
    "project-config/inference/**/*.yml",
    "document/adr/*.md",
]

DEFAULT_PROMPTS = [
    {
        "id": "repo_state",
        "title": "Repo State Summary",
        "prompt": (
            "Based only on the repository context below, summarize the current state of the project in 8 bullets.\n"
            "Do not describe your plan.\n"
            "Do not say what you will check.\n"
            "Answer directly.\n"
        ),
    },
    {
        "id": "pr_slice",
        "title": "Smallest Safe PR Slice",
        "prompt": (
            "Given the repository context below, propose the smallest safe pull request for the next meaningful improvement.\n"
            "Include:\n"
            "- likely affected files\n"
            "- tests to run\n"
            "- docs/config updates\n"
            "- risks if done incorrectly\n"
            "Keep the scope tight.\n"
        ),
    },
    {
        "id": "change_impact",
        "title": "Adjacent Surfaces Review",
        "prompt": (
            "Given the repository context below, assume a small change is being made in the retrieval service.\n"
            "List the adjacent surfaces that should also be reviewed so the change is not incomplete.\n"
            "Be concrete and concise.\n"
        ),
    },
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen Hugging Face chat models against local repo context."
    )
    parser.add_argument(
        "--repo", required=True, help="Absolute path to your local repository checkout"
    )
    parser.add_argument("--outdir", default="out", help="Output directory for reports")
    parser.add_argument(
        "--models", nargs="*", default=DEFAULT_MODELS, help="Model IDs to test"
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="How many times to run each prompt per model",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=36000,
        help="Max total chars of repo context",
    )
    parser.add_argument(
        "--max-file-chars", type=int, default=3500, help="Max chars per included file"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=900, help="max_tokens for completion"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, dest="top_p", default=0.95, help="Sampling top_p"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        dest="top_k",
        default=20,
        help="Provider-specific top_k via extra_body",
    )
    parser.add_argument(
        "--token", default=None, help="HF token; otherwise reads HF_TOKEN env var"
    )
    parser.add_argument(
        "--provider", default=None, help="Optional provider hint, e.g. hf-inference"
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
        help="Save assembled repo context to disk",
    )
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    # Common local storage path used by Hugging Face CLI
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
    repo_root: Path, extra_files: List[str], extra_globs: List[str]
) -> List[Path]:
    found: List[Path] = []

    for rel in DEFAULT_INCLUDE_FILES + extra_files:
        p = repo_root / rel
        if p.exists() and p.is_file():
            found.append(p)

    for pattern in DEFAULT_INCLUDE_GLOBS + extra_globs:
        for p in repo_root.glob(pattern):
            if p.is_file():
                found.append(p)

    # Stable unique ordering
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
    repo_root: Path, files: List[Path], max_context_chars: int, max_file_chars: int
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
    r"\bgit status\b",  # Often suspicious if not actually grounded in provided context
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
    # Look at repeated 8-char windows
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
        # Provider hint can be encoded in model selection on HF, but leave open for future use.
        # We do not force provider routing here by default.

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

        # Heuristic composite scores
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
            f"- **{row['model']}** — score {row['heuristic_composite_score_100']}/100, "
            f"avg elapsed {row['avg_elapsed_seconds']}s, "
            f"avg prompt tokens {row['avg_prompt_tokens']}, "
            f"avg completion tokens {row['avg_completion_tokens']}"
        )
    lines.append("")

    lines.append("## Per-run Metrics\n")
    for m in metrics_rows:
        lines.append(
            f"- **{m.model} / {m.prompt_id} / pass {m.pass_index}** — "
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


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not repo_root.exists() or not repo_root.is_dir():
        print(
            f"Repo path does not exist or is not a directory: {repo_root}",
            file=sys.stderr,
        )
        return 2

    token = resolve_token(args.token)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    repo_files = gather_repo_files(repo_root, args.extra_file, args.extra_glob)
    repo_context = build_repo_context(
        repo_root=repo_root,
        files=repo_files,
        max_context_chars=args.max_context_chars,
        max_file_chars=args.max_file_chars,
    )

    if args.save_context:
        (outdir / "repo_context.txt").write_text(repo_context, encoding="utf-8")

    metrics_rows: List[RunMetrics] = []
    raw_outputs: List[Dict[str, Any]] = []

    for model in args.models:
        for prompt_cfg in DEFAULT_PROMPTS:
            messages = build_messages(repo_context, prompt_cfg["prompt"])
            for pass_index in range(1, args.passes + 1):
                print(
                    f"Running model={model} prompt={prompt_cfg['id']} pass={pass_index} ..."
                )
                response, output, elapsed, error = run_one(
                    client=client,
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    provider=args.provider,
                )

                metrics = analyze_output(
                    model=model,
                    prompt_id=prompt_cfg["id"],
                    prompt_title=prompt_cfg["title"],
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
                        "prompt_id": prompt_cfg["id"],
                        "prompt_title": prompt_cfg["title"],
                        "pass_index": pass_index,
                        "output": output,
                        "error": error,
                    }
                )

    summary_rows = summarize_scores(metrics_rows)

    metrics_dicts = [asdict(m) for m in metrics_rows]
    write_json(
        outdir / "results.json",
        {
            "repo": str(repo_root),
            "models": args.models,
            "config": {
                "passes": args.passes,
                "max_context_chars": args.max_context_chars,
                "max_file_chars": args.max_file_chars,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            },
            "summary": summary_rows,
            "metrics": metrics_dicts,
            "outputs": raw_outputs,
        },
    )
    write_csv(outdir / "results.csv", metrics_dicts)
    write_markdown_report(
        outdir / "results.md",
        repo_root=repo_root,
        models=args.models,
        metrics_rows=metrics_rows,
        raw_outputs=raw_outputs,
        summary_rows=summary_rows,
    )

    print(f"\nDone. Reports written to: {outdir}")
    print(f"- {outdir / 'results.json'}")
    print(f"- {outdir / 'results.csv'}")
    print(f"- {outdir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
