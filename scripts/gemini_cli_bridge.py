"""
Gemini CLI Bridge Script for Codex Skills.

Runs the `gemini` (Google Gemini CLI) in headless mode and returns a JSON
envelope suitable for multi-model collaboration.

This bridge is tuned for Gemini's practical / effective context constraints: 
it defaults to read-only (`--no-full-access`) and encourages file-scoped, 
one-shot requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODEL = "gemini-3-pro-preview"

# User-reported effective context (conservative) when using sliding windows.
DEFAULT_EFFECTIVE_CONTEXT_TOKENS = 48_000

# Heuristic: ~4 chars per token in many tokenizers. Used only for warnings.
DEFAULT_MAX_FOCUS_BYTES = 200_000

DEFAULT_GEMINI_HOME_MODE = "auto"  # auto|system|sandbox
DEFAULT_GEMINI_HOME_BASE = "/tmp/codex-gemini-home"


def _readable_path_env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val is None:
        return None
    stripped = val.strip()
    return stripped or None


def _is_writable_dir(path: Path) -> bool:
    try:
        return bool(path.is_dir() and os.access(str(path), os.W_OK))
    except OSError:
        return False


def _safe_gemini_home_for_cd(cd_path: Path, base: Path) -> Path:
    digest = hashlib.sha256(str(cd_path.resolve()).encode("utf-8")).hexdigest()[:24]
    return base / f"gemini-home-{digest}"


def _copy_if_newer(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
    except OSError:
        # Best effort only; missing auth may force re-login.
        pass


def _sync_gemini_state(*, system_home: Path, sandbox_home: Path) -> None:
    """
    Make Gemini CLI usable when HOME isn't writable by copying auth/config into a writable HOME.

    Gemini CLI stores state under ~/.gemini by default; we copy a small allowlist of
    auth/config files (not large profiles/history).
    """
    sandbox_home.mkdir(parents=True, exist_ok=True)

    src_root = system_home / ".gemini"
    if not src_root.is_dir():
        return

    dst_root = sandbox_home / ".gemini"
    dst_root.mkdir(parents=True, exist_ok=True)

    # Common top-level state/auth files.
    for filename in (
        "oauth_creds.json",
        "settings.json",
        "settings.json.orig",
        "state.json",
        "google_accounts.json",
        "installation_id",
        "user_id",
        ".env",
    ):
        _copy_if_newer(src_root / filename, dst_root / filename)

    # Minimal subset of ~/.gemini/antigravity sometimes referenced by the CLI.
    src_ag = src_root / "antigravity"
    if src_ag.is_dir():
        dst_ag = dst_root / "antigravity"
        dst_ag.mkdir(parents=True, exist_ok=True)
        for filename in ("installation_id", "mcp_config.json", "onboarding.json", "user_settings.pb"):
            _copy_if_newer(src_ag / filename, dst_ag / filename)


def _apply_gemini_home_mode(
    env: Dict[str, str],
    *,
    cd_path: Path,
    home_mode: str,
    home_base: Path,
) -> Dict[str, str]:
    """
    Ensure Gemini CLI has a writable HOME for ~/.gemini writes under sandboxed runners.

    - system: do not modify HOME/XDG paths
    - sandbox: force a writable home under `home_base`
    - auto: use sandbox home if the current HOME isn't writable
    """
    mode = (home_mode or DEFAULT_GEMINI_HOME_MODE).strip().lower()
    if mode not in {"auto", "system", "sandbox"}:
        mode = DEFAULT_GEMINI_HOME_MODE

    if mode == "system":
        return env

    current_home = Path(env.get("HOME", str(Path.home()))).expanduser()
    needs_sandbox = mode == "sandbox" or not _is_writable_dir(current_home)
    if not needs_sandbox:
        return env

    sandbox_home = _safe_gemini_home_for_cd(cd_path, home_base)

    # Copy auth/config from the real system home (readable in Codex) so Gemini stays logged in.
    system_home = Path(os.path.expanduser("~"))
    _sync_gemini_state(system_home=system_home, sandbox_home=sandbox_home)

    updated = env.copy()
    updated["HOME"] = str(sandbox_home)
    updated["XDG_CONFIG_HOME"] = str(sandbox_home / ".config")
    updated["XDG_STATE_HOME"] = str(sandbox_home / ".state")
    updated["XDG_CACHE_HOME"] = str(sandbox_home / ".cache")
    return updated


def _get_windows_npm_paths() -> List[Path]:
    """Return candidate directories for npm global installs on Windows."""
    if os.name != "nt":
        return []
    env = os.environ
    paths: List[Path] = []
    if prefix := env.get("NPM_CONFIG_PREFIX") or env.get("npm_config_prefix"):
        paths.append(Path(prefix))
    if appdata := env.get("APPDATA"):
        paths.append(Path(appdata) / "npm")
    if localappdata := env.get("LOCALAPPDATA"):
        paths.append(Path(localappdata) / "npm")
    if programfiles := env.get("ProgramFiles"):
        paths.append(Path(programfiles) / "nodejs")
    return paths


def _augment_path_env(env: Dict[str, str]) -> None:
    """Prepend npm global directories to PATH if missing (Windows only)."""
    if os.name != "nt":
        return
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    path_entries = [p for p in env.get(path_key, "").split(os.pathsep) if p]
    lower_set = {p.lower() for p in path_entries}
    for candidate in _get_windows_npm_paths():
        if candidate.is_dir() and str(candidate).lower() not in lower_set:
            path_entries.insert(0, str(candidate))
            lower_set.add(str(candidate).lower())
    env[path_key] = os.pathsep.join(path_entries)


def _resolve_executable(name: str, env: Dict[str, str]) -> str:
    """Resolve executable path, checking npm directories for .cmd/.bat on Windows."""
    if os.path.isabs(name) or os.sep in name or (os.altsep and os.altsep in name):
        return name
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    path_val = env.get(path_key)
    if resolved := shutil.which(name, path=path_val):
        return resolved
    if os.name == "nt":
        for base in _get_windows_npm_paths():
            for ext in (".cmd", ".bat", ".exe", ".com"):
                candidate = base / f"{name}{ext}"
                if candidate.is_file():
                    return str(candidate)
    return name


def _configure_windows_stdio() -> None:
    """Configure stdout/stderr to use UTF-8 encoding on Windows."""
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _run(
    cmd: List[str],
    timeout_s: Optional[float],
    cwd: Optional[Path],
    *,
    stdin_text: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    env = (env or os.environ.copy()).copy()
    _augment_path_env(env)
    cmd = cmd.copy()
    cmd[0] = _resolve_executable(cmd[0], env)

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )
    try:
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return 124, stdout, (stderr + "\n[timeout] Gemini CLI process timed out.").strip()
    return process.returncode, stdout, stderr


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract JSON objects from a mixed stdout/stderr blob.

    Gemini CLI usually prints exactly one JSON object for `--output-format json`,
    but may prepend warnings or retry logs. This scans the whole text and
    returns any parsed JSON objects (dicts only).
    """
    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            i = start + 1
            continue
        i = start + end
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def _parse_single_json(stdout: str, stderr: str) -> Dict[str, Any]:
    """Parse the most relevant JSON object from Gemini CLI json output."""
    blob = "\n".join([stdout.strip(), stderr.strip()]).strip()
    if not blob:
        raise ValueError("Empty stdout/stderr")
    objects = _extract_json_objects(blob)
    if not objects:
        raise ValueError("No JSON object found in output")

    # Prefer the last object that looks like a headless response.
    for obj in reversed(objects):
        if "session_id" in obj and ("response" in obj or "error" in obj):
            return obj
    return objects[-1]


def _parse_stream_json(stdout: str, stderr: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    def _feed(lines: List[str], source: str) -> None:
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    events.append(data)
                else:
                    events.append({"type": "non_dict_json", "value": data, "source": source})
            except json.JSONDecodeError:
                events.append({"type": "non_json_line", "text": raw_line, "source": source})

    _feed(stdout.splitlines(), "stdout")
    _feed(stderr.splitlines(), "stderr")
    return events


def _extract_session_id(events: List[Dict[str, Any]]) -> Optional[str]:
    for ev in events:
        sid = ev.get("session_id")
        if isinstance(sid, str) and sid:
            return sid
    return None


def _extract_agent_messages_from_stream(events: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for ev in events:
        if ev.get("type") == "message" and ev.get("role") == "assistant":
            content = ev.get("content")
            if isinstance(content, str):
                chunks.append(content)
    return "".join(chunks).strip()


def _extract_stats_from_stream(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for ev in reversed(events):
        if ev.get("type") == "result" and isinstance(ev.get("stats"), dict):
            return ev.get("stats")
    return None


def _extract_prompt_tokens(stats: Optional[Dict[str, Any]], model: str) -> Optional[int]:
    if not isinstance(stats, dict):
        return None
    models = stats.get("models")
    if isinstance(models, dict):
        # Preferred: exact key match.
        model_info = models.get(model)
        if isinstance(model_info, dict):
            tokens = model_info.get("tokens")
            if isinstance(tokens, dict):
                prompt = tokens.get("prompt")
                if isinstance(prompt, int):
                    return prompt

        # Fallbacks: some Gemini CLI modes may report a different model key
        # (e.g. `gemini-3-pro-high`) even when a `--model` alias was requested.
        candidates: List[Tuple[str, int]] = []
        for key, info in models.items():
            if not isinstance(info, dict):
                continue
            tokens = info.get("tokens")
            if not isinstance(tokens, dict):
                continue
            prompt = tokens.get("prompt")
            if isinstance(prompt, int):
                candidates.append((str(key), prompt))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][1]

        # Heuristic: prefer keys that contain the requested model string.
        lower_model = model.lower()
        for key, prompt in candidates:
            if lower_model and lower_model in key.lower():
                return prompt
        # Heuristic: or keys that are contained by the requested model string.
        for key, prompt in candidates:
            if key.lower() in lower_model:
                return prompt

        # Last resort: take the largest prompt token count.
        return max(prompt for _, prompt in candidates)
    # stream-json `result.stats` shape
    input_tokens = stats.get("input_tokens")
    if isinstance(input_tokens, int):
        return input_tokens
    return None


_CANDIDATE_PATH_RE = re.compile(
    r"""
    (?:
        # path with at least one directory separator and a file-ish tail
        (?:\./)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+
        |
        # simple filename with an extension
        [A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}
    )
    """,
    re.VERBOSE,
)


def _extract_focus_files_from_prompt(prompt: str, cd_path: Path) -> List[str]:
    candidates: List[str] = []
    for match in _CANDIDATE_PATH_RE.finditer(prompt):
        token = match.group(0).strip()
        if not token or "://" in token:
            continue
        candidates.append(token)

    # De-dupe while preserving order; keep only existing files.
    seen: set[str] = set()
    focus: List[str] = []
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)

        p = Path(token)
        if not p.is_absolute():
            p = (cd_path / p).resolve()
        try:
            p.relative_to(cd_path.resolve())
        except ValueError:
            continue
        if p.is_file():
            focus.append(str(p.relative_to(cd_path.resolve())))
    return focus


def _normalize_focus_files(cd_path: Path, files: List[str]) -> List[str]:
    """Normalize file paths to be relative to cd_path, ensuring they exist and are files."""
    normalized: List[str] = []
    seen: set[str] = set()
    cd_resolved = cd_path.resolve()
    for raw in files:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (cd_resolved / p).resolve()
        try:
            rel = p.relative_to(cd_resolved)
        except ValueError:
            continue
        if not p.is_file():
            continue
        rel_str = str(rel)
        if rel_str not in seen:
            normalized.append(rel_str)
            seen.add(rel_str)
    return normalized


def _estimate_focus_bytes(cd_path: Path, focus_files: List[str]) -> int:
    total = 0
    for rel in focus_files:
        try:
            total += (cd_path / rel).stat().st_size
        except OSError:
            continue
    return total


def _build_guardrail_preamble(
    *,
    preferred_max_files: int,
    effective_context_tokens: int,
    focus_files: List[str],
    explicit_files_count: int,
    auto_extracted_files_count: int,
) -> str:
    files_hint = ", ".join(focus_files) if focus_files else "(none provided)"
    return (
        "CONTEXT / SCOPE GUARDRAILS (important):\n"
        f"- Practical effective context may degrade beyond ~{effective_context_tokens:,} tokens; keep this run small.\n"
        f"- Prefer scoping each run to ≤ {preferred_max_files} file(s) unless the caller explicitly chooses otherwise.\n"
        f"- Focus files provided: {len(focus_files)} (explicit: {explicit_files_count}, auto: {auto_extracted_files_count}).\n"
        f"- Focus files: {files_hint}\n"
        "- If you need additional files, STOP and ask which file to inspect next (do not read more by yourself).\n"
        "- Prefer returning a unified diff for changes instead of rewriting whole files.\n"
    )


def main() -> None:
    _configure_windows_stdio()

    parser = argparse.ArgumentParser(description="Gemini CLI Bridge")
    parser.add_argument("--PROMPT", required=True, help="Instruction for the task to send to Gemini CLI.")
    parser.add_argument("--cd", required=True, help="Working directory to run Gemini CLI in (typically the repo root).")
    parser.add_argument("--SESSION_ID", default="", help="Resume the specified Gemini CLI session (uuid). Defaults to empty: start a new session.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use. Defaults to `{DEFAULT_MODEL}`.")

    access_group = parser.add_mutually_exclusive_group()
    access_group.add_argument(
        "--full-access",
        dest="full_access",
        action="store_true",
        help="Enable edit/shell tool usage (mapped to `--approval-mode auto_edit`).",
    )
    access_group.add_argument(
        "--no-full-access",
        dest="full_access",
        action="store_false",
        help="Disable full access (read-only review). Defaults to --no-full-access.",
    )
    parser.set_defaults(full_access=False)

    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Enable YOLO approval mode (auto-approve all tools). Implies --full-access. Use only in trusted repos.",
    )

    parser.add_argument(
        "--approval-mode",
        default=None,
        choices=["default", "auto_edit", "yolo"],
        help="Override Gemini CLI approval mode. Default: `default` for --no-full-access; `auto_edit` for --full-access; `yolo` for --yolo.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        default=False,
        help="Run Gemini CLI in sandbox mode (requires Docker). Defaults to False.",
    )
    parser.add_argument(
        "--extensions",
        default=None,
        help='Gemini CLI extensions to use. Examples: "none", or "extensionA,extensionB". Default: all.',
    )
    parser.add_argument(
        "--allowed-tools",
        default=None,
        help='Comma-separated tools to allow without confirmation (passes through to `--allowed-tools`). Default: unset.',
    )
    parser.add_argument(
        "--return-all-messages",
        action="store_true",
        help="Return the full streamed JSON event list from Gemini CLI (useful for debugging).",
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0, help="Process timeout in seconds. Defaults to 1800 (30 minutes).")
    parser.add_argument("--gemini-bin", default="gemini", help="Gemini CLI executable name/path. Defaults to `gemini`.")
    parser.add_argument(
        "--gemini-home-mode",
        choices=["auto", "system", "sandbox"],
        default=_readable_path_env("CODEX_GEMINI_HOME_MODE") or DEFAULT_GEMINI_HOME_MODE,
        help=(
            "Where Gemini CLI should store its per-run state/config (e.g. ~/.gemini). "
            "Default: auto (use sandbox HOME if current HOME isn't writable). "
            "Set to system to preserve the current HOME, or sandbox to force a writable HOME under --gemini-home-base."
        ),
    )
    parser.add_argument(
        "--gemini-home-base",
        default=_readable_path_env("CODEX_GEMINI_HOME_BASE") or DEFAULT_GEMINI_HOME_BASE,
        help=f"Base directory for sandbox Gemini HOME (default: {DEFAULT_GEMINI_HOME_BASE}).",
    )

    # Guardrails for effective context + file scoping
    parser.add_argument(
        "--guardrails",
        dest="guardrails",
        action="store_true",
        default=True,
        help="Enable conservative scope guardrails (default).",
    )
    parser.add_argument(
        "--no-guardrails",
        dest="guardrails",
        action="store_false",
        help="Disable guardrails (Gemini may read many files; treat like normal agent).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help=(
            "Preferred focus-file cap used by guardrails. When no explicit --file is provided, auto-extracted focus files from PROMPT "
            "are limited to this count. Explicit --file entries are always included (not blocked). Defaults to 5."
        ),
    )
    parser.add_argument(
        "--effective-context-tokens",
        type=int,
        default=DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
        help=f"Effective context threshold used for warnings. Defaults to {DEFAULT_EFFECTIVE_CONTEXT_TOKENS}.",
    )
    parser.add_argument(
        "--max-focus-bytes",
        type=int,
        default=DEFAULT_MAX_FOCUS_BYTES,
        help=f"Heuristic byte limit for inlined focus files (warning only). Defaults to {DEFAULT_MAX_FOCUS_BYTES}.",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        default=[],
        help="Focus file (repeatable). When provided, the bridge will prepend @file references to scope context.",
    )
    parser.add_argument(
        "--auto-extract-files",
        dest="auto_extract_files",
        action="store_true",
        default=True,
        help="Auto-detect file paths from PROMPT and treat them as focus files (default).",
    )
    parser.add_argument(
        "--no-auto-extract-files",
        dest="auto_extract_files",
        action="store_false",
        help="Disable auto file extraction from PROMPT.",
    )

    args = parser.parse_args()

    cd_path = Path(args.cd).expanduser()
    if not cd_path.is_dir():
        print(
            json.dumps(
                {"success": False, "error": f"`--cd` must be an existing directory. Got: {args.cd}"},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    # Resolve / normalize focus files.
    explicit_focus_files = _normalize_focus_files(cd_path, args.files)
    auto_extracted_focus_files: List[str] = []
    auto_extracted_candidates: List[str] = []

    # Auto-extract is only applied when the caller did NOT explicitly provide --file.
    # This keeps the bridge predictable and allows Codex (the main agent) to decide
    # the exact scope when needed.
    if args.guardrails and args.auto_extract_files and not explicit_focus_files:
        for rel in _extract_focus_files_from_prompt(args.PROMPT, cd_path):
            if rel not in auto_extracted_candidates:
                auto_extracted_candidates.append(rel)

        if len(auto_extracted_candidates) > max(0, args.max_files):
            auto_extracted_focus_files = auto_extracted_candidates[: max(0, args.max_files)]
        else:
            auto_extracted_focus_files = auto_extracted_candidates

    # Combined focus list (explicit files first).
    focus_files: List[str] = []
    for rel in [*explicit_focus_files, *auto_extracted_focus_files]:
        if rel not in focus_files:
            focus_files.append(rel)

    approval_mode = args.approval_mode
    if approval_mode is None:
        if args.yolo:
            approval_mode = "yolo"
        elif args.full_access:
            approval_mode = "auto_edit"
        else:
            approval_mode = "default"

    # If user asked for YOLO, make sure full access is effectively enabled.
    full_access = bool(args.full_access or args.yolo or approval_mode in {"auto_edit", "yolo"})

    # Build prompt with optional @file scoping + guardrails.
    prompt = args.PROMPT
    meta: Dict[str, Any] = {
        "model": args.model,
        "full_access": full_access,
        "approval_mode": approval_mode,
        "guardrails": args.guardrails,
        "max_files": args.max_files,
        "effective_context_tokens": args.effective_context_tokens,
        "focus_files": focus_files,
        "explicit_focus_files": explicit_focus_files,
        "auto_extracted_focus_files": auto_extracted_focus_files,
    }
    if approval_mode == "yolo":
        meta.setdefault("warnings", []).append(
            "YOLO approval mode is enabled: Gemini CLI may edit files and run commands without prompting. Use only in trusted repos."
        )
    if args.guardrails and explicit_focus_files and len(explicit_focus_files) > max(0, args.max_files):
        meta.setdefault("warnings", []).append(
            f"Explicit --file count ({len(explicit_focus_files)}) exceeds preferred --max-files ({args.max_files}). "
            "This is allowed; consider splitting into smaller turns if context quality degrades."
        )
    if args.guardrails and (not explicit_focus_files) and auto_extracted_candidates and len(auto_extracted_candidates) > len(auto_extracted_focus_files):
        meta.setdefault("warnings", []).append(
            f"Auto-extracted {len(auto_extracted_candidates)} file paths from PROMPT, but only included the first {len(auto_extracted_focus_files)} "
            f"due to preferred --max-files ({args.max_files}). To include more, pass explicit --file, increase --max-files, or disable guardrails."
        )

    if args.guardrails:
        preamble = _build_guardrail_preamble(
            preferred_max_files=args.max_files,
            effective_context_tokens=args.effective_context_tokens,
            focus_files=focus_files,
            explicit_files_count=len(explicit_focus_files),
            auto_extracted_files_count=len(auto_extracted_focus_files),
        )
        prompt = f"{preamble}\n\nUSER TASK:\n{prompt}".strip()

    if focus_files:
        # Gemini CLI supports @path to inline file contents into the query.
        refs = "\n".join(f"@{p}" for p in focus_files)
        prompt = f"{refs}\n\n{prompt}".strip()

        focus_bytes = _estimate_focus_bytes(cd_path, focus_files)
        meta["focus_bytes"] = focus_bytes
        if args.guardrails and focus_bytes > args.max_focus_bytes:
            meta["warnings"] = meta.get("warnings", [])
            meta["warnings"].append(
                f"Focus files total size is ~{focus_bytes:,} bytes, which may exceed practical effective context. "
                "Consider narrowing files or splitting the task."
            )

    output_format = "stream-json" if args.return_all_messages else "json"

    cmd: List[str] = [
        args.gemini_bin,
        "--output-format",
        output_format,
        "--model",
        args.model,
        "--approval-mode",
        approval_mode,
    ]

    if args.sandbox:
        cmd.append("--sandbox")

    if args.extensions is not None:
        ext = args.extensions.strip()
        if ext:
            cmd.extend(["--extensions", ext])

    if args.allowed_tools is not None:
        # yargs "array" flag: accept comma-separated list here for convenience.
        tool_str = args.allowed_tools.strip()
        if tool_str:
            for t in [x.strip() for x in tool_str.split(",") if x.strip()]:
                cmd.extend(["--allowed-tools", t])

    if args.SESSION_ID:
        cmd.extend(["--resume", args.SESSION_ID])

    # Use positional prompt (preferred; --prompt is deprecated).
    cmd.append(prompt)

    try:
        base_env = os.environ.copy()
        _augment_path_env(base_env)
        base_env = _apply_gemini_home_mode(
            base_env,
            cd_path=cd_path,
            home_mode=str(args.gemini_home_mode),
            home_base=Path(str(args.gemini_home_base)).expanduser(),
        )
        rc, stdout, stderr = _run(cmd, timeout_s=args.timeout_s, cwd=cd_path, env=base_env)
    except FileNotFoundError as error:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Failed to execute Gemini CLI. Is `gemini` installed and on PATH?\n\n{error}",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    meta["exit_code"] = rc

    # Gemini CLI may try to perform a web-based OAuth flow by binding a local port.
    # Some sandboxed runners disallow listening sockets, causing EPERM. Surface an
    # actionable error before attempting JSON parsing (Gemini may also crash and emit non-JSON).
    combined_err = "\n".join([stderr.strip(), stdout.strip()]).strip()
    if rc != 0 and "Error authenticating" in combined_err and "listen EPERM" in combined_err:
        result = {
            "success": False,
            "error": (
                "Gemini CLI authentication failed because it attempted a web-based OAuth flow that binds a local port, "
                "but the current sandbox disallows listening sockets (EPERM).\n\n"
                "Workarounds:\n"
                "- Run this skill with network-enabled/escalated permissions and authenticate once.\n"
                "- Or configure Gemini CLI to use API key auth (so no local OAuth callback server is needed).\n"
            ).strip(),
            "meta": {**meta, "stdout": stdout.strip(), "stderr": stderr.strip()},
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # If the subprocess timed out, return a deterministic envelope instead of
    # failing JSON parsing later.
    #
    # Note: Some Gemini CLI builds can emit a complete JSON response but then
    # crash/hang due to internal telemetry/session cleanup errors. In that case,
    # treat the already-emitted response as success (with warnings) so this skill
    # remains usable in sandboxed Codex runs.
    if rc == 124:
        meta_timeout = {**meta, "exit_code": rc, "timed_out": True}
        timeout_warning = (
            f"Gemini CLI timed out after --timeout-s={args.timeout_s}, but a response may have been emitted before the hang."
        )

        try:
            if output_format == "json":
                payload = _parse_single_json(stdout, stderr)
                session_id = payload.get("session_id")
                agent_messages = payload.get("response")
                stats = payload.get("stats")

                if isinstance(stats, dict):
                    meta_timeout["stats"] = stats
                prompt_tokens = _extract_prompt_tokens(stats if isinstance(stats, dict) else None, args.model)
                if prompt_tokens is not None:
                    meta_timeout["prompt_tokens"] = prompt_tokens
                    meta_timeout["over_effective_context_limit"] = bool(prompt_tokens > args.effective_context_tokens)

                if isinstance(session_id, str) and session_id and isinstance(agent_messages, str) and agent_messages.strip():
                    meta_timeout.setdefault("warnings", []).append(timeout_warning)
                    result: Dict[str, Any] = {"success": True, "SESSION_ID": session_id, "agent_messages": agent_messages, "meta": meta_timeout}
                    if args.return_all_messages:
                        result["all_messages"] = [payload]
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    return

            else:
                events = _parse_stream_json(stdout, stderr)
                session_id = _extract_session_id(events)
                agent_messages = _extract_agent_messages_from_stream(events)
                stats = _extract_stats_from_stream(events)

                if isinstance(stats, dict):
                    meta_timeout["stats"] = stats
                prompt_tokens = _extract_prompt_tokens(stats if isinstance(stats, dict) else None, args.model)
                if prompt_tokens is not None:
                    meta_timeout["prompt_tokens"] = prompt_tokens
                    meta_timeout["over_effective_context_limit"] = bool(prompt_tokens > args.effective_context_tokens)

                if session_id and agent_messages:
                    meta_timeout.setdefault("warnings", []).append(timeout_warning)
                    result = {"success": True, "SESSION_ID": session_id, "agent_messages": agent_messages, "meta": meta_timeout, "all_messages": events}
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    return
        except Exception:
            # Fall through to deterministic timeout error.
            pass

        result = {
            "success": False,
            "error": "Gemini CLI process timed out.",
            "meta": {**meta_timeout, "stdout": stdout.strip(), "stderr": stderr.strip()},
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    try:
        if output_format == "json":
            payload = _parse_single_json(stdout, stderr)
            session_id = payload.get("session_id")
            agent_messages = payload.get("response")
            stats = payload.get("stats")

            meta["stats"] = stats
            prompt_tokens = _extract_prompt_tokens(stats, args.model)
            if prompt_tokens is not None:
                meta["prompt_tokens"] = prompt_tokens
                meta["over_effective_context_limit"] = bool(prompt_tokens > args.effective_context_tokens)
                if meta["over_effective_context_limit"]:
                    meta.setdefault("warnings", []).append(
                        "Prompt tokens exceed effective-context threshold; consider starting a NEW session for the next turn (omit --SESSION_ID) "
                        "and/or reducing focus files."
                    )

            success = bool(
                rc == 0
                and isinstance(session_id, str)
                and session_id
                and isinstance(agent_messages, str)
                and agent_messages.strip()
            )
            if success:
                result: Dict[str, Any] = {"success": True, "SESSION_ID": session_id, "agent_messages": agent_messages}
            else:
                error_bits = []
                if rc != 0:
                    error_bits.append(f"[exit_code] {rc}")
                if stderr.strip():
                    error_bits.append(f"[stderr] {stderr.strip()}")
                error_bits.append(f"[stdout] {stdout.strip()}")
                result = {"success": False, "error": "\n".join(error_bits).strip()}

            if args.return_all_messages:
                result["all_messages"] = [payload]

        else:
            events = _parse_stream_json(stdout, stderr)
            session_id = _extract_session_id(events)
            agent_messages = _extract_agent_messages_from_stream(events)
            stats = _extract_stats_from_stream(events)
            meta["stats"] = stats

            prompt_tokens = _extract_prompt_tokens(stats, args.model)
            if prompt_tokens is not None:
                meta["prompt_tokens"] = prompt_tokens
                meta["over_effective_context_limit"] = bool(prompt_tokens > args.effective_context_tokens)
                if meta["over_effective_context_limit"]:
                    meta.setdefault("warnings", []).append(
                        "Prompt tokens exceed effective-context threshold; consider starting a NEW session for the next turn (omit --SESSION_ID) "
                        "and/or reducing focus files."
                    )

            # Detect error status if present.
            status: Optional[str] = None
            for ev in reversed(events):
                if ev.get("type") == "result":
                    status = ev.get("status")
                    break
            meta["status"] = status

            success = bool(rc == 0 and session_id and agent_messages and (status in (None, "success")))
            if success:
                result = {"success": True, "SESSION_ID": session_id, "agent_messages": agent_messages}
            else:
                error_bits = []
                if status and status != "success":
                    error_bits.append(f"[status] {status}")
                if stderr.strip():
                    error_bits.append(f"[stderr] {stderr.strip()}")
                if stdout.strip():
                    error_bits.append(f"[stdout] {stdout.strip()}")
                if rc != 0:
                    error_bits.append(f"[exit_code] {rc}")
                result = {"success": False, "error": "\n".join(error_bits).strip()}

            result["all_messages"] = events

        result["meta"] = meta

    except Exception as error:  # noqa: BLE001 - keep bridge resilient
        # If parsing fails, still return as much context as possible so users
        # can debug, instead of only surfacing a parsing exception.
        result = {
            "success": False,
            "error": f"Bridge failed to parse Gemini CLI output: {error}\n\n[stderr]\n{stderr.strip()}\n\n[stdout]\n{stdout.strip()}".strip(),
        }
        result["meta"] = meta

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
