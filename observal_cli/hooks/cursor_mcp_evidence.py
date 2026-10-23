# SPDX-FileCopyrightText: 2026 Observal Contributors
# SPDX-License-Identifier: Apache-2.0

"""Private, bounded Cursor MCP hook journal; no hook input or result is retained.

This is a *source collector*, not an invocation extractor. Only the tested CLI
version supplies the observed hook shape. A server may use these sanitized lines
only after independently checking lifecycle completeness and the native JSONL.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from observal_cli.harness.protocol import SessionSource

_TESTED_VERSION = "2026.10.14-941492e"
_MODULE = "observal_cli.hooks.cursor_mcp_evidence"
_EVENTS = {"sessionStart": "start", "beforeMCPExecution": "before", "afterMCPExecution": "after", "sessionEnd": "end"}
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SESSION = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_ALIAS = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_TOOL = re.compile(r"[A-Za-z0-9_.-]{1,256}\Z")
_MAX_INPUT = 1024 * 1024
_MAX_JOURNAL = 256 * 1024
_MAX_RECORDS = 1000
_KEY = "_observal_cursor_mcp_v1"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _binding(event: dict, home: Path) -> tuple[str, str, str, str] | None:
    """Bind hook input to one native transcript and one existing workspace.

    Only hashes of paths enter the journal; no project, transcript or home path
    can be emitted as evidence. Resolve /tmp aliases before checking ancestry.
    """
    from observal_cli.sessions.cursor import project_key_from_cwd

    session = event.get("session_id")
    roots = event.get("workspace_roots")
    transcript = event.get("transcript_path")
    if (
        not isinstance(session, str)
        or not _SESSION.fullmatch(session)
        or event.get("conversation_id") != session
        or event.get("cursor_version") != _TESTED_VERSION
        or not isinstance(roots, list)
        or len(roots) != 1
        or not isinstance(roots[0], str)
        or not Path(roots[0]).is_absolute()
    ):
        return None
    root = Path(roots[0]).resolve()
    projects = (home / ".cursor" / "projects").resolve()
    if not root.is_dir():
        return None
    expected = projects / project_key_from_cwd(str(root)) / "agent-transcripts" / session / f"{session}.jsonl"
    # In this CLI build sessionStart has no absolute transcript path yet. Its
    # one-root/session binding identifies the *expected* native location; all
    # later events must name that exact path before their records can pair.
    stage = event.get("hook_event_name")
    if stage == "sessionStart" and (not isinstance(transcript, str) or not Path(transcript).is_absolute()):
        path = expected
    elif isinstance(transcript, str) and Path(transcript).is_absolute():
        path = Path(transcript).resolve()
        if path != expected:
            return None
    else:
        return None
    # A second project-local hook file could shadow the managed user hooks.
    if (root / ".cursor" / "hooks.json").exists():
        return None
    return session, _hash(str(root)), _hash(str(path)), str(root)


def _hooks_hash(home: Path) -> str | None:
    """Require the exact installed collector as sole handler for every trigger."""
    from observal_cli.shared.launcher import module_command

    expected = module_command(_MODULE)
    path = home / ".cursor" / "hooks.json"
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_JOURNAL or path.is_symlink():
            return None
        hooks = json.loads(raw).get("hooks")
        if not isinstance(hooks, dict):
            return None
        for event, stage in _EVENTS.items():
            entries = hooks.get(event)
            if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
                return None
            command = entries[0].get("command")
            if entries[0].get("type") != "command" or command != f"{expected} --event {stage}":
                return None
    except (OSError, UnicodeError, ValueError, AttributeError):
        return None
    return hashlib.sha256(raw).hexdigest()


def _user_hash(home: Path) -> str | None:
    from observal_cli.sessions.base import load_config

    config = load_config(home=home)
    user_id = config.get("user_id") if config else None
    return _hash(user_id) if isinstance(user_id, str) and _SESSION.fullmatch(user_id) else None


def _journal_path(home: Path, session: str) -> Path | None:
    if not _SESSION.fullmatch(session):
        return None
    parent = home / ".observal"
    directory = parent / "cursor-mcp-evidence"
    try:
        if parent.is_symlink() or not parent.is_dir():
            return None
        if stat.S_IMODE(parent.stat().st_mode) & 0o077:
            return None
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir() or stat.S_IMODE(directory.stat().st_mode) != 0o700:
            return None
    except OSError:
        return None
    return directory / f"{session}.jsonl"


def _append(path: Path, record: dict) -> bool:
    """Append one sanitized line under a per-file lock; refuse unsafe files."""
    try:
        import fcntl

        line = json.dumps({_KEY: record}, sort_keys=True, separators=(",", ":")) + "\n"
        if len(line) > 2048:
            return False
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size + len(line) > _MAX_JOURNAL
            ):
                return False
            encoded = line.encode()
            if os.write(fd, encoded) != len(encoded):
                return False
            os.fsync(fd)
            return True
        finally:
            os.close(fd)
    except (OSError, ImportError):
        return False


def record_hook(event: dict, *, home: Path | None = None) -> bool:
    """Sanitize a native hook event before *any* persistent write."""
    home = home or Path.home()
    hook_name = event.get("hook_event_name")
    stage = _EVENTS.get(hook_name) if isinstance(hook_name, str) else None
    binding = _binding(event, home)
    config_hash = _hooks_hash(home)
    user_hash = _user_hash(home)
    if stage is None or binding is None or config_hash is None or user_hash is None:
        return False
    session, root_hash, path_hash, _root = binding
    journal = _journal_path(home, session)
    if journal is None:
        return False
    record: dict = {
        "event": stage,
        "session": session,
        "version": _TESTED_VERSION,
        "platform": sys.platform,
        "root_sha256": root_hash,
        "transcript_sha256": path_hash,
        "hooks_sha256": config_hash,
        "user_sha256": user_hash,
        "observed_at": datetime.now(UTC).isoformat(),
    }
    if stage in {"before", "after"}:
        call_id, server, tool = event.get("tool_use_id"), event.get("mcp_server_name"), event.get("tool_name")
        if (
            not isinstance(call_id, str)
            or not _ID.fullmatch(call_id)
            or not isinstance(server, str)
            or not _ALIAS.fullmatch(server)
            or not isinstance(tool, str)
            or not _TOOL.fullmatch(tool)
        ):
            return False
        record.update(call_sha256=_hash(f"cursor-call:{session}:{call_id}"), server=server, tool=tool)
    if stage == "after":
        raw = event.get("result_json")
        if not isinstance(raw, str) or len(raw) > _MAX_INPUT:
            return False
        try:
            result = json.loads(raw)
        except (ValueError, TypeError):
            return False
        if not isinstance(result, dict):
            return False
        if result.get("rejected") is True and set(result) == {"reason", "rejected"}:
            record["outcome"] = "rejected"
        elif (
            set(result) == {"content", "isError"}
            and type(result.get("isError")) is bool
            and isinstance(result.get("content"), list)
        ):
            record["outcome"] = "error" if result["isError"] else "success"
        else:
            return False
    return _append(journal, record)


def source_records(source: SessionSource, event: dict, *, home: Path | None = None) -> tuple[str, ...]:
    """Return only bounded sanitized journal lines for the native sessionEnd.

    Missing/unsafe/incomplete journals return no evidence; a future extractor
    must require both lifecycle records and every native attempted call.
    """
    home = home or Path.home()
    if event.get("hook_event_name") != "sessionEnd" or source.parent_session_id is not None:
        return ()
    binding = _binding(event, home)
    if binding is None or source.session_id != binding[0] or not source.path or not source.path.is_file():
        return ()
    if (
        _hash(str(source.path.resolve())) != binding[2]
        or not source.cwd
        or _hash(str(Path(source.cwd).resolve())) != binding[1]
    ):
        return ()
    journal = _journal_path(home, source.session_id)
    if journal is None:
        return ()
    try:
        import fcntl

        fd = os.open(journal, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > _MAX_JOURNAL
            ):
                return ()
            raw = os.read(fd, _MAX_JOURNAL + 1)
        finally:
            os.close(fd)
        if len(raw) != info.st_size or not raw.endswith(b"\n"):
            return ()
        lines = raw.decode("utf-8").splitlines()
        if len(lines) < 2 or len(lines) > _MAX_RECORDS:
            return ()
        records = [json.loads(line)[_KEY] for line in lines]
        expected = (source.session_id, binding[1], binding[2], _hooks_hash(home), _user_hash(home))
        if (
            expected[3] is None
            or expected[4] is None
            or any(
                not isinstance(record, dict)
                or tuple(
                    record.get(key)
                    for key in ("session", "root_sha256", "transcript_sha256", "hooks_sha256", "user_sha256")
                )
                != expected
                or record.get("version") != _TESTED_VERSION
                for record in records
            )
        ):
            return ()
        if records[0].get("event") != "start" or records[-1].get("event") != "end":
            return ()
        before: dict[str, dict] = {}
        after: set[str] = set()
        for record in records[1:-1]:
            event_name = record.get("event")
            call = record.get("call_sha256")
            if (
                event_name not in {"before", "after"}
                or not isinstance(call, str)
                or not re.fullmatch(r"[0-9a-f]{64}", call)
            ):
                return ()
            if event_name == "before":
                if call in before or call in after:
                    return ()
                before[call] = record
            else:
                prior = before.get(call)
                if (
                    prior is None
                    or call in after
                    or prior.get("server") != record.get("server")
                    or prior.get("tool") != record.get("tool")
                    or record.get("outcome") not in {"success", "error", "rejected"}
                ):
                    return ()
                after.add(call)
        if len(before) != len(after) or _native_attempt_count(source.path) != len(before):
            return ()
    except (OSError, ImportError, UnicodeError, ValueError, KeyError, TypeError):
        return ()
    return tuple(lines)


def _native_attempt_count(path: Path) -> int:
    """Count only the tested persisted Cursor dynamic MCP call form."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("invalid native record")
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                raise ValueError("invalid native content")
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name")
                if name == "CallDynamicTool":
                    count += 1
                elif name != "GetDynamicTools":
                    # An unrecorded/new tool form might itself be an MCP call.
                    # Preserve a no-tool denominator only for the tested shape.
                    raise ValueError("untested native tool call form")
    return count


def main() -> None:
    """Cursor command-hook entry point. Never changes tool permission or logs input."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--event", choices=tuple(_EVENTS.values()), required=True)
    args = parser.parse_args()
    data = sys.stdin.buffer.read(_MAX_INPUT + 1)
    event: dict = {}
    if len(data) <= _MAX_INPUT:
        try:
            parsed = json.loads(data)
            hook_name = parsed.get("hook_event_name") if isinstance(parsed, dict) else None
            if isinstance(hook_name, str) and _EVENTS.get(hook_name) == args.event:
                event = parsed
                record_hook(event)
        except (ValueError, UnicodeError, OSError, TypeError):
            pass
    if args.event == "end":
        # Preserve session delivery even when evidence collection failed.
        from observal_cli.hooks.session_push import _run_hook

        try:
            _run_hook(event, harness="cursor")
        except Exception:
            pass  # Hooks must never interrupt Cursor; absence stays unsupported.
    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()
