# SPDX-FileCopyrightText: 2026 Observal Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cursor 2026.10.14 private hook source, not a published MCP-use extractor."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from observal_cli.harness import ensure_loaded, get_adapter
from observal_cli.harness.protocol import SessionSource
from observal_cli.hooks import cursor_mcp_evidence as evidence
from observal_cli.sessions.cursor import project_key_from_cwd
from observal_cli.shared.launcher import module_command

if TYPE_CHECKING:
    from collections.abc import Callable

_SESSION = "11111111-1111-4111-8111-111111111111"
_PRIVATE = "PRIVATE_SENTINEL_MUST_NOT_LEAVE_HOOK_MEMORY"


@pytest.fixture
def source_environment(tmp_path: Path) -> tuple[Path, Path, SessionSource, Callable[..., dict]]:
    home, project = tmp_path / "home", tmp_path / "workspace"
    (home / ".observal").mkdir(parents=True, mode=0o700)
    (home / ".observal/config.json").write_text(
        json.dumps(
            {
                "server_url": "http://127.0.0.1:8000",
                "access_token": "test-token",
                "user_id": "22222222-2222-4222-8222-222222222222",
            }
        )
    )
    (home / ".cursor").mkdir()
    project.mkdir()
    key = project_key_from_cwd(str(project.resolve()))
    path = home / ".cursor/projects" / key / "agent-transcripts" / _SESSION / f"{_SESSION}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"role":"assistant","message":{"content":[]}}\n')
    hooks = {
        event: [{"command": f"{module_command(evidence._MODULE)} --event {stage}", "type": "command"}]
        for event, stage in evidence._EVENTS.items()
    }
    (home / ".cursor/hooks.json").write_text(json.dumps({"version": 1, "hooks": hooks}))

    def event(stage: str, **extra: object) -> dict:
        return {
            "hook_event_name": stage,
            "session_id": _SESSION,
            "conversation_id": _SESSION,
            "cursor_version": evidence._TESTED_VERSION,
            "workspace_roots": [str(project)],
            "transcript_path": str(path),
            "user_email": _PRIVATE,
            "prompt": _PRIVATE,
            **extra,
        }

    return home, project, SessionSource("cursor", _SESSION, path, cwd=str(project)), event


def _call(event: Callable[..., dict], stage: str, *, result: dict | None = None, server: str = "probe") -> dict:
    extra: dict = {"tool_use_id": "call_001", "mcp_server_name": server, "tool_name": "ping", "tool_input": _PRIVATE}
    if result is not None:
        extra["result_json"] = json.dumps(result)
    return event(stage, **extra)


def _attempt(source: SessionSource, count: int = 1) -> None:
    assert source.path is not None
    with source.path.open("a") as handle:
        for _ in range(count):
            handle.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "message": {
                            "content": [{"type": "tool_use", "name": "CallDynamicTool", "input": {"secret": _PRIVATE}}]
                        },
                    }
                )
                + "\n"
            )


def _journal(home: Path) -> Path:
    return home / ".observal/cursor-mcp-evidence" / f"{_SESSION}.jsonl"


def test_no_tool_has_private_start_end_and_only_sanitized_source(source_environment):
    home, _project, source, event = source_environment
    assert evidence.record_hook(event("sessionStart", transcript_path=None), home=home)
    assert evidence.record_hook(event("sessionEnd"), home=home)
    lines = evidence.source_records(source, event("sessionEnd"), home=home)
    assert [json.loads(line)[evidence._KEY]["event"] for line in lines] == ["start", "end"]
    assert _PRIVATE not in _journal(home).read_text() and _PRIVATE not in "".join(lines)
    assert stat.S_IMODE(_journal(home).stat().st_mode) == 0o600
    assert stat.S_IMODE(_journal(home).parent.stat().st_mode) == 0o700
    ensure_loaded()
    adapter = get_adapter("cursor")
    assert adapter.session_extra_records(source, event("sessionEnd"), True, home=home) == lines
    assert adapter.session_extra_records(source, event("sessionStart"), False, home=home) == ()
    assert adapter.session_extra_records(source, event("sessionEnd"), False, home=home) == ()


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        ({"content": [{"text": _PRIVATE}], "isError": False}, "success"),
        ({"content": [{"text": _PRIVATE}], "isError": True}, "error"),
        ({"reason": _PRIVATE, "rejected": True}, "rejected"),
    ],
)
def test_call_pair_only_emits_server_tool_hash_and_outcome(source_environment, result, outcome):
    home, _project, source, event = source_environment
    _attempt(source)
    for item in (
        event("sessionStart"),
        _call(event, "beforeMCPExecution"),
        _call(event, "afterMCPExecution", result=result),
        event("sessionEnd"),
    ):
        assert evidence.record_hook(item, home=home)
    lines = evidence.source_records(source, event("sessionEnd"), home=home)
    assert len(lines) == 4
    before, after = (json.loads(lines[index])[evidence._KEY] for index in (1, 2))
    assert before["call_sha256"] == after["call_sha256"] and len(after["call_sha256"]) == 64
    assert (after["server"], after["tool"], after["outcome"]) == ("probe", "ping", outcome)
    assert _PRIVATE not in _journal(home).read_text() and _PRIVATE not in "".join(lines)
    assert "tool_input" not in "".join(lines) and "result_json" not in "".join(lines)


def test_same_tool_on_second_server_is_distinct_and_native_count_is_required(source_environment):
    home, _project, source, event = source_environment
    _attempt(source, count=2)
    assert evidence.record_hook(event("sessionStart"), home=home)
    for server, call_id in (("probe", "call_1"), ("collision", "call_2")):
        for stage in ("beforeMCPExecution", "afterMCPExecution"):
            item = _call(event, stage, result={"content": [], "isError": False}, server=server)
            item["tool_use_id"] = call_id
            assert evidence.record_hook(item, home=home)
    assert evidence.record_hook(event("sessionEnd"), home=home)
    lines = evidence.source_records(source, event("sessionEnd"), home=home)
    assert len(lines) == 6
    records = [json.loads(line)[evidence._KEY] for line in lines]
    assert records[1]["server"] != records[3]["server"]
    assert records[1]["call_sha256"] != records[3]["call_sha256"]
    _attempt(source)
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()


@pytest.mark.parametrize("missing", ["start", "before", "after", "end"])
def test_missing_lifecycle_or_result_never_exposes_complete_source(source_environment, missing):
    home, _project, source, event = source_environment
    _attempt(source)
    sequence = [
        ("start", event("sessionStart")),
        ("before", _call(event, "beforeMCPExecution")),
        ("after", _call(event, "afterMCPExecution", result={"content": [], "isError": False})),
        ("end", event("sessionEnd")),
    ]
    for label, item in sequence:
        if label != missing:
            assert evidence.record_hook(item, home=home)
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()


def test_duplicate_call_id_mismatched_server_and_config_change_fail_closed(source_environment):
    home, project, source, event = source_environment
    _attempt(source)
    assert evidence.record_hook(event("sessionStart"), home=home)
    assert evidence.record_hook(_call(event, "beforeMCPExecution"), home=home)
    assert evidence.record_hook(
        _call(event, "afterMCPExecution", result={"content": [], "isError": False}, server="other"), home=home
    )
    assert evidence.record_hook(event("sessionEnd"), home=home)
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()
    project_hooks = project / ".cursor/hooks.json"
    project_hooks.parent.mkdir()
    project_hooks.write_text("{}")
    assert not evidence.record_hook(event("sessionStart"), home=home)
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()
    project_hooks.unlink()
    hooks = home / ".cursor/hooks.json"
    hooks.write_text(hooks.read_text() + " ")
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()
    hooks.write_text(hooks.read_text().rstrip())
    config = home / ".observal/config.json"
    data = json.loads(config.read_text())
    data["user_id"] = "33333333-3333-4333-8333-333333333333"
    config.write_text(json.dumps(data))
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()


def test_duplicate_before_after_and_bad_result_fail_closed(source_environment):
    home, _project, source, event = source_environment
    _attempt(source)
    before = _call(event, "beforeMCPExecution")
    after = _call(event, "afterMCPExecution", result={"content": [], "isError": False})
    for item in (event("sessionStart"), before, before, after, event("sessionEnd")):
        assert evidence.record_hook(item, home=home)
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()
    assert not evidence.record_hook(
        _call(event, "afterMCPExecution", result={"rejected": False, "reason": "x"}), home=home
    )
    assert not evidence.record_hook(
        _call(event, "afterMCPExecution", result={"content": [], "isError": "false"}), home=home
    )
    assert not evidence.record_hook(
        _call(event, "afterMCPExecution", result={"content": [], "isError": False, "rejected": True}),
        home=home,
    )


@pytest.mark.parametrize("change", ["version", "session", "multiple_roots", "transcript", "tool_id"])
def test_unverified_binding_cannot_write_evidence(source_environment, change):
    home, _project, source, event = source_environment
    item = event("sessionStart")
    if change == "version":
        item["cursor_version"] = "future"
    elif change == "session":
        item["conversation_id"] = "other"
    elif change == "multiple_roots":
        item["workspace_roots"] *= 2
    elif change == "transcript":
        item["transcript_path"] = str(source.path.parent.parent / _SESSION / "outside.jsonl")
    elif change == "tool_id":
        item = _call(event, "beforeMCPExecution")
        item["tool_use_id"] = "../PRIVATE_SENTINEL"
    assert not evidence.record_hook(item, home=home)
    assert not _journal(home).exists()


def test_non_command_or_forged_hook_is_not_a_no_tool_heartbeat(source_environment):
    home, _project, source, event = source_environment
    p = home / ".cursor/hooks.json"
    d = json.loads(p.read_text())
    d["hooks"]["beforeMCPExecution"][0]["type"] = "prompt"
    p.write_text(json.dumps(d))
    assert not evidence.record_hook(event("sessionStart"), home=home)
    assert not evidence.record_hook(event("sessionEnd"), home=home)
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()
    d["hooks"]["beforeMCPExecution"][0]["type"] = "command"
    d["hooks"]["beforeMCPExecution"][0]["command"] = "echo -m " + evidence._MODULE + " --event before"
    p.write_text(json.dumps(d))
    assert not evidence.record_hook(event("sessionStart"), home=home)


def test_symlink_or_world_readable_journal_and_untrusted_native_form_fail_closed(source_environment):
    home, _project, source, event = source_environment
    journal = _journal(home)
    journal.parent.mkdir(mode=0o700)
    journal.symlink_to(home / ".cursor/hooks.json")
    assert not evidence.record_hook(event("sessionStart"), home=home)
    journal.unlink()
    assert evidence.record_hook(event("sessionStart"), home=home)
    os.chmod(journal, 0o644)
    assert not evidence.record_hook(event("sessionEnd"), home=home)
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()
    os.chmod(journal, 0o600)
    assert evidence.record_hook(event("sessionEnd"), home=home)
    assert source.path is not None
    source.path.write_text('{"role":"assistant","message":{"content":[{"type":"tool_use","name":"MCP:ping"}]}}\n')
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()


def test_other_native_tool_form_is_unknown_not_zero(source_environment):
    home, _project, source, event = source_environment
    assert evidence.record_hook(event("sessionStart"), home=home)
    assert evidence.record_hook(event("sessionEnd"), home=home)
    assert source.path is not None
    source.path.write_text('{"role":"assistant","message":{"content":[{"type":"tool_use","name":"Read"}]}}\n')
    assert evidence.source_records(source, event("sessionEnd"), home=home) == ()


def test_doctor_installs_and_removes_only_managed_cursor_evidence(tmp_path, monkeypatch):
    from observal_cli import cmd_doctor

    home = tmp_path / "home"
    (home / ".cursor").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    assert cmd_doctor._patch_cursor(dry_run=False)
    data = json.loads((home / ".cursor/hooks.json").read_text())
    for event, stage in evidence._EVENTS.items():
        assert len(data["hooks"][event]) == 1
        assert data["hooks"][event][0]["command"].endswith(f"-m {evidence._MODULE} --event {stage}")
    assert evidence._hooks_hash(home) is not None
    assert cmd_doctor._patch_cursor(dry_run=False) is False
    data["hooks"]["sessionStart"][0]["type"] = "prompt"
    (home / ".cursor/hooks.json").write_text(json.dumps(data))
    assert cmd_doctor._patch_cursor(dry_run=False) is True
    assert evidence._hooks_hash(home) is not None
    data = json.loads((home / ".cursor/hooks.json").read_text())
    issues, warnings = [], []
    cmd_doctor._check_cursor(issues, warnings)
    assert not issues and not warnings
    data["hooks"]["sessionEnd"].insert(0, {"command": "foreign"})
    data["hooks"]["sessionEnd"].insert(0, {"command": "someone_else_session_push_helper"})
    (home / ".cursor/hooks.json").write_text(json.dumps(data))
    cmd_doctor._check_cursor(issues, warnings)
    assert any("evidence hooks are incomplete" in warning for warning in warnings)
    assert cmd_doctor._cleanup_cursor(dry_run=False)
    assert json.loads((home / ".cursor/hooks.json").read_text())["hooks"] == {
        "sessionEnd": [{"command": "someone_else_session_push_helper"}, {"command": "foreign"}]
    }


def test_doctor_refuses_malformed_cursor_hook_mapping(tmp_path, monkeypatch):
    from observal_cli import cmd_doctor

    home = tmp_path / "home"
    (home / ".cursor").mkdir(parents=True)
    path = home / ".cursor/hooks.json"
    path.write_text('{"hooks":[]}')
    monkeypatch.setattr(Path, "home", lambda: home)
    with pytest.raises(ValueError, match="Cursor hooks must be a JSON object"):
        cmd_doctor._patch_cursor(dry_run=False)
    with pytest.raises(ValueError, match="Cursor hooks must be a JSON object"):
        cmd_doctor._cleanup_cursor(dry_run=False)
    issues, warnings = [], []
    cmd_doctor._check_cursor(issues, warnings)
    assert issues and not warnings
    assert path.read_text() == '{"hooks":[]}'


def test_unsupported_harness_remains_unpublished_and_local_hook_is_passive(source_environment, monkeypatch, capsys):
    from observal_cli.hooks import session_push
    from observal_shared.harness_registry import HARNESS_REGISTRY

    home, _project, _source, event = source_environment
    assert HARNESS_REGISTRY["cursor"].get("invocation_extractor") is None
    monkeypatch.setattr(Path, "home", lambda: home)
    seen = []
    monkeypatch.setattr(session_push, "_run_hook", lambda payload, harness: seen.append((payload, harness)))
    payload = event("sessionEnd")
    monkeypatch.setattr(sys, "argv", ["collector", "--event", "end"])
    import io

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode())))
    evidence.main()
    assert seen == [(payload, "cursor")]
    assert capsys.readouterr().out == "{}\n"
    assert evidence.source_records(_source, payload, home=home) == ()  # no start heartbeat
