# SPDX-FileCopyrightText: 2026 Observal Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail closed on shared harness MCP settings; never upload their credentials."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from observal_cli import layer
from observal_cli.harness import ensure_loaded, get_adapter

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _recorded_cli_versions(monkeypatch):
    versions = {
        "kiro-cli": "kiro-cli 2.32.0",
        "codex": "codex-cli 0.163.0",
        "cursor-agent": "2026.10.14-941492e",
    }
    monkeypatch.setattr("observal_cli.harness.mcp_binding.native_mcp_version", versions.get)


@pytest.mark.parametrize(
    ("harness", "user_path", "project_path"),
    [
        ("kiro", ".kiro/settings/mcp.json", ".kiro/settings/mcp.json"),
        ("cursor", ".cursor/mcp.json", ".cursor/mcp.json"),
        ("codex", ".codex/config.toml", ".codex/config.toml"),
    ],
)
def test_shared_mcp_settings_are_hash_only_in_layer_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness: str, user_path: str, project_path: str
):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(layer, "_FILE_HASH_CACHE_PATH", tmp_path / "hash-cache.json")
    sentinel = "MCP_SECRET_VALUE_NEVER_UPLOAD"
    for root, relative in ((home, user_path), (project, project_path)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"header = '{sentinel}'\n" if harness == "codex" else json.dumps({"secret": sentinel}))
    manifest = layer.build_layer_manifest(harness, str(project), include_content=True, registry_data={"harnesses": {}})
    paths = {row["path"]: row for row in manifest}
    expected = {f"user:{user_path.removeprefix('.' + harness + '/')}", f"project:{project_path}"}
    assert expected <= paths.keys()
    for path in expected:
        assert paths[path]["content"] == ""
        assert paths[path]["hash"].startswith("sha256-")
    assert sentinel not in json.dumps(manifest)


def _write_mcp_config(path: Path, harness: str, alias: str | None, *, command: str = "inert") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if harness == "codex":
        rows = [f'[mcp_servers."{alias}"]', f'command = "{command}"', 'args = ["probe"]'] if alias else []
        path.write_text("\n".join(rows) + "\n")
    else:
        path.write_text(json.dumps({"mcpServers": {alias: {"command": command, "args": ["probe"]}} if alias else {}}))
    path.chmod(0o600)


@pytest.mark.parametrize("harness", ["kiro", "codex", "cursor"])
def test_mcp_binding_needs_exact_effective_scope_and_unshadowed_name(tmp_path, monkeypatch, harness):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    if harness == "kiro":
        monkeypatch.setattr("observal_cli.harness.kiro.kiro_cli_major", lambda *_args: 2)
    paths = {
        "kiro": ".kiro/settings/mcp.json",
        "codex": ".codex/config.toml",
        "cursor": ".cursor/mcp.json",
    }
    alias = "matrix_scope_probe"
    user, workspace = home / paths[harness], project / paths[harness]
    _write_mcp_config(user, harness, None)
    _write_mcp_config(workspace, harness, alias)
    if harness == "codex":
        # Without explicit trust, this present project file is ignored by the tested CLI.
        ensure_loaded()
        assert get_adapter(harness).read_installed_mcp("project", str(project), alias)[0] == "unverified"
        user.write_text(f'[projects."{project.resolve()}"]\ntrust_level = "trusted"\n')
    ensure_loaded()
    adapter = get_adapter(harness)
    assert adapter.read_installed_mcp("project", str(project), alias)[0] == "verified"
    assert adapter.mcp_manifest_path("project") == f"project:{paths[harness]}"
    status, fingerprint = layer.verify_installed_mcp(harness, "project", str(project), alias)
    assert status == "verified" and fingerprint and fingerprint.startswith("sha256-")
    _write_mcp_config(workspace, harness, alias, command="changed")
    assert layer.verify_installed_mcp(harness, "project", str(project), alias, fingerprint)[0] == "drifted"
    _write_mcp_config(user, harness, alias)
    if harness == "codex":
        with user.open("a") as file:
            file.write(f'\n[projects."{project.resolve()}"]\ntrust_level = "trusted"\n')
    assert adapter.read_installed_mcp("project", str(project), alias)[0] == "unverified"


def test_kiro_agent_duplicate_does_not_verify_even_with_include_mcp_json_disabled(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("observal_cli.harness.kiro.kiro_cli_major", lambda *_args: 2)
    _write_mcp_config(home / ".kiro/settings/mcp.json", "kiro", None)
    _write_mcp_config(project / ".kiro/settings/mcp.json", "kiro", "probe")
    agent = project / ".kiro/agents/other.json"
    agent.parent.mkdir(parents=True)
    agent.write_text(json.dumps({"includeMcpJson": False, "mcpServers": {"probe": {"command": "other"}}}))
    ensure_loaded()
    assert get_adapter("kiro").read_installed_mcp("project", str(project), "probe")[0] == "unverified"
    agent.write_text("not json")
    assert get_adapter("kiro").read_installed_mcp("project", str(project), "probe")[0] == "unverified"


@pytest.mark.parametrize("harness", ["kiro", "codex", "cursor"])
def test_malformed_or_disabled_mcp_entry_is_not_verified(tmp_path, monkeypatch, harness):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr("observal_cli.harness.kiro.kiro_cli_major", lambda *_args: 2)
    user = home / (
        ".codex/config.toml"
        if harness == "codex"
        else f".{harness}/" + ("settings/mcp.json" if harness == "kiro" else "mcp.json")
    )
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text("not valid TOML = [" if harness == "codex" else "{bad json")
    ensure_loaded()
    assert get_adapter(harness).read_installed_mcp("user", str(project), "probe")[0] == "unverified"
    if harness == "codex":
        user.write_text('[mcp_servers.probe]\ncommand = "inert"\nenabled = false\n')
    else:
        user.write_text(json.dumps({"mcpServers": {"probe": {"command": "inert", "disabled": True}}}))
    assert get_adapter(harness).read_installed_mcp("user", str(project), "probe")[0] == "unverified"


@pytest.mark.parametrize("harness", ["kiro", "codex", "cursor"])
def test_native_stdio_install_refuses_collisions_and_leaves_credentials_out_of_snapshot(tmp_path, monkeypatch, harness):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr("observal_cli.harness.kiro.kiro_cli_major", lambda *_args: 2)
    monkeypatch.setattr(layer, "_FILE_HASH_CACHE_PATH", tmp_path / "hash-cache.json")
    ensure_loaded()
    adapter = get_adapter(harness)
    if harness == "codex":
        user = home / ".codex/config.toml"
        user.parent.mkdir()
        user.write_text(f'[projects."{project.resolve()}"]\ntrust_level = "trusted"\n')
    alias = "tracked_probe"
    entry = {"command": "inert", "args": ["probe"], "env": {"TOKEN": "MCP_SECRET_VALUE_NEVER_UPLOAD"}}
    key = "mcp_servers" if harness == "codex" else "mcpServers"
    snippet = {key: {alias: entry}}
    adapter.configure_tracked_mcp("project", str(project), alias, snippet)
    assert layer.verify_installed_mcp(harness, "project", str(project), alias)[0] == "verified"
    manifest = layer.build_layer_manifest(harness, str(project), include_content=True, registry_data={"harnesses": {}})
    assert "MCP_SECRET_VALUE_NEVER_UPLOAD" not in json.dumps(manifest)
    assert adapter.mcp_manifest_path("project") in {row["path"] for row in manifest}
    with pytest.raises(ValueError, match="alias"):
        adapter.configure_tracked_mcp("project", str(project), alias, snippet)
    for invalid in (
        {key: {alias: {"url": "https://example.test"}}},
        {key: {"other": entry}},
        {key: {alias: entry, "other": entry}},
    ):
        with pytest.raises(ValueError):
            adapter.configure_tracked_mcp("project", str(project), alias + "-new", invalid)


def test_codex_project_install_refuses_untrusted_workspace_before_writing(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    ensure_loaded()
    adapter = get_adapter("codex")
    with pytest.raises(ValueError, match="untrusted"):
        adapter.configure_tracked_mcp(
            "project", str(project), "probe", {"mcp_servers": {"probe": {"command": "inert", "args": [], "env": {}}}}
        )
    assert not (project / ".codex/config.toml").exists()


def test_native_install_does_not_follow_direct_config_symlink(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"mcpServers": {}}))
    target = project / ".cursor/mcp.json"
    target.parent.mkdir()
    target.symlink_to(outside)
    ensure_loaded()
    with pytest.raises(ValueError, match="symlink"):
        get_adapter("cursor").configure_tracked_mcp(
            "project", str(project), "probe", {"mcpServers": {"probe": {"command": "inert", "args": [], "env": {}}}}
        )
    assert json.loads(outside.read_text()) == {"mcpServers": {}}


@pytest.mark.parametrize("harness", ["kiro", "codex", "cursor"])
def test_tracked_install_rejects_unrecorded_native_version_without_writing(tmp_path, monkeypatch, harness):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr("observal_cli.harness.mcp_binding.native_mcp_version", lambda _binary: None)
    ensure_loaded()
    key = "mcp_servers" if harness == "codex" else "mcpServers"
    with pytest.raises(ValueError, match="requires the tested"):
        get_adapter(harness).configure_tracked_mcp(
            "project", str(project), "probe", {key: {"probe": {"command": "inert", "args": [], "env": {}}}}
        )
    config_path = {"kiro": ".kiro/settings/mcp.json", "codex": ".codex/config.toml", "cursor": ".cursor/mcp.json"}[
        harness
    ]
    assert not (project / config_path).exists()


@pytest.mark.parametrize("harness", ["kiro", "codex", "cursor"])
def test_tracked_user_install_is_project_bound_and_shadow_fails_closed(tmp_path, monkeypatch, harness):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr("observal_cli.harness.kiro.kiro_cli_major", lambda *_args: 2)
    ensure_loaded()
    adapter = get_adapter(harness)
    alias = "user_probe"
    key = "mcp_servers" if harness == "codex" else "mcpServers"
    snippet = {key: {alias: {"command": "inert", "args": ["probe"], "env": {}}}}
    adapter.configure_tracked_mcp("user", str(project), alias, snippet)
    assert layer.verify_installed_mcp(harness, "user", str(project), alias)[0] == "verified"
    assert adapter.mcp_manifest_path("user") is not None
    other = (
        project
        / {"kiro": ".kiro/settings/mcp.json", "codex": ".codex/config.toml", "cursor": ".cursor/mcp.json"}[harness]
    )
    _write_mcp_config(other, harness, alias, command="other")
    assert adapter.read_installed_mcp("user", str(project), alias)[0] == "unverified"


@pytest.mark.parametrize("harness", ["kiro", "codex", "cursor"])
def test_tracked_install_undo_restores_exact_original_and_refuses_concurrent_edit(tmp_path, monkeypatch, harness):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr("observal_cli.harness.kiro.kiro_cli_major", lambda *_args: 2)
    ensure_loaded()
    adapter = get_adapter(harness)
    if harness == "codex":
        trusted = home / ".codex/config.toml"
        trusted.parent.mkdir()
        trusted.write_text(f'[projects."{project.resolve()}"]\ntrust_level = "trusted"\n')
    config = (
        project
        / {"kiro": ".kiro/settings/mcp.json", "codex": ".codex/config.toml", "cursor": ".cursor/mcp.json"}[harness]
    )
    config.parent.mkdir(parents=True)
    original = b'[other]\nkeep = "yes"\n' if harness == "codex" else b'{"mcpServers": {}, "keep": true}\n'
    config.write_bytes(original)
    key = "mcp_servers" if harness == "codex" else "mcpServers"
    snippet = {key: {"probe": {"command": "inert", "args": [], "env": {}}}}
    undo = adapter.configure_tracked_mcp("project", str(project), "probe", snippet)
    assert undo.rollback() is True
    assert config.read_bytes() == original
    undo = adapter.configure_tracked_mcp("project", str(project), "probe", snippet)
    config.write_bytes(config.read_bytes() + (b"\n# edited\n" if harness == "codex" else b" "))
    assert undo.rollback() is False
    assert config.read_bytes().endswith(b"# edited\n" if harness == "codex" else b" ")
