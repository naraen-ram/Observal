# SPDX-FileCopyrightText: 2026 Observal Contributors
# SPDX-License-Identifier: Apache-2.0

"""Version-scoped, sanitized native Codex skills inventory and install proof."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from observal_cli.harness.codex import CodexAdapter
from observal_cli.layer import _skill_status, build_layer_manifest, skill_location_sha256, skill_verification_entry
from services.component_activity import skill_queries
from services.component_activity.skill_matcher import match_skill_evidence
from services.session_parsers.skill_evidence import extract_skill_evidence, kind_recorded

FIXTURE = Path(__file__).parent / "fixtures/component_insights/codex/skill_available.jsonl"
ALIAS = "matrix-probe-skill"


def rows():
    return [
        {"is_source_record": 1, "line_offset": index, "raw_line": raw, "line_hash": f"hash-{index}"}
        for index, raw in enumerate(FIXTURE.read_text().splitlines())
    ]


def changed(rows_, index, change):
    items = [dict(row) for row in rows_]
    raw = json.loads(items[index]["raw_line"])
    change(raw)
    items[index]["raw_line"] = json.dumps(raw)
    return items


def test_native_inventory_is_availability_only_and_never_uses_user_prompt():
    extracted = extract_skill_evidence("codex", rows())
    assert extracted.status == "supported"
    assert extracted.malformed_source_records == 0
    assert len(extracted.evidence) == 1
    fact = extracted.evidence[0]
    assert (fact.kind, fact.scope, fact.alias, fact.source_line_offset) == ("available", "project", ALIAS, 1)
    assert (
        fact.location_sha256
        == hashlib.sha256(f"/private/tmp/observal-codex-proof/.agents/skills/{ALIAS}/SKILL.md".encode()).hexdigest()
    )
    assert "tmp" not in str(fact)
    assert kind_recorded("codex", "available")
    assert not kind_recorded("codex", "load")
    assert not kind_recorded("codex", "invoked")
    assert extract_skill_evidence("kiro", rows()).status == "unsupported"


def test_malformed_missing_or_unverified_version_is_not_zero():
    assert extract_skill_evidence("codex", rows()[1:]).status == "unsupported"
    assert (
        extract_skill_evidence(
            "codex", changed(rows(), 0, lambda record: record["payload"].update(cli_version="0.164.0"))
        ).status
        == "unsupported"
    )
    assert (
        extract_skill_evidence(
            "codex", changed(rows(), 1, lambda record: record["payload"]["content"][0].update(text="other"))
        ).status
        == "unsupported"
    )
    assert extract_skill_evidence("codex", rows() + rows()[:1]).status == "unsupported"


def test_cannot_spoof_inventory_with_user_or_assistant_message():
    changed_role = changed(rows(), 1, lambda record: record["payload"].update(role="user"))
    assert extract_skill_evidence("codex", changed_role).status == "unsupported"
    changed_type = changed(rows(), 1, lambda record: record.update(type="event_msg"))
    assert extract_skill_evidence("codex", changed_type).status == "unsupported"


def test_other_roots_or_ambiguous_aliases_never_match():
    different = changed(
        rows(),
        1,
        lambda record: record["payload"]["content"][0].update(
            text=record["payload"]["content"][0]["text"].replace(
                "/observal-codex-proof/.agents/skills", "/other/.agents/skills"
            )
        ),
    )
    assert extract_skill_evidence("codex", different).status == "unsupported"
    for extra_line in (
        "- matrix-probe-skill: Another scope (file: r1/matrix-probe-skill/SKILL.md)\n",
        "- matrix-probe-skill: Duplicate (file: r0/matrix-probe-skill/SKILL.md)\n",
    ):

        def duplicate(record, line=extra_line):
            text = record["payload"]["content"][0]["text"]
            record["payload"]["content"][0]["text"] = (
                text.replace("### Available skills\n", "- `r1` = `/tmp/user/.agents/skills`\n### Available skills\n")
                + line
            )

        assert extract_skill_evidence("codex", changed(rows(), 1, duplicate)).status == "unsupported"
    # Parent skills are real native listings, but cannot establish this project's availability.
    for replaced in (
        "- `r0` = `/tmp/another/.agents/skills`\n",  # reused root id
        "- `r9` = `/tmp/project/../different/.agents/skills`\n",  # noncanonical root
        "- other: malformed project entry\n",  # unparsed inventory line
    ):
        malformed = changed(
            rows(),
            1,
            lambda record, line=replaced: record["payload"]["content"][0].update(
                text=record["payload"]["content"][0]["text"].replace(
                    "### Available skills\n", line + "### Available skills\n"
                )
                if line.startswith("- `r")
                else record["payload"]["content"][0]["text"] + line
            ),
        )
        assert extract_skill_evidence("codex", malformed).status == "unsupported"


def test_verified_project_file_location_and_shadow_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "workspace"
    file = project / ".agents" / "skills" / ALIAS / "SKILL.md"
    file.parent.mkdir(parents=True)
    file.write_text("---\nname: matrix-probe-skill\n---\nSafe fixture\n")
    expected = f"sha256-{hashlib.sha256(file.read_bytes()).hexdigest()}"
    active = f"project:.agents/skills/{ALIAS}/SKILL.md"
    assert CodexAdapter().skill_manifest_path("project", ALIAS) == active
    assert CodexAdapter().skill_manifest_path("user", ALIAS) is None
    assert _skill_status("codex", "project", str(project), ALIAS, expected, {active: expected}) == "verified"
    assert (
        skill_location_sha256("codex", "project", str(project), ALIAS)
        == hashlib.sha256(str(file.resolve()).encode()).hexdigest()
    )
    pin = {
        "harnesses": {
            "codex": {
                "agents": [],
                "standalone": [
                    {
                        "type": "skill",
                        "id": "5d0c5ffb-c875-4e1e-9948-e53b4c0eeaa4",
                        "local_name": ALIAS,
                        "scope": "project",
                        "directory": str(project),
                        "skill_integrity": expected,
                    }
                ],
            }
        }
    }
    fingerprint = skill_verification_entry("codex", pin, str(project))
    assert fingerprint["path"] == "observal:skill-verification"
    original = build_layer_manifest("codex", str(project), registry_data=pin)
    assert {row["path"]: row["hash"] for row in original}[active] == expected
    assert fingerprint in original
    file.write_text("modified")
    modified = build_layer_manifest("codex", str(project), registry_data=pin)
    new_hash = {row["path"]: row["hash"] for row in modified}[active]
    assert new_hash != expected
    assert _skill_status("codex", "project", str(project), ALIAS, expected, {active: new_hash}) == "drifted"
    file.write_text("---\nname: matrix-probe-skill\n---\nSafe fixture\n")
    shadow = Path.home() / ".codex" / "skills" / ALIAS / "SKILL.md"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("shadow")
    assert _skill_status("codex", "project", str(project), ALIAS, expected, {active: expected}) == "unverified"


def test_matching_requires_verified_identity_exact_location_and_snapshot():
    fact = extract_skill_evidence("codex", rows()).evidence[0]
    candidate = {
        "component_id": "component-id",
        "component_version_id": "version-id",
        "local_name": ALIAS,
        "scope": "project",
        "location_sha256": fact.location_sha256,
        "identity_status": "resolved",
        "verification_status": "verified",
    }
    assert match_skill_evidence((fact,), {"v2_layer": [candidate]}, {1: "v2_layer"}).rows[0]["evidence_kind"] == (
        "skill_available"
    )
    for replacement in (
        {"verification_status": "unverified"},
        {"identity_status": "unknown"},
        {"location_sha256": "0" * 64},
        {"scope": "user"},
    ):
        result = match_skill_evidence((fact,), {"v2_layer": [candidate | replacement]}, {1: "v2_layer"})
        assert result.rows == ()
        assert result.unmatched_count == 1
    other = candidate | {"component_id": "other"}
    result = match_skill_evidence((fact,), {"v2_layer": [candidate, other]}, {1: "v2_layer"})
    assert result.rows == () and result.collision_count == 1


@pytest.mark.asyncio
async def test_codex_only_cohort_reports_offered_but_unknown_use(monkeypatch):
    async def warehouse(sql, _params):
        if sql == skill_queries._SKILL_SUMMARY:
            return [
                {
                    "supported_present_sessions": 1,
                    "projection_complete_sessions": 1,
                    "usage_measurable_sessions": 0,
                    "available_sessions": 1,
                    "loaded_sessions": 0,
                    "total_confirmed_loads": 0,
                    "total_load_attempts": 0,
                    "observed_sessions": 0,
                    "candidate_facts": 2,
                    "attributed_facts": 1,
                    "unmatched_facts": 1,
                }
            ]
        if sql == skill_queries._HARNESSES:
            return [{"harness": "codex", "sessions": 1}]
        return [{}]

    monkeypatch.setattr(skill_queries, "_rows", warehouse)
    monkeypatch.setattr(
        skill_queries,
        "presence_coverage",
        AsyncMock(
            return_value={
                "present_sessions": 1,
                "present_users": 1,
            }
        ),
    )
    monkeypatch.setattr(skill_queries, "presence_version_distribution", AsyncMock(return_value={}))
    now = datetime.now(UTC)
    listing = "11111111-1111-4111-8111-111111111111"
    result = await skill_queries.skill_activity_summary("default", listing, None, (now - timedelta(days=1), now))
    assert result["available_sessions"] == 1
    assert (result["loaded_sessions"], result["confirmed_loads"], result["load_attempts"]) == (None, None, None)
    assert (result["invoked_sessions"], result["invocations"]) == (None, None)
    assert result["coverage"].usage_rate_denominator_sessions == 0
    assert result["coverage"].attribution_state == "attribution_not_possible"
    assert result["coverage"].usage_rate is None
    assert "availability_only_harness_sessions" in result["coverage"].reasons
    assert any(x.startswith("loads_not_recorded") for x in result["coverage"].limitations)
    params = skill_queries._params("default", listing, None, (now - timedelta(days=1), now))
    assert "codex" in params["param_supported"] and "codex" not in params["param_use_capable"]
    assert "has({use_capable:Array(String)}, harness)" in skill_queries._SKILL_SUMMARY

    from services.insights import component_report

    narrative = component_report.generate_skill_sections(result, result["coverage"].model_dump(mode="json"))
    assert "skill was offered" in narrative["summary"]
    assert "No use conclusion can be drawn" in narrative["summary"]


@pytest.mark.asyncio
async def test_unprojected_codex_availability_is_unknown_not_zero(monkeypatch):
    async def warehouse(sql, _params):
        if sql == skill_queries._SKILL_SUMMARY:
            return [{"supported_present_sessions": 1, "projection_pending_sessions": 1}]
        if sql == skill_queries._HARNESSES:
            return [{"harness": "codex", "sessions": 1}]
        return [{}]

    monkeypatch.setattr(skill_queries, "_rows", warehouse)
    monkeypatch.setattr(skill_queries, "presence_coverage", AsyncMock(return_value={"present_sessions": 1}))
    monkeypatch.setattr(skill_queries, "presence_version_distribution", AsyncMock(return_value={}))
    now = datetime.now(UTC)
    summary = await skill_queries.skill_activity_summary(
        "default", "11111111-1111-4111-8111-111111111111", None, (now - timedelta(days=1), now)
    )
    assert all(
        summary[key] is None
        for key in (
            "available_sessions",
            "loaded_sessions",
            "confirmed_loads",
            "load_attempts",
            "invoked_sessions",
            "invocations",
        )
    )
    assert summary["coverage"].attribution_state == "attribution_not_possible"
