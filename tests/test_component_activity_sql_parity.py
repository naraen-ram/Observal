# SPDX-FileCopyrightText: 2026 Observal Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pre-consolidation SQL fingerprints for the three scoped session reads.

These hash the entire query (including cohort and joins), not synthetic
execution results. If query semantics deliberately change, update the
fingerprints only after checking all three evidence types and running the
opt-in ClickHouse integration suites against disposable databases.
"""

import hashlib

import pytest

from services.component_activity import hook_queries, queries, skill_queries


@pytest.mark.parametrize(
    ("module", "attribute", "expected"),
    [
        (queries, "_SESSIONS", "83cb71d1c38f34e4af7fada363ad36d1ee770d381db98e1a72be8002f95eba9d"),
        (skill_queries, "_SKILL_SESSIONS", "5d0d9d9c4f206d0d1697bd1570b730dbfcdb97dbd3c723b920730fd05a7dd2a6"),
        (hook_queries, "_HOOK_SESSIONS", "c39efbf248d5604c646ad363aa1266076510743823b43fffbd7d168f0cbfb4f6"),
    ],
)
def test_session_cohort_sql_unchanged(module, attribute, expected):
    assert hashlib.sha256(getattr(module, attribute).encode()).hexdigest() == expected


def test_project_only_skill_scope_is_bound_to_current_verified_listing():
    scopes = skill_queries._SKILL_SCOPES
    sessions = skill_queries._SKILL_SESSIONS
    summary = skill_queries._SKILL_SUMMARY
    for fragment in (
        "c.extraction_generation = published.generation",
        "published.conflict = 0",
        "c.identity_status = 'resolved' AND c.verification_status = 'verified'",
        "c.component_id = {component_id:String}",
        "c.component_version_id = {component_version_id:String}",
        "GROUP BY c.user_id, c.harness, c.layer_hash",
    ):
        assert fragment in scopes
    assert "scope_map.project_installs > 0 AND scope_map.other_installs = 0" in sessions
    assert "scope_map.layer_hash" in sessions
    assert "AND scope_measurable) AS usage_measurable_sessions" in summary
    assert "AND scope_measurable) AS invoked_measurable_sessions" in summary


def test_evidence_kind_is_scoped_per_publication():
    for evidence_type in ("mcp", "skill", "hook"):
        publication = queries.latest_publication_sql(evidence_type)
        markers = queries.marker_states_sql(evidence_type)
        assert f"evidence_type = '{evidence_type}'" in publication
        assert f"evidence_type = '{evidence_type}'" in markers
    with pytest.raises(ValueError):
        queries.latest_publication_sql("mcp' OR 1=1")
    with pytest.raises(ValueError):
        queries.cohort_session_joins_sql("", "", "", "other")
