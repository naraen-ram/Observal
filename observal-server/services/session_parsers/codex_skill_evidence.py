# SPDX-FileCopyrightText: 2026 Observal Contributors
# SPDX-License-Identifier: Apache-2.0

"""Codex 0.163.0 project skill *availability* from native rollout JSONL.

Codex writes the skills inventory in a developer ``response_item`` as a
``## Skills`` block with a ``### Skill roots`` table and an ``### Available
skills`` list. The inventory names a root and SKILL.md for each entry. Only a
single, exact project ``<cwd>/.agents/skills/<alias>/SKILL.md`` location can
be a candidate; the verified layer pin and its file/location fingerprint are
still required by the shared skill matcher. We do not interpret a user message
containing ``$name`` or an expanded SKILL.md as a native invocation: pasted
text can reproduce those records. No prompt, description or path is published.

Other Codex versions and missing/ambiguous inventories are unsupported, not
measured zero. User-scope and parent-workspace skills are not attributed.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .base import load_line
from .skill_evidence import SkillEvidence, SkillEvidenceExtraction

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_TESTED_VERSION = "0.163.0"
_ALIAS = r"[A-Za-z0-9_-]{1,128}"
_ROOT = re.compile(r"- `(r[0-9]{1,3})` = `(/[^`\r\n]{1,4096})`\Z")
_ENTRY = re.compile(
    rf"- (?P<alias>{_ALIAS}): [^\r\n]{{1,4096}} \(file: (?P<root>r[0-9]{{1,3}})/(?P=alias)/SKILL\.md\)\Z"
)


def _mac_tmp(path: str) -> str:
    """Compare macOS's canonical /private/tmp to the recorded /tmp cwd."""
    return path.removeprefix("/private") if path.startswith("/private/tmp/") else path


def _time(record: dict, row: Mapping[str, object]) -> datetime | None:
    for value in (record.get("timestamp"), row.get("timestamp")):
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
        else:
            continue
        if parsed.year != 1970:
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


class CodexSkillEvidenceExtractor:
    observed_kinds: frozenset = frozenset({"available"})

    def extract(self, rows: Sequence[Mapping[str, object]]) -> SkillEvidenceExtraction:
        records: list[tuple[int, dict, Mapping[str, object]]] = []
        malformed = 0
        for row in rows:
            if row.get("is_source_record") != 1:
                continue
            offset, raw = row.get("line_offset"), row.get("raw_line")
            if type(offset) is not int or offset < 0 or not isinstance(raw, str):
                malformed += 1
                continue
            record = load_line(raw)
            if record is None:
                malformed += 1
                continue
            records.append((offset, record, row))
        records.sort(key=lambda item: item[0])

        metas = [record.get("payload") for _, record, _ in records if record.get("type") == "session_meta"]
        if len(metas) != 1 or not isinstance(metas[0], dict) or metas[0].get("cli_version") != _TESTED_VERSION:
            return SkillEvidenceExtraction("unsupported", (), malformed)
        cwd = metas[0].get("cwd")
        if not isinstance(cwd, str) or not cwd.startswith("/") or posixpath.normpath(cwd) != cwd:
            return SkillEvidenceExtraction("unsupported", (), malformed)
        project_skills = _mac_tmp(cwd).rstrip("/") + "/.agents/skills"
        candidates: dict[str, list[SkillEvidence]] = defaultdict(list)
        alias_occurrences: dict[str, int] = defaultdict(int)
        inventories = 0
        project_root_seen = False
        ambiguous = False
        for offset, record, row in records:
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "message" or payload.get("role") != "developer":
                continue
            blocks = payload.get("content")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "input_text":
                    continue
                text = block.get("text")
                if not isinstance(text, str):
                    continue
                # Codex prefixes this native developer block with a short
                # internal header before the skills section. Never search
                # arbitrary prompt text for a pasted imitation of the list.
                heading = text.find("## Skills\n")
                if heading not in (0, 22) or (heading and text[heading - 1] != "\n"):
                    continue
                roots_text, separator, entries_text = text[heading:].partition("### Available skills\n")
                if not separator or "### Skill roots\n" not in roots_text:
                    continue
                inventories += 1
                roots: dict[str, list[str]] = defaultdict(list)
                for line in roots_text.splitlines():
                    match = _ROOT.fullmatch(line)
                    if line.startswith("- `r") and not match:
                        ambiguous = True
                    if match:
                        root = match.group(2)
                        if posixpath.normpath(root) != root or root.startswith("//"):
                            ambiguous = True
                            continue
                        roots[match.group(1)].append(root)
                        project_root_seen |= _mac_tmp(root) == project_skills
                ambiguous |= any(len(locations) != 1 for locations in roots.values())
                for line in entries_text.splitlines():
                    match = _ENTRY.fullmatch(line)
                    if line.startswith("- ") and not match:
                        ambiguous = True
                    if not match:
                        continue
                    alias, root_id = match.group("alias", "root")
                    alias_occurrences[alias] += 1
                    if len(roots[root_id]) != 1:
                        ambiguous = True
                        continue
                    if _mac_tmp(roots[root_id][0]) != project_skills:
                        continue
                    location = f"{roots[root_id][0]}/{alias}/SKILL.md"
                    candidates[alias].append(
                        SkillEvidence(
                            "available",
                            "project",
                            alias,
                            offset,
                            f"skill-available:project:{alias}",
                            _time(record, row),
                            location_sha256=hashlib.sha256(location.encode()).hexdigest(),
                        )
                    )
        # A missing project root could be a symlinked workspace or changed
        # inventory layout, not evidence that this project's skill was absent.
        # Duplicated roots/aliases also make absence and scope ambiguous.
        if inventories != 1 or not project_root_seen or ambiguous or any(n != 1 for n in alias_occurrences.values()):
            return SkillEvidenceExtraction("unsupported", (), malformed)
        evidence = tuple(items[0] for _, items in sorted(candidates.items()))
        return SkillEvidenceExtraction("supported", evidence, malformed)
