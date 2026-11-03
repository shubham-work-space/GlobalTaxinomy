"""Broader O*NET / ESCO occupation context for gated JEV calls (jev_context_v2).

Loads from raw O*NET 31.0 SQL dumps and ESCO v1.2.1 CSVs (see export_excel.py).
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import ROOT, get_settings

JEV_CONTEXT_VERSION = "jev_context_v2"

CAP_TASKS = 15
CAP_ABILITIES = 15
CAP_WORK_ACTIVITIES = 15
CAP_SOFTWARE_SKILLS = 20
CAP_LIST_ITEMS = 25
CAP_TITLES = 30

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _cap(items: list[str], limit: int) -> list[str]:
    return items[: max(0, limit)]


def _join(items: list[str]) -> str:
    return "; ".join(_cap(_dedupe(items), CAP_LIST_ITEMS))


def unescape_sql(value: str) -> str:
    return value.replace("''", "'")


def resolve_onet_sql_dir() -> Path | None:
    raw = (os.environ.get("ONET_SQL_DIR") or "").strip()
    if not raw:
        try:
            raw = (get_settings().onet_sql_dir or "").strip()
        except Exception:
            raw = ""
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.extend([p, p / "db_31_0_mssql"])
    candidates.extend(
        [
            ROOT / "db_31_0_mssql" / "db_31_0_mssql",
            Path.home() / "Downloads" / "db_31_0_mssql" / "db_31_0_mssql",
        ]
    )
    for path in candidates:
        if (path / "03_occupation_data.sql").is_file():
            return path
    return None


def resolve_esco_csv_dir() -> Path | None:
    raw = (os.environ.get("ESCO_CSV_DIR") or "").strip()
    if not raw:
        try:
            raw = (get_settings().esco_csv_dir or "").strip()
        except Exception:
            raw = ""
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
    candidates.extend(
        [
            ROOT / "ESCO dataset - v1.2.1 - classification - en - csv",
            Path.home()
            / "Downloads"
            / "ESCO dataset - v1.2.1 - classification - en - csv",
        ]
    )
    for path in candidates:
        if (path / "occupations_en.csv").is_file():
            return path
    return None


def _load_onet_content_model(onet_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    path = onet_dir / "01_content_model_reference.sql"
    if not path.is_file():
        return mapping
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("INSERT INTO content_model_reference"):
                continue
            match = re.search(
                r"VALUES \('([^']+)', '((?:''|[^'])*)', '((?:''|[^'])*)'\);",
                line,
            )
            if match:
                mapping[match.group(1)] = unescape_sql(match.group(2))
    return mapping


def _load_onet_occupations(onet_dir: Path) -> dict[str, tuple[str, str]]:
    occupations: dict[str, tuple[str, str]] = {}
    path = onet_dir / "03_occupation_data.sql"
    if not path.is_file():
        return occupations
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("INSERT INTO occupation_data"):
                continue
            match = re.search(
                r"VALUES \('([^']+)', '((?:''|[^'])*)', '((?:''|[^'])*)'\);",
                line,
            )
            if match:
                code = match.group(1)
                occupations[code] = (
                    unescape_sql(match.group(2)),
                    unescape_sql(match.group(3)),
                )
    return occupations


def _load_onet_rated_items(
    onet_dir: Path,
    table_file: str,
    content_model: dict[str, str],
) -> dict[str, list[tuple[str, float]]]:
    by_code: dict[str, list[tuple[str, float]]] = {}
    path = onet_dir / table_file
    if not path.is_file():
        return by_code
    pattern = re.compile(
        r"VALUES \('([^']+)', '([^']+)', '([^']+)', ([0-9.]+),"
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("INSERT INTO"):
                continue
            match = pattern.search(line)
            if not match:
                continue
            code, element_id, scale_id, data_value = (
                match.group(1),
                match.group(2),
                match.group(3),
                float(match.group(4)),
            )
            if scale_id != "IM":
                continue
            label = content_model.get(element_id)
            if not label:
                continue
            by_code.setdefault(code, []).append((label, data_value))
    return by_code


def _sort_labels(items: list[tuple[str, float]]) -> list[str]:
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return _dedupe([label for label, _ in items])


def _load_onet_tasks(onet_dir: Path) -> dict[str, list[str]]:
    by_code: dict[str, list[str]] = {}
    path = onet_dir / "26_task_statements.sql"
    if not path.is_file():
        return by_code
    pattern = re.compile(
        r"VALUES \('([^']+)', [^,]+, '((?:''|[^'])*)',"
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("INSERT INTO task_statements"):
                continue
            match = pattern.search(line)
            if not match:
                continue
            code, task = match.group(1), unescape_sql(match.group(2))
            by_code.setdefault(code, []).append(task)
    return by_code


def _load_onet_software_skills(onet_dir: Path) -> dict[str, list[str]]:
    by_code: dict[str, list[str]] = {}
    path = onet_dir / "23_software_skills.sql"
    if not path.is_file():
        return by_code
    pattern = re.compile(
        r"VALUES \('([^']+)', '((?:''|[^'])*)',"
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("INSERT INTO software_skills"):
                continue
            match = pattern.search(line)
            if not match:
                continue
            code, example = match.group(1), unescape_sql(match.group(2))
            by_code.setdefault(code, []).append(example)
    return by_code


def _load_onet_reported_titles(onet_dir: Path) -> dict[str, list[str]]:
    by_code: dict[str, list[str]] = {}
    for fname in ("37_sample_of_reported_titles.sql", "36_job_titles.sql"):
        path = onet_dir / fname
        if not path.is_file():
            continue
        pattern = re.compile(
            r"VALUES \('([^']+)', '((?:''|[^'])*)',"
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("INSERT INTO"):
                    continue
                match = pattern.search(line)
                if not match:
                    continue
                code, title = match.group(1), unescape_sql(match.group(2))
                by_code.setdefault(code, []).append(title)
    return by_code


def normalize_esco_alt_labels(alt_labels: str) -> str:
    if not alt_labels or not alt_labels.strip():
        return ""
    parts = [part.strip() for part in alt_labels.replace("\r\n", "\n").split("\n")]
    parts = [part for part in parts if part]
    return ", ".join(_cap(parts, CAP_TITLES))


@dataclass
class _OnetStore:
    occupations: dict[str, tuple[str, str]] = field(default_factory=dict)
    essential_skills: dict[str, list[str]] = field(default_factory=dict)
    transferable_skills: dict[str, list[str]] = field(default_factory=dict)
    knowledge: dict[str, list[str]] = field(default_factory=dict)
    abilities: dict[str, list[str]] = field(default_factory=dict)
    work_activities: dict[str, list[str]] = field(default_factory=dict)
    tasks: dict[str, list[str]] = field(default_factory=dict)
    software_skills: dict[str, list[str]] = field(default_factory=dict)
    titles: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _EscoStore:
    occupations: dict[str, dict[str, str]] = field(default_factory=dict)
    relations: dict[str, dict[str, list[str]]] = field(default_factory=dict)


_onet_store: _OnetStore | None = None
_esco_store: _EscoStore | None = None


def _get_onet_store() -> _OnetStore | None:
    global _onet_store
    if _onet_store is not None:
        return _onet_store
    onet_dir = resolve_onet_sql_dir()
    if not onet_dir:
        return None
    cm = _load_onet_content_model(onet_dir)
    store = _OnetStore(
        occupations=_load_onet_occupations(onet_dir),
        essential_skills={
            k: _sort_labels(v)
            for k, v in _load_onet_rated_items(
                onet_dir, "24_essential_skills.sql", cm
            ).items()
        },
        transferable_skills={
            k: _sort_labels(v)
            for k, v in _load_onet_rated_items(
                onet_dir, "25_transferable_skills.sql", cm
            ).items()
        },
        knowledge={
            k: _sort_labels(v)
            for k, v in _load_onet_rated_items(onet_dir, "22_knowledge.sql", cm).items()
        },
        abilities={
            k: _sort_labels(v)
            for k, v in _load_onet_rated_items(onet_dir, "12_abilities.sql", cm).items()
        },
        work_activities={
            k: _sort_labels(v)
            for k, v in _load_onet_rated_items(onet_dir, "28_work_activities.sql", cm).items()
        },
        tasks=_load_onet_tasks(onet_dir),
        software_skills=_load_onet_software_skills(onet_dir),
        titles=_load_onet_reported_titles(onet_dir),
    )
    _onet_store = store
    return store


def _load_esco_skill_labels(esco_dir: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    path = esco_dir / "skills_en.csv"
    if not path.is_file():
        return labels
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["conceptUri"]] = row["preferredLabel"]
    return labels


def _load_esco_relations(
    esco_dir: Path,
    skill_labels: dict[str, str],
) -> dict[str, dict[str, list[str]]]:
    relations: dict[str, dict[str, list[str]]] = {}
    path = esco_dir / "occupationSkillRelations_en.csv"
    if not path.is_file():
        return relations
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            occupation_uri = row["occupationUri"]
            relation_type = row["relationType"]
            skill_type = row["skillType"]
            label = skill_labels.get(row["skillUri"])
            if not label:
                continue
            if skill_type not in {"skill/competence", "knowledge"}:
                continue
            if relation_type not in {"essential", "optional"}:
                continue
            if skill_type == "skill/competence":
                bucket = f"{relation_type}_skill"
            else:
                bucket = f"{relation_type}_knowledge"
            relations.setdefault(occupation_uri, {}).setdefault(bucket, []).append(label)
    return relations


def _get_esco_store() -> _EscoStore | None:
    global _esco_store
    if _esco_store is not None:
        return _esco_store
    esco_dir = resolve_esco_csv_dir()
    if not esco_dir:
        return None
    skill_labels = _load_esco_skill_labels(esco_dir)
    relations = _load_esco_relations(esco_dir, skill_labels)
    occupations: dict[str, dict[str, str]] = {}
    path = esco_dir / "occupations_en.csv"
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                uri = row["conceptUri"]
                occupations[uri] = {
                    "preferred_title": row.get("preferredLabel") or "",
                    "alternate_titles": normalize_esco_alt_labels(row.get("altLabels") or ""),
                    "description": (row.get("description") or "").strip(),
                }
    store = _EscoStore(occupations=occupations, relations=relations)
    _esco_store = store
    return store


def get_onet_jev_context(onetsoc_code: str) -> dict[str, Any]:
    """Return jev_context_v2 fields for an O*NET occupation."""
    store = _get_onet_store()
    code = str(onetsoc_code).strip()
    if not store or code not in store.occupations:
        return {}
    title, description = store.occupations[code]
    return {
        "occupation_id": code,
        "title": title,
        "description": description,
        "titles": _join(store.titles.get(code, [])),
        "tasks": _join(_cap(store.tasks.get(code, []), CAP_TASKS)),
        "abilities": _join(_cap(store.abilities.get(code, []), CAP_ABILITIES)),
        "work_activities": _join(
            _cap(store.work_activities.get(code, []), CAP_WORK_ACTIVITIES)
        ),
        "software_skills": _join(
            _cap(store.software_skills.get(code, []), CAP_SOFTWARE_SKILLS)
        ),
        "essential_skills": _join(store.essential_skills.get(code, [])),
        "transferable_skills": _join(store.transferable_skills.get(code, [])),
        "knowledge": _join(store.knowledge.get(code, [])),
    }


def get_esco_jev_context(occupation_uri: str) -> dict[str, Any]:
    """Return jev_context_v2 fields for an ESCO occupation."""
    store = _get_esco_store()
    uri = str(occupation_uri).strip().rstrip("/")
    if not store:
        return {}
    occ = store.occupations.get(uri) or store.occupations.get(f"{uri}/")
    if not occ:
        # try trailing uuid match
        for key, val in store.occupations.items():
            if key.rstrip("/") == uri or key.endswith(uri.rsplit("/", 1)[-1]):
                occ = val
                uri = key
                break
    if not occ:
        return {}
    rel = store.relations.get(uri) or store.relations.get(uri.rstrip("/")) or {}
    return {
        "occupation_id": uri,
        "preferred_title": occ.get("preferred_title") or "",
        "alternate_titles": occ.get("alternate_titles") or "",
        "description": occ.get("description") or "",
        "essential_skills": _join(rel.get("essential_skill", [])),
        "optional_skills": _join(rel.get("optional_skill", [])),
        "essential_knowledge": _join(rel.get("essential_knowledge", [])),
        "optional_knowledge": _join(rel.get("optional_knowledge", [])),
    }


def _merge_input(existing: dict[str, Any], enriched: dict[str, Any]) -> dict[str, Any]:
    out = dict(existing or {})
    if enriched:
        out.update({k: v for k, v in enriched.items() if v})
    return out


def _merge_candidate(existing: dict[str, Any], enriched: dict[str, Any]) -> dict[str, Any]:
    out = dict(existing or {})
    if not enriched:
        return out
    mapping = {
        "preferred_title": enriched.get("preferred_title"),
        "alternate_title": enriched.get("alternate_titles") or enriched.get("alternate_title"),
        "description": enriched.get("description"),
        "essential_skills": enriched.get("essential_skills"),
        "optional_skills": enriched.get("optional_skills"),
        "essential_knowledge": enriched.get("essential_knowledge"),
        "optional_knowledge": enriched.get("optional_knowledge"),
        "occupation_id": enriched.get("occupation_id"),
    }
    for key, val in mapping.items():
        if val:
            out[key] = val
    return out


def enrich_gated_record(
    record: dict[str, Any],
    *,
    input_taxonomy: str = "onet",
    output_taxonomy: str = "esco",
) -> dict[str, Any]:
    """Attach broader context to a gated JEV record (source + candidates)."""
    row = dict(record)
    sid = str(row.get("source_id") or "")
    if input_taxonomy == "onet" and sid:
        onet_ctx = get_onet_jev_context(sid)
        row["input"] = _merge_input(row.get("input") or {}, onet_ctx)
        if onet_ctx.get("title") and not row.get("source_title"):
            row["source_title"] = onet_ctx["title"]
    enriched_cands: list[dict[str, Any]] = []
    for cand in row.get("candidates") or []:
        tid = cand.get("target_id") or cand.get("occupation_id") or ""
        esco_ctx = {}
        if output_taxonomy == "esco" and tid:
            esco_ctx = get_esco_jev_context(str(tid))
        enriched_cands.append(_merge_candidate(cand, esco_ctx))
    row["candidates"] = enriched_cands
    row["jev_context_version"] = JEV_CONTEXT_VERSION
    return row


@lru_cache(maxsize=1)
def context_sources_available() -> bool:
    return resolve_onet_sql_dir() is not None and resolve_esco_csv_dir() is not None


def reset_context_cache() -> None:
    """Clear in-memory stores (tests)."""
    global _onet_store, _esco_store
    _onet_store = None
    _esco_store = None
    context_sources_available.cache_clear()
