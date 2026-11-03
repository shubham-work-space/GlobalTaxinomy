"""Materialize ESCO v1.1.0 facet text dirs + label index from ESCO_jobs_v1.1.0.xlsx.

Writes:
  data/esco_1.1.0/<uuid>/{title,description,skills,knowledge}.txt
  data/item_index/esco_labels_1.1.0.json

Usage:
  python scripts/prepare_esco_v110_facets.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "V1.1.0" / "ESCO_jobs_v1.1.0.xlsx"
OUT_FACETS = ROOT / "data" / "esco_1.1.0"
OUT_LABELS = ROOT / "data" / "item_index" / "esco_labels_1.1.0.json"
URI_PREFIX = "http://data.europa.eu/esco/occupation/"

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def _split_lines(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    seen: set[str] = set()
    for line in text.split("\n"):
        item = line.strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _label_id(uuid: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"esco_label:{uuid}:{digest}"


def _prefixed_list(prefix: str, items: list[str]) -> str:
    if not items:
        return f"{prefix}:"
    return f"{prefix}:\n" + "\n".join(items)


def main() -> None:
    if not XLSX.is_file():
        raise SystemExit(f"Missing workbook: {XLSX}")

    wb = load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)
    if not header:
        raise SystemExit("Empty workbook")
    cols = {str(c).strip(): i for i, c in enumerate(header) if c is not None}

    required = [
        "ESCO Job ID",
        "Preferred Title",
        "Alternative Titles",
        "Description",
        "Skills",
        "Knowledge",
    ]
    for name in required:
        if name not in cols:
            raise SystemExit(f"Missing column {name!r} in {XLSX.name}")

    OUT_FACETS.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list[dict[str, object]]] = {}
    n_jobs = 0
    n_labels = 0

    for row in rows:
        if not row:
            continue
        raw_id = row[cols["ESCO Job ID"]]
        if raw_id is None:
            continue
        uuid = str(raw_id).strip().lower()
        if not UUID_RE.match(uuid):
            print(f"skip invalid id: {raw_id!r}", file=sys.stderr)
            continue

        preferred = str(row[cols["Preferred Title"]] or "").strip()
        description = str(row[cols["Description"]] or "").strip()
        alt_titles = _split_lines(row[cols["Alternative Titles"]])
        skills = _split_lines(row[cols["Skills"]])
        knowledge = _split_lines(row[cols["Knowledge"]])

        # Preferred title should lead the label list; keep alts that differ.
        label_texts: list[tuple[str, bool]] = []
        if preferred:
            label_texts.append((preferred, True))
        for alt in alt_titles:
            if preferred and alt.casefold() == preferred.casefold():
                continue
            label_texts.append((alt, False))

        occ_dir = OUT_FACETS / uuid
        occ_dir.mkdir(parents=True, exist_ok=True)
        (occ_dir / "title.txt").write_text(preferred, encoding="utf-8")
        (occ_dir / "description.txt").write_text(description, encoding="utf-8")
        (occ_dir / "skills.txt").write_text(_prefixed_list("Skills", skills), encoding="utf-8")
        (occ_dir / "knowledge.txt").write_text(
            _prefixed_list("Knowledge", knowledge), encoding="utf-8"
        )

        uri = f"{URI_PREFIX}{uuid}"
        label_rows: list[dict[str, object]] = []
        for text, is_pref in label_texts:
            label_rows.append(
                {
                    "label_id": _label_id(uuid, text),
                    "text": text,
                    "is_preferred": is_pref,
                }
            )
            n_labels += 1
        labels[uri] = label_rows
        n_jobs += 1

    wb.close()
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {n_jobs} occupations under {OUT_FACETS}")
    print(f"Wrote {n_labels} labels to {OUT_LABELS}")


if __name__ == "__main__":
    main()
