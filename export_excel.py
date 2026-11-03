"""
Export O*NET and ESCO occupation data to Excel files with position-aligned
Skills and Knowledge rows.

Output:
  output/ONET.xlsx
  output/ESCO.xlsx

Columns:
  ONET: ID, Title, Description, Skills, Knowledge
  ESCO: ID, Preferred Title, Alternate Titles, Description, Skills, Knowledge
"""

import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
ONET_DIR = ROOT / "db_31_0_mssql" / "db_31_0_mssql"
ESCO_DIR = ROOT / "ESCO dataset - v1.2.1 - classification - en - csv"
OUTPUT_DIR = ROOT / "output"

ONET_COLUMNS = ["ID", "Title", "Description", "Skills", "Knowledge"]
ESCO_COLUMNS = [
    "ID",
    "Preferred Title",
    "Alternate Titles",
    "Description",
    "Skills",
    "Knowledge",
]

# ESCO CSV fields can be very large on Windows.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class OccupationRecord:
    occupation_id: str
    description: str
    skills: list[str] = field(default_factory=list)
    knowledge: list[str] = field(default_factory=list)
    title: str = ""
    preferred_title: str = ""
    alternate_titles: str = ""


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def occupation_field_values(record: OccupationRecord, columns: list[str]) -> dict[str, str]:
    values = {
        "ID": record.occupation_id,
        "Title": record.title,
        "Preferred Title": record.preferred_title,
        "Alternate Titles": record.alternate_titles,
        "Description": record.description,
    }
    return {column: values[column] for column in columns if column in values}


def build_aligned_rows(
    record: OccupationRecord,
    columns: list[str],
) -> list[dict[str, str]]:
    """Emit position-aligned rows for one occupation."""
    row_count = max(len(record.skills), len(record.knowledge), 1)
    rows: list[dict[str, str]] = []
    occupation_fields = occupation_field_values(record, columns)

    for index in range(row_count):
        skill = record.skills[index] if index < len(record.skills) else ""
        knowledge = record.knowledge[index] if index < len(record.knowledge) else ""
        row = {column: "" for column in columns}
        if index == 0:
            row.update(occupation_fields)
        row["Skills"] = skill
        row["Knowledge"] = knowledge
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# O*NET
# ---------------------------------------------------------------------------

SQL_VALUES_RE = re.compile(
    r"VALUES \('((?:''|[^'])*)', '((?:''|[^'])*)', '((?:''|[^'])*)'"
)


def unescape_sql(value: str) -> str:
    return value.replace("''", "'")


def load_onet_content_model() -> dict[str, str]:
    """element_id -> element_name"""
    mapping: dict[str, str] = {}
    path = ONET_DIR / "01_content_model_reference.sql"
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


def load_onet_occupations() -> dict[str, tuple[str, str]]:
    """onetsoc_code -> (title, description)"""
    occupations: dict[str, tuple[str, str]] = {}
    path = ONET_DIR / "03_occupation_data.sql"
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


def load_onet_rated_items(
    table_file: str,
    content_model: dict[str, str],
) -> dict[str, list[tuple[str, float]]]:
    """
    onetsoc_code -> [(element_name, importance), ...]
    Keeps only IM scale rows.
    """
    by_code: dict[str, list[tuple[str, float]]] = {}
    path = ONET_DIR / table_file
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


def sort_by_importance(items: list[tuple[str, float]]) -> list[str]:
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return dedupe_preserve_order([label for label, _ in items])


def build_onet_records() -> list[OccupationRecord]:
    content_model = load_onet_content_model()
    occupations = load_onet_occupations()
    essential = load_onet_rated_items("24_essential_skills.sql", content_model)
    transferable = load_onet_rated_items("25_transferable_skills.sql", content_model)
    knowledge = load_onet_rated_items("22_knowledge.sql", content_model)

    records: list[OccupationRecord] = []
    for code in sorted(occupations):
        title, description = occupations[code]
        essential_skills = sort_by_importance(essential.get(code, []))
        transferable_skills = sort_by_importance(transferable.get(code, []))
        skills = dedupe_preserve_order(essential_skills + transferable_skills)
        knowledge_items = sort_by_importance(knowledge.get(code, []))

        records.append(
            OccupationRecord(
                occupation_id=code,
                title=title,
                description=description,
                skills=skills,
                knowledge=knowledge_items,
            )
        )

    return records


# ---------------------------------------------------------------------------
# ESCO
# ---------------------------------------------------------------------------

def normalize_alt_labels(alt_labels: str) -> str:
    if not alt_labels or not alt_labels.strip():
        return ""
    parts = [part.strip() for part in alt_labels.replace("\r\n", "\n").split("\n")]
    parts = [part for part in parts if part]
    return ", ".join(parts)


def load_esco_skill_labels() -> dict[str, str]:
    """conceptUri -> preferredLabel"""
    labels: dict[str, str] = {}
    path = ESCO_DIR / "skills_en.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["conceptUri"]] = row["preferredLabel"]
    return labels


def load_esco_relations(
    skill_labels: dict[str, str],
) -> dict[str, dict[str, list[str]]]:
    """
    occupationUri -> {
      essential_skill: [...],
      optional_skill: [...],
      essential_knowledge: [...],
      optional_knowledge: [...],
    }
    """
    relations: dict[str, dict[str, list[str]]] = {}
    path = ESCO_DIR / "occupationSkillRelations_en.csv"

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


def build_esco_records() -> list[OccupationRecord]:
    skill_labels = load_esco_skill_labels()
    relations = load_esco_relations(skill_labels)
    records: list[OccupationRecord] = []

    path = ESCO_DIR / "occupations_en.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uri = row["conceptUri"]
            rel = relations.get(uri, {})

            skills = dedupe_preserve_order(
                rel.get("essential_skill", []) + rel.get("optional_skill", [])
            )
            knowledge = dedupe_preserve_order(
                rel.get("essential_knowledge", []) + rel.get("optional_knowledge", [])
            )

            records.append(
                OccupationRecord(
                    occupation_id=uri,
                    preferred_title=row["preferredLabel"],
                    alternate_titles=normalize_alt_labels(row.get("altLabels", "")),
                    description=(row.get("description") or "").strip(),
                    skills=skills,
                    knowledge=knowledge,
                )
            )

    records.sort(key=lambda record: record.preferred_title.lower())
    return records


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def records_to_dataframe(
    records: list[OccupationRecord],
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for record in records:
        rows.extend(build_aligned_rows(record, columns))
    return pd.DataFrame(rows, columns=columns)


def export_excel(
    records: list[OccupationRecord],
    output_path: Path,
    columns: list[str],
) -> Path:
    dataframe = records_to_dataframe(records, columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}._tmp{output_path.suffix}")
    dataframe.to_excel(temp_path, index=False, sheet_name="Occupations")
    try:
        os.replace(temp_path, output_path)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_name(
            f"{output_path.stem}_regenerated{output_path.suffix}"
        )
        os.replace(temp_path, fallback_path)
        print(
            f"Warning: could not overwrite locked file {output_path.name}. "
            f"Wrote {fallback_path.name} instead. Close the open Excel file "
            f"and rerun export to replace {output_path.name}."
        )
        return fallback_path


def validate_export(
    path: Path,
    occupation_count: int,
    columns: list[str],
) -> None:
    df = pd.read_excel(path)
    assert list(df.columns) == columns, f"Unexpected columns in {path.name}"

    id_rows = df["ID"].notna() & (df["ID"].astype(str).str.strip() != "")
    assert id_rows.sum() == occupation_count, (
        f"{path.name}: expected {occupation_count} ID rows, got {id_rows.sum()}"
    )

    occupation_only_columns = [column for column in columns if column not in {"Skills", "Knowledge"}]
    block_starts = df.index[id_rows].tolist()
    for start_index, next_start in zip(block_starts, block_starts[1:] + [len(df)]):
        block = df.iloc[start_index:next_start]
        for column in occupation_only_columns:
            blank_values = block.iloc[1:][column].fillna("").astype(str).str.strip()
            assert blank_values.eq("").all(), (
                f"{path.name}: non-blank {column} in occupation block"
            )

    print(
        f"Validated {path.name}: {occupation_count} occupations, "
        f"{len(df)} total rows"
    )


def main() -> None:
    print("Building O*NET export...")
    onet_records = build_onet_records()
    onet_path = OUTPUT_DIR / "ONET.xlsx"
    onet_written = export_excel(onet_records, onet_path, ONET_COLUMNS)
    validate_export(onet_written, len(onet_records), ONET_COLUMNS)

    print("Building ESCO export...")
    esco_records = build_esco_records()
    esco_path = OUTPUT_DIR / "ESCO.xlsx"
    esco_written = export_excel(esco_records, esco_path, ESCO_COLUMNS)
    validate_export(esco_written, len(esco_records), ESCO_COLUMNS)

    # Spot-check samples
    sample_onet = next(r for r in onet_records if r.title == "Chief Executives")
    sample_esco = next(
        r for r in esco_records if r.preferred_title == "technical director"
    )
    print("\nSample O*NET (Chief Executives):")
    print(f"  id={sample_onet.occupation_id}")
    print(f"  skills={len(sample_onet.skills)}, knowledge={len(sample_onet.knowledge)}")
    print(f"  first skill={sample_onet.skills[:1]}, first knowledge={sample_onet.knowledge[:1]}")

    print("\nSample ESCO (technical director):")
    print(f"  id={sample_esco.occupation_id}")
    print(f"  alternate titles={sample_esco.alternate_titles[:80]}...")
    print(f"  skills={len(sample_esco.skills)}, knowledge={len(sample_esco.knowledge)}")
    print(f"  first skill={sample_esco.skills[:1]}, first knowledge={sample_esco.knowledge[:1]}")

    print(f"\nWrote {onet_written}")
    print(f"Wrote {esco_written}")


if __name__ == "__main__":
    main()
