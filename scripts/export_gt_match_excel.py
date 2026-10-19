"""Export O*NET→ESCO predictions vs ground truth to Excel (run 9 by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from backend.app.accuracy import (
    _gold_has_exact,
    _normalize_id,
    _pick_final_target,
    _resolve_gold,
    _sent_to_resolver,
    _target_hits_gold,
)
from backend.app.ground_truth import direction_map
from backend.app.item_index import esco_labels

ROOT = Path(__file__).resolve().parents[1]


def esco_title(tid: str | None, label_index: dict) -> str:
    if not tid:
        return ""
    rows = label_index.get(tid) or label_index.get(str(tid).rstrip("/")) or []
    for row in rows:
        if row.get("is_preferred"):
            return row.get("text") or ""
    if rows:
        return rows[0].get("text") or ""
    uuid = str(tid).rstrip("/").rsplit("/", 1)[-1]
    for key, key_rows in label_index.items():
        if key.rstrip("/").rsplit("/", 1)[-1] == uuid:
            for row in key_rows:
                if row.get("is_preferred"):
                    return row.get("text") or ""
            return (key_rows[0].get("text") if key_rows else "") or ""
    return ""


def yn(val: bool | None) -> str:
    if val is None:
        return ""
    return "Yes" if val else "No"


def match_type_for(tid: str | None, gold: dict[str, str]) -> str:
    if not tid or not gold:
        return ""
    tid_n = _normalize_id(tid)
    uuid = tid_n.rsplit("/", 1)[-1]
    for gid, mt in gold.items():
        g = _normalize_id(gid)
        gu = g.rsplit("/", 1)[-1]
        if tid_n == g or tid_n == gu or uuid == gu or uuid == g:
            return mt
    return "no_match"


def write_sheet(ws, frame: pd.DataFrame) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True)
    yes_fill = PatternFill("solid", fgColor="C6EFCE")
    no_fill = PatternFill("solid", fgColor="FFC7CE")
    thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    match_cols = {
        "Embedding matches GT (exact)?",
        "Jev matches GT (exact)?",
        "Final matches GT (exact)?",
    }
    for r_idx, row in enumerate(dataframe_to_rows(frame, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.border = thin
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if r_idx == 1:
                cell.fill = header_fill
                cell.font = header_font
            else:
                header = frame.columns[c_idx - 1]
                if header in match_cols:
                    if value == "Yes":
                        cell.fill = yes_fill
                    elif value == "No":
                        cell.fill = no_fill
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"
    widths = {
        "O*NET Code": 14,
        "O*NET Title": 36,
        "Sent to Jev": 12,
        "Top1 Similarity": 14,
        "Embedding ESCO Title": 32,
        "Embedding ESCO ID": 42,
        "Embedding matches GT (exact)?": 22,
        "Jev ESCO Title": 32,
        "Jev ESCO ID": 42,
        "Jev matches GT (exact)?": 22,
        "Jev Confidence": 14,
        "Final ESCO Title": 32,
        "Final ESCO ID": 42,
        "Final matches GT (exact)?": 22,
        "Final GT match type": 16,
        "Ground Truth exactMatch Titles": 40,
        "Ground Truth exactMatch IDs": 50,
        "All Ground Truth (title [type])": 55,
        "Metric": 42,
        "Value": 12,
    }
    for i, col in enumerate(frame.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 18)


def build_records(run_dir: Path) -> pd.DataFrame:
    gold_map = direction_map("onet_to_esco")
    label_index = esco_labels()
    rows_raw = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[dict] = []
    for r in rows_raw:
        gold = _resolve_gold(gold_map, r.get("source_id"))
        if not gold or not _gold_has_exact(gold):
            continue

        sent = _sent_to_resolver(r)
        top1_id = r.get("top1_target_id")
        jev_id = r.get("jev_target_id") if sent else None
        final_id = _pick_final_target(r)
        exact_golds = [(gid, mt) for gid, mt in gold.items() if mt == "exactMatch"]

        emb_hit = _target_hits_gold(top1_id, gold, exact_only=True)
        jev_hit = _target_hits_gold(jev_id, gold, exact_only=True) if sent else None
        final_hit = _target_hits_gold(final_id, gold, exact_only=True)

        records.append(
            {
                "O*NET Code": r.get("source_id") or "",
                "O*NET Title": r.get("source_title") or "",
                "Sent to Jev": yn(sent),
                "Top1 Similarity": r.get("top1_similarity"),
                "Embedding ESCO Title": r.get("top1_preferred_title")
                or esco_title(top1_id, label_index),
                "Embedding ESCO ID": top1_id or "",
                "Embedding matches GT (exact)?": yn(emb_hit),
                "Jev ESCO Title": (
                    (r.get("jev_preferred_title") or esco_title(jev_id, label_index))
                    if sent
                    else ""
                ),
                "Jev ESCO ID": jev_id or "",
                "Jev matches GT (exact)?": yn(jev_hit),
                "Jev Confidence": r.get("jev_confidence") if sent else "",
                "Final ESCO Title": r.get("final_preferred_title")
                or esco_title(final_id, label_index),
                "Final ESCO ID": final_id or "",
                "Final matches GT (exact)?": yn(final_hit),
                "Final GT match type": match_type_for(final_id, gold),
                "Ground Truth exactMatch Titles": "; ".join(
                    esco_title(gid, label_index) for gid, _ in exact_golds
                ),
                "Ground Truth exactMatch IDs": "; ".join(gid for gid, _ in exact_golds),
                "All Ground Truth (title [type])": "; ".join(
                    f"{esco_title(gid, label_index)} [{mt}]" for gid, mt in gold.items()
                ),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "output/runs/run_0009_20261015T061000Z",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output xlsx path (default: <run-dir>/onet_esco_ground_truth_match.xlsx)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    out = args.out or (run_dir / "onet_esco_ground_truth_match.xlsx")

    df = build_records(run_dir)
    df_jev = df[df["Sent to Jev"] == "Yes"].copy()
    summary = pd.DataFrame(
        [
            {"Metric": "ExactMatch O*NET rows (denominator)", "Value": len(df)},
            {"Metric": "Sent to Jev", "Value": int((df["Sent to Jev"] == "Yes").sum())},
            {"Metric": "Not sent to Jev", "Value": int((df["Sent to Jev"] == "No").sum())},
            {
                "Metric": "Embedding Top1 correct (all exactMatch)",
                "Value": int((df["Embedding matches GT (exact)?"] == "Yes").sum()),
            },
            {
                "Metric": "Final Acc@1 correct (all exactMatch)",
                "Value": int((df["Final matches GT (exact)?"] == "Yes").sum()),
            },
            {
                "Metric": "Jev subset — embedding correct",
                "Value": int((df_jev["Embedding matches GT (exact)?"] == "Yes").sum()),
            },
            {
                "Metric": "Jev subset — Jev answer correct",
                "Value": int((df_jev["Jev matches GT (exact)?"] == "Yes").sum()),
            },
            {
                "Metric": "Non-Jev — final correct",
                "Value": int(
                    (
                        (df["Sent to Jev"] == "No")
                        & (df["Final matches GT (exact)?"] == "Yes")
                    ).sum()
                ),
            },
        ]
    )

    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Summary"
    write_sheet(ws0, summary)
    ws1 = wb.create_sheet("All_ExactMatch_490")
    write_sheet(ws1, df)
    ws2 = wb.create_sheet("Jev_Sent_180")
    write_sheet(ws2, df_jev)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"Wrote {out}")
    print(f"rows={len(df)} jev={len(df_jev)}")
    print(
        "jev correct:",
        int((df_jev["Jev matches GT (exact)?"] == "Yes").sum()),
    )


if __name__ == "__main__":
    main()
