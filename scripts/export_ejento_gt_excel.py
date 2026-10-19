"""Export Ejento subset (180) ground-truth match Excel for I/O pair runs."""

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
    compute_accuracy_metrics,
    _gold_has_exact,
    _resolve_gold,
    _target_hits_gold,
)
from backend.app.ground_truth import direction_map
from backend.app.item_index import esco_labels
from scripts.import_ejento_eval import (
    ROOT,
    _norm_title,
    _preferred_title_index,
    build_results,
    esco_title,
    load_ejento_picks,
)

DEFAULT_BASE = ROOT / "output/runs/run_0009_20261015T061000Z"


def write_sheet(ws, frame: pd.DataFrame, match_cols: set[str]) -> None:
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
    for r_idx, row in enumerate(dataframe_to_rows(frame, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.border = thin
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if r_idx == 1:
                cell.fill = header_fill
                cell.font = header_font
            elif frame.columns[c_idx - 1] in match_cols:
                if value == "Yes":
                    cell.fill = yes_fill
                elif value == "No":
                    cell.fill = no_fill
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"
    for i, col in enumerate(frame.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = min(42, max(14, len(col) + 2))


def export_pairs(pairs_dir: Path, base_run_dir: Path, out: Path | None = None) -> Path:
    base_rows = [
        json.loads(line)
        for line in (base_run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_title = {
        _norm_title(r.get("source_title")): r for r in base_rows if r.get("source_title")
    }
    picks = load_ejento_picks(
        pairs_dir, by_title=by_title, pref_index=_preferred_title_index()
    )
    if len(picks) != 180:
        raise RuntimeError(f"{pairs_dir.name}: expected 180 picks, got {len(picks)}")

    results = build_results(base_rows, picks)
    metrics = compute_accuracy_metrics(results, direction="onet_to_esco")
    label_index = esco_labels()
    gold_map = direction_map("onet_to_esco")

    records: list[dict] = []
    for r in results:
        if r.get("resolver") != "ejento":
            continue
        gold = _resolve_gold(gold_map, r.get("source_id"))
        if not gold or not _gold_has_exact(gold):
            continue
        exact_golds = [gid for gid, mt in gold.items() if mt == "exactMatch"]
        emb_id = r.get("top1_target_id")
        llm_id = r.get("llm_target_id")
        final_id = r.get("final_target_id")
        records.append(
            {
                "O*NET Code": r.get("source_id") or "",
                "O*NET Title": r.get("source_title") or "",
                "Top1 Similarity": r.get("top1_similarity"),
                "Embedding ESCO Title": r.get("top1_preferred_title")
                or esco_title(emb_id, label_index),
                "Embedding matches GT (exact)?": "Yes"
                if _target_hits_gold(emb_id, gold, exact_only=True)
                else "No",
                "Ejento Match Letter": r.get("ejento_match_letter") or "",
                "Ejento ESCO Title": r.get("llm_preferred_title") or "",
                "Ejento ESCO ID": llm_id or "",
                "Ejento matches GT (exact)?": "Yes"
                if _target_hits_gold(llm_id, gold, exact_only=True)
                else "No",
                "Final ESCO Title": r.get("final_preferred_title")
                or esco_title(final_id, label_index),
                "Final matches GT (exact)?": "Yes"
                if _target_hits_gold(final_id, gold, exact_only=True)
                else "No",
                "Ground Truth exactMatch Titles": "; ".join(
                    esco_title(gid, label_index) for gid in exact_golds
                ),
                "Ground Truth exactMatch IDs": "; ".join(exact_golds),
                "All Ground Truth (title [type])": "; ".join(
                    f"{esco_title(gid, label_index)} [{mt}]" for gid, mt in gold.items()
                ),
            }
        )

    df = pd.DataFrame(records)
    ejento_correct = int((df["Ejento matches GT (exact)?"] == "Yes").sum())
    emb_correct = int((df["Embedding matches GT (exact)?"] == "Yes").sum())
    summary = pd.DataFrame(
        [
            {"Metric": "Ejento subset rows (exactMatch)", "Value": len(df)},
            {"Metric": "Embedding Top1 correct (subset)", "Value": emb_correct},
            {"Metric": "Ejento Acc@1 correct (subset)", "Value": ejento_correct},
            {
                "Metric": "Ejento subset Acc@1",
                "Value": metrics.get("llm_subset_accuracy_after"),
            },
            {
                "Metric": "Hybrid Acc@1 (full 490)",
                "Value": metrics.get("accuracy_at_1"),
            },
            {"Metric": "Source pairs dir", "Value": pairs_dir.name},
        ]
    )

    out_path = out or (pairs_dir / "onet_esco_ejento_ground_truth_match.xlsx")
    match_cols = {
        "Embedding matches GT (exact)?",
        "Ejento matches GT (exact)?",
        "Final matches GT (exact)?",
    }
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Summary"
    write_sheet(ws0, summary, match_cols)
    ws1 = wb.create_sheet("Ejento_Sent_180")
    write_sheet(ws1, df, match_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(
        f"Wrote {out_path} rows={len(df)} "
        f"ejento_correct={ejento_correct} emb_correct={emb_correct}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        nargs="+",
        default=[
            ROOT / "output/ejento_io_pairs_run_1",
            ROOT / "output/ejento_io_pairs_run_2",
            ROOT / "output/ejento_io_pairs_run_3",
        ],
    )
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    for pairs_dir in args.pairs_dir:
        export_pairs(pairs_dir, args.base_run_dir)


if __name__ == "__main__":
    main()
