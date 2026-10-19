"""Build an Evals run from historical Ejento I/O pairs (Top-1 only).

Uses run_0009 embedding results for the ≥threshold path, and replaces the
below-threshold resolver with Ejento's single Match letter → ESCO title.
Accuracy uses the same exactMatch-any ground-truth rule as Jev Acc@1.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd

from backend.app.accuracy import (
    annotate_results_with_ground_truth,
    compute_accuracy_metrics,
    _gold_has_exact,
    _resolve_gold,
    _sent_to_resolver,
    _target_hits_gold,
)
from backend.app.ground_truth import direction_map
from backend.app.item_index import esco_labels
from backend.app.runs import create_run, metrics_summary_from_metrics, save_run_artifacts, update_run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "output" / "ejento_io_pairs_run_3"
DEFAULT_BASE_RUN = ROOT / "output" / "runs" / "run_0009_20261015T061000Z"


def _norm_title(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _preferred_title_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for oid, rows in esco_labels().items():
        for row in rows:
            if row.get("is_preferred") and row.get("text"):
                index[row["text"].strip().lower()].append(oid)
    return index


def esco_title(tid: str | None, label_index: dict[str, list[dict[str, Any]]]) -> str:
    if not tid:
        return ""
    rows = label_index.get(tid) or label_index.get(str(tid).rstrip("/")) or []
    for row in rows:
        if row.get("is_preferred"):
            return row.get("text") or ""
    return (rows[0].get("text") if rows else "") or ""


def parse_ejento_input(text: str) -> tuple[str | None, dict[str, str]]:
    m = re.search(r"^Input Occupation\s*\n+Title:\s*\n(.+)", text, re.M)
    onet_title = m.group(1).strip() if m else None
    cands: dict[str, str] = {}
    for letter in "ABCD":
        mm = re.search(rf"^Job {letter}\n\nPreferred Title:\n(.+)\n", text, re.M)
        if mm:
            cands[letter] = mm.group(1).strip()
    return onet_title, cands


def parse_ejento_output(text: str) -> tuple[str | None, str | None]:
    m = re.search(r"Match:\s*\**([A-D])\**", text, re.I)
    letter = m.group(1).upper() if m else None
    m2 = re.search(r"Job Name:\s*\**(.+?)\**\s*(?:\n|$)", text)
    name = m2.group(1).strip() if m2 else None
    return letter, name


def load_ejento_picks(
    pairs_dir: Path,
    *,
    by_title: dict[str, dict[str, Any]],
    pref_index: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    picks: dict[str, dict[str, Any]] = {}
    for folder in sorted(p for p in pairs_dir.iterdir() if p.is_dir()):
        inp = (folder / "Input.txt").read_text(encoding="utf-8", errors="replace")
        out = (folder / "Output.txt").read_text(encoding="utf-8", errors="replace")
        onet_title, cands = parse_ejento_input(inp)
        letter, job_name = parse_ejento_output(out)
        row = by_title.get(_norm_title(onet_title))
        if not row:
            raise LookupError(f"No base-run row for O*NET title {onet_title!r} ({folder.name})")
        pref_title = cands.get(letter or "") if letter else None
        ids = pref_index.get((pref_title or "").lower(), [])
        if not ids and job_name:
            ids = pref_index.get(job_name.lower(), [])
        if not ids:
            raise LookupError(
                f"Could not resolve ESCO for {folder.name}: letter={letter} title={pref_title!r}"
            )
        tid = ids[0]
        picks[row["source_id"]] = {
            "folder": folder.name,
            "onet_title": onet_title,
            "match_letter": letter,
            "preferred_title": pref_title or job_name or esco_title(tid, esco_labels()),
            "target_id": tid,
            "candidates_abcd": cands,
        }
    return picks


def build_results(
    base_rows: list[dict[str, Any]],
    picks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in base_rows:
        r = deepcopy(row)
        sid = r.get("source_id")
        pick = picks.get(sid) if sid else None
        if pick and _sent_to_resolver(r):
            tid = pick["target_id"]
            title = pick["preferred_title"]
            r["sent_to_llm"] = True
            r["sent_to_jev"] = True  # reuse Acc@K path; resolver noted in metrics
            r["llm_target_id"] = tid
            r["llm_preferred_title"] = title
            r["llm_match_label"] = pick["match_letter"]
            r["llm_primary_label"] = pick["match_letter"]
            r["jev_target_id"] = tid
            r["jev_preferred_title"] = title
            r["jev_match_label"] = pick["match_letter"]
            r["jev_ranked_target_ids"] = [tid]  # Top-1 only from Ejento
            r["jev_ranked_labels"] = [pick["match_letter"]] if pick["match_letter"] else []
            r["final_target_id"] = tid
            r["final_preferred_title"] = title
            r["resolver"] = "ejento"
            r["ejento_folder"] = pick["folder"]
            r["ejento_match_letter"] = pick["match_letter"]
        else:
            # High-similarity path: keep embedding final
            r["resolver"] = "embedding"
        out.append(r)
    return annotate_results_with_ground_truth(out, direction="onet_to_esco")


def export_excel(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    out_path: Path,
) -> None:
    label_index = esco_labels()
    gold_map = direction_map("onet_to_esco")
    records: list[dict[str, Any]] = []
    for r in results:
        gold = _resolve_gold(gold_map, r.get("source_id"))
        if not gold or not _gold_has_exact(gold):
            continue
        exact_golds = [gid for gid, mt in gold.items() if mt == "exactMatch"]
        sent = r.get("resolver") == "ejento"
        emb_id = r.get("top1_target_id")
        llm_id = r.get("llm_target_id") if sent else None
        final_id = r.get("final_target_id")
        records.append(
            {
                "O*NET Code": r.get("source_id") or "",
                "O*NET Title": r.get("source_title") or "",
                "Sent to Ejento": "Yes" if sent else "No",
                "Top1 Similarity": r.get("top1_similarity"),
                "Embedding ESCO Title": r.get("top1_preferred_title")
                or esco_title(emb_id, label_index),
                "Embedding matches GT (exact)?": "Yes"
                if _target_hits_gold(emb_id, gold, exact_only=True)
                else "No",
                "Ejento Match Letter": r.get("ejento_match_letter") or "",
                "Ejento ESCO Title": r.get("llm_preferred_title") or "",
                "Ejento ESCO ID": llm_id or "",
                "Ejento matches GT (exact)?": (
                    "Yes"
                    if sent and _target_hits_gold(llm_id, gold, exact_only=True)
                    else ("No" if sent else "")
                ),
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
    df_ej = df[df["Sent to Ejento"] == "Yes"].copy()
    summary = pd.DataFrame(
        [
            {"Metric": k, "Value": v}
            for k, v in [
                ("ExactMatch O*NET rows", metrics.get("total_records")),
                ("Sent to Ejento", metrics.get("records_sent_to_llm")),
                ("Embedding Top1 Acc", metrics.get("embedding_top1_accuracy")),
                ("Ejento subset Acc@1", metrics.get("llm_subset_accuracy_after")),
                ("Hybrid Acc@1 (Ejento)", metrics.get("accuracy_at_1")),
            ]
        ]
    )

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
        "Ejento matches GT (exact)?",
        "Final matches GT (exact)?",
    }

    def write_sheet(ws, frame: pd.DataFrame) -> None:
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

    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Summary"
    write_sheet(ws0, summary)
    ws1 = wb.create_sheet("All_ExactMatch_490")
    write_sheet(ws1, df)
    ws2 = wb.create_sheet("Ejento_Sent_180")
    write_sheet(ws2, df_ej)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN)
    args = parser.parse_args()

    base_cfg = json.loads((args.base_run_dir / "config.json").read_text(encoding="utf-8"))
    base_rows = [
        json.loads(line)
        for line in (args.base_run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_title = {
        _norm_title(r.get("source_title")): r
        for r in base_rows
        if r.get("source_title")
    }
    picks = load_ejento_picks(args.pairs_dir, by_title=by_title, pref_index=_preferred_title_index())
    if len(picks) != 180:
        raise RuntimeError(f"Expected 180 Ejento picks, got {len(picks)}")

    results = build_results(base_rows, picks)
    metrics = compute_accuracy_metrics(results, direction="onet_to_esco")
    # Acc@K>1 not meaningful for Ejento single-choice rows — keep Acc@1 only in summary UX
    metrics["resolver"] = "ejento"
    metrics["accuracy_at_2"] = None
    metrics["accuracy_at_3"] = None
    metrics["accuracy_at_5"] = None
    metrics["accuracy_at_10"] = None
    metrics["source_pairs_dir"] = str(args.pairs_dir.as_posix())
    metrics["base_run_id"] = base_cfg.get("run_id")
    metrics["notes"] = (
        "Hybrid Acc@1: embedding Top-1 when similarity≥threshold; "
        "Ejento single Match (A–D) when below threshold. "
        "Hit = predicted ESCO ∈ any exactMatch ground-truth for that O*NET."
    )

    cfg = {
        "model": base_cfg["model"],
        "input_taxonomy": base_cfg["input_taxonomy"],
        "output_taxonomy": base_cfg["output_taxonomy"],
        "weights": base_cfg["weights"],
        "threshold": base_cfg["threshold"],
        "top_k": base_cfg.get("top_k", 20),
        "resolver": "ejento",
        "force": True,
    }
    entry = create_run(cfg)
    run_id = entry["run_id"]
    save_run_artifacts(run_id, results=results, metrics=metrics)
    # Enrich summary for the Evals table
    update_run(
        run_id,
        label=f"{entry['label']} (Ejento)",
        message="Imported Ejento Top-1 I/O · Acc@1 vs exactMatch GT",
        metrics_summary={
            **metrics_summary_from_metrics(metrics),
            "accuracy_at_2": None,
            "accuracy_at_3": None,
            "accuracy_at_5": None,
            "accuracy_at_10": None,
            "resolver": "ejento",
        },
    )

    xlsx = Path(f"output/runs/{run_id}/onet_esco_ejento_ground_truth_match.xlsx")
    export_excel(results, metrics, ROOT / xlsx if not xlsx.is_absolute() else xlsx)

    print(json.dumps(
        {
            "run_id": run_id,
            "label": f"{entry['label']} (Ejento)",
            "accuracy_at_1": metrics.get("accuracy_at_1"),
            "embedding_top1_accuracy": metrics.get("embedding_top1_accuracy"),
            "llm_subset_accuracy_after": metrics.get("llm_subset_accuracy_after"),
            "records_sent_to_llm": metrics.get("records_sent_to_llm"),
            "exact_hits_after": metrics.get("exact_hits_after"),
            "excel": str(xlsx),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
