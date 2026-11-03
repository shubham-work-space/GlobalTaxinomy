"""Export ESCO v1.1.0 stage Excels + Acc@K summary from a completed run.

Writes into the run directory:
  embedding_accuracy_1.1.0.xlsx
  jev_candidates_1.1.0.xlsx
  onet_esco_ground_truth_match_1.1.0.xlsx
  metrics_summary_1.1.0.xlsx
  analysis_report_1.1.0.md

Usage:
  python scripts/export_esco_v110_stage_excels.py --run-dir output/runs/run_XXXX_...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Prefer v1.1.0 labels for title resolution
os.environ.setdefault(
    "ESCO_LABELS_JSON",
    str(ROOT / "data" / "item_index" / "esco_labels_1.1.0.json"),
)

from backend.app.accuracy import (  # noqa: E402
    ACCURACY_KS,
    _embedding_ranked_ids,
    _gold_has_exact,
    _hit_at_k,
    _pick_final_target,
    _resolve_gold,
    _sent_to_resolver,
    _target_hits_gold,
    final_ranked_target_ids,
)
from backend.app.ground_truth import direction_map  # noqa: E402
from backend.app.item_index import clear_item_index_cache, esco_labels  # noqa: E402

# Load sibling export helpers without requiring scripts/ as a package
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "export_gt_match_excel",
    ROOT / "scripts" / "export_gt_match_excel.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)
build_records = _mod.build_records
esco_title = _mod.esco_title
match_type_for = _mod.match_type_for
write_sheet = _mod.write_sheet
yn = _mod.yn


def _load_rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _eligible(rows: list[dict]) -> list[dict]:
    gold_map = direction_map("onet_to_esco")
    out = []
    for r in rows:
        gold = _resolve_gold(gold_map, r.get("source_id"))
        if gold and _gold_has_exact(gold):
            out.append({**r, "_gold": gold})
    return out


def embedding_accuracy_frame(eligible: list[dict], label_index: dict) -> pd.DataFrame:
    gold_map = direction_map("onet_to_esco")
    records = []
    for r in eligible:
        gold = r["_gold"]
        ranked = _embedding_ranked_ids(r, limit=10)
        hits = {k: _hit_at_k(ranked, gold, k) for k in ACCURACY_KS}
        top1 = ranked[0] if ranked else None
        records.append(
            {
                "O*NET Code": r.get("source_id") or "",
                "O*NET Title": r.get("source_title") or "",
                "Top1 Similarity": r.get("top1_similarity"),
                "Embedding Top-1 Title": r.get("top1_preferred_title")
                or esco_title(top1, label_index),
                "Embedding Top-1 ID": top1 or "",
                "Emb Acc@1": yn(hits[1]),
                "Emb Acc@2": yn(hits[2]),
                "Emb Acc@3": yn(hits[3]),
                "Emb Acc@5": yn(hits[5]),
                "Emb Acc@10": yn(hits[10]),
                "Ground Truth exactMatch Titles": "; ".join(
                    esco_title(gid, label_index)
                    for gid, mt in gold.items()
                    if mt == "exactMatch"
                ),
                "Ground Truth exactMatch IDs": "; ".join(
                    gid for gid, mt in gold.items() if mt == "exactMatch"
                ),
            }
        )
    _ = gold_map
    return pd.DataFrame(records)


def jev_candidates_frame(eligible: list[dict], label_index: dict) -> pd.DataFrame:
    records = []
    for r in eligible:
        if not _sent_to_resolver(r):
            continue
        gold = r["_gold"]
        for cand in r.get("candidates") or []:
            tid = cand.get("target_id") or cand.get("occupation_id")
            records.append(
                {
                    "O*NET Code": r.get("source_id") or "",
                    "O*NET Title": r.get("source_title") or "",
                    "Top1 Similarity": r.get("top1_similarity"),
                    "Candidate Rank": cand.get("rank"),
                    "Candidate ESCO Title": cand.get("preferred_title")
                    or esco_title(tid, label_index),
                    "Candidate ESCO ID": tid or "",
                    "Candidate Similarity": cand.get("similarity"),
                    "Is exactMatch GT?": yn(
                        _target_hits_gold(tid, gold, exact_only=True)
                    ),
                    "GT match type": match_type_for(tid, gold),
                }
            )
    return pd.DataFrame(records)


def metrics_summary(
    eligible: list[dict],
    metrics: dict,
) -> tuple[pd.DataFrame, dict]:
    n = len(eligible)
    emb_hits = {
        k: sum(
            1
            for r in eligible
            if _hit_at_k(_embedding_ranked_ids(r, limit=10), r["_gold"], k)
        )
        for k in ACCURACY_KS
    }
    emb_wrong_top1 = n - emb_hits[1]
    gated_hits = {
        k: sum(
            1
            for r in eligible
            if _hit_at_k(final_ranked_target_ids(r, limit=10), r["_gold"], k)
        )
        for k in ACCURACY_KS
    }
    gated_correct_top1 = gated_hits[1]
    sent = sum(1 for r in eligible if _sent_to_resolver(r))
    lift_pp = (
        round(100.0 * (gated_hits[1] - emb_hits[1]) / n, 4) if n else 0.0
    )
    report = {
        "eligible_n": n,
        "embedding_wrong_top1": emb_wrong_top1,
        "embedding_acc": {k: emb_hits[k] / n if n else None for k in ACCURACY_KS},
        "embedding_hits": emb_hits,
        "gated_acc": {k: gated_hits[k] / n if n else None for k in ACCURACY_KS},
        "gated_hits": gated_hits,
        "gated_correct_top1": gated_correct_top1,
        "records_sent_to_jev": sent,
        "high_confidence": n - sent,
        "lift_absolute": gated_hits[1] - emb_hits[1],
        "lift_pp": lift_pp,
        "pipeline_metrics": metrics,
    }
    rows = [
        {"Metric": "ExactMatch-eligible O*NET (N)", "Value": n},
        {"Metric": "Embedding Top-1 incorrect", "Value": emb_wrong_top1},
        {"Metric": "Embedding Top-1 correct", "Value": emb_hits[1]},
    ]
    for k in ACCURACY_KS:
        rows.append(
            {
                "Metric": f"Embedding Acc@{k}",
                "Value": round(emb_hits[k] / n, 6) if n else None,
            }
        )
        rows.append({"Metric": f"Embedding hits@{k}", "Value": emb_hits[k]})
    rows.extend(
        [
            {"Metric": "Sent to JEV (below θ)", "Value": sent},
            {"Metric": "High-confidence (embedding kept)", "Value": n - sent},
            {"Metric": "Gated / final Top-1 correct", "Value": gated_correct_top1},
        ]
    )
    for k in ACCURACY_KS:
        rows.append(
            {
                "Metric": f"Gated Acc@{k}",
                "Value": round(gated_hits[k] / n, 6) if n else None,
            }
        )
        rows.append({"Metric": f"Gated hits@{k}", "Value": gated_hits[k]})
    rows.extend(
        [
            {
                "Metric": "Lift vs embedding-only (absolute Top-1)",
                "Value": gated_hits[1] - emb_hits[1],
            },
            {"Metric": "Lift vs embedding-only (pp)", "Value": lift_pp},
            {
                "Metric": "Pipeline embedding_top1_accuracy",
                "Value": metrics.get("embedding_top1_accuracy"),
            },
            {
                "Metric": "Pipeline accuracy_at_1",
                "Value": metrics.get("accuracy_at_1"),
            },
            {
                "Metric": "Pipeline records_sent_to_jev",
                "Value": metrics.get("records_sent_to_jev"),
            },
            {"Metric": "Pipeline threshold", "Value": metrics.get("threshold")},
        ]
    )
    return pd.DataFrame(rows), report


def write_report_md(path: Path, report: dict, run_id: str) -> None:
    emb = report["embedding_acc"]
    gat = report["gated_acc"]
    lines = [
        f"# ESCO v1.1.0 Analysis Report — `{run_id}`",
        "",
        f"- Eligible exactMatch O*NET sources (N): **{report['eligible_n']}**",
        f"- Embedding Top-1 incorrect: **{report['embedding_wrong_top1']}**",
        f"- Records sent to JEV: **{report['records_sent_to_jev']}** "
        f"(high-confidence kept: {report['high_confidence']})",
        "",
        "## Embedding-only Acc@K",
        "",
        "| K | Accuracy | Hits |",
        "|---|----------|------|",
    ]
    for k in ACCURACY_KS:
        lines.append(
            f"| {k} | {emb[k]:.4%} | {report['embedding_hits'][k]} |"
            if emb[k] is not None
            else f"| {k} | — | {report['embedding_hits'][k]} |"
        )
    lines.extend(
        [
            "",
            "## After JEV gate (hybrid)",
            "",
            "| K | Accuracy | Hits |",
            "|---|----------|------|",
        ]
    )
    for k in ACCURACY_KS:
        lines.append(
            f"| {k} | {gat[k]:.4%} | {report['gated_hits'][k]} |"
            if gat[k] is not None
            else f"| {k} | — | {report['gated_hits'][k]} |"
        )
    lines.extend(
        [
            "",
            "## Improvement vs embedding-only",
            "",
            f"- Absolute Top-1 gains: **{report['lift_absolute']}**",
            f"- Percentage-point lift: **{report['lift_pp']} pp**",
            f"- Gated Top-1 correct: **{report['gated_correct_top1']}** / {report['eligible_n']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    if not (run_dir / "results.jsonl").is_file():
        raise SystemExit(f"Missing results.jsonl in {run_dir}")

    clear_item_index_cache()
    label_index = esco_labels()
    rows = _load_rows(run_dir)
    eligible = _eligible(rows)
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    emb_df = embedding_accuracy_frame(eligible, label_index)
    jev_df = jev_candidates_frame(eligible, label_index)
    summary_df, report = metrics_summary(eligible, metrics)
    gt_df = build_records(run_dir)

    # 1) embedding accuracy
    out1 = run_dir / "embedding_accuracy_1.1.0.xlsx"
    wb1 = Workbook()
    ws = wb1.active
    ws.title = "Embedding_AccK"
    write_sheet(ws, emb_df)
    wb1.save(out1)
    print(f"Wrote {out1}")

    # 2) JEV candidates
    out2 = run_dir / "jev_candidates_1.1.0.xlsx"
    wb2 = Workbook()
    ws = wb2.active
    ws.title = "JEV_Candidates"
    write_sheet(ws, jev_df if len(jev_df) else pd.DataFrame([{"Note": "No JEV rows"}]))
    wb2.save(out2)
    print(f"Wrote {out2}")

    # 3) Final GT match (same as export_gt_match_excel)
    out3 = run_dir / "onet_esco_ground_truth_match_1.1.0.xlsx"
    wb3 = Workbook()
    ws0 = wb3.active
    ws0.title = "Summary"
    write_sheet(ws0, summary_df)
    ws1 = wb3.create_sheet("All_ExactMatch")
    write_sheet(ws1, gt_df)
    df_jev = gt_df[gt_df["Sent to Jev"] == "Yes"].copy() if len(gt_df) else gt_df
    ws2 = wb3.create_sheet("Jev_Sent")
    write_sheet(ws2, df_jev if len(df_jev) else pd.DataFrame([{"Note": "empty"}]))
    wb3.save(out3)
    print(f"Wrote {out3}")

    # 4) metrics summary
    out4 = run_dir / "metrics_summary_1.1.0.xlsx"
    wb4 = Workbook()
    ws = wb4.active
    ws.title = "Metrics"
    write_sheet(ws, summary_df)
    wb4.save(out4)
    print(f"Wrote {out4}")

    run_id = run_dir.name
    md = run_dir / "analysis_report_1.1.0.md"
    write_report_md(md, report, run_id)
    print(f"Wrote {md}")
    print(
        f"N={report['eligible_n']} emb_wrong@1={report['embedding_wrong_top1']} "
        f"gated@1={report['gated_correct_top1']} lift={report['lift_absolute']} "
        f"({report['lift_pp']} pp) jev={report['records_sent_to_jev']}"
    )


if __name__ == "__main__":
    main()
