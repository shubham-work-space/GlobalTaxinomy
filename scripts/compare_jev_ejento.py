"""O*NETs where Jev (run9) is exact-correct but Ejento (run10) is not."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.accuracy import (
    _gold_has_exact,
    _pick_final_target,
    _resolve_gold,
    _sent_to_resolver,
    _target_hits_gold,
)
from backend.app.ground_truth import direction_map
from backend.app.item_index import esco_labels

ROOT = Path(__file__).resolve().parents[1]
JEV_RUN = ROOT / "output/runs/run_0009_20261015T061000Z"
EJ_RUN = ROOT / "output/runs/run_0010_20261015T075751Z"


def load_rows(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("source_id")
        if sid:
            out[sid] = row
    return out


def title_for(tid: str | None, labs: dict) -> str:
    if not tid:
        return ""
    rows = labs.get(tid) or labs.get(str(tid).rstrip("/")) or []
    for row in rows:
        if row.get("is_preferred"):
            return row.get("text") or ""
    return (rows[0].get("text") if rows else "") or ""


def main() -> None:
    labs = esco_labels()
    gold_map = direction_map("onet_to_esco")
    jev_rows = load_rows(JEV_RUN)
    ej_rows = load_rows(EJ_RUN)

    only_jev: list[dict] = []
    only_ej: list[dict] = []
    both_ok = 0
    both_bad = 0

    for sid, jr in jev_rows.items():
        er = ej_rows.get(sid)
        if not er:
            continue
        gold = _resolve_gold(gold_map, sid)
        if not gold or not _gold_has_exact(gold):
            continue
        if not _sent_to_resolver(jr):
            continue

        j_tid = _pick_final_target(jr)
        e_tid = er.get("llm_target_id") or er.get("jev_target_id") or _pick_final_target(er)
        j_ok = _target_hits_gold(j_tid, gold, exact_only=True)
        e_ok = _target_hits_gold(e_tid, gold, exact_only=True)

        exact_ids = [gid for gid, mt in gold.items() if mt == "exactMatch"]
        gt_titles = "; ".join(title_for(x, labs) for x in exact_ids)
        rec = {
            "source_id": sid,
            "onet": jr.get("source_title") or "",
            "jev": jr.get("jev_preferred_title")
            or jr.get("final_preferred_title")
            or title_for(j_tid, labs),
            "ejento": er.get("llm_preferred_title")
            or er.get("jev_preferred_title")
            or title_for(e_tid, labs),
            "gt": gt_titles,
        }
        if j_ok and not e_ok:
            only_jev.append(rec)
        elif e_ok and not j_ok:
            only_ej.append(rec)
        elif j_ok and e_ok:
            both_ok += 1
        else:
            both_bad += 1

    only_jev.sort(key=lambda r: r["onet"].lower())
    only_ej.sort(key=lambda r: r["onet"].lower())

    print(f"Jev correct, Ejento wrong: {len(only_jev)}")
    print(f"Ejento correct, Jev wrong: {len(only_ej)}")
    print(f"Both correct: {both_ok}")
    print(f"Both wrong: {both_bad}")
    print()
    print("=== Jev YES / Ejento NO ===")
    for i, r in enumerate(only_jev, 1):
        print(f"{i:2d}. {r['source_id']}  {r['onet']}")
        print(f"    Jev:    {r['jev']}")
        print(f"    Ejento: {r['ejento']}")
        print(f"    GT:     {r['gt']}")

    out_path = ROOT / "output/runs/jev_correct_ejento_wrong.json"
    out_path.write_text(json.dumps(only_jev, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
