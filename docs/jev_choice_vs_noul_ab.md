# JEV Choice vs Noul A/B (gated enriched)

Live A/B comparing TypeSafe System One **choice** vs **noul** ranking under the same Model P embeddings, gate, and `jev_context_v2` enrichment.

Script: `scripts/run_jev_choice_noul_ab.py`  
Smoke check: `scripts/jev_smoke_test.py` (HTTP 200 before rerun)

---

## Setup

| Item | Value |
|------|-------|
| Direction | ONET → ESCO |
| Eligible sources | \(N = 490\) with `exactMatch` GT |
| Model | P |
| Gate \(\theta\) | 0.60 |
| Top-K candidates | 20 |
| Context | `jev_context_v2` |
| Gated to JEV | 180 sources |
| Date | 2026-11-03 |

### Valid runs (post API restart)

| Mode | Run ID | I/O pairs |
|------|--------|-----------|
| Choice | `run_0027_20261103T102427Z` | `output/jev_io_pairs/run_27/` |
| Noul | `run_0028_20261103T102610Z` | `output/jev_io_pairs/run_28/` |

### Invalid earlier runs (do not use)

| Runs | Issue |
|------|--------|
| `run_0024` | HTTP **402** billing / no TypeSafe credits |
| `run_0025` / `run_0026` | HTTP **403** on all JEV calls (stale API process key; Acc@K matched embedding-only) |

Restart the API after changing `JEV_API_KEY` so settings reload.

---

## Acc@K (exactMatch)

| Metric | Choice | Noul | Δ (choice − noul) |
|--------|--------|------|-------------------|
| Acc@1 | **0.883673** | 0.863265 | +0.020408 |
| Acc@2 | 0.932653 | 0.932653 | 0 |
| Acc@3 | **0.963265** | 0.953061 | +0.010204 |
| Acc@5 | **0.979592** | 0.977551 | +0.002041 |
| Acc@10 | 0.989796 | **0.993878** | −0.004082 |

Choice wins Acc@1 / @3 / @5; Noul slightly higher Acc@10; Acc@2 tied.

---

## JEV usage (gated subset)

| Metric | Choice | Noul |
|--------|--------|------|
| API calls | 180 | 3600 |
| Input tokens | 3,929,975 | 6,454,774 |
| Output tokens | 27,475 | 72,000 |
| Total tokens | 3,957,450 | 6,526,774 |
| Avg total tokens / gated source | 21,985.83 | 36,259.86 |

Noul uses one call per candidate (180 × 20 = 3600), so ~1.65× Choice total tokens and 20× API calls.

---

## JEV response health

| Run | Outputs | Status codes |
|-----|---------|--------------|
| Choice (`run_27`) | 180 | **200**: 175; **400**: 5 |
| Noul (`run_28`) | 3600 | **200**: 3600 |

No 402/403 on the valid pair.

---

## How to reproduce

1. Set `.env`: `JEV_API_KEY` (optional `ONET_SQL_DIR` / `ESCO_CSV_DIR`).
2. Start API (restart if key changed): `python -m backend`
3. Smoke: `python scripts/jev_smoke_test.py`
4. A/B: `python scripts/run_jev_choice_noul_ab.py --base-url http://127.0.0.1:8001 --force`
