# Exact McNemar Test: Embedding-Only vs Confidence-Gated Re-Ranking

This note documents **how** the McNemar \(p\)-values in `Techinal_research_summary.md` (Section 18) were computed for:

- **Embedding-only vs JEV-gated** (JEV-12/13/14)
- **Embedding-only vs Ejento Claude-gated** (Ejento-1/2/3)
- **Alternative re-ranker vs gated choice:** Ejento Claude-gated vs JEV-gated (paper Table 6 row 2)

---

## 1. What question are we asking?

For each O\*NET source with an official ESCO `exactMatch` ground-truth link, we observe two binary outcomes under the **same** Model P embedding weights and threshold \(\theta = 0.60\):

| Configuration | Prediction used for Acc@1 |
|---------------|---------------------------|
| **Embedding-only** | Stage-3 Top-1 (`top1_exact_hit`) |
| **Gated** | If \(S_{\max} \ge 0.60\): keep embedding Top-1; else: resolver Top-1 (`final_exact_hit`) |

**Null hypothesis \(H_0\):** the two configurations have the **same** Acc@1 (on these paired sources). Equivalently, among sources where they disagree, each direction of disagreement is equally likely.

**Alternative \(H_1\):** Acc@1 differs (two-sided).

We use **McNemar’s test** because outcomes are **paired** (same sources, same embeddings; only the gated resolver path can change the answer).

---

## 2. Evaluation set

| Item | Value |
|------|-------|
| Direction | O\*NET → ESCO |
| Eligible sources | \(N = 490\) with at least one `exactMatch` in the official crosswalk |
| Model / weights | Model P: `label_max=0.32`, `label_median=0.12`, `description_pool=0.26`, `skills_pool=0.24`, `knowledge_pool=0.06` |
| Gate | \(\theta = 0.60\) |
| High-confidence (embedding accepted) | \(n = 310\) |
| Low-confidence (sent to resolver) | \(n = 180\) |

Correctness is **exactMatch Acc@1**: the chosen ESCO URI must appear among that source’s `exactMatch` ground-truth IDs.

---

## 3. Data sources

### JEV runs

| Label | Run directory |
|-------|---------------|
| JEV-12 | `output/runs/run_0012_20261015T102942Z/` |
| JEV-13 | `output/runs/run_0013_20261015T105210Z/` |
| JEV-14 | `output/runs/run_0014_20261015T135838Z/` |

Per-source fields in `results.jsonl` (exactMatch-eligible rows only, `has_exact_match == true`):

| Field | Role |
|-------|------|
| `top1_exact_hit` | Embedding-only correct? |
| `final_exact_hit` | Gated final correct? |
| `sent_to_jev` | Whether the source was routed to JEV |

### Ejento (Claude) runs

| Label | Workbook |
|-------|----------|
| Ejento-1 | `output/ejento_io_pairs_run_1/onet_esco_ejento_ground_truth_match.xlsx` |
| Ejento-2 | `output/ejento_io_pairs_run_2/onet_esco_ejento_ground_truth_match.xlsx` |
| Ejento-3 | `output/ejento_io_pairs_run_3/onet_esco_ejento_ground_truth_match.xlsx` |

Sheet `Ejento_Sent_180` covers **only** the 180 low-confidence sources. Columns used:

| Column | Role |
|--------|------|
| `Embedding matches GT (exact)?` | Embedding-only correct on that source |
| `Final matches GT (exact)?` | Gated Claude final correct on that source |

For the **full** \(N = 490\) gated pipeline, the 310 high-confidence sources are taken from the JEV run artifacts (same embeddings / gate): for those rows, gated ≡ embedding, so they add **no** discordant pairs. Numerically:

\[
\text{Hybrid Acc@1} = \frac{286 + \text{(Ejento correct on 180)}}{490}
\]

where \(286\) is the number of high-confidence exactMatch hits (shared across these runs).

---

## 4. McNemar contingency table

For each source \(i = 1,\ldots,N\), let:

- \(E_i = 1\) if embedding Top-1 is exactMatch-correct, else \(0\)
- \(G_i = 1\) if gated final is exactMatch-correct, else \(0\)

Count the four cells:

|  | Gated correct (\(G=1\)) | Gated wrong (\(G=0\)) |
|--|-------------------------|------------------------|
| **Emb correct (\(E=1\))** | \(a\) (both OK) | \(c\) (emb OK → gated wrong) |
| **Emb wrong (\(E=0\))** | \(b\) (emb wrong → gated OK) | \(d\) (both wrong) |

Definitions used in the paper summary:

- \(b\) = **fixes** (embedding miss that gating recovers)
- \(c\) = **regressions** (embedding hit that gating breaks)
- Discordant total: \(n_d = b + c\)

**Important:** Concordant cells \(a\) and \(d\) do **not** enter the McNemar statistic. Only flips matter.

Because high-confidence sources never change prediction, they contribute only to \(a\) or \(d\). All of \(b\) and \(c\) come from the 180 resolver-routed sources.

---

## 5. Exact (binomial) McNemar \(p\)-value

Under \(H_0\), each discordant case is a fair coin flip toward \(b\) or \(c\):

\[
B \sim \mathrm{Binomial}(n_d,\, 1/2), \quad n_d = b + c.
\]

The **two-sided exact** \(p\)-value is:

\[
p = \min\Bigl(1,\; 2 \sum_{i=0}^{k} \binom{n_d}{i} \Bigl(\tfrac{1}{2}\Bigr)^{n_d}\Bigr),
\quad k = \min(b, c).
\]

Equivalently: twice the probability of observing a split as extreme as (or more extreme than) the observed \((b,c)\) in the direction of the smaller cell, capped at 1.

We intentionally use the **exact** test (not the \(\chi^2\) approximation with continuity correction), which is preferred when \(n_d\) is moderate (\(\sim 30\)).

### Decision rule

| Result | Meaning |
|--------|---------|
| \(p < 0.05\) | Reject \(H_0\): Acc@1 of embedding-only and gated configs differ significantly |
| \(p \ge 0.05\) | Do **not** reject \(H_0\): the observed lift is not significant at \(\alpha = 0.05\) (it may still be positive) |

Failing to reject \(H_0\) is **not** proof that the methods are identical; it means the evidence is insufficient at that threshold.

---

## 6. Worked example (JEV-14)

From `results.jsonl` (exactMatch-eligible):

| Cell | Count | Meaning |
|------|------:|---------|
| \(a\) both OK | 407 | |
| \(d\) both wrong | 52 | |
| \(b\) emb wrong, gated OK | 26 | net fixes |
| \(c\) emb OK, gated wrong | 5 | regressions |
| \(N = a+b+c+d\) | 490 | |
| \(n_d = b+c\) | 31 | |

Accuracies:

\[
\text{Emb Acc@1} = \frac{a+c}{N} = \frac{412}{490} = 0.8408
\]

\[
\text{Gated Acc@1} = \frac{a+b}{N} = \frac{433}{490} = 0.8837
\]

Net Acc@1 change in absolute hits: \(b - c = 21\) (matches `exact_hits_after - exact_hits_before` in `metrics.json`).

Exact McNemar:

\[
k = \min(26,5) = 5,\quad n_d = 31
\]

\[
p = 2 \sum_{i=0}^{5} \binom{31}{i} / 2^{31} \approx 1.92 \times 10^{-4}
\]

Rounded in prose: \(p = 1.9 \times 10^{-4}\).

---

## 7. Full results

### JEV (gated vs embedding, \(N = 490\))

| Run | Emb Acc@1 | Gated Acc@1 | \(a\) | \(d\) | \(b\) | \(c\) | \(n_d\) | Exact \(p\) |
|-----|-----------|-------------|------:|------:|------:|------:|--------:|------------|
| JEV-12 | 0.8408 | 0.8796 | 407 | 54 | 24 | 5 | 29 | \(5.5 \times 10^{-4}\) |
| JEV-13 | 0.8408 | 0.8837 | 407 | 52 | 26 | 5 | 31 | \(1.9 \times 10^{-4}\) |
| JEV-14 | 0.8408 | 0.8837 | 407 | 52 | 26 | 5 | 31 | \(1.9 \times 10^{-4}\) |

All three **reject** \(H_0\) at \(\alpha = 0.05\). Pattern: \(b \gg c\) (many more fixes than regressions).

### Ejento Claude (gated vs embedding, \(N = 490\))

| Run | Emb Acc@1 | Gated Acc@1 | \(b\) | \(c\) | \(n_d\) | Exact \(p\) |
|-----|-----------|-------------|------:|------:|--------:|------------|
| Ejento-1 | 0.8408 | 0.8633 | 21 | 10 | 31 | 0.071 |
| Ejento-2 | 0.8408 | 0.8633 | 22 | 11 | 33 | 0.080 |
| Ejento-3 | 0.8408 | 0.8612 | 21 | 11 | 32 | 0.110 |

All three **fail to reject** \(H_0\) at \(\alpha = 0.05\). Acc@1 still rises (~+2.0–2.2 pp), but regressions (\(c \approx 10\)–11) keep \(b\) and \(c\) close enough that the two-sided exact test is not significant.

On the **180-only** subset the \(p\)-values are identical (same \(b,c\)); high-confidence rows only inflate \(a\) and \(d\).

---

## 8. Alternative re-ranker vs gated choice (Claude vs JEV)

Table 6’s second O\*NET→ESCO row compares two **gated** pipelines that share the same embeddings and \(\theta = 0.60\), and differ only in the low-confidence resolver:

| Role in Table 6 | Configuration |
|-----------------|---------------|
| **Alternative re-ranker** (first system) | Gated Claude: high-conf → embedding Top-1; else → Ejento final (`Final matches GT (exact)?`) |
| **Gated choice** (second system) | Gated JEV: high-conf → embedding Top-1; else → JEV final (`final_exact_hit`) |

Definitions (same McNemar cells as above, but now first = Claude-gated, second = JEV-gated):

- \(b\) = **Fixed**: Claude-gated wrong, JEV-gated correct
- \(c\) = **Broken**: Claude-gated correct, JEV-gated wrong

All discordant pairs sit in the 180 resolver-routed sources (high-confidence rows have identical predictions under both gates).

| Claude run | JEV run | Fixed \(b\) | Broken \(c\) | \(n_d\) | Exact \(p\) |
|------------|---------|------------:|-------------:|--------:|------------|
| Ejento-1 | JEV-12 | 14 | 6 | 20 | \(0.115\) |
| Ejento-1 | JEV-13/14 | 15 | 5 | 20 | \(0.041\) |
| Ejento-2 | JEV-12 | 16 | 8 | 24 | \(0.152\) |
| Ejento-2 | JEV-13/14 | 17 | 7 | 24 | \(0.064\) |
| Ejento-3 | JEV-12 | 14 | 5 | 19 | \(0.064\) |
| **Ejento-3** | **JEV-13/14** | **15** | **4** | **19** | \(1.9 \times 10^{-2}\) |

The paper Table 6 numbers (**Fixed 15, Broken 4, \(p = 1.9 \times 10^{-2}\)**) match **Ejento-3 vs JEV-13/14**. At \(\alpha = 0.05\), that pairing rejects equality: JEV-gated Acc@1 is significantly higher than Claude-gated Acc@1 on the same sources.

**Suggested paper wording.** Against the matched Claude Opus (Ejento) gated backup, JEV-gated Top-1 is also significantly better on the strongest Claude replicate (Ejento-3 vs JEV-13/14: 15 fixes / 4 regressions, exact McNemar \(p = 1.9 \times 10^{-2}\)).

---

## 9. Pseudocode

```python
from math import comb

def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial test under p=0.5
    p = 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, p)

# For each exactMatch-eligible source:
#   emb_ok   = top1_exact_hit
#   gated_ok = final_exact_hit
# Tally a, b, c, d then report exact_mcnemar(b, c)
```

---

## 10. What this supports (and what it does not)

**Supports**

- JEV-gated Acc@1 is significantly higher than the paired embedding-only Acc@1 on JEV-12/13/14.
- Under the same gate, Ejento Claude’s Acc@1 lift is smaller and **not** significant at 0.05 — so the significant gated gain is tied to **JEV**, not to “any LLM backup.”
- On the strongest Claude replicate, JEV-gated also significantly beats Claude-gated (Ejento-3 vs JEV-13/14: \(p = 1.9 \times 10^{-2}\)).

**Does not claim**

- That Claude is harmful or useless (point estimates still improve vs embedding).
- That gated Acc@1 beats the **SME baseline** (that is a different comparison).
- That every Claude×JEV pairing is significant (Ejento-1/2 vs some JEV runs are n.s. at 0.05).
- That Config-1/2/3 “embedding-only” Acc@1 figures (0.8612–0.8633 in the methodology Acc@K table) are the same as this paired embedding baseline (0.8408). Those configs use different embedding setups; McNemar deliberately pairs against the **same** Stage-3 ranking that feeds the gate.

---

## 11. Reproducibility checklist

1. Load exactMatch-eligible rows from JEV `results.jsonl` (`has_exact_match`).
2. Build \((E_i, G_i)\) from `top1_exact_hit` / `final_exact_hit`.
3. Count \(b,c\); compute exact McNemar as above.
4. For Ejento vs embedding: count \(b,c\) from the 180-row sheet; prepend 310 high-conf pairs with \(E=G\) from any matching JEV run if reporting full-pipeline Acc@1 (optional for \(p\), since \(p\) depends only on \(b,c\)).
5. For Claude vs JEV: on the same 180 codes, pair Ejento `Final matches GT (exact)?` with JEV `final_exact_hit`; \(b\) = Claude wrong / JEV right, \(c\) = Claude right / JEV wrong.

---

## 12. Suggested paper wording

> An exact McNemar test over paired per-source outcomes rejects equality of the embedding-only and JEV-gated configurations on JEV-12–14 (\(p = 5.5 \times 10^{-4}\), \(1.9 \times 10^{-4}\), \(1.9 \times 10^{-4}\)). Matched Ejento Claude replicates under the same gate do not reach significance at \(\alpha = 0.05\) (\(p = 0.071\), \(0.080\), \(0.110\)), despite modest Acc@1 gains. Directly comparing the two gated resolvers, JEV also significantly outperforms Claude on Ejento-3 vs JEV-13/14 (15 fixes / 4 regressions, \(p = 1.9 \times 10^{-2}\)).
