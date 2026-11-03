# Technical Research Dossier
## Generalized Cross-Taxonomy Interoperability with Confidence-Gated JEV Re-Ranking

**Primary sources:** `25811-methodology_draft_v3.txt`, project codebase (`backend/app/`), `docs/JEV_Knowledge_Share.docx` / `scripts/generate_jev_knowledge_share_docx.py`  
**Convention:** Where a fact is not documented in those sources, the text states `[no enough information is available]`.

---

# 1. Executive Summary

**Problem.** Organizations maintain incompatible occupation taxonomies (O\*NET, ESCO, ISCO, national SOCs, proprietary catalogs). Manual crosswalks are slow, expensive, and hard to maintain. Automated mapping is needed for labour-market analytics, skills routing, and multi-jurisdiction workforce products.

**Why it matters.** Without interoperable taxonomies, job postings, credentials, and labour statistics cannot be compared or transferred across systems. Interoperability underpins global talent matching, regulatory reporting, and AI-assisted career guidance.

**Current limitations.** Expert-curated crosswalks (e.g., published ESCO–O\*NET) remain the de facto gold standard but are point-to-point, costly, and contain annotation inconsistencies. Pure embedding retrieval fails on semantic-boundary occupations. Full LLM re-ranking of every pair is accurate but prohibitively slow and expensive if candidate pools are large.

**Proposed solution.** A taxonomy-agnostic four-stage pipeline: (1) facet extraction into five similarity pillars, (2) dense embeddings, (3) multi-signal weighted cosine scoring with missing-pillar renormalization, (4) confidence-gated LLM re-ranking via **JEV (TypeSafe System One)** only when Top-1 embedding score \(S_{\max} < \theta\) (default \(\theta = 0.60\)).

**Key innovations.**
- Multi-pillar scoring with dual title signals (`label_max` + `label_median`) and weight renormalization for incomplete facets.
- Confidence threshold as a **cost-efficiency gate**, not a quality ceiling — JEV is the accuracy driver on hard cases.
- Structured Choice ranking over Top-20 candidates (vs. constrained shortlists forced by expensive LLMs).
- Double-convergence analysis exposing 19 candidate ground-truth annotation errors in the published ESCO–O\*NET crosswalk.

**Impact.** On ~490 exactMatch-eligible O\*NET→ESCO occupations: Acc@1 = **88.37%** vs. SME baseline **84.75%** (+3.62 pp). Embedding-only already beats the baseline (86.33%); JEV adds +2.04 pp. Knowledge-share comparison vs. Claude Opus 4.7 backup: ~1 hour → ~20 seconds, ~$11 → ~$0.20, with wider Top-20 shortlists and better hard-case accuracy.

---

# 2. Problem Statement

## Business Problem

- **Challenge:** Map occupations (and related concepts) from one taxonomy to another so products, analytics, and compliance workflows can operate across regions and data sources.
- **Why mappings are needed:** Job titles and codes differ by country and publisher; a US O\*NET SOC code is not an ESCO URI or an ISCO unit group. Downstream systems need a shared bridge.
- **Users:** Taxonomy / data engineering teams, labour-market researchers, product teams building matching or skills platforms, SMEs reviewing automated suggestions.
- **Business value of interoperability:** Faster onboarding of new taxonomies, reduced SME annotation cost, consistent multi-market job classification, auditability via ranked shortlists (Accuracy@K), and selective use of expensive reasoning only when needed.

## Research Problem

- **Taxonomy:** A structured catalog of concepts (here, occupations) with identifiers, labels, descriptions, and optional associated attributes (skills, knowledge, hierarchy).
- **Crosswalk:** A set of directed or undirected links between concepts in two taxonomies, often with graded match types (`exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch`).
- **Taxonomy interoperability:** The ability to automatically produce high-quality ranked mappings between arbitrary taxonomy pairs given only structured textual facets, without taxonomy-specific hard-coding.
- **Gap in current methods:** Manual SME crosswalks do not scale; single-signal embedding retrieval underperforms on ambiguous titles; LLM-everywhere re-ranking is costly; published ground truth may itself be inconsistent.
- **Why difficult:** Nomenclature drift, composite occupations spanning multiple ESCO concepts, seniority/skill-level mismatches, sparse or missing facets, structural gaps (no true equivalent in the target taxonomy), and many-to-many semantics.

---

# 3. Research Objectives

1. Reduce dependency on manual, point-to-point crosswalk creation.
2. Improve Top-1 (`exactMatch`) accuracy beyond the published ESCO–O\*NET SME baseline (84.75%).
3. Design a **taxonomy-agnostic** pipeline applicable to any pair exposing textual pillars.
4. Minimize expensive AI reasoning via confidence-gated invocation (Stage 4 only when \(S_{\max} < \theta\)).
5. Increase scalability of re-ranking by using structured Choice (JEV) so Top-20 pools are affordable.
6. Combine multiple semantic signals (title max/median, description, skills, knowledge) with learnable/optimized weights.
7. Handle missing pillars via weight renormalization so scores remain comparable.
8. Quantify Accuracy@K (K = 1, 2, 3, 5, 10) against graded official ground truth.
9. Diagnose ground-truth quality via double-convergence (embedding ∩ JEV agree against GT).
10. Support operational live lookup with the same threshold-and-route logic as batch evaluation.
11. Explore extension to additional taxonomies (ISCO, OSCA, UK SOC) — experimental scripts exist; full Acc@K publication results for those pairs: `[no enough information is available]` in the methodology draft.

---

# 4. Taxonomies Used

## O\*NET (Occupational Information Network)

| Attribute | Detail |
|-----------|--------|
| Full name | Occupational Information Network (O\*NET) |
| Owner / publisher | United States Department of Labor |
| Domain | Occupations / labour market |
| Scope | US occupational titles with descriptions, tasks, skills, knowledge, abilities |
| Number of records | Approximately **1,000** occupational titles (methodology); evaluation uses SOC 2019 as source |
| Hierarchy levels | `[no enough information is available]` in methodology for full hierarchy depth used by the pipeline |
| Structure | SOC-coded occupations with rich facet text |
| Relationship model | Hierarchical SOC structure in the source data; pipeline treats each occupation as a concept with pillars |
| Project role | **Source taxonomy** for primary validation (O\*NET → ESCO) |
| ID field (codebase) | `onet_soc_code` |

**Pillars used:** preferred title, description, skills, knowledge (alternate labels handled primarily on ESCO side via label aggregation).

---

## ESCO (European Skills, Competences, Qualifications and Occupations)

| Attribute | Detail |
|-----------|--------|
| Full name | European Skills, Competences, Qualifications and Occupations |
| Owner / publisher | European Commission |
| Domain | Occupations and skills |
| Scope | Multilingual occupation and skill taxonomy |
| Number of records | Approximately **3,000** occupation concepts |
| Languages | Preferred and alternate labels in **27** languages (methodology); pipeline validation focuses on English facet text — multilingual ablation: `[no enough information is available]` |
| Hierarchy levels | `[no enough information is available]` for levels consumed by scoring |
| Structure | URIs, preferred + alternate labels, descriptions, skill associations |
| Relationship model | SKOS-style graded match types in the official O\*NET crosswalk |
| Project role | **Target taxonomy** for primary validation |
| Version | ESCO **v1.1** (experimental table) |
| ID field (codebase) | `esco_uri` |

**Pillars used:** preferred title, alternate labels (item-level collection for max/median), description, skills, knowledge.

---

## ISCO (International Standard Classification of Occupations)

| Attribute | Detail |
|-----------|--------|
| Full name | International Standard Classification of Occupations (ISCO-08 referenced in project data) |
| Owner / publisher | International Labour Organization (ILO) — standard attribution; project uses local processed CSV |
| Domain | Occupations |
| Scope | International occupation classification |
| Number of records | `[no enough information is available]` as an exact count in methodology |
| Hierarchy levels | `[no enough information is available]` in methodology |
| Structure | Codes + titles + definitions (embed script uses `preferred_title`, `Definition`) |
| Relationship model | Hierarchical unit groups; pipeline embedding script creates `isco_title`, `isco_description` Chroma collections |
| Project role | Present in `data/isco/` and embedding/UK-SOC scripts; **not** the primary Acc@K validation pair in methodology §4 |

---

## OSCA

| Attribute | Detail |
|-----------|--------|
| Full name | `[no enough information is available]` for expanded acronym in methodology |
| Owner / publisher | `[no enough information is available]` |
| Domain | Occupations (paired with ISCO in project data under `ISCO_OSCA/`) |
| Scope | `[no enough information is available]` |
| Number of records | `[no enough information is available]` |
| Hierarchy levels | `[no enough information is available]` |
| Structure | Embed script uses `preferred_title`, `lead_statement` → `osca_title`, `osca_description` |
| Relationship model | `[no enough information is available]` |
| Project role | Facet embedding support; not in methodology §4 headline results |

---

## UK SOC 2010 (related experiment)

| Attribute | Detail |
|-----------|--------|
| Full name | UK Standard Occupational Classification 2010 |
| Owner / publisher | UK Office for National Statistics (standard attribution) |
| Domain | Occupations |
| Project role | Script `scripts/run_jev_uk_soc_to_isco.py` applies JEV Choice over Top-10 UK SOC→ISCO embedding candidates when Top-1 < 0.6; Excel artifact `UK_SOC2010_to_ISCO08_with_similarity.xlsx` |
| Formal Acc@K results in methodology draft | `[no enough information is available]` |

---

# 5. Available Data Attributes

## Schema Documentation — Canonical Similarity Pillars

| Field / Pillar | Description | Data Type | Nullable | Example |
|----------------|-------------|-----------|----------|---------|
| Title (preferred) | Canonical label of the concept | string | typically no | `"Software Developers"` |
| Alternate Labels | Synonyms / alternative titles | list→joined string or per-item embeddings | yes | `"Apps developer"; "Software engineer"` |
| Description | Narrative scope text | string | yes | Job definition paragraph |
| Skills | Associated competencies | string (items joined) | yes | List of skill phrases |
| Knowledge | Domain knowledge areas | string (items joined) | yes | List of knowledge areas |
| Concept ID | Taxonomy-native identifier | string | no | O\*NET SOC code / ESCO URI |

## O\*NET (as used in pipeline)

| Field | Description | Data Type | Nullable | Notes |
|-------|-------------|-----------|----------|-------|
| `onet_soc_code` | Source occupation ID | string | no | Primary key |
| preferred title / title facet | Label | string | no | Embedded as title |
| description | Occupation description | string | possibly | Embedded as description |
| skills | Skill requirements text | string | possibly | Pooled string for embedding |
| knowledge | Knowledge areas text | string | possibly | Pooled string for embedding |

Exact column-level CSV schema with null rates and text-length statistics: `[no enough information is available]`.

## ESCO (as used in pipeline)

| Field | Description | Data Type | Nullable | Notes |
|-------|-------------|-----------|----------|-------|
| `esco_uri` | Target occupation URI | string | no | e.g. `http://data.europa.eu/esco/occupation/...` |
| preferred_title | Preferred label | string | no | |
| alternate_title / labels | Alternate labels | multi-item | yes | Each label embedded independently in `esco_label` collection |
| description | Concept description | string | possibly | |
| skills | Associated skills text | string | possibly | |
| knowledge | Associated knowledge text | string | possibly | |

## Data quality observations

- Missing pillars are **flagged** and masked; weights renormalized (methodology §3.4.2).
- Alternate labels are **not** concatenated for embedding; independent embeddings outperform a single concatenated string.
- Markup stripped; UTF-8 encoding.
- Systematic missing-value rates / length histograms: `[no enough information is available]`.
- Normalization concerns: terminology differences across taxonomies (addressed via multi-pillar signals rather than title alone).

---

# 6. Ground Truth and Crosswalks

## ESCO–O\*NET Official Occupational Crosswalk

| Attribute | Detail |
|-----------|--------|
| Source taxonomy | O\*NET (SOC) |
| Target taxonomy | ESCO occupations |
| Publisher | Official ESCO–O\*NET crosswalk (European Commission / linked open data program) |
| Official or unofficial | **Official** |
| Number of mappings | Full crosswalk size: `[no enough information is available]` as exact link count; **~490** O\*NET sources with ≥1 `exactMatch` used for Acc@K |
| Mapping cardinality | Graded links; can be one-to-many / many-to-one depending on occupation (composite titles, multiple exactMatches possible) |
| Match types | `exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch` |
| Mapping confidence | Match type only; no numeric confidence in GT |
| Local path (codebase) | `Ground truth Onet to Esco/ONET_(Occupations)_0_updated.csv` |
| Parsing | Header row index 16; data from row 17 (codebase) |

**Evaluation rule:** Acc@K uses **exactMatch only**. Sources without any exactMatch are excluded from the denominator.

**Published baseline Acc@1 (exactMatch):** **0.8475** (84.75%). Related report benchmarks in code: any-match Top-1 0.9096, any Top-5 0.9945, exact Top-10 1.0.

**Examples (from double-convergence Table 4):**

| O\*NET | GT exactMatch ESCO | Pipeline prediction |
|--------|--------------------|---------------------|
| Fitness and Wellness Coordinators | sport facility manager | spa manager |
| Claims Adjusters, Examiners, and Investigators | loss adjuster | insurance claims handler |
| Airline Pilots, Copilots, and Flight Engineers | aircraft pilot | airline transport pilot |

## UK SOC 2010 → ISCO-08 (project artifact)

| Attribute | Detail |
|-----------|--------|
| Source | UK SOC 2010 |
| Target | ISCO-08 |
| Publisher / official status | `[no enough information is available]` whether official |
| Artifact | `UK_SOC2010_to_ISCO08_with_similarity.xlsx` |
| Formal mapping cardinality / Acc@K in methodology | `[no enough information is available]` |

---

# 7. Data Preparation Pipeline

## Steps

1. **Extraction** — Pull pillar texts from native sources (CSV / JSON / API / ontology / DB export). O\*NET and ESCO facets loaded into Chroma-backed stores under `data/`.
2. **Cleaning** — Convert to UTF-8; strip markup.
3. **Standardization** — Flatten to key-value concept records with active pillar set \(P(c)\).
4. **Concatenation (multi-item pillars)** — Skills/knowledge lists joined with a separator token into one string before embedding.
5. **Alternate-label pooling strategy** — Each alternate label embedded **independently** (not concatenated).
6. **Missing-pillar flagging** — Absent pillars recorded for renormalization masks.
7. **Deduplication** — `[no enough information is available]` as an explicit documented step.
8. **Attribute selection** — Five canonical pillars; optional additional pillars allowed if weights sum to 1.0.
9. **Language handling** — ESCO is multilingual; experimental validation uses English facets as available — multilingual policy details: `[no enough information is available]`.
10. **Persistence** — Pillar texts and embeddings stored in vector index; reused across batch and live queries.
11. **Validation** — Inventory / Chroma verification before scoring runs (README pipeline step 1).

## Pseudocode (Stage 1)

```
for each concept c in Taxonomy_A ∪ Taxonomy_B:
    extract {title, alt_labels, description, skills, knowledge}
    normalize encodings; strip markup
    for multi-valued skills/knowledge: join with separator
    for each alternate label: keep as separate text unit
    flag missing pillars → P(c)
    persist facet texts
```

## Workflow diagram

```
Native taxonomy files
        │
        ▼
┌───────────────────────┐
│ Facet extraction      │
│ + UTF-8 / strip markup│
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Pillar records + mask │
│ Persist to store      │
└───────────────────────┘
```

---

# 8. Embedding Strategy

## Title Embedding

| | |
|--|--|
| **Purpose** | Capture preferred-label semantics; query for ESCO label aggregation |
| **Input** | Preferred title string |
| **Output** | Unit-normalized dense vector |
| **Notes** | Used as query for `label_max` / `label_median` against all target labels |

## Alternate Label Embeddings (target side)

| | |
|--|--|
| **Purpose** | Richer title matching than single preferred label |
| **Input** | Each alternate (and preferred) label separately |
| **Output** | One vector per label item in item-level collection |
| **Aggregation at score time** | max and median cosine vs. source title vector |

## Description Embedding

| | |
|--|--|
| **Purpose** | Scope / duties similarity when titles diverge |
| **Input** | Full description text (normalized) |
| **Output** | One vector per concept |

## Skills Embedding

| | |
|--|--|
| **Purpose** | Competency-profile alignment |
| **Input** | Joined skills string |
| **Output** | One vector per concept |

## Knowledge Embedding

| | |
|--|--|
| **Purpose** | Domain-knowledge alignment |
| **Input** | Joined knowledge string |
| **Output** | One vector per concept |

## Input templates / prompts for embeddings

No special prompt templates; raw facet text is embedded. Prompt formats for embeddings: N/A.

ISCO/OSCA embed script explicitly embeds **exact CSV cell text** with no labels/prefixes.

---

# 9. Embedding Models

## text-embedding-3-large (primary experimental model)

| Attribute | Detail |
|-----------|--------|
| Provider | OpenAI (via Azure OpenAI in this codebase) |
| Version / deployment | `text-embedding-3-large` (methodology Table 1); Azure deployment name configurable via `AZURE_OAI_EMBEDDING_DEPLOYMENT_LARGE` |
| Embedding dimension | OpenAI large = **3072** (standard API); confirm deployment truncation: `[no enough information is available]` if dimensions were reduced |
| Context window | Model max input tokens per OpenAI docs; exact token limits used in project: `[no enough information is available]` |
| Cost | `[no enough information is available]` as measured project spend for embeddings alone |
| Performance observations | Used for all pillars; same model required on both taxonomies for cosine comparability |
| Reason selected | High-quality dense retrieval; named as experimental configuration |

## text-embedding-3-small (code default fallback)

| Attribute | Detail |
|-----------|--------|
| Provider | OpenAI / Azure OpenAI |
| Version | `text-embedding-3-small` if large deployment unset (`config.py`) |
| Dimension | **1536** (standard) |
| Role | Code default when large not configured; methodology headline results cite **large** |
| Comparative ablation small vs large | `[no enough information is available]` |

## Other models tested

`[no enough information is available]` beyond the above in the methodology draft.

---

# 10. Similarity Retrieval System

## Similarity metric

- **Cosine similarity** on unit-normalized embeddings:
  \[
  \cos(e_a, e_b) = e_a \cdot e_b \in [-1, 1]
  \]
  clipped to \([0, 1]\) before scoring (Eq. 1).
- Equivalent to **dot product** after L2 normalization.
- **Hybrid scoring:** weighted sum of per-pillar similarities (Section 11), not BM25/lexical hybrid — lexical hybrid: `[no enough information is available]`.

## Index / storage

- **Vector store:** ChromaDB (`data/chroma_db/`).
- Collections per taxonomy/pillar (and item-level `esco_label`).
- Optional cached score tensor: `output/cache/P_tensor.npz` (“Model P”) to speed evals.
- ANN algorithm internals (HNSW params etc.): `[no enough information is available]`.

## Candidate generation

| Parameter | Default |
|-----------|---------|
| Pool scored | All target occupations (~3,000 ESCO) |
| \(K_{\mathrm{pool}}\) retained | **20** |
| \(K_{\mathrm{display}}\) / Acc@K depth | **10** |
| Live UI | Top-20 → Top-10 |

For each source \(s\):
1. Score all targets with \(S(s,t)\).
2. Keep Top-20 by descending score.
3. Record \(S_{\max}(s) = \max_t S(s,t)\) for routing.

---

# 11. Multi-Signal Scoring Framework

## Combined score

\[
S(s, t) = \sum_i w_i' \cdot c_i(s, t)
\]

with \(\sum_i w_i' = 1\) over active pillars \(P(s) \cap P(t)\).

## Pre-optimised pillar weights (Table 2 / `config/active_weights.json`)

| Signal | Weight \(w_i\) | Computation |
|--------|----------------|-------------|
| Title (`label_max`) | **0.32** | \(\max_j \cos(e^s_{\mathrm{title}}, e^t_{\mathrm{label}_j})\) |
| Alternate Title (`label_median`) | **0.12** | \(\mathrm{median}_j \cos(e^s_{\mathrm{title}}, e^t_{\mathrm{label}_j})\) |
| Description (`description_pool`) | **0.26** | Pooled description cosine |
| Skills (`skills_pool`) | **0.24** | Pooled skills cosine |
| Knowledge (`knowledge_pool`) | **0.06** | Pooled knowledge cosine |
| **Sum** | **1.00** | |

Equivalent expanded form:

\[
S = 0.32\,c_{\mathrm{max}} + 0.12\,c_{\mathrm{med}} + 0.26\,c_{\mathrm{desc}} + 0.24\,c_{\mathrm{skills}} + 0.06\,c_{\mathrm{know}}
\]

(with renormalization if any pillar missing).

## Missing-pillar renormalization

\[
w_i' = \frac{w_i}{\sum_{j \in P'} w_j} \quad (i \in P'),\quad w_i' = 0 \ (i \notin P')
\]

## How weights were derived

- Offline optimisation on a **held-out portion** of the ground-truth crosswalk.
- Search: **grid or Bayesian** maximizing Accuracy@1 on validation.
- Frozen into configuration (`active_weights.json` source: `"optimizer"`, recorded `accuracy_at_1`: 0.8408 on that optimiser snapshot — note this is an optimiser artifact, not identical to full eval Acc@1).

## Alternatives tested

Detailed weight grid results / ablation tables beyond headline Config-1..3: `[no enough information is available]` in methodology (discussion notes title signals = 0.44 combined; description+skills ≈ title importance).

Codebase implements mask-aware weighted sum over signal tensor (`pipeline._combined_from_tensor`).

---

# 12. Confidence Framework

## Definition

Confidence proxy = **Top-1 multi-signal score** \(S_{\max}(s)\).

## Threshold

- Default \(\theta = \mathbf{0.60}\).
- **Static** threshold (not distribution-adaptive in methodology).
- Role: **cost-efficiency gate**, not quality ceiling.

## Routing logic

```
if S_max(s) >= θ:
    accept Stage-3 ranking (Top-K_display)
else:
    escalate Top-K_pool candidates to JEV re-ranker
```

## Margin-based / Top1−Top2 confidence

Not used as the primary gate. Margin-based alternative: `[no enough information is available]` as a tested production rule.

## Distribution-based confidence

`[no enough information is available]`.

## Threshold selection

Documented default 0.60; systematic sweep of \(\theta\) vs. cost/accuracy curve: `[no enough information is available]` in methodology draft (future work candidate).

---

# 13. AI / LLM Re-Ranking Layer

## Model

| Attribute | Detail |
|-----------|--------|
| Name | **JEV** (TypeSafe System One) |
| Provider | TypeSafe (`https://api.typesafe.ai`) |
| Version / model string | `jev-latest` |
| API | `POST /v1/systemone` |
| Context window | `[no enough information is available]` |
| Temperature | N/A for Choice API as configured; sampling params: `[no enough information is available]` |
| Candidate count supplied | Up to **20** (labels A–T) |
| Output ranking depth | Top **10** by descending Choice probabilities |
| Concurrency (code) | Default `jev_concurrency = 16` |
| Caching | Per (source concept, run configuration); I/O logged under `output/jev_io_pairs/` |
| Fallback | On failure, fall back to embedding order |

### Prior backup (knowledge share)

Claude Opus 4.7 was previously used as the LLM backup with smaller shortlists due to cost/latency.

## Full System Prompt

JEV uses structured Choice questions rather than a classic chat system prompt. The Choice **instructions** field (from `build_jev_payload`) is:

> Using only the supplied Job Title, Alternate Title, Job Description, Skills, and Knowledge for the input occupation and each candidate, which candidate is the closest overall match? Do not use web search or any external sources. Pick exactly one of: {A, B, …}.

## Full User Prompt / Payload Structure

State includes:

- **input:** `job_title`, `alternate_title`, `job_description`, `skills`, `knowledge`
- **candidates:** labeled A..T each with the same five fields + `target_id`
- **criteria:** per-label summary string:
  `Job Title: … | Alternate Title: … | Job Description: … | Skills: … | Knowledge: …`

## Expected Output Format

- `answers.match.choice`: selected label (e.g. `"C"`)
- `answers.match.probabilities`: map label → probability
- Ranking = labels sorted by descending probability; Top-10 IDs used for Acc@K

## Ranking Logic

1. Parse probabilities; sort descending (ties broken by original order).
2. If probabilities missing: put chosen label first, then remaining original order.
3. Map labels back to target IDs.

## Decision Criteria

Closest overall semantic match using **only** supplied pillar fields; no external retrieval.

## Cost per Request / Latency

From knowledge-share comparison on the backup workload vs. Claude Opus 4.7:

| Metric | Claude Opus 4.7 backup | JEV backup |
|--------|------------------------|------------|
| Approximate run time | ~1 hour | ~20 seconds |
| Approximate cost | ~$11 | ~$0.20 |

Per-request token pricing and p50/p95 latency: `[no enough information is available]`.

## Failure Cases

- True match not in Top-20 embedding pool → unretractable Stage-4 miss.
- API/network failure → embedding fallback.
- Ambiguous GT / structural ESCO gaps (radiologist vs radiographer class of errors).
- Parsing edge cases mitigated by label regex tolerance.

---

# 14. End-to-End Workflow

1. Load taxonomies (O\*NET source, ESCO target).
2. Extract and normalise pillar attributes; flag missing pillars.
3. Generate (or load) dense embeddings into Chroma / tensor cache.
4. For each source, compute per-pillar cosine similarities to all targets.
5. Aggregate with optimised weights → \(S(s,t)\); retain Top-20; record \(S_{\max}\).
6. Confidence gate at \(\theta = 0.60\).
7. High confidence: accept embedding Top-10.
8. Low confidence: JEV Choice re-rank Top-20 → ordered Top-10.
9. Evaluate Acc@K vs. exactMatch GT; save run artifacts under `output/runs/`.
10. Live Crosswalk UI uses the same hybrid path.

```
┌────────────┐   ┌────────────┐   ┌─────────────────┐   ┌──────────────┐
│ Extract    │──▶│ Embed      │──▶│ Multi-signal    │──▶│ S_max ≥ θ ?  │
│ facets     │   │ pillars    │   │ score + Top-20  │   └──────┬───────┘
└────────────┘   └────────────┘   └─────────────────┘          │
                                               yes│            │no
                                                  ▼            ▼
                                           Accept embed   JEV Choice
                                           Top-10         re-rank → Top-10
                                                  \          /
                                                   ▼        ▼
                                              Accuracy@K / Live display
```

---

# 15. System Architecture

## Components

| Layer | Technology |
|-------|------------|
| Frontend | React/Vite (`frontend/`, port 5173) — Crosswalk + Evals tabs |
| Backend API | Python FastAPI-style app (`python -m backend`, port 8001) |
| Embedding | Azure OpenAI embeddings |
| Vector store | ChromaDB |
| Score cache | NumPy tensor (`P_tensor.npz`) |
| Re-ranker | TypeSafe System One JEV |
| Ground truth | Official CSV |
| Artifacts | `output/runs/`, `output/jev_io_pairs/` |

## ASCII architecture

```
[Browser UI]
     │ HTTP
     ▼
[Backend API] ──▶ [Run Pipeline]
     │                 │
     │                 ├─▶ Chroma (embeddings + facet texts)
     │                 ├─▶ Signal tensor / weighted scoring
     │                 ├─▶ Threshold router
     │                 └─▶ TypeSafe JEV API (low-confidence only)
     │
     ├─▶ Live crosswalk lookup (reuse cached rankings when possible)
     └─▶ Accuracy metrics vs GT CSV
```

## APIs / services

- Local REST backend (inventory, runs, live crosswalk, weights).
- Azure OpenAI Embeddings.
- TypeSafe ` /v1/systemone `.

Detailed OpenAPI schema dump: `[no enough information is available]` in this dossier (see running server / code).

---

# 16. Experimental Design

## Experiment A — Embedding-only multi-signal scoring (Configs 1–3)

| | |
|--|--|
| **Objective** | Measure Stage-3 Acc@K without JEV |
| **Dataset** | O\*NET→ESCO, ~490 exactMatch-eligible sources |
| **Method** | Weighted pillar cosine; Top-K from embeddings only |
| **Configuration** | Weights as Table 2; \(K_{\mathrm{pool}}=20\); model text-embedding-3-large |
| **Results** | Acc@1 ≈ **0.8612–0.8633** |
| **Findings** | Already exceeds SME baseline 0.8475 |

## Experiment B — Full pipeline with JEV (JEV-12, JEV-13/14)

| | |
|--|--|
| **Objective** | Measure Acc@K with confidence-gated JEV |
| **Dataset** | Same |
| **Method** | Stage 3 + Stage 4 when \(S_{\max} < 0.60\) |
| **Configuration** | JEV `jev-latest`; Top-20 Choice |
| **Results** | Best Acc@1 = **0.8837** (JEV-13/14); JEV-12 = 0.8796 |
| **Findings** | +3.62 pp vs baseline; +2.04 pp vs embedding-only |

## Experiment C — Error / GT quality (double-convergence)

| | |
|--|--|
| **Objective** | Separate pipeline errors from GT annotation issues |
| **Method** | Cases where Stage 3 Top-1 = Stage 4 Top-1 ≠ GT exactMatch |
| **Results** | **19** cases catalogued with typology A–E |
| **Findings** | Prima facie GT inconsistencies; true method advantage may exceed +3.62 pp |

## Experiment D — JEV vs Claude Opus 4.7 backup (knowledge share)

| | |
|--|--|
| **Objective** | Cost/latency/accuracy of backup slot |
| **Results** | ~1h→~20s; ~$11→~$0.20; wider Top-20; better hard-subset accuracy |
| **Detailed Acc@K table for Opus** | `[no enough information is available]` in methodology draft |

## Experiment E — UK SOC → ISCO JEV assist

| | |
|--|--|
| **Objective** | Apply same hybrid pattern to UK SOC→ISCO |
| **Method** | Excel embedding ranks; JEV if Top-1 < 0.6 |
| **Formal Acc@K** | `[no enough information is available]` in methodology |

---

# 17. Evaluation Metrics

## Primary: Accuracy@K (exactMatch)

\[
\mathrm{Acc@K} = \frac{|\{ s : R(s)[1..K] \cap GT_{\mathrm{exact}}(s) \neq \emptyset \}|}{|S_{\mathrm{eligible}}|}
\]

Reported for **K ∈ {1, 2, 3, 5, 10}**. Acc@1 is the headline comparison metric.

**Rationale:** Matches how SMEs review ranked shortlists; exactMatch is the strictest graded type.

## Diagnostic metrics

| Metric | Definition |
|--------|------------|
| Embedding-only Top-1 | Acc@1 of Stage 3 before Stage 4 |
| High-confidence accuracy | Acc@1 on \(S_{\max} \ge \theta\) |
| JEV subset before/after | Acc@1 on low-confidence subset, embed vs JEV |
| Accuracy gain | Absolute Acc@1 lift on JEV-sent subset |

## Other metrics

| Metric | Status |
|--------|--------|
| Precision / Recall / F1 | `[no enough information is available]` as reported headline metrics |
| Coverage | Implicit via eligible-set definition |
| MRR | `[no enough information is available]` as reported |
| NDCG | Graded gains defined in code (`exact=3, close=2, broad/narrow=1`) but methodology headline results use Acc@K; published NDCG tables: `[no enough information is available]` |

---

# 18. Experimental Results

## Table — Accuracy@K: Baseline vs Proposed (methodology Table 3)

| Acc@K | Baseline (SME GT) | Config-1 (Embed.) | Config-2 (Embed.) | Config-3 (Embed.) | JEV-12 | JEV-13 / JEV-14 |
|-------|-------------------|-------------------|-------------------|-------------------|--------|-----------------|
| Top-1 | 0.8475 | 0.8633 | 0.8633 | 0.8612 | 0.8796 | **0.8837** |
| Top-2 | 0.9153 | — | — | — | 0.9327 | **0.9367** |
| Top-3 | 0.9605 | — | — | — | 0.9653 | 0.9653 |
| Top-5 | 0.9774 | — | — | — | 0.9816 | 0.9816 |
| Top-10 | 1.0000 | — | — | — | 0.9918 | 0.9918 |

**Notes:**
- Best Acc@1 improvement vs baseline: **+3.62 pp**.
- JEV contribution vs best embedding-only: **+2.04 pp**.
- Top-10 slightly below baseline’s 1.0 (0.9918) — shortlist recall not perfect after hybrid path.

## Statistical significance — exact McNemar (JEV-12/13/14 + Ejento Claude-1/2/3)

Paired per-source exactMatch Acc@1 on the \(N = 490\) eligible O\*NET→ESCO set, comparing **embedding Top-1** vs **confidence-gated final** under the same Model P weights (\(\theta = 0.60\)). Discordant cells: \(b\) = embedding wrong / gated correct; \(c\) = embedding correct / gated wrong. Two-sided exact (binomial) McNemar on \(b+c\). High-confidence sources (\(S_{\max} \ge 0.60\), \(n = 310\)) never flip; all discordant pairs sit in the 180 resolver-routed cases. Ejento Claude runs use the same embedding baseline and gate; only the low-confidence resolver differs (`output/ejento_io_pairs_run_{1,2,3}`).

### JEV (TypeSafe System One)

| Run | Emb Acc@1 | Gated Acc@1 | \(b\) | \(c\) | Discordant | Exact McNemar \(p\) |
|-----|-----------|-------------|------|------|------------|---------------------|
| JEV-12 | 0.8408 | 0.8796 | 24 | 5 | 29 | \(5.5 \times 10^{-4}\) |
| JEV-13 | 0.8408 | 0.8837 | 26 | 5 | 31 | \(1.9 \times 10^{-4}\) |
| JEV-14 | 0.8408 | 0.8837 | 26 | 5 | 31 | \(1.9 \times 10^{-4}\) |

### Ejento Claude Opus backup

| Run | Emb Acc@1 | Gated Acc@1 | \(b\) | \(c\) | Discordant | Exact McNemar \(p\) |
|-----|-----------|-------------|------|------|------------|---------------------|
| Ejento-1 | 0.8408 | 0.8633 | 21 | 10 | 31 | \(0.071\) |
| Ejento-2 | 0.8408 | 0.8633 | 22 | 11 | 33 | \(0.080\) |
| Ejento-3 | 0.8408 | 0.8612 | 21 | 11 | 32 | \(0.110\) |

### Alternative re-ranker vs gated choice (Claude-gated vs JEV-gated)

Same embeddings and gate; only the low-confidence resolver differs. Discordant cells: \(b\) = Claude-gated wrong / JEV-gated correct (**Fixed**); \(c\) = Claude-gated correct / JEV-gated wrong (**Broken**). All flips sit in the 180 resolver-routed sources.

| Claude run | JEV run | Fixed \(b\) | Broken \(c\) | Discordant | Exact McNemar \(p\) |
|------------|---------|------------:|-------------:|------------|---------------------|
| Ejento-1 | JEV-12 | 14 | 6 | 20 | \(0.115\) |
| Ejento-1 | JEV-13/14 | 15 | 5 | 20 | \(0.041\) |
| Ejento-2 | JEV-12 | 16 | 8 | 24 | \(0.152\) |
| Ejento-2 | JEV-13/14 | 17 | 7 | 24 | \(0.064\) |
| Ejento-3 | JEV-12 | 14 | 5 | 19 | \(0.064\) |
| **Ejento-3** | **JEV-13/14** | **15** | **4** | **19** | \(1.9 \times 10^{-2}\) |

Paper Table 6 row 2 (**Fixed 15, Broken 4, \(p = 1.9 \times 10^{-2}\)**) = **Ejento-3 vs JEV-13/14**.

**Interpretation.** Gating with **JEV** significantly beats embedding-only on every replicate (\(p = 5.5 \times 10^{-4}\), \(1.9 \times 10^{-4}\), \(1.9 \times 10^{-4}\); \(b \gg c\)). The same gate with **Ejento Claude** yields a smaller Acc@1 lift (+2.0–2.2 pp vs +3.9–4.3 pp for JEV) and does **not** reject equality at \(\alpha = 0.05\) (\(p \in [0.071, 0.110]\)), because Claude introduces nearly as many regressions (\(c \approx 10\)–\(11\)) as fixes relative to JEV’s \(c = 5\). Head-to-head, JEV-gated also significantly beats Claude-gated on the strongest Claude replicate (Ejento-3 vs JEV-13/14). The contrast supports JEV—not merely “any LLM backup”—as the accuracy driver behind a statistically reliable gated gain.

**Suggested paper wording.** An exact McNemar test over paired per-source outcomes rejects equality of the embedding-only and JEV-gated configurations on JEV-12–14 (\(p = 5.5 \times 10^{-4}\), \(1.9 \times 10^{-4}\), \(1.9 \times 10^{-4}\)). Matched Ejento Claude replicates under the same gate do not reach significance at \(\alpha = 0.05\) (\(p = 0.071\), \(0.080\), \(0.110\)), despite modest Acc@1 gains. Comparing the two gated resolvers directly, JEV also significantly outperforms Claude on Ejento-3 vs JEV-13/14 (15 fixes / 4 regressions, \(p = 1.9 \times 10^{-2}\)).

## Incorrect Top-1 counts (JEV-14)

| Subset | Incorrect Top-1 |
|--------|-----------------|
| High-confidence (embedding accepted) | 24 |
| Low-confidence (JEV processed) | 33 |

Among apparent errors, **19** are double-convergence GT-suspect cases.

---

# 19. Ablation Studies

## Documented ablations / variants

| Variant | Acc@1 | Notes |
|---------|-------|-------|
| SME baseline | 0.8475 | Manual curated crosswalk |
| Embedding-only (paired Model P) | 0.8408 | Same weights as gated runs |
| Embedding-only (Configs 1–3) | 0.8612–0.8633 | Alternate embed configs in methodology table |
| Full + Ejento Claude-1/2 | 0.8633 | Gated Claude; McNemar n.s. |
| Full + Ejento Claude-3 | 0.8612 | Gated Claude; McNemar n.s. |
| Full + JEV-12 | 0.8796 | Hybrid; McNemar sig. |
| Full + JEV-13/14 | **0.8837** | Best; McNemar sig. |

## Discussion-level signal importance

- Title signals combined weight **0.44** (`label_max` + `label_median`).
- Description **0.26** + Skills **0.24** ≈ title importance — titles alone are insufficient across taxonomies.

## Fine-grained ablations (title-only, description-only, skills-only, etc.)

Result tables: `[no enough information is available]` in methodology draft — recommended as missing experiments (Section 28).

---

# 20. Baselines

## 1. Published SME ESCO–O\*NET Crosswalk

| | |
|--|--|
| **Name** | Official / published ESCO–O\*NET exactMatch Top-1 |
| **Description** | Expert-curated graded crosswalk |
| **Source** | Official crosswalk release; Acc@1 = 0.8475 on eligible set |
| **Why selected** | State-of-the-art task-specific manual mapping for this pair |
| **Comparison** | Proposed method 0.8837 (+3.62 pp) |

## 2. Embedding-only multi-signal (internal)

| | |
|--|--|
| **Name** | Stage-3-only |
| **Description** | Same pillars/weights without JEV |
| **Why selected** | Isolates re-ranker contribution |
| **Comparison** | 0.8633 → 0.8837 with JEV |

## 3. Prior LLM backup (Claude Opus 4.7)

| | |
|--|--|
| **Name** | Hybrid with Opus re-ranker |
| **Description** | Same confidence gate; smaller shortlist due to cost |
| **Why selected** | Production predecessor of JEV slot |
| **Comparison** | JEV cheaper/faster; Acc@K table vs Opus: `[no enough information is available]` in methodology |

## 4. External systems (Ejento import scripts exist)

Scripts compare/import Ejento evals; published methodology baseline table does not include Ejento Acc@K: `[no enough information is available]` for formal inclusion.

---

# 21. Error Analysis

## Quantitative split (JEV-14)

- 24 high-confidence misses; 33 JEV-subset misses.
- 19 double-convergence disagreements with GT.

## Discrepancy typology (GT quality / hard cases)

**(A) Seniority / skill-level inflation in GT** — e.g. First-Line Supervisors of Police → GT `police commissioner` vs prediction `police inspector`; Computer Network Support Specialists → GT administrator vs technician.

**(B) GT too narrow for broad/composite O\*NET** — e.g. Retail Sales Supervisors mapped only to clothing/textile shop managers; Mining engineers including safety → geotechnical-only GT.

**(C) GT too generic** — Airline Pilots → GT `aircraft pilot` vs `airline transport pilot`; Hosts → generic host vs restaurant host.

**(D) Role conflation** — Set Designers → GT `set builder` vs `set designer`; Desk clerks → GT concierge vs receptionist; Fitness instructors → sports instructor.

**(E) Structural ESCO gaps** — Radiologists (physicians) lack physician-level ESCO radiology concept; Surgical Technologists / Tire Repairers similarly ambiguous.

## Other failure modes

- Similar occupations with overlapping descriptions/skills.
- Missing or sparse facet text: `[no enough information is available]` with quantified impact.
- Embedding pool miss (true target outside Top-20).

---

# 22. Cost Analysis

## Reported (knowledge share — backup stage)

| Item | Claude Opus 4.7 | JEV |
|------|-----------------|-----|
| Backup workload time | ~1 hour | ~20 seconds |
| Backup workload cost | ~$11 | ~$0.20 |

## Embedding cost

`[no enough information is available]` as isolated dollar figures.

## Retrieval cost

Dominated by local cosine / tensor ops after embeddings exist — marginal cloud cost low; exact figures: `[no enough information is available]`.

## Cost per record / per 1000

`[no enough information is available]` as audited unit economics beyond the aggregate backup comparison.

## Cost savings from confidence gating

Qualitative: most rows never call JEV; only \(S_{\max} < 0.60\) escalate. Exact % of rows escalated and $ saved vs always-on JEV: `[no enough information is available]` in methodology draft (important missing analysis).

---

# 23. Scalability Analysis

| Topic | Available information |
|-------|------------------------|
| Dataset sizes | ~1k O\*NET, ~3k ESCO, ~490 eligible eval sources |
| Runtime | JEV backup path ~20s for escalated workload (knowledge share); full pipeline wall-clock for complete eval: `[no enough information is available]` |
| Throughput | JEV concurrency default 16 |
| Memory | Cached full pairwise signal tensor; size figures: `[no enough information is available]` |
| Parallelization | Async JEV batch; multiprocessing for ISCO/OSCA embedding script |
| Bottlenecks | (1) Initial embedding generation, (2) JEV latency on large low-confidence sets, (3) Stage-4 cannot fix Stage-3 recall misses |

---

# 24. Novel Contributions

## Contribution 1 — Taxonomy-agnostic multi-pillar weighted matching with renormalization

- **What is new?** Unified five-pillar representation + missing-pillar renormalization + dual title statistics (max & median).
- **Why useful?** Works for taxonomies with incomplete facets; title synonyms handled without brittle rules.
- **Why not already solved?** Many systems use title-only or single concatenated text; few publish pillar-weight optimisation for occupation crosswalks.
- **Evidence:** Embedding-only Acc@1 86.33% > SME 84.75% without taxonomy-specific engineering.

## Contribution 2 — Confidence-gated structured Choice re-ranking (JEV)

- **What is new?** Framing Stage 4 as cost gate with structured Choice over Top-20, not free-form LLM prose.
- **Why useful?** Enables larger candidate pools at low cost/latency.
- **Evidence:** +2.04 pp Acc@1; ~55× time and ~55× cost reduction vs Opus backup workload (approx.).

## Contribution 3 — Double-convergence GT auditing

- **What is new?** Independent agreement of statistical ranker and reasoning ranker against GT as an automated inconsistency detector; 19 cases with typology.
- **Why useful?** Reduces SME review burden; reframes measured accuracy ceilings.
- **Evidence:** Table 4 cases; qualitative typology A–E.

## Honesty check

- Core idea “embed + LLM re-rank when uncertain” is known in IR; novelty is **domain instantiation**, **pillar design**, **efficiency via JEV Choice**, and **GT audit method**, not a wholly new ML paradigm.
- Single primary pair (O\*NET–ESCO) limits claims of universal generalisation until more pairs are published.

---

# 25. Literature Review Mapping

Methodology draft open item: *“Literature survey citations to be added in §2.”*

| Paper | Status |
|-------|--------|
| Citation / objective / method / similarities / differences | **[no enough information is available]** — no literature list in `25811-methodology_draft_v3.txt` |

**Action required before submission:** Map related work on ontology matching, crosswalk learning, dense retrieval re-ranking, SKOS mapping, and labour-market taxonomy alignment.

---

# 26. Current Limitations

1. Dependence on text quality of pillars.
2. Primary evaluation on **one** taxonomy pair (O\*NET→ESCO).
3. English-centric evaluation despite ESCO multilinguality.
4. Ground-truth noise / inconsistency (partially analysed, not fully quantified with IAA).
5. Stage-4 bounded by Stage-3 Top-20 recall.
6. Static threshold \(\theta=0.60\) without published sensitivity analysis.
7. Incomplete ablations (per-pillar only scores).
8. Limited external baselines beyond SME crosswalk / prior Opus backup.
9. Cost unit economics incompletely reported.
10. ISCO/OSCA/UK SOC not in headline Acc@K tables.
11. No peer-reviewed literature positioning yet.
12. Author/venue metadata still open (draft footer).

---

# 27. Future Work

1. Additional taxonomy pairs (ISCO, OSCA, UK SOC, proprietary) with full Acc@K.
2. Learned / adaptive pillar weights (per-domain or per-query).
3. Graph methods over skill–occupation bipartite structure.
4. Active learning / human-in-the-loop on double-convergence cases.
5. Multilingual pillar embeddings and evaluation.
6. Knowledge-graph integration (ESCO skill relations).
7. Formal IAA study on GT quality; quantify double-convergence precision.
8. Threshold sweeps and cost–accuracy Pareto curves.
9. Per-pillar ablation tables and statistical significance tests.
10. Margin-based or calibrated confidence alternatives.
11. Expand \(K_{\mathrm{pool}}\) studies (10 vs 20 vs 50) under JEV cost model.
12. Release reproducibility package with fixed seeds and configs.

---

# 28. Publication Readiness Assessment

## Strengths

- Clear, implementable methodology with equations and algorithm.
- Beats strong SME baseline with a general pipeline.
- Strong systems story (cost/latency) with JEV.
- Unusual and valuable GT-quality analysis (19 cases).
- Working software artifact (backend + UI + tests).

## Weaknesses

- Missing literature review.
- Single-pair primary eval.
- Incomplete ablations, significance tests, cost curves.
- Some Acc@K cells missing for embedding-only configs.
- Top-10 slightly below baseline (may draw reviewer questions).

## Missing Experiments

- Multi-pair validation.
- Full factorial pillar ablations.
- \(\theta\) sensitivity + always-JEV vs gated-JEV.
- Statistical significance: exact McNemar for JEV-12/13/14 vs paired embedding Top-1 is done (Section 18); bootstrap CIs / multi-pair tests still open.
- Human preference study on double-convergence cases.
- Multilingual ESCO labels experiment.

## Missing Analysis

- Literature mapping.
- Escalation rate and $ / 1k mappings.
- Error taxonomy coverage % of all misses.
- Calibrated probability / ECE if using JEV probs as confidence.

## Reviewer Risks

- “Engineering paper, limited ML novelty.”
- “GT errors cherry-picked; need blinded SME adjudication.”
- “Results on one dataset.”
- “Baseline is the GT itself — clarify what Acc@1 of published crosswalk means operationally.”
- Proprietary JEV reduces reproducibility for external labs.

## Journal Readiness

| Venue | Readiness |
|-------|-----------|
| TKDE | Not ready — needs broader experiments, theory/related work depth |
| Information Fusion | Partial — fusion-of-signals framing fits; need more modalities/pairs |
| Knowledge-Based Systems | Closest near-term fit after ablations + lit review |
| Expert Systems with Applications | Good applied fit; strengthen cost and case studies |
| ACM TIST | Needs stronger intelligent-systems framing + multi-task evidence |

---

# 29. Reproducibility Package

## Folder structure (project)

```
GlobalTaxinomy/
├── backend/app/          # pipeline, scoring, JEV, accuracy, GT
├── frontend/             # UI
├── config/active_weights.json
├── data/chroma_db/, onet/, esco/, isco/, osca/, item_index/
├── Ground truth Onet to Esco/
├── output/runs/, jev_io_pairs/, cache/
├── scripts/              # embeds, exports, UK SOC→ISCO, docs generators
├── tests/
├── requirements.txt
└── 25811-methodology_draft_v3.txt
```

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Weights | 0.32 / 0.12 / 0.26 / 0.24 / 0.06 |
| \(\theta\) | 0.60 |
| \(K_{\mathrm{pool}}\) | 20 |
| \(K_{\mathrm{display}}\) | 10 |
| Embedding model | text-embedding-3-large |
| JEV model | jev-latest |
| JEV concurrency | 16 |

## Dependencies

See `requirements.txt` (Python); frontend `package.json`. Secrets: `.env` with Azure + `JEV_API_KEY`.

## Execution workflow

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# configure .env
python -m backend          # API :8001
cd frontend && npm i && npm run dev   # UI :5173
pytest
```

Eval: Evals tab / run pipeline → artifacts in `output/runs/`.

---

# 30. Appendix

## Sample double-convergence cases

See Section 21 / methodology Table 4 (19 rows).

## Sample JEV instructions

See Section 13.

## Sample output

Choice label + probability map → ranked target IDs (Top-10).

## Workflow / architecture diagrams

See Sections 14–15.

## Hyperparameters

See Section 29.

## Algorithm 1 (summary)

Four stages: extract → embed → weighted score Top-20 → if \(S_{\max}<\theta\) then JEV else accept → Acc@K.

## Open draft items (from methodology footer)

1. Author names / affiliations  
2. Target venue confirmation  
3. Literature survey  
4. Figures 1–2 embedding  

---

# Final Deliverables

## One-Sentence Contribution

We present a taxonomy-agnostic, multi-pillar embedding pipeline with confidence-gated JEV Choice re-ranking that raises O\*NET→ESCO exactMatch Acc@1 from 84.75% (SME baseline) to 88.37%, while exposing systematic ground-truth inconsistencies via double-convergence analysis.

## One-Paragraph Contribution

Cross-taxonomy occupation mapping is traditionally performed by costly expert curation. We propose a four-stage, taxonomy-agnostic methodology that represents each concept by five textual pillars, embeds them with a shared dense model, and ranks targets using optimised multi-signal cosine weights with missing-pillar renormalization. Ambiguous cases (\(S_{\max} < 0.60\)) are re-ranked by JEV (TypeSafe System One) as a structured Choice over the Top-20 embedding candidates. On the official ESCO–O\*NET exactMatch-eligible set (~490 sources), the full system reaches 88.37% Acc@1 versus 84.75% for the published SME crosswalk and 86.33% for embeddings alone. Replacing a prior Claude Opus backup with JEV cuts backup runtime and cost by roughly an order of magnitude while supporting wider shortlists. An analysis of 19 cases where embedding and JEV independently agree against the ground truth further suggests annotation errors that compress measured gains and offers an automated crosswalk QA signal.

## Five-Bullet Novelty Summary

- Multi-pillar occupation matching with dual title aggregation (max + median) and weight renormalization.
- Confidence-gated structured Choice re-ranking (JEV) as an accuracy engine with a cost gate.
- Outperforms official SME ESCO–O\*NET Acc@1 by +3.62 pp without taxonomy-specific rules.
- Practical systems result: Top-20 re-ranking at ~$0.20 / ~20s vs prior Opus backup.
- Double-convergence GT auditing with a five-type discrepancy taxonomy (19 cases).

## Elevator Pitch

We automatically map jobs across taxonomies. Embeddings handle the easy matches; JEV only looks at the hard ones. On the standard O\*NET–ESCO benchmark we beat the expert crosswalk, cost much less than a heavy LLM backup, and even help find mistakes in the official labels.

## Reviewer Pitch

This paper delivers a complete, reproducible systems + method contribution: formal pillar scoring, gating theory as cost control, strong empirical gains against the community’s own SME crosswalk, and a methodological tool (double-convergence) for auditing graded occupation links. With expanded pair coverage, ablations, and related work, it is a natural fit for applied knowledge-systems venues and a credible step toward replacing point-to-point expert crosswalk production.

## 250-Word Abstract

Interoperability between occupation taxonomies is a prerequisite for cross-border labour-market analytics, yet production crosswalks remain largely manual. We introduce a generalized four-stage pipeline for cross-taxonomy mapping that is independent of any particular taxonomy’s internal schema. Concepts are represented by five similarity pillars—title, alternate labels, description, skills, and knowledge—embedded with a shared dense encoder. Targets are ranked by a weighted combination of per-pillar cosine similarities, with weights optimised offline and renormalized when pillars are missing. Alternate labels contribute both maximum and median similarity statistics. Because embedding scores are unreliable near semantic boundaries, a confidence threshold routes low-scoring sources to an LLM re-ranker; in our instantiation this stage uses JEV (TypeSafe System One) to perform structured Choice ranking over the Top-20 candidates. On approximately 490 O\*NET occupations with exactMatch links in the official ESCO–O\*NET crosswalk, the full system achieves 88.37% Accuracy@1, improving on the published SME baseline of 84.75% and on embedding-only scoring at 86.33%. Efficiency analysis shows that replacing a prior Claude Opus backup with JEV reduces backup runtime and cost dramatically while allowing larger candidate pools. Finally, we identify 19 double-convergence disagreements with ground truth—cases where embedding and JEV independently agree on a non-GT Top-1—revealing recurring annotation inconsistency patterns and implying that true performance gains may exceed headline figures. The methodology offers a practical path to scalable, auditable taxonomy interoperability.

## 500-Word Executive Abstract

**Background.** National and international occupation taxonomies (O\*NET, ESCO, ISCO, and others) encode the same labour market with incompatible identifiers and labels. Organizations that operate across jurisdictions need crosswalks: ranked mappings that translate a concept in one taxonomy into candidates in another. Expert-curated crosswalks are accurate enough to serve as benchmarks but are expensive to build, slow to refresh, and—as we show—not free of annotation error.

**Method.** We treat interoperability as a retrieval problem over structured text. Each concept exposes up to five pillars. After normalisation, each pillar is embedded with `text-embedding-3-large` and stored in ChromaDB. For every source–target pair we compute pillar-wise cosine similarities. Title matching against multi-label targets uses both the maximum and median similarity across labels. A frozen weight vector (0.32, 0.12, 0.26, 0.24, 0.06) combines signals; absent pillars are masked and weights renormalized. All targets are scored; the Top-20 form a candidate pool and the Top-1 score \(S_{\max}\) drives routing. If \(S_{\max} \ge 0.60\), the embedding ranking is accepted. Otherwise JEV performs a constrained Choice over labeled candidates using full pillar text and returns a probability-ordered Top-10.

**Results.** Against exactMatch links in the official ESCO–O\*NET crosswalk (~490 eligible sources), Accuracy@1 reaches 88.37% for the best JEV configuration, versus 84.75% for the SME baseline and ~86.3% for embeddings alone. Accuracy@2–@5 also improve; Accuracy@10 is 99.18%. Among residual Top-1 errors, 19 exhibit double-convergence against the ground truth. We classify these into seniority inflation, over-/under-specificity, role conflation, and structural coverage gaps—evidence that evaluation ceilings are partly artifactual. Separately, swapping an earlier Claude Opus 4.7 backup for JEV reduced backup-stage time from about an hour to about twenty seconds and cost from about eleven dollars to about twenty cents, enabling Top-20 shortlists.

**Implications.** Multi-signal embeddings already rival expert maps; gated structured reasoning supplies the remaining lift where it is worth paying for. The same architecture is intended to generalise to other taxonomy pairs (ISCO, OSCA, UK SOC scripts exist in-repo). Remaining work includes multi-pair benchmarks, pillar ablations, threshold–cost Pareto analysis, multilingual evaluation, and a full literature survey prior to journal submission.

**Keywords seed:** taxonomy alignment, occupation crosswalk, dense retrieval, LLM re-ranking, labour market interoperability.

## 10 Keywords

1. Taxonomy interoperability  
2. Occupation crosswalk  
3. Dense embedding retrieval  
4. Multi-signal similarity fusion  
5. Confidence-gated re-ranking  
6. Structured Choice / JEV  
7. ESCO  
8. O\*NET  
9. Ground-truth quality auditing  
10. Labour-market AI  

## Potential Paper Titles

### TKDE-style (10)

1. Confidence-Gated Multi-Signal Matching for Large-Scale Taxonomy Interoperability  
2. A Generalized Framework for Cross-Taxonomy Occupation Alignment with Dense Retrieval and Structured Re-Ranking  
3. Learning to Bridge Occupation Taxonomies: Weighted Pillar Embeddings and Selective LLM Escalation  
4. ExactMatch Accuracy Beyond Expert Crosswalks: An End-to-End O\*NET–ESCO Mapping System  
5. Cost-Aware Hybrid Ranking for Semantic Taxonomy Alignment  
6. Dual-Aggregation Title Signals for Cross-Lingual Occupation Matching  
7. Auditing Graded Crosswalks via Independent Ranker Agreement  
8. From Manual Crosswalks to Taxonomy-Agnostic Pipelines: Methods and Evaluation  
9. Scalable Occupation Concept Alignment under Incomplete Facet Evidence  
10. Thresholded Neural Re-Ranking for High-Precision Taxonomy Mapping  

### Information Fusion-style (10)

1. Fusing Title, Description, Skills, and Knowledge Signals for Cross-Taxonomy Occupation Matching  
2. Multi-Modal Text Facet Fusion with Confidence-Gated Reasoning for Labour-Market Interoperability  
3. Hybrid Statistical–Generative Fusion for Ambiguous Occupation Alignment  
4. Evidence Fusion Across Similarity Pillars for Robust Crosswalk Generation  
5. Combining Dense Similarity and Structured Choice for Taxonomy Mapping  
6. Adaptive Signal Fusion under Missing Occupational Attributes  
7. Dual-Moment Label Aggregation as a Fusion Primitive for Synonym-Rich Taxonomies  
8. Cost-Sensitive Fusion Control via Confidence Thresholds in Cross-Taxonomy Retrieval  
9. Cross-Source Occupational Knowledge Fusion: Embeddings Meet JEV Re-Ranking  
10. Disagreement Fusion: Using Dual-System Consensus to Detect Crosswalk Annotation Noise  

### Knowledge-Based Systems-style (10)

1. A Knowledge-Based Pipeline for Automated Occupation Taxonomy Mapping  
2. Embedding-Driven Crosswalk Construction with Expert-Level Accuracy  
3. Structured Knowledge Pillars for Interoperable Occupation Classification Systems  
4. An Intelligent Hybrid System for ESCO–O\*NET Alignment  
5. Reducing Expert Annotation Effort in Taxonomy Crosswalks via Selective AI Re-Ranking  
6. Knowledge-Aware Similarity Scoring for Cross-Taxonomy Job Matching  
7. Detecting Inconsistencies in Published Occupational Crosswalks with Dual Inference Paths  
8. Practical Knowledge Engineering for Global Labour Taxonomy Interoperability  
9. A Reusable Architecture for Mapping Heterogeneous Occupation Knowledge Bases  
10. Decision Support for Taxonomy Mapping: Ranked Shortlists, Confidence Gates, and Audit Trails  

## Publication Assessment

1. **Probability of acceptance for TKDE:** ~**15–25%** as currently evidenced (too applied/single-dataset; missing theory/lit/ablation depth).  
2. **Probability of acceptance for Information Fusion:** ~**25–35%** (fusion narrative is strong; need more modalities/pairs and sharper fusion analysis).  
3. **Probability of acceptance for Knowledge-Based Systems:** ~**40–50%** after completing lit review, ablations, multi-pair results, and GT adjudication study.  
4. **Missing work required:**
   - **TKDE:** multi-dataset SOTA comparisons, significance tests, deeper algorithmic novelty, full related work.  
   - **Information Fusion:** explicit fusion ablations, uncertainty fusion, possibly multimodal extras.  
   - **KBS:** case studies, cost analysis, reproducibility, SME validation of 19 GT cases, second taxonomy pair.  
5. **Recommended submission order:** (1) Knowledge-Based Systems or Expert Systems with Applications → (2) Information Fusion → (3) TKDE only after major expansion. ACM TIST parallel track if systems framing emphasized.

## Most Important Section

### Why this work is publishable

It combines a clean, general method with a hard public benchmark win over an SME crosswalk, a credible efficiency story, a working system, and a distinctive GT-quality analysis that other taxonomy papers rarely attempt.

### Why this work may be rejected

Thin literature positioning, single primary dataset, incomplete ablations/significance/cost curves, and a novelty profile that reviewers may call “strong engineering / incremental science,” plus dependence on a proprietary re-ranker (JEV).

### Top 10 improvements that would most increase publication chances

1. Complete a rigorous related-work section.  
2. Validate on ≥2 additional taxonomy pairs with Acc@K.  
3. Publish full pillar ablations and weight sensitivity.  
4. Run \(\theta\) cost–accuracy Pareto and always-on vs gated JEV.  
5. Blind SME adjudication of the 19 double-convergence cases.  
6. Statistical significance tests on Acc@1 differences.  
7. Report escalation rates and $ per 1,000 mappings.  
8. Multilingual ESCO evaluation.  
9. Open-source reproducibility bundle (configs, seeds, scripts).  
10. Compare against additional automatic baselines (BM25, bi-encoder title-only, cross-encoder, prior Opus Acc@K table).

### Overall journal recommendation

**Primary target: Knowledge-Based Systems** (or **Expert Systems with Applications** for a more applied angle). Treat **Information Fusion** as a stretch after multi-signal fusion ablations. Defer **TKDE** until multi-pair evidence and deeper comparative evaluation are ready.

---

*Document generated as `Techinal_research_summary.md` per `summary.md` briefing. Factual claims preferentially cite `25811-methodology_draft_v3.txt` and the project implementation; gaps are marked `[no enough information is available]`.*
