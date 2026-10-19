"""Generate O*NET → ESCO methodology Word document for Taxonomy Crosswalk 4.0."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "ONET_to_ESCO_Methodology.docx"


def _set_run_font(run, *, size: int | None = None, bold: bool = False, color: RGBColor | None = None):
    run.bold = bold
    if size is not None:
        run.font.size = Pt(size)
    run.font.name = "Calibri"
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), "Calibri")
    rFonts.set(qn("w:hAnsi"), "Calibri")
    if color is not None:
        run.font.color.rgb = color


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        _set_run_font(run, bold=True)


def add_para(doc: Document, text: str, *, bold: bool = False, space_after: int = 8) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, size=11, bold=bold)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)


def add_bullets(doc: Document, items: list[str], *, level: int = 0) -> None:
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        if level:
            p.paragraph_format.left_indent = Inches(0.25 * level)
        for run in p.runs:
            _set_run_font(run, size=11)


def add_numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(item, style="List Number")
        for run in p.runs:
            _set_run_font(run, size=11)


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for run in p.runs:
                _set_run_font(run, size=10, bold=True)
    for r_i, row in enumerate(rows):
        cells = table.rows[r_i + 1].cells
        for c_i, val in enumerate(row):
            cells[c_i].text = val
            for p in cells[c_i].paragraphs:
                for run in p.runs:
                    _set_run_font(run, size=10)
    doc.add_paragraph()


def build_document() -> Document:
    doc = Document()

    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("O*NET → ESCO Crosswalk Methodology")
    _set_run_font(run, size=22, bold=True, color=RGBColor(0x1F, 0x4E, 0x79))

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Taxonomy Crosswalk 4.0 — Model P + TypeSafe Jev Hybrid Pipeline")
    _set_run_font(run, size=12, color=RGBColor(0x5B, 0x5B, 0x5B))

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run(f"Document date: {date.today().isoformat()}")
    _set_run_font(run, size=10, color=RGBColor(0x5B, 0x5B, 0x5B))
    doc.add_paragraph()

    # 1. Overview
    add_heading(doc, "1. Overview", 1)
    add_para(
        doc,
        "This document describes the production methodology used in Taxonomy Crosswalk 4.0 "
        "to map O*NET occupations to ESCO occupations. The approach is a hybrid of "
        "embedding-based similarity ranking (Model P) and selective TypeSafe Jev Choice "
        "re-ranking for low-confidence cases.",
    )
    add_para(doc, "Design goals:", bold=True)
    add_bullets(
        doc,
        [
            "Produce a ranked list of ESCO candidates for each O*NET occupation.",
            "Prefer high-confidence embedding matches when Top-1 similarity is strong.",
            "Escalate ambiguous cases (Top-1 similarity below threshold) to Jev for Choice ranking.",
            "Evaluate results against official ESCO–O*NET ground truth using Accuracy@K on exactMatch links.",
        ],
    )

    # 2. Data inputs
    add_heading(doc, "2. Data Inputs", 1)
    add_heading(doc, "2.1 Occupation content facets", 2)
    add_para(
        doc,
        "Each occupation in both taxonomies is represented by structured text facets that are "
        "embedded and stored in ChromaDB:",
    )
    add_table(
        doc,
        ["Facet", "O*NET", "ESCO", "Role in matching"],
        [
            [
                "Preferred title",
                "Yes",
                "Yes",
                "Primary title signal; ESCO also uses alternate labels",
            ],
            [
                "Alternate titles / labels",
                "Limited",
                "Yes (esco_label collection)",
                "label_max / label_median signals over preferred + alternate labels",
            ],
            [
                "Description",
                "Yes",
                "Yes",
                "Pooled description cosine similarity",
            ],
            [
                "Skills",
                "Yes",
                "Yes",
                "Pooled skills cosine similarity",
            ],
            [
                "Knowledge",
                "Yes",
                "Yes",
                "Pooled knowledge cosine similarity",
            ],
        ],
    )
    add_para(
        doc,
        "Local data layout: data/onet/, data/esco/, data/item_index/, and embeddings in "
        "data/chroma_db/ (collections for onet/esco facet embeddings and esco_label).",
    )

    add_heading(doc, "2.2 Ground truth", 2)
    add_para(
        doc,
        "Official ESCO–O*NET crosswalk CSV is used for evaluation "
        "(Ground truth Onet to Esco/ONET_(Occupations)_0_updated.csv). "
        "Each O*NET→ESCO link has a graded match type:",
    )
    add_bullets(
        doc,
        [
            "exactMatch — primary success criterion for Accuracy@K",
            "closeMatch",
            "broadMatch",
            "narrowMatch",
        ],
    )
    add_para(
        doc,
        "Accuracy metrics use only O*NET sources that have at least one exactMatch target "
        "(approximately 490 occupations). Sources with only close/broad/narrow links are "
        "excluded from the Accuracy@K denominator.",
    )

    # 3. Embeddings
    add_heading(doc, "3. Embeddings", 1)
    add_para(
        doc,
        "Facet texts are embedded with Azure OpenAI text-embedding models "
        "(configured deployment; typically text-embedding-3-small or text-embedding-3-large). "
        "Vectors are persisted in Chroma and reused across live lookups and batch evaluation runs. "
        "The pipeline verifies that O*NET, ESCO, and esco_label collections are present before scoring.",
    )

    # 4. Model P
    add_heading(doc, "4. Model P Scoring", 1)
    add_para(
        doc,
        "Model P is the sole production embedding/scoring model. For each O*NET source occupation, "
        "it scores every ESCO target occupation using five weighted signals, then combines them "
        "into a single similarity score.",
    )

    add_heading(doc, "4.1 Signals", 2)
    add_table(
        doc,
        ["Signal", "Default weight", "How it is computed"],
        [
            [
                "label_max",
                "0.32",
                "Max cosine similarity between the O*NET title embedding and all ESCO labels "
                "(preferred + alternate) for a target occupation",
            ],
            [
                "label_median",
                "0.12",
                "Median cosine similarity over the same ESCO label set",
            ],
            [
                "description_pool",
                "0.26",
                "Cosine similarity of description embeddings (O*NET ↔ ESCO)",
            ],
            [
                "skills_pool",
                "0.24",
                "Cosine similarity of skills embeddings (O*NET ↔ ESCO)",
            ],
            [
                "knowledge_pool",
                "0.06",
                "Cosine similarity of knowledge embeddings (O*NET ↔ ESCO)",
            ],
        ],
    )
    add_para(
        doc,
        "Weights sum to 1.0. Production defaults (Optimized-P) can be overridden via "
        "config/active_weights.json. Missing facets are masked and remaining weights are "
        "renormalized so occupations without every facet still receive a fair combined score.",
    )

    add_heading(doc, "4.2 Combined score and candidate selection", 2)
    add_numbered(
        doc,
        [
            "Build or load a cached signal tensor (output/cache/) covering all sources × all targets × signals.",
            "Compute the weighted combined similarity matrix.",
            "For each O*NET occupation, select the Top-20 ESCO candidates by combined score.",
            "Retain Top-1 similarity for the confidence threshold decision.",
        ],
    )

    # 5. Hybrid threshold + Jev
    add_heading(doc, "5. Hybrid Resolution (Threshold + Jev)", 1)
    add_para(
        doc,
        "Not every occupation relies on embeddings alone. A confidence threshold (default 0.60) "
        "gates whether Model P’s ranking is accepted or Jev re-ranks the Top-20 pool.",
    )

    add_heading(doc, "5.1 Decision rule", 2)
    add_bullets(
        doc,
        [
            "If Top-1 Model P similarity ≥ 0.60 → accept embedding order; return Top-10 for display / Acc@K.",
            "If Top-1 Model P similarity < 0.60 → send the Top-20 candidates to TypeSafe Jev Choice ranking "
            "(or reuse a prior Jev ranking from live cache / completed eval runs when available).",
        ],
    )
    add_para(
        doc,
        "In batch evaluation runs, Jev is applied only to sources that have at least one exactMatch "
        "ground-truth target and whose Top-1 score is below the threshold. Live crosswalk applies the "
        "same threshold logic per queried occupation.",
    )

    add_heading(doc, "5.2 Jev Choice ranking", 2)
    add_para(
        doc,
        "Low-confidence rows are submitted to the TypeSafe System One API "
        "(POST https://api.typesafe.ai/v1/systemone) using model jev-latest. "
        "The request is a Choice question over up to 20 labeled candidates (A–T).",
    )
    add_para(doc, "Input occupation and each candidate include:", bold=True)
    add_bullets(
        doc,
        [
            "Job title (preferred)",
            "Alternate title(s)",
            "Job description (full text)",
            "Skills (full text)",
            "Knowledge (full text)",
        ],
    )
    add_para(
        doc,
        "Jev is instructed to pick the closest overall match using only the supplied fields "
        "(no web search). The API returns a chosen label plus per-option probabilities. "
        "Candidates are sorted by descending probability to produce an ordered ranking; "
        "the Top-10 of that ranking is used for Accuracy@K and live UI display. "
        "Jev I/O pairs are audited under output/jev_io_pairs/run_N/.",
    )

    # 6. Pipeline phases
    add_heading(doc, "6. End-to-End Eval Pipeline Phases", 1)
    add_para(doc, "Batch evaluation follows seven phases:")
    add_numbered(
        doc,
        [
            "Generate / verify embeddings — confirm Chroma inventory for O*NET, ESCO, and labels.",
            "Calculate similarities — build or load Model P signal tensor.",
            "Select top candidates — Top-20 per source from weighted scores.",
            "Apply threshold — flag exact-GT rows with Top-1 < threshold for Jev.",
            "Run Jev evaluation — Choice ranking over Top-20; write ordered Top-10.",
            "Calculate Accuracy@K — Acc@1 / @2 / @3 / @5 / @10 on exactMatch gold.",
            "Save run — persist results and metrics under output/runs/.",
        ],
    )

    # 7. Live crosswalk
    add_heading(doc, "7. Live Crosswalk Flow", 1)
    add_para(
        doc,
        "The interactive Crosswalk tab mirrors the same methodology for a single occupation:",
    )
    add_numbered(
        doc,
        [
            "Score the full ESCO pool with Model P; take Top-20.",
            "If Top-1 ≥ threshold → return embedding Top-10.",
            "If Top-1 < threshold → reuse a cached Jev ranking when available; otherwise call Jev and cache the result.",
            "Return ordered Top-10 with facet/signal diagnostics for UI inspection.",
        ],
    )

    # 8. Evaluation
    add_heading(doc, "8. Evaluation Methodology", 1)
    add_heading(doc, "8.1 Accuracy@K", 2)
    add_para(
        doc,
        "For each eligible O*NET source (has ≥1 exactMatch gold target), the system produces a "
        "final ranked list of up to 10 ESCO IDs:",
    )
    add_bullets(
        doc,
        [
            "High-confidence rows: embedding candidate order.",
            "Jev rows: jev_ranked_target_ids from Choice probabilities.",
        ],
    )
    add_para(
        doc,
        "Accuracy@K is the fraction of eligible sources for which at least one exactMatch "
        "gold target appears in the first K positions of that ranked list. Reported K values: "
        "1, 2, 3, 5, and 10.",
    )

    add_heading(doc, "8.2 Additional reported metrics", 2)
    add_bullets(
        doc,
        [
            "Embedding Top-1 accuracy — exactMatch hit rate of Model P Top-1 before Jev.",
            "High-confidence accuracy — Acc@1 among rows that did not go to Jev.",
            "Jev subset before/after — exactMatch Acc@1 on the Jev-sent subset, comparing embedding Top-1 vs Jev final pick.",
            "Accuracy gain — improvement on the Jev subset after re-ranking.",
        ],
    )

    # 9. Architecture summary
    add_heading(doc, "9. Architecture Summary", 1)
    add_table(
        doc,
        ["Component", "Role"],
        [
            ["ChromaDB (data/chroma_db)", "Stores facet and label embeddings"],
            ["Model P signal tensor", "Full-pool multi-signal cosine scoring with cached NPZ"],
            ["Weight config (active_weights.json)", "Production signal weights (sum to 1.0)"],
            ["Threshold (default 0.60)", "Routes low-confidence rows to Jev"],
            ["TypeSafe Jev (System One)", "Choice re-ranking of Top-20 → ordered Top-10"],
            ["Ground truth CSV", "exactMatch / close / broad / narrow evaluation labels"],
            ["Run artifacts (output/runs)", "Batch results, metrics, reproducibility"],
        ],
    )

    # 10. Configuration defaults
    add_heading(doc, "10. Default Configuration", 1)
    add_table(
        doc,
        ["Parameter", "Default"],
        [
            ["Direction", "onet_to_esco"],
            ["Model", "P"],
            ["Top-K candidate pool", "20"],
            ["Displayed / Acc ranked depth", "10"],
            ["Confidence threshold", "0.60"],
            ["label_max weight", "0.32"],
            ["label_median weight", "0.12"],
            ["description_pool weight", "0.26"],
            ["skills_pool weight", "0.24"],
            ["knowledge_pool weight", "0.06"],
            ["Jev model", "jev-latest"],
            ["Jev concurrency", "16 (configurable)"],
        ],
    )

    # 11. Limitations
    add_heading(doc, "11. Scope and Limitations", 1)
    add_bullets(
        doc,
        [
            "Primary production path documented here is O*NET → ESCO; the codebase retains "
            "direction plumbing for related experiments but Model P pair defaults target this direction.",
            "Accuracy@K focuses on exactMatch links only; close/broad/narrow matches are available "
            "in ground truth for graded analysis but are not the primary Acc@K hit criterion.",
            "Jev quality depends on candidate pool quality from Model P — if the true match is "
            "outside Top-20, re-ranking cannot recover it.",
            "Embedding quality depends on facet text completeness and the chosen Azure embedding deployment.",
        ],
    )

    add_heading(doc, "12. References (in-repo)", 1)
    add_bullets(
        doc,
        [
            "README.md — setup, run instructions, pipeline overview",
            "backend/app/pipeline.py — seven-phase batch pipeline",
            "backend/app/live_crosswalk.py — live Top-20 → Top-10 hybrid flow",
            "backend/app/variants.py / taxonomies.py / weights.py — Model P signals and weights",
            "backend/app/jev_resolve.py — TypeSafe Choice payload and ranking",
            "backend/app/accuracy.py — Accuracy@K computation",
            "backend/app/ground_truth.py — official crosswalk CSV parsing",
        ],
    )

    return doc


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = build_document()
    doc.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
