# Taxonomy Crosswalk 4.0

Self-contained O\*NET → ESCO crosswalk using **Model P** embeddings and **TypeSafe Jev** for low-confidence rows (similarity &lt; 0.60). Eval runs report **Accuracy@1 / @2 / @3 / @5 / @10**.

This project does **not** depend on the 3.0 folder at runtime. All data and code live under this directory.

## Features

- **Crosswalk** tab: live Top-20 → Top-10 (embedding if Top-1 ≥ 0.60; else Jev, reusing prior rankings when available)
- **Evals** tab: configure weights/threshold, run hybrid pipeline, view Accuracy@K
- Jev Choice ranking over Top-20 candidates when Top-1 score &lt; threshold (default 0.60); ordered Top-10 used for Acc@K and live display
- No Embeddings / Optimize tabs

## Setup

```bash
cd TaxonomyCrosswalk-4.0
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

Copy `.env.example` → `.env` and set `JEV_API_KEY`.
Default API: `https://api.typesafe.ai/v1/systemone` ([TypeSafe docs](https://docs.typesafe.ai/api)).

Required local data (already bootstrapped if you followed the scaffold):

- `data/chroma_db/`
- `data/onet/`, `data/esco/`, `data/item_index/`
- `Ground truth Onet to Esco/*.csv`
- Optional: `output/cache/P_tensor.npz` (speeds first eval)

## Run

Backend (default port **8001**):

```bash
python -m backend
```

Frontend:

```bash
cd frontend
npm run dev
```

Open http://localhost:5173

## Tests

```bash
pytest
```

Default tests mock Jev HTTP (no network). Optional live checks:

```bash
python scripts/check_typesafe_connect.py
python scripts/jev_smoke_test.py
```

## Pipeline

1. Verify Chroma embeddings  
2. Model P similarities (cached tensor under `output/cache/`)  
3. Select Top-20  
4. Apply threshold (default 0.60)  
5. Jev Choice ranking for rows below threshold (probabilities → ordered Top-10)  
6. Accuracy@1 / @2 / @3 / @5 / @10  
7. Save run under `output/runs/`

Jev I/O audit: `output/jev_io_pairs/run_N/`
