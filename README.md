# GridOps Copilot

GridOps Copilot is an AMD-oriented decision-support backend for EHV substation
maintenance. It validates DCRM and FRA diagnostic CSVs, combines waveform evidence
with SCADA events and maintenance history, calculates a transparent risk score, and
produces an engineer-facing Markdown report.

It is aimed at AMD Developer Hackathon ACT II **Track 3 — Unicorn Track**: a
containerized, market-oriented maintenance copilot with visible AMD/ROCm evidence.

The MVP uses a training-first AMD strategy: train and benchmark the FRA 1D CNN in an
AMD GPU Jupyter environment with ROCm PyTorch, export `models/fra_cnn_rocm.pt`, and
run the containerized application on ordinary CPU infrastructure. Fireworks AI can
generate reports from structured evidence, with a deterministic offline fallback.

> This system is a decision-support prototype for hackathon demonstration. It does
> not control grid equipment, does not replace certified engineering analysis, and
> does not provide final fault certification. All high-risk, critical, or
> low-confidence findings require confirmation by a qualified human engineer before
> action.

## Architecture

```text
CSV upload -> ingestion/schema validation -> asset mapping
                                      |
                  +-------------------+-------------------+
                  |                                       |
             DCRM agent                         FRA artifact agent
       resistance/travel/coil             AMD-trained CNN or rule fallback
                  +-------------------+-------------------+
                                      |
                 SCADA + maintenance context -> risk scoring
                                      |
                    persisted digital-twin health state
                                      |
          deterministic report + optional Fireworks narrative
```

The live application performs CPU inference. AMD GPU time is reserved for training
and benchmarking, which keeps the demo reliable and preserves cloud credits.

## Backend quick start

Python 3.11–3.13 is supported.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
uvicorn app.main:app --reload
```

Open the API documentation at `http://localhost:8000/docs` and health check at
`http://localhost:8000/api/health`.

To enable `.pt` artifact inference outside Docker, install the optional ML dependency:

```bash
cd backend
python -m pip install -e ".[dev,ml]"
```

If PyTorch, the artifact, or the label map is unavailable, FRA diagnosis safely uses
the deterministic rule-based fallback.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

Compose installs CPU PyTorch by default so it can load an AMD-trained state dict
without a live GPU. For a smaller fallback-only image:

```bash
INSTALL_ML=false docker compose up --build
```

The container mounts `data/`, `models/`, and `reports/`, so uploads, generated reports,
the model artifact, and AMD evidence remain outside the image.

## API workflow

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service readiness |
| `GET` | `/api/assets` | Dashboard asset summaries |
| `POST` | `/api/upload` | Validate and persist a CSV upload |
| `POST` | `/api/diagnose` | Run FRA/DCRM agents and persist the diagnosis |
| `POST` | `/api/reports/generate` | Generate a report from a diagnosis ID |
| `GET` | `/api/runtime/amd-evidence` | Read notebook-exported AMD evidence |

Example DCRM flow:

```bash
curl -sS -X POST http://localhost:8000/api/upload \
  -F file=@data/synthetic/dcrm_fault_contact_wear.csv \
  -F file_type=dcrm

curl -sS -X POST http://localhost:8000/api/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"asset_id":"CB-402","diagnostic_type":"dcrm","upload_id":"UPLOAD_ID"}'

curl -sS -X POST http://localhost:8000/api/reports/generate \
  -H 'Content-Type: application/json' \
  -d '{"diagnosis_id":"DIAGNOSIS_ID"}'
```

Replace `UPLOAD_ID` and `DIAGNOSIS_ID` with IDs from the preceding responses. The
same flow works for `TX-1` with `fra_fault_winding_shift.csv` and `diagnostic_type=fra`.

`POST /api/diagnose` also accepts optional `scada_upload_id`,
`maintenance_upload_id`, and `assets_upload_id` fields. When supplied, those
validated uploads replace the canonical context for that diagnosis. Persisted
diagnoses are overlaid on `/api/assets`, where `health_score = 100 - risk_score`, so
the asset list acts as the MVP digital-twin state.

## AMD ROCm notebook workflow

1. Upload or clone this repository into AMD AI Notebooks / AMD Developer Cloud.
2. Open `notebooks/fra_training_rocm.ipynb` and select the ROCm PyTorch GPU kernel.
3. Run all cells and confirm `torch.cuda.is_available()` is true. ROCm exposes AMD
   GPUs through PyTorch's `cuda` API.
4. Download or commit the generated artifacts before stopping the GPU session:

   - `models/fra_cnn_rocm.pt`
   - `models/fra_label_map.json`
   - `reports/amd_training_evidence.json`
   - `reports/amd_benchmark.md`
   - `plots/fra_training_curve.png`
   - `plots/fra_confusion_matrix.png`

5. Stop the notebook/GPU resource so credits are not consumed while idle.

The evidence endpoint reports `pending` until the notebook output exists. A CPU-only
notebook run is explicitly marked incomplete and is never presented as AMD GPU proof.

## Fireworks report generation

Copy `.env.example` to `.env`, then set:

```dotenv
USE_FIREWORKS=true
FIREWORKS_API_KEY=your_key
FIREWORKS_MODEL=model-id-from-your-fireworks-dashboard
```

Do not hardcode an unverified model ID. If Fireworks is disabled, misconfigured, or
temporarily unavailable, report generation returns the deterministic Markdown
template instead of breaking the demo. Evidence, risk, and recommended-action
sections always remain deterministic. Fireworks is confined to a labeled
supplemental narrative and unsafe control language is discarded.

## Data and persistence

- Canonical demo inputs live in `data/synthetic/`.
- Validated uploads and diagnoses are written below `data/runtime/`.
- Generated Markdown/JSON reports are written below `reports/generated/`.
- Runtime data and handoff notes are intentionally excluded from Git.

Regenerate all canonical synthetic inputs deterministically with:

```bash
cd backend
python -m app.services.synthetic_data
```

The generated data is demo-only and not field validated. FRA labels cover healthy,
winding deformation, core/clamping, insulation-related abnormality, and human-review
cases. DCRM data covers healthy, contact-wear, and mechanism-delay cases.

## Submission checklist

- Run the full backend test/lint commands below.
- Run the ROCm notebook on an actual AMD GPU; retain its artifact, evidence JSON,
  benchmark report, plots, terminal output, and screenshots.
- Set a real Fireworks model ID from the hackathon account, or demonstrate the
  deterministic fallback.
- Run `docker compose up --build` on a fresh checkout and demo CB-402 plus TX-1.
- Confirm `/api/runtime/amd-evidence` reports verified HIP/ROCm evidence rather than
  `pending` or `incomplete`.
- Keep the exact decision-support disclaimer visible in the UI/video/report.
- Stop the AMD notebook/GPU resource after downloading every submission artifact.

Run backend quality checks with:

```bash
cd backend
ruff check app tests
pytest -q
```

The full project requirements and AMD rationale are documented in
`GridOps_Copilot_AMD_ACTII_Codex_Guide.md`.
