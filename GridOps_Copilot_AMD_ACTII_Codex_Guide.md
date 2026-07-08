# GridOps Copilot — AMD Developer Hackathon ACT II Codex Build Guide

**Project:** GridOps Copilot  
**Hackathon:** AMD Developer Hackathon ACT II by lablab.ai  
**Recommended track:** Track 3 — Unicorn Track  
**Primary strategy:** Use AMD GPU credits for training + benchmarking in web-based Jupyter notebooks, then ship a containerized app that loads the trained model artifact.  
**LLM strategy:** Use Fireworks AI for engineer-facing report generation and optional Gemma usage.  
**Project positioning:** Decision-support system for EHV substation maintenance, not an autonomous grid controller.

---

## 0. Why this guide exists

This file is meant to be read by both:

1. **You / the human builder** — to understand how AMD Developer Cloud, AMD AI Notebooks, ROCm, Fireworks AI, and Docker fit into the project.
2. **ChatGPT Codex / coding agent** — to implement the repository structure, backend, frontend, notebooks, agents, API routes, and demo assets.

The key idea is simple:

> Do not make the live app depend on always-on AMD GPU hosting. Use AMD GPU where it matters most: training, fine-tuning, benchmarking, and producing evidence. Then export the trained model and integrate it into a normal FastAPI + Next.js app.

---

## 1. Official hackathon facts to keep in mind

As of the checked ACT II materials:

- The hackathon is centered on **AI agents** and high-performance AI apps using **AMD Developer Cloud, ROCm, Fireworks AI API, and open-source AI frameworks**.
- New AMD AI Developer Program sign-ups can receive **$100 AMD Developer Cloud GPU credits**.
- New sign-ups can also receive **$50 Fireworks AI API credits**.
- The ACT II page also says all participants receive hackathon Fireworks credits, with additional infrastructure details shared through Discord.
- **All submissions must be containerized.**
- Track 3 — Unicorn Track is judged on creativity/originality, product or market potential, completeness, and meaningful use of AMD platforms.
- AMD AI Notebooks provide **GPU-powered Jupyter notebooks**, **AMD ROCm**, and **JupyterLab** access.
- AMD Developer Cloud credits can be used for training, fine-tuning, benchmarking, inference, and custom GPU workloads.
- Fireworks AI is allowed and provides managed LLM inference backed by AMD Instinct GPUs.

Important: always re-check the event dashboard and Discord for the latest event-specific access details before submission.

---

## 2. Final project summary

**GridOps Copilot** is an agentic AI digital twin and fault diagnosis assistant for EHV substations.

It ingests:

- DCRM waveform data for circuit breakers
- FRA frequency-response curves for transformers
- Simulated SCADA/event data
- Asset metadata
- Maintenance history
- Baseline/reference diagnostic data

It produces:

- Asset health score
- Risk level
- Detected anomaly
- Likely fault category
- Confidence score
- Evidence summary
- Recommended maintenance action
- Digital twin dashboard state
- Engineer-ready report

The project must always use conservative language:

- “possible fault”
- “suspected issue”
- “requires engineer confirmation”
- “recommend inspection”
- “do not take automated control action”

---

## 3. AMD-specific project strategy

### Recommended strategy

Use AMD resources in three visible ways:

```text
1. AMD AI Notebooks / AMD Developer Cloud
   Use web-based GPU Jupyter notebooks to train and benchmark a small FRA diagnostic model.

2. ROCm PyTorch
   Train the FRA model using PyTorch on AMD ROCm.

3. Fireworks AI
   Use Fireworks AI for the Report Agent and optional Gemma-powered explanation/recommendation layer.
```

### Why this is the best strategy

This project does not need an AMD GPU running forever. EHV diagnosis in the MVP can run with a small exported model artifact.

A strong engineering decision is:

```text
Train on AMD GPU -> save model artifact -> run app with CPU fallback -> use Fireworks AI for reports
```

This gives you credible AMD usage without making the live demo fragile.

---

## 4. What AMD GPU should do

The AMD GPU should be used for the most compute-heavy project component:

> Training and benchmarking a 1D CNN model for FRA curve classification.

### FRA classes

Use the following MVP taxonomy:

```text
0: healthy
1: winding_deformation_suspected
2: core_clamping_issue_suspected
3: insulation_related_abnormality_suspected
4: needs_human_review
```

### Model

```text
Model type: 1D CNN
Input: FRA curve sequence
Input channels: magnitude_db, phase_deg
Optional input: log_frequency_hz as a feature channel
Output: fault class + confidence
Training platform: AMD AI Notebooks / AMD Developer Cloud
Runtime: ROCm PyTorch
Saved artifact: models/fra_cnn_rocm.pt
```

### What the trained model does in the app

The FastAPI backend loads `models/fra_cnn_rocm.pt` and uses it when a user uploads an FRA file.

The model returns structured evidence:

```json
{
  "asset_id": "TX-1",
  "asset_type": "Transformer",
  "fault_class": "winding_deformation_suspected",
  "confidence": 0.82,
  "anomaly_score": 0.71,
  "evidence": [
    "Mid-frequency FRA magnitude deviation exceeds synthetic baseline tolerance.",
    "Curve shift is strongest between 10 kHz and 100 kHz.",
    "Model confidence is above the medium-risk threshold."
  ],
  "requires_human_review": true
}
```

---

## 5. How to claim and use AMD credits

### Step 1 — Register for ACT II

Register on the lablab.ai ACT II page and join the event dashboard.

### Step 2 — Join AMD AI Developer Program

Create or log into your AMD account and join the AMD AI Developer Program.

### Step 3 — Request credits

Go to the AMD Developer Portal / Member Perks page and request cloud credits.

Recommended selection for this project:

```text
Choose AMD Developer Cloud credit / Direct GPU Access first.
```

Reason:

```text
You need real AMD GPU access for ROCm PyTorch training and benchmarking.
```

Fireworks AI is still allowed and useful. You may receive hackathon Fireworks credits separately, and new-member Fireworks credits may also be available depending on the onboarding path.

### Step 4 — Wait for approval

Approval may be manual. Plan for a few business days if the page says so.

### Step 5 — Use the web-based GPU notebooks

Use AMD AI Notebooks or the JupyterLab environment provided through AMD Developer Cloud.

The human should run the training notebook. Codex should only generate the notebook and code.

### Step 6 — Save and download artifacts

Before ending the GPU session, download or commit:

```text
models/fra_cnn_rocm.pt
models/fra_label_map.json
reports/amd_training_evidence.json
reports/amd_benchmark.md
notebooks/fra_training_rocm.ipynb
plots/fra_training_curve.png
plots/fra_confusion_matrix.png
```

### Step 7 — Stop spending credits

If using an environment that creates a VM or GPU instance, destroy/terminate it according to the provider instructions. Do not merely power it off if the provider says powered-off instances still consume credit.

If using AMD AI Notebooks, end/stop the session as instructed by that environment.

---

## 6. Important credit safety rules

Follow these rules to avoid wasting the $100 credit:

```text
Use the smallest available single-GPU environment.
Do not train a large LLM.
Do not leave the GPU session idle.
Do not keep the GPU running for the live demo unless absolutely required.
Always save artifacts before ending the session.
If a VM/Droplet is created, destroy it when done.
Keep screenshots/logs proving AMD usage.
```

Recommended usage pattern:

```text
Session 1: verify ROCm + run smoke test
Session 2: train FRA model
Session 3: benchmark CPU vs AMD GPU and export evidence
```

---

## 7. Fireworks AI usage

Fireworks AI should be used for language-generation tasks, not for raw signal classification.

Use Fireworks for:

```text
Report Agent
Recommendation Agent
Explanation Agent
Optional Gemma-powered diagnostic Q&A
```

Do not use Fireworks to invent evidence. Only pass structured evidence produced by your agents.

### Environment variables

```bash
FIREWORKS_API_KEY=your_key_here
FIREWORKS_MODEL=your_selected_fireworks_model_id
USE_FIREWORKS=true
```

Do not hardcode a model ID unless you confirm it in the Fireworks dashboard. For Gemma prize attempts, set `FIREWORKS_MODEL` to the available Gemma model shown in your Fireworks account or hackathon instructions.

### Fireworks code pattern

Use OpenAI-compatible calling style:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url="https://api.fireworks.ai/inference/v1"
)

response = client.chat.completions.create(
    model=os.environ.get("FIREWORKS_MODEL"),
    messages=[
        {
            "role": "system",
            "content": (
                "You are GridOps Copilot's report agent. Generate conservative, "
                "evidence-backed maintenance recommendations. Never claim certified diagnosis. "
                "Always require human engineer confirmation for high-risk or uncertain cases."
            ),
        },
        {
            "role": "user",
            "content": "Generate a diagnostic report from this structured evidence: ...",
        },
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)
```

### Fireworks fallback

If `FIREWORKS_API_KEY` is missing, the app should still work by generating a deterministic template report.

Codex must implement this fallback so the demo does not break.

---

## 8. Recommended architecture

```text
Frontend: Next.js + Tailwind + React Flow + Plotly
Backend: FastAPI + Pydantic
Agents: LangGraph or custom deterministic agent pipeline
ML: PyTorch + scikit-learn + pandas + numpy + scipy
AMD training: ROCm PyTorch in AMD AI Notebooks/JupyterLab
LLM reporting: Fireworks AI API
Storage: local JSON or SQLite for MVP
Reports: Markdown/HTML export, optional PDF later
Deployment: Docker + docker-compose
```

### Runtime flow

```text
User opens dashboard
        ↓
User uploads DCRM/FRA/SCADA/maintenance data
        ↓
FastAPI validates files
        ↓
Asset Mapping Agent links data to asset registry
        ↓
DCRM Agent analyzes breaker data
        ↓
FRA Agent loads AMD-trained model artifact and analyzes transformer data
        ↓
SCADA Agent checks operational context
        ↓
Maintenance History Agent checks recurrence and overdue maintenance
        ↓
Risk Scoring Agent combines evidence
        ↓
Digital Twin Agent updates health state
        ↓
Fireworks Report Agent generates engineer-ready report
        ↓
Frontend displays dashboard + report export
```

---

## 9. Repository structure Codex should create

```text
gridops-copilot/
  README.md
  .env.example
  .gitignore
  docker-compose.yml

  backend/
    Dockerfile
    requirements.txt
    app/
      main.py
      config.py
      api/
        routes_health.py
        routes_upload.py
        routes_assets.py
        routes_diagnostics.py
        routes_reports.py
        routes_runtime.py
      agents/
        ingestion_agent.py
        asset_mapping_agent.py
        dcrm_agent.py
        fra_agent.py
        scada_agent.py
        maintenance_agent.py
        risk_agent.py
        digital_twin_agent.py
        recommendation_agent.py
        report_agent.py
        orchestrator.py
      models/
        fra_cnn.py
        model_loader.py
      schemas/
        asset.py
        diagnostic.py
        report.py
        runtime.py
      services/
        data_validation.py
        synthetic_data.py
        feature_engineering.py
        fra_inference.py
        dcrm_analysis.py
        risk_scoring.py
        fireworks_client.py
        report_export.py
      storage/
        json_store.py
      utils/
        logging.py

  frontend/
    Dockerfile
    package.json
    next.config.js
    tailwind.config.ts
    src/
      app/
        page.tsx
        assets/[assetId]/page.tsx
        upload/page.tsx
        reports/page.tsx
      components/
        AssetCard.tsx
        DigitalTwinGraph.tsx
        FRAChart.tsx
        DCRMChart.tsx
        RiskBadge.tsx
        EvidencePanel.tsx
        AMDRuntimePanel.tsx
        ReportViewer.tsx
      lib/
        api.ts
        types.ts

  data/
    synthetic/
      assets.csv
      maintenance_logs.csv
      scada_events.csv
      fra_healthy.csv
      fra_fault_winding_shift.csv
      fra_fault_core_clamping.csv
      fra_fault_insulation.csv
      dcrm_healthy.csv
      dcrm_fault_contact_wear.csv
      dcrm_fault_mechanism_delay.csv

  models/
    .gitkeep
    fra_label_map.json
    fra_cnn_rocm.pt       # generated from AMD notebook; can be mocked until trained

  notebooks/
    data_generation.ipynb
    fra_training_rocm.ipynb
    amd_benchmark.ipynb

  reports/
    amd_training_evidence.json
    amd_benchmark.md
    sample_report.md

  plots/
    .gitkeep
```

---

## 10. Backend API contract

Codex should implement these endpoints.

### Health

```text
GET /api/health
```

Response:

```json
{
  "status": "ok",
  "service": "gridops-copilot-backend"
}
```

### List assets

```text
GET /api/assets
```

Response:

```json
[
  {
    "asset_id": "TX-1",
    "asset_type": "Transformer",
    "voltage_level": "400kV",
    "criticality": 4,
    "health_score": 68,
    "risk_level": "High"
  }
]
```

### Upload diagnostic file

```text
POST /api/upload
```

Multipart fields:

```text
file: CSV file
file_type: fra | dcrm | scada | maintenance | assets
asset_id: optional
```

Response:

```json
{
  "upload_id": "upload_abc123",
  "file_type": "fra",
  "asset_id": "TX-1",
  "validation_status": "valid",
  "rows": 1024,
  "warnings": []
}
```

### Run diagnosis

```text
POST /api/diagnose
```

Request:

```json
{
  "asset_id": "TX-1",
  "diagnostic_type": "fra",
  "upload_id": "upload_abc123"
}
```

Response:

```json
{
  "asset_id": "TX-1",
  "asset_type": "Transformer",
  "diagnostic_type": "fra",
  "fault_class": "winding_deformation_suspected",
  "confidence": 0.82,
  "anomaly_score": 0.71,
  "risk_score": 67,
  "risk_level": "High",
  "evidence": [
    "Mid-frequency FRA deviation exceeded baseline tolerance.",
    "Maintenance inspection is overdue by 43 days.",
    "Recent SCADA temperature trend is mildly elevated."
  ],
  "recommended_action": "Schedule transformer diagnostic inspection and confirm with qualified engineer.",
  "requires_human_review": true
}
```

### Generate report

```text
POST /api/reports/generate
```

Request:

```json
{
  "diagnosis_id": "diag_abc123"
}
```

Response:

```json
{
  "report_id": "report_abc123",
  "format": "markdown",
  "content": "# GridOps Copilot Diagnostic Report\n..."
}
```

### AMD evidence endpoint

```text
GET /api/runtime/amd-evidence
```

Response should read from `reports/amd_training_evidence.json`.

Example:

```json
{
  "amd_usage_claim": "FRA model trained and benchmarked on AMD GPU using ROCm PyTorch.",
  "training_platform": "AMD AI Notebooks / AMD Developer Cloud",
  "framework": "PyTorch ROCm",
  "device_name": "AMD Instinct GPU, value captured by notebook",
  "torch_version": "captured_by_notebook",
  "hip_version": "captured_by_notebook",
  "model_artifact": "models/fra_cnn_rocm.pt",
  "metrics": {
    "accuracy": 0.91,
    "f1_macro": 0.89
  },
  "benchmarks": {
    "cpu_batch_ms": 120.0,
    "amd_gpu_batch_ms": 18.0,
    "speedup": 6.67
  }
}
```

If the evidence file is missing, return:

```json
{
  "amd_usage_claim": "No AMD training evidence file found yet.",
  "status": "pending"
}
```

---

## 11. Synthetic data plan

Codex should create a synthetic data generator.

### Assets

Create a small substation:

```text
400 kV Bus
  TX-1
  CB-401
  CB-402

220 kV Bus
  TX-2
  CB-221
```

### Asset registry fields

```text
asset_id
asset_type
voltage_level
manufacturer
age_years
criticality
bus_group
connected_to
```

### FRA CSV schema

```text
transformer_id
timestamp
frequency_hz
magnitude_db
phase_deg
winding
label
```

### DCRM CSV schema

```text
breaker_id
timestamp
time_ms
resistance_micro_ohm
travel_mm
coil_current_A
operation_type
label
```

### SCADA CSV schema

```text
asset_id
timestamp
voltage_kv
current_a
temperature_c
status
alarm_code
```

### Maintenance CSV schema

```text
asset_id
date
issue
action_taken
severity
next_due_date
```

### FRA anomaly simulation

Healthy curve:

```text
Smooth frequency response with small noise.
```

Winding deformation suspected:

```text
Add mid-frequency magnitude shift and localized resonance displacement.
```

Core/clamping issue suspected:

```text
Add low-to-mid frequency deviation.
```

Insulation-related abnormality suspected:

```text
Add high-frequency attenuation and phase irregularity.
```

Needs human review:

```text
Inject missing data, too much noise, or inconsistent sampling.
```

---

## 12. AMD notebook workflow Codex should generate

Create `notebooks/fra_training_rocm.ipynb` with these sections.

### Cell 1 — environment check

```python
import os
import json
import platform
from datetime import datetime, timezone

import torch

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("Torch HIP:", getattr(torch.version, "hip", None))
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. This notebook can run on CPU, but AMD evidence will be incomplete.")
```

### Cell 2 — imports

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
```

### Cell 3 — load or generate synthetic FRA dataset

The notebook should either load CSVs from `data/synthetic/` or generate a labeled dataset in memory.

### Cell 4 — dataset class

Use a PyTorch `Dataset` class that returns:

```text
x: tensor shape [channels, sequence_length]
y: class index
```

### Cell 5 — model

```python
class FRA1DCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.head(x)
```

### Cell 6 — train

Use:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FRA1DCNN(in_channels=3, num_classes=5).to(device)
```

Note: in ROCm PyTorch, AMD GPUs are usually exposed through PyTorch's `cuda` device API.

### Cell 7 — evaluate

Save:

```text
accuracy
macro_f1
classification_report
confusion_matrix
```

### Cell 8 — benchmark CPU vs GPU

Benchmark batch inference on CPU and GPU when GPU is available.

```python
import time

def benchmark(model, batch, device, repeats=50):
    model = model.to(device)
    batch = batch.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return ((end - start) / repeats) * 1000
```

### Cell 9 — save model + evidence

Save these files:

```python
from pathlib import Path
import hashlib

Path("../models").mkdir(exist_ok=True)
Path("../reports").mkdir(exist_ok=True)

model_path = Path("../models/fra_cnn_rocm.pt")
torch.save(model.state_dict(), model_path)

label_map = {
    "0": "healthy",
    "1": "winding_deformation_suspected",
    "2": "core_clamping_issue_suspected",
    "3": "insulation_related_abnormality_suspected",
    "4": "needs_human_review"
}
Path("../models/fra_label_map.json").write_text(json.dumps(label_map, indent=2))

artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

evidence = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "training_platform": "AMD AI Notebooks / AMD Developer Cloud",
    "framework": "PyTorch ROCm",
    "torch_version": torch.__version__,
    "hip_version": getattr(torch.version, "hip", None),
    "gpu_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only / not captured",
    "model_artifact": "models/fra_cnn_rocm.pt",
    "artifact_sha256": artifact_hash,
    "metrics": {
        "accuracy": float(acc),
        "f1_macro": float(f1)
    },
    "benchmarks": {
        "cpu_batch_ms": float(cpu_ms),
        "amd_gpu_batch_ms": float(gpu_ms) if torch.cuda.is_available() else None,
        "speedup": float(cpu_ms / gpu_ms) if torch.cuda.is_available() and gpu_ms > 0 else None
    }
}

Path("../reports/amd_training_evidence.json").write_text(json.dumps(evidence, indent=2))
```

---

## 13. Agent design

Use structured agents. Each agent should return JSON-like Python dictionaries.

### Ingestion Agent

Responsibilities:

```text
Parse CSV
Validate schema
Check missing values
Normalize column names
Return data quality warnings
```

### Asset Mapping Agent

Responsibilities:

```text
Map uploaded file to asset_id
Verify asset exists in registry
Resolve transformer_id/breaker_id fields
```

### FRA Agent

Responsibilities:

```text
Load FRA curve
Extract/resample features
Run fra_cnn_rocm.pt if available
Fallback to rule-based curve comparison if model missing
Return fault class, confidence, anomaly score, evidence
```

### DCRM Agent

Responsibilities:

```text
Analyze resistance spikes
Analyze operation delay using time/travel/coil current
Detect contact wear or mechanism delay patterns
Return confidence and evidence
```

### SCADA Agent

Responsibilities:

```text
Find recent alarms
Check temperature/current deviations
Find recent asset status changes
Return supporting context
```

### Maintenance Agent

Responsibilities:

```text
Check previous similar issues
Check next_due_date
Detect overdue maintenance
Return recurrence and overdue evidence
```

### Risk Agent

Use this weighted logic:

```text
risk_score =
  diagnostic_anomaly_score
+ asset_criticality_weight
+ recent_alarm_weight
+ maintenance_overdue_weight
+ historical_recurrence_weight
- data_quality_penalty
```

Risk mapping:

```text
0-30: Low
31-60: Medium
61-80: High
81-100: Critical
```

### Report Agent

Responsibilities:

```text
Take structured diagnosis JSON
Use Fireworks AI when configured
Fallback to deterministic markdown template
Never invent evidence
Use conservative language
Require human review for high-risk/critical/low-confidence cases
```

---

## 14. Frontend requirements

Build a clean, simple dashboard.

### Pages

```text
/                 Overview dashboard
/upload           Upload diagnostic files
/assets/[id]      Asset detail page
/reports          Generated reports
```

### Components

```text
AssetCard
RiskBadge
DigitalTwinGraph
FRAChart
DCRMChart
EvidencePanel
AMDRuntimePanel
ReportViewer
UploadPanel
```

### Digital twin view

Use React Flow or a simple tree if React Flow takes too long.

Example:

```text
400 kV Bus
  CB-401: Healthy
  CB-402: High Risk
  TX-1: Medium Risk

220 kV Bus
  CB-221: Healthy
  TX-2: Low Risk
```

Use labels/colors:

```text
Low / Healthy: green
Medium: yellow
High: orange
Critical: red
Insufficient data: gray
```

### AMD Runtime Panel

Show this prominently for judges:

```text
AMD Platform Evidence
- Training platform: AMD AI Notebooks / AMD Developer Cloud
- Framework: ROCm PyTorch
- Model: FRA 1D CNN
- Artifact: fra_cnn_rocm.pt
- Device captured: from amd_training_evidence.json
- Accuracy/F1: from evidence JSON
- CPU vs AMD GPU benchmark: from evidence JSON
- Report model: Fireworks AI / Gemma if configured
```

If no evidence file exists, show:

```text
AMD training evidence pending. Run notebooks/fra_training_rocm.ipynb on AMD GPU and save reports/amd_training_evidence.json.
```

---

## 15. Docker and submission requirements

All submissions must be containerized, so Codex must provide:

```text
backend/Dockerfile
frontend/Dockerfile
docker-compose.yml
README.md with setup instructions
```

### docker-compose services

```yaml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
    depends_on:
      - backend
```

The submitted app should be runnable with:

```bash
cp .env.example .env
docker compose up --build
```

The app must work even without `FIREWORKS_API_KEY` by using the deterministic report fallback.

---

## 16. README requirements

Codex must write a README with:

```text
Project summary
Hackathon track
AMD usage explanation
Architecture diagram
Setup instructions
How to run with Docker
How to run backend locally
How to run frontend locally
How to generate synthetic data
How to run diagnosis demo
How to use Fireworks AI
How to run AMD notebook training
How to view AMD evidence panel
Safety disclaimer
Submission checklist
```

### README AMD usage wording

Use this wording:

> GridOps Copilot uses AMD resources where they matter most: model training and benchmarking. We trained a 1D CNN FRA diagnostic model on AMD GPU-powered Jupyter notebooks using ROCm PyTorch, exported the model artifact, and integrated it into an agentic FastAPI diagnostic workflow. Fireworks AI is used for engineer-facing diagnostic report generation from structured evidence. The live MVP can run without always-on GPU hosting, making it cost-efficient and reliable for demo use.

---

## 17. Demo flow

Use two demo cases.

### Demo case 1 — circuit breaker DCRM

```text
1. Open dashboard.
2. Load synthetic demo data.
3. Select CB-402.
4. Upload or load dcrm_fault_contact_wear.csv.
5. DCRM Agent detects abnormal resistance spike.
6. SCADA Agent finds recent operation delay alarm.
7. Maintenance Agent finds overdue inspection.
8. Risk Agent marks High Risk.
9. Digital twin highlights CB-402.
10. Report Agent generates conservative maintenance report.
```

### Demo case 2 — transformer FRA

```text
1. Select TX-1.
2. Upload or load fra_fault_winding_shift.csv.
3. FRA Agent uses AMD-trained model artifact.
4. System detects suspected winding deformation.
5. Risk Agent marks Medium or High Risk.
6. AMD Runtime Panel shows training artifact and benchmark evidence.
7. Fireworks Report Agent generates engineer-ready report.
```

---

## 18. Submission assets to prepare

Prepare these before submitting:

```text
Public GitHub repository
Application URL
Demo video
Slide deck
Cover image
README with setup instructions
Dockerized app
AMD evidence JSON
AMD benchmark report
Screenshots of AMD notebook training
Screenshot of AMD Runtime Panel
Sample engineer report
```

### Video script outline

```text
1. Problem: utilities have scattered diagnostic data.
2. Solution: GridOps Copilot combines DCRM, FRA, SCADA, metadata, and history.
3. AMD usage: FRA model trained and benchmarked on AMD GPU notebooks using ROCm PyTorch.
4. Agent workflow: ingestion -> diagnostics -> context -> risk -> report.
5. Demo: CB-402 high risk and TX-1 suspected FRA issue.
6. Safety: decision support only; human engineer confirmation required.
7. Market: predictive maintenance for transmission utilities and industrial asset owners.
```

---

## 19. Codex implementation instructions

Give Codex this instruction:

```text
Build the GridOps Copilot repository exactly according to this guide.

Priority order:
1. Create backend FastAPI service with health, assets, upload, diagnose, reports, and AMD evidence endpoints.
2. Create synthetic data generator and seed demo files.
3. Implement deterministic DCRM, SCADA, maintenance, and risk agents.
4. Implement FRA agent with model artifact loading and rule-based fallback.
5. Implement Fireworks report agent with deterministic fallback when API key is absent.
6. Create Next.js frontend with overview, upload, asset detail, report, digital twin, and AMD Runtime Panel.
7. Add Dockerfiles and docker-compose.
8. Add notebooks for synthetic data, FRA ROCm training, and benchmarking.
9. Add README and sample reports.
10. Make the app runnable with one Docker command.

Do not require live AMD GPU for the Docker app.
Do not require Fireworks API key for basic demo.
Do not invent certified fault diagnosis.
Use conservative engineering language.
Include clear TODO comments where human AMD notebook execution is required.
```

---

## 20. Codex acceptance criteria

The implementation is acceptable only if:

```text
[ ] docker compose up --build starts backend and frontend.
[ ] GET /api/health returns ok.
[ ] Frontend dashboard loads demo assets.
[ ] User can run a demo diagnosis for CB-402 and TX-1.
[ ] DCRM Agent returns evidence-backed diagnosis.
[ ] FRA Agent works with model artifact if present.
[ ] FRA Agent has a rule-based fallback if model is absent.
[ ] Risk Agent returns score and risk level.
[ ] Report Agent works without Fireworks API key using template fallback.
[ ] Report Agent uses Fireworks API when FIREWORKS_API_KEY is present.
[ ] AMD Runtime Panel reads reports/amd_training_evidence.json.
[ ] README explains AMD GPU training-first strategy.
[ ] No part of the app claims autonomous grid control.
[ ] All recommendations require human confirmation.
```

---

## 21. What you personally should research before building

### AMD / hackathon

```text
AMD AI Developer Program signup flow
AMD Member Perks credit request form
AMD AI Notebooks login flow
How to download files from JupyterLab
How to stop/destroy a GPU session safely
Fireworks API key setup
Available Fireworks/Gemma model IDs in your account
lablab.ai submission fields and deadline
```

### Technical

```text
PyTorch basics
ROCm PyTorch device detection
1D CNN for time-series classification
Curve similarity metrics
FastAPI file upload handling
Next.js API integration
Docker Compose basics
```

### Domain

```text
What FRA curve deviation means at a high level
What DCRM resistance spikes suggest at a high level
Why SCADA alarms add context but do not prove fault
Why maintenance history changes risk priority
Why human engineer review is mandatory
```

---

## 22. Human build checklist

### Before coding

```text
[ ] Register on lablab.ai ACT II.
[ ] Join AMD AI Developer Program.
[ ] Request AMD Developer Cloud credits.
[ ] Join lablab Discord.
[ ] Join AMD Discord.
[ ] Create Fireworks account / redeem credits if available.
[ ] Create GitHub repository.
```

### During coding

```text
[ ] Build app with CPU-safe fallback.
[ ] Add synthetic demo data.
[ ] Add model placeholder or fallback.
[ ] Add AMD evidence panel.
[ ] Add report fallback.
[ ] Add Docker setup.
```

### During AMD notebook session

```text
[ ] Open AMD AI Notebooks / JupyterLab.
[ ] Verify GPU using torch.cuda.is_available().
[ ] Train FRA 1D CNN.
[ ] Save model artifact.
[ ] Save label map.
[ ] Save benchmark report.
[ ] Save AMD evidence JSON.
[ ] Download artifacts.
[ ] Stop/destroy session according to platform instructions.
```

### Before submission

```text
[ ] Push public GitHub repo.
[ ] Confirm Docker works on a fresh machine.
[ ] Record demo video.
[ ] Create slide deck.
[ ] Add app URL.
[ ] Add cover image.
[ ] Include AMD evidence in README and UI.
[ ] Confirm no unsupported safety claims.
```

---

## 23. Recommended final pitch

Use this in your README, video, and slides:

> GridOps Copilot is an AMD-powered agentic digital twin for EHV substation maintenance. It uses ROCm PyTorch on AMD GPU-powered Jupyter notebooks to train and benchmark a diagnostic FRA waveform model, then integrates the exported model into a containerized FastAPI and Next.js application. Specialized agents combine FRA, DCRM, SCADA, asset metadata, and maintenance history into evidence-backed risk scores, while Fireworks AI generates conservative engineer-ready reports. The system is designed as decision support only and requires human expert confirmation before maintenance action.

---

## 24. Sources to verify again before final submission

Use these pages to verify current rules and access details:

```text
lablab.ai ACT II page:
https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii

AMD AI Developer Program:
https://www.amd.com/en/developer/ai-dev-program.html

AMD Developer Cloud:
https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html

AMD AI Notebooks:
https://notebooks.amd.com/

AMD cloud credit claiming article:
https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-claim-amd-cloud-credits.html

AMD Member Perks:
https://developer.amd.com/member-perks/

Fireworks + AMD:
https://fireworks.ai/partners/amd

ROCm AI Developer Hub tutorials:
https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/
```

---

## 25. Non-negotiable safety statement

GridOps Copilot must always state:

```text
This system is a decision-support prototype for hackathon demonstration. It does not control grid equipment, does not replace certified engineering analysis, and does not provide final fault certification. All high-risk, critical, or low-confidence findings require confirmation by a qualified human engineer before action.
```

