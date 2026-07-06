# GridOps Copilot

## Agentic Digital Twin and Fault Diagnosis Assistant for EHV Substations

**Document type:** Lean project brief  
**Audience:** Hackathon team, mentors, evaluators, software/AI builders  
**Focus:** Only project-relevant details. No basic power-transmission explanations.  
**Project category:** Industrial AI, predictive maintenance, time-series diagnostics, digital twin, agentic workflow

---

## 1. Simple Summary

**GridOps Copilot** is an AI assistant for power-transmission maintenance teams.

It analyzes technical diagnostic data from high-voltage substation assets and helps engineers answer:

- Which asset is becoming risky?
- What fault may be developing?
- How confident is the system?
- What evidence supports the finding?
- What maintenance action should be taken?
- Should the case be escalated to a human expert?

The project should be positioned as a **decision-support system**, not an autonomous grid controller.

---

## 2. One-Line Pitch

> GridOps Copilot is an agentic AI system that analyzes DCRM, FRA, SCADA-like readings, asset metadata, and maintenance history to detect equipment anomalies, classify likely faults, maintain a digital-twin view of a substation, and generate explainable maintenance recommendations.

---

## 3. Problem Statement

Power-transmission utilities collect many types of equipment-health data, but the data is usually scattered across different files, tools, and inspection records. Engineers must manually compare test curves, read alarms, check past maintenance logs, and decide whether a transformer or circuit breaker needs attention.

This process can be slow, expert-dependent, and inconsistent. Early warning signs may be missed if data is not analyzed together.

**Goal:** Build an AI copilot that combines diagnostic test data, sensor readings, and maintenance history into one explainable asset-health workflow.

---

## 4. Public-Sector / SIH-Style Inspiration

GridOps Copilot combines three related public-sector power-transmission problem areas:

| Source problem area | What it contributes to GridOps Copilot |
|---|---|
| **AI-Based DCRM Analysis for EHV Circuit Breakers** | Detect abnormal resistance patterns, contact wear, or operating-mechanism issues in circuit breakers. |
| **AI-Driven Frequency Response Analysis for Transformer Diagnostics** | Detect possible transformer internal abnormalities such as winding deformation, core displacement, or insulation-related issues. |
| **AI/ML-enabled Digital Twin for EHV 400/220 kV Substation** | Create a virtual asset-health view of a substation and support remote diagnostics, maintenance planning, and training. |

The project is also aligned with POWERGRID-style asset-management priorities such as condition monitoring, remote diagnostics, early fault detection, and predictive maintenance.

---

## 5. Key Terms Needed for This Project

Only these technical terms are required to understand the project:

| Term | Project meaning |
|---|---|
| **DCRM** | Dynamic Contact Resistance Measurement. A diagnostic test used to analyze circuit-breaker contact/mechanism health. |
| **FRA** | Frequency Response Analysis. A diagnostic test used to analyze transformer internal mechanical/electrical condition. |
| **SCADA-like data** | Operational readings such as voltage, current, temperature, breaker status, alarms, and event logs. For the hackathon, this can be simulated. |
| **Digital twin** | A virtual representation of the substation assets and their current health/risk status. |
| **Predictive maintenance** | Using data to detect risk before failure and recommend inspection or repair actions. |
| **Agentic workflow** | A workflow where multiple AI/tool-using agents validate data, run analysis, combine evidence, reason about risk, and generate reports. |

---

## 6. Target Users

| User | What they need |
|---|---|
| **Maintenance engineer** | Understand which asset needs inspection or repair. |
| **Asset manager** | Prioritize maintenance based on risk and asset criticality. |
| **Control-room / monitoring team** | View abnormal assets and recent alerts in one dashboard. |
| **Field engineer** | Receive an evidence-backed report before visiting the site. |
| **Training team** | Use the digital twin for fault-scenario learning. |

---

## 7. Project Scope

### In Scope

The MVP should support:

- Uploading or loading DCRM data for circuit breakers
- Uploading or loading FRA data for transformers
- Loading simulated SCADA/event data
- Loading asset metadata and maintenance history
- Detecting anomalies in DCRM and FRA signals
- Assigning asset health score and risk level
- Highlighting affected assets in a digital-twin dashboard
- Generating explainable maintenance recommendations
- Exporting a diagnostic report

### Out of Scope

The MVP should not:

- Operate real substations
- Send control commands to live grid equipment
- Replace protection relays or human engineering judgment
- Claim certified fault diagnosis without validation
- Use real sensitive utility data unless explicitly permitted

---

## 8. Core Data Inputs

| Input | Example fields | Purpose |
|---|---|---|
| **Asset registry** | asset_id, asset_type, voltage_level, manufacturer, age_years, criticality | Defines assets in the digital twin. |
| **DCRM waveform** | breaker_id, timestamp, time_ms, resistance_micro_ohm, travel_mm, coil_current_A, operation_type | Detects circuit-breaker contact/mechanism anomalies. |
| **FRA curve** | transformer_id, timestamp, frequency_hz, magnitude_db, phase_deg, winding | Detects transformer frequency-response deviations. |
| **SCADA-like stream** | asset_id, timestamp, voltage, current, temperature, status, alarm_code | Adds operational context. |
| **Maintenance log** | asset_id, date, issue, action_taken, severity, next_due_date | Helps determine whether the issue is recurring or overdue. |
| **Baseline/reference data** | historical healthy curves, previous test results, expected thresholds | Enables comparison against normal behavior. |

For a hackathon, all data can be **synthetic but realistic**.

---

## 9. Expected Outputs

The system should produce:

| Output | Description |
|---|---|
| **Asset health score** | Numerical score, for example 0-100. |
| **Risk level** | Low, Medium, High, or Critical. |
| **Detected anomaly** | What abnormal pattern was found. |
| **Likely fault category** | Example: contact wear, mechanism delay, winding deformation, insulation-related issue, sensor anomaly. |
| **Confidence score** | How confident the model/agent is. |
| **Evidence summary** | Key curve deviation, alarm, historical pattern, or maintenance fact. |
| **Recommended action** | Monitor, retest, inspect, repair, replace, or escalate. |
| **Digital twin status** | Visual asset state in a dashboard. |
| **Engineer report** | Exportable PDF/HTML/Markdown diagnostic summary. |

---

## 10. Agentic Workflow

GridOps Copilot should not be just one classifier. It should be a coordinated agent workflow.

```text
User uploads diagnostic data
        ↓
Data Ingestion Agent validates files
        ↓
Asset Mapping Agent links data to assets
        ↓
DCRM Agent analyzes breaker signatures
        ↓
FRA Agent analyzes transformer signatures
        ↓
SCADA Agent checks operational context
        ↓
Maintenance History Agent checks past issues
        ↓
Risk Scoring Agent combines all evidence
        ↓
Digital Twin Agent updates asset health view
        ↓
Recommendation Agent suggests action
        ↓
Report Agent generates engineer-ready output
```

### Agent Responsibilities

| Agent | Responsibility |
|---|---|
| **Data Ingestion Agent** | Parses CSV/XML/JSON files, checks missing values, validates schema. |
| **Asset Mapping Agent** | Maps uploaded data to the correct asset in the substation model. |
| **DCRM Agent** | Detects abnormal resistance/time/travel patterns in breaker data. |
| **FRA Agent** | Detects abnormal frequency-response shifts in transformer data. |
| **SCADA Agent** | Looks for supporting alarms, temperature/current deviations, or recent status changes. |
| **Maintenance History Agent** | Checks whether similar issues occurred before or maintenance is overdue. |
| **Risk Scoring Agent** | Combines model outputs, asset criticality, and evidence into final risk. |
| **Digital Twin Agent** | Updates the virtual substation map with health/risk state. |
| **Recommendation Agent** | Suggests next action and urgency. |
| **Report Agent** | Generates human-readable explanation and report. |

---

## 11. Fault Categories for MVP

Keep the initial fault taxonomy small.

### Circuit Breaker / DCRM

| Class | Meaning for project |
|---|---|
| **Healthy** | No major deviation from baseline. |
| **Contact wear** | Resistance pattern suggests possible contact degradation. |
| **Mechanism delay** | Timing/travel pattern suggests slow or abnormal operation. |
| **Abnormal resistance spike** | Unexpected resistance peak during operation. |
| **Needs human review** | Data is unclear or low quality. |

### Transformer / FRA

| Class | Meaning for project |
|---|---|
| **Healthy** | FRA curve is close to baseline. |
| **Winding deformation suspected** | Frequency-response shift suggests possible mechanical displacement. |
| **Core/clamping issue suspected** | Certain curve deviations suggest structural/internal issue. |
| **Insulation-related abnormality suspected** | Deviation pattern may indicate insulation-related concern. |
| **Needs human review** | Data is unclear or low quality. |

Use careful wording such as **“suspected”**, **“possible”**, and **“requires engineer confirmation.”**

---

## 12. Risk Scoring Logic

A practical MVP risk score can combine:

```text
Risk Score =
  diagnostic anomaly score
+ asset criticality weight
+ recent alarm weight
+ maintenance overdue weight
+ historical recurrence weight
- data quality penalty
```

Example risk interpretation:

| Score | Risk level | Suggested action |
|---:|---|---|
| 0-30 | Low | Continue routine monitoring. |
| 31-60 | Medium | Review during next maintenance window. |
| 61-80 | High | Schedule inspection soon. |
| 81-100 | Critical | Escalate to senior engineer immediately. |

---

## 13. ML / AI Approach

### MVP Approach

Use a hybrid method:

| Task | Suggested approach |
|---|---|
| File parsing | Python parsers for CSV/XML/JSON. |
| DCRM anomaly detection | Feature extraction + Isolation Forest / Random Forest / simple threshold rules. |
| FRA anomaly detection | Curve comparison + feature extraction + classifier. |
| SCADA anomaly detection | Rolling thresholds, z-score, or Isolation Forest. |
| Risk scoring | Rule-based weighted scoring. |
| Report generation | LLM-based explanation using structured evidence. |
| Agent orchestration | LangGraph, CrewAI, AutoGen, or a custom Python workflow. |

### Optional Advanced Approach

If time permits:

- 1D CNN or LSTM for waveform classification
- Autoencoder for anomaly detection
- RAG over maintenance manuals or diagnostic notes
- Interactive natural-language query interface
- Multi-step what-if simulation in the digital twin

---

## 14. Digital Twin Requirements

The digital twin does not need full electrical simulation for the MVP.

It should represent:

- assets in the substation
- asset connections or grouping
- current health score
- risk level
- active alarms
- last diagnostic test result
- maintenance recommendation

### MVP Digital Twin View

A simple 2D dashboard is enough:

```text
400 kV Bus
  ├── CB-401: Healthy
  ├── CB-402: High Risk
  └── TX-1: Medium Risk

220 kV Bus
  ├── CB-221: Healthy
  └── TX-2: Low Risk
```

Use colors/status labels in the UI:

- Green: healthy
- Yellow: medium risk
- Orange: high risk
- Red: critical
- Gray: insufficient data

---

## 15. MVP Demo Plan

### Demo Scenario

Create a synthetic substation with:

- 2 transformers
- 3 circuit breakers
- 1 busbar group
- simulated SCADA values
- 5-10 historical maintenance records
- healthy and faulty DCRM/FRA sample files

### Demo Flow

```text
1. User opens dashboard.
2. User uploads DCRM file for CB-402.
3. System detects abnormal resistance spike.
4. DCRM Agent marks possible contact wear.
5. SCADA Agent finds recent operation delay alarm.
6. Maintenance Agent finds inspection overdue.
7. Risk Agent marks asset as High Risk.
8. Digital Twin highlights CB-402.
9. Recommendation Agent suggests inspection within 7 days.
10. Report Agent exports diagnostic summary.
```

### Second Demo Case

```text
1. User uploads FRA file for TX-1.
2. FRA Agent compares current curve with baseline.
3. System detects mid-frequency deviation.
4. Risk Agent marks Medium Risk.
5. Report says possible winding deformation suspected, confirm with expert review.
```

---

## 16. Example System Output

```text
Asset: CB-402
Asset Type: Circuit Breaker
Risk Level: High
Health Score: 64/100

Detected Issue:
DCRM signature shows abnormal resistance spike during close operation.

Possible Cause:
Contact wear or operating-mechanism misalignment.

Supporting Evidence:
- Resistance peak is 38% above baseline.
- Closing operation is slower than previous test.
- Similar minor deviation appeared in previous maintenance record.

Recommended Action:
Schedule inspection of breaker contacts and operating mechanism.
Repeat DCRM after maintenance.
Escalate if next operation shows further delay.

Confidence:
0.78

Human Review:
Required before final maintenance decision.
```

---

## 17. Dashboard Requirements

| Screen | What it should show |
|---|---|
| **Overview** | Substation asset list, health score, risk level, recent alerts. |
| **Upload / Ingestion** | Upload DCRM/FRA/SCADA files and show validation status. |
| **Asset Detail** | Asset metadata, curve plots, alarms, maintenance history. |
| **Diagnostic Result** | Fault class, confidence, evidence, recommendation. |
| **Digital Twin View** | Visual layout or asset tree with health colors. |
| **Report Export** | Downloadable diagnostic report. |

---

## 18. Report Format

Each generated report should include:

1. Asset ID and asset type
2. Uploaded file details
3. Data quality status
4. Detected anomaly
5. Likely fault category
6. Confidence score
7. Evidence summary
8. Risk level and health score
9. Recommended action
10. Whether human expert review is required
11. Timestamp and model/version details

---

## 19. Evaluation Metrics

| Area | Metric |
|---|---|
| **File handling** | Percentage of valid files parsed correctly. |
| **Anomaly detection** | Precision, recall, F1 score on synthetic labeled cases. |
| **Fault classification** | Accuracy/F1 across MVP fault classes. |
| **Risk scoring** | Agreement with predefined synthetic ground truth. |
| **Report quality** | Whether recommendation is evidence-backed and understandable. |
| **Agent workflow** | Whether each agent produces traceable intermediate output. |
| **Safety** | Whether system avoids unsupported claims and requires human review for uncertain cases. |

---

## 20. Recommended Tech Stack

| Layer | Suggested tools |
|---|---|
| **Frontend** | React / Next.js, Tailwind, Plotly/Recharts for curves. |
| **Backend** | FastAPI / Python. |
| **Data processing** | pandas, numpy, scipy. |
| **ML** | scikit-learn, PyTorch if using deep models. |
| **Agents** | LangGraph, CrewAI, AutoGen, or custom orchestrator. |
| **Vector/RAG optional** | FAISS, Chroma, LanceDB. |
| **Visualization** | Plotly, D3.js, React Flow for digital twin graph. |
| **Reports** | Markdown/HTML/PDF export. |
| **Deployment** | Docker container. |

---

## 21. Suggested Repository Structure

```text
gridops-copilot/
  README.md
  docker-compose.yml
  backend/
    app.py
    agents/
      ingestion_agent.py
      dcrm_agent.py
      fra_agent.py
      scada_agent.py
      risk_agent.py
      report_agent.py
    models/
    schemas/
    services/
  frontend/
    src/
      pages/
      components/
      charts/
      digital_twin/
  data/
    synthetic/
      assets.csv
      dcrm_healthy.csv
      dcrm_fault_contact_wear.csv
      fra_healthy.csv
      fra_fault_winding_shift.csv
      scada_events.csv
      maintenance_logs.csv
  notebooks/
    data_generation.ipynb
    model_training.ipynb
  reports/
    sample_report.md
```

---

## 22. Safety and Positioning

GridOps Copilot must clearly state:

- It is a decision-support tool.
- It does not control grid equipment.
- It does not replace certified engineering analysis.
- It should use synthetic or approved data only.
- High-risk or low-confidence results require human expert review.
- Recommendations should be explainable and evidence-backed.

Use conservative language:

- “possible fault”
- “suspected issue”
- “requires confirmation”
- “recommend inspection”
- “do not take automated control action”

---

## 23. What Makes This Project Distinctive

GridOps Copilot is less generic than typical hackathon projects because it combines:

- industrial asset diagnostics
- time-series / waveform analysis
- multi-agent reasoning
- digital twin visualization
- predictive maintenance
- explainable AI reports
- public-sector power-transmission relevance

It is not just a chatbot or dashboard. The core value is **combining multiple diagnostic signals into a traceable maintenance decision workflow**.

---

## 24. Final Project Problem Statement

**GridOps Copilot: Agentic Digital Twin and Fault Diagnosis Assistant for EHV Substations**

Power-transmission utilities collect diagnostic and operational data from critical substation assets, including DCRM data for circuit breakers, FRA data for transformers, SCADA-like readings, maintenance logs, and asset metadata. These data sources are often scattered, multi-format, and dependent on expert manual interpretation.

Build an agentic AI copilot that ingests these data sources, validates them, detects abnormal patterns, classifies likely equipment faults, combines evidence with asset history and criticality, updates a digital-twin dashboard, and generates explainable maintenance recommendations for engineers.

The system should support predictive maintenance and remote diagnostics while keeping human expert review mandatory for final decisions.

---

## 25. Hackathon MVP Checklist

A strong MVP should include:

- [ ] Synthetic substation asset registry
- [ ] DCRM sample files: healthy + faulty
- [ ] FRA sample files: healthy + faulty
- [ ] Simulated SCADA/event logs
- [ ] Maintenance history sample data
- [ ] File upload and validation
- [ ] DCRM anomaly detection
- [ ] FRA anomaly detection
- [ ] Risk score calculation
- [ ] Digital twin dashboard
- [ ] Explainable recommendation output
- [ ] Exportable report
- [ ] Dockerized setup
- [ ] README with clear demo instructions

---

## 26. References to Verify Before Submission

Use these only as project inspiration and verify current official wording before final submission:

- Smart India Hackathon official problem-statement page: `https://www.sih.gov.in/sih2025PS`
- POWERGRID transmission and asset-management information: `https://www.powergrid.in/en/transmission`
- POWERGRID NTAMC overview: `https://www.powergrid.in/en/ntamc-overview`

