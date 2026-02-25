# FOIA Dump Investigative Grapher

## Phase 2: Preliminary Results & Prototype Checkout 1

This repository hosts the prototype implementation of the **FOIA Dump Investigative Grapher** ML pipeline and extraction backend tailored for constrained hardware (RTX 4070 8GB VRAM). 

### Prerequisites
- Python 3.10+
- NVIDIA GPU (8GB VRAM min) with standard CUDA drivers
- Docker Desktop (for Neo4j graph database hosting)

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Launch the Graph Database
We use a Docker setup to host our Neo4j instance locally.
```powershell
docker-compose up -d
```
The database admin panel is hosted at `http://localhost:7474`.

### 3. Running the Dataset Engine
To download the `tonytan48/Re-DocRED` dataset and format it for relations:
```powershell
python -m ml_pipeline.process_docred
```

### 4. Running the Checkpoint 1 Scoring Pipeline (F1 Metrics)
The Judgment Pipeline executes automated NLP queries asking LLaMA-3 (via 4-bit config) to extract JSON entities and relations. It then parses these triples and compares them to the DocRED human-annotated ground truths.
```powershell
python -m ml_pipeline.judge_pipeline
```
*Note: Depending on the target document size, LLM inference on 8GB VRAM will output F1 metrics synchronously.*

### 5. Storing to the Graph DB
Run the extraction integration script. This logic feeds the `InferenceService` module and pushes all identified Nodes and Relations straight to Neo4j.
```powershell
python -m backend.extract_and_store
```