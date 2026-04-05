# GPUFlow v1 — Technical Specification

## Overview

GPUFlow is a lightweight GPU job scheduler for ML teams running on a single multi-GPU machine. It replaces ad-hoc GPU coordination (SSH + nvidia-smi + Slack) with a CLI, a REST API, a live web dashboard, and built-in experiment tracking. The design goal is minimal infrastructure — any machine with Docker and an NVIDIA GPU can run it.

---

## Problem Statement

ML teams sharing GPU servers face three recurring issues:

1. **No visibility** — no one knows which GPUs are free without SSH-ing in
2. **Wasted capacity** — jobs accidentally share memory, OOM each other, or idle GPUs go unnoticed
3. **Slow iteration** — developers SSH into servers to debug, with no interactive environment

GPUFlow solves all three with a single `pip install`.

---

## Architecture

```
Developer CLI / REST client
        │
        │ HTTP  (X-API-Key auth)
        ▼
┌─────────────────────────────────────┐
│          FastAPI Server :8000        │
│                                     │
│  /api/v1/jobs   /api/v1/gpus        │
│  /api/v1/mlflow /api/v1/debug       │
│  /mlflow/{path} (reverse proxy)     │
│  /dashboard     (static)            │
│  /landing       (static)            │
└────────┬───────────────┬────────────┘
         │               │
         ▼               ▼
  FIFO Scheduler    SessionManager
  (2s poll loop)    (debug containers)
         │
         ▼
    DockerRunner
    (blocking, thread pool)
         │
         ▼
  Docker containers
  (--gpus device=N,
   -v $CWD:/workspace,
   MLFLOW_TRACKING_URI injected)
         │
         ▼
  MLflow Server :5001
  (subprocess, auto-started)
```

---

## Components

### 1. API Server (`gpuflow/api/`)

FastAPI application. Started via `uvicorn`. All state is initialised in the `lifespan` context manager on startup.

**Startup sequence:**
1. Connect to SQLite DB (`aiosqlite`, WAL mode)
2. Start FIFO scheduler as an `asyncio.Task`
3. Start MLflow tracking server as an `asyncio` subprocess on port 5001
4. Instantiate `SessionManager` for debug containers

**Shutdown sequence:**
1. Cancel scheduler task
2. Kill all active debug containers
3. Terminate MLflow subprocess (5s timeout, then `kill`)
4. Close DB connection

**Authentication:** Every API route (except static file mounts) requires `X-API-Key` header matching `settings.API_KEY`.

**Routes:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/jobs` | Submit a job |
| GET | `/api/v1/jobs` | List jobs (optional `?status_filter=`) |
| GET | `/api/v1/jobs/{id}` | Get job |
| DELETE | `/api/v1/jobs/{id}` | Cancel job |
| GET | `/api/v1/jobs/{id}/logs` | Stream logs (`?follow=true` for tail) |
| GET | `/api/v1/gpus` | List GPUs + utilization |
| GET | `/api/v1/mlflow/experiments` | Proxied MLflow experiments |
| GET | `/api/v1/mlflow/runs` | Proxied MLflow runs |
| GET | `/api/v1/debug/images` | List local Docker images |
| POST | `/api/v1/debug/sessions` | Launch a debug container |
| GET | `/api/v1/debug/sessions` | List active debug sessions |
| DELETE | `/api/v1/debug/sessions/{id}` | Kill a debug session |
| ANY | `/mlflow/{path}` | Full reverse proxy to MLflow UI |
| GET | `/dashboard` | Live operations dashboard (static) |
| GET | `/landing` | Public showcase page (static) |
| GET | `/` | Redirect → `/landing` |

---

### 2. Job Data Model (`gpuflow/models/job.py`)

```python
class Job:
    id: str              # UUID4
    name: str
    status: JobStatus    # queued | running | completed | failed | cancelled
    entrypoint: str      # legacy field, defaults to ""
    command: str         # the shell command run inside the container
    requested_gpus: int  # 1–16
    requested_nodes: int # default 1
    assigned_gpus: List[int]
    docker_image: str
    log_path: Optional[str]
    error_message: Optional[str]
    submitted_by: str    # username attribution, default "anonymous"
    created_at: datetime
```

**Status transitions:**
```
QUEUED → RUNNING → COMPLETED
                 → FAILED
       → CANCELLED
```

---

### 3. Scheduler (`gpuflow/scheduler/scheduler.py`)

Async FIFO loop. Polls every `SCHEDULER_POLL_INTERVAL` seconds (default 2s).

**Tick logic:**
```python
queued_jobs = store.list(status=QUEUED)
running_jobs = store.get_running_jobs()
allocated_gpus = {gpu for job in running_jobs for gpu in job.assigned_gpus}
free_gpus = [g.index for g in all_gpus if g.index not in allocated_gpus]

for job in queued_jobs:           # FIFO — first job blocks the queue
    if len(free_gpus) < job.requested_gpus:
        break
    assign(job, free_gpus[:job.requested_gpus])
    asyncio.create_task(worker.execute(job))
```

No preemption. No priority weights. Pure FIFO — predictable and auditable.

---

### 4. Docker Runner (`gpuflow/runner/docker_runner.py`)

Executes jobs as Docker containers. Runs blocking Docker SDK calls in a thread pool executor to avoid blocking the event loop.

**Container configuration per job:**
- Image: `job.docker_image`
- Command: `bash -c '{job.command}'`
- GPUs: `DeviceRequest(device_ids=[str(i) for i in job.assigned_gpus], capabilities=[["gpu"]])`
- Volume: `{server_cwd: {bind: "/workspace", mode: "rw"}}`
- Working directory: `/workspace`
- Environment:
  - `CUDA_VISIBLE_DEVICES` — comma-separated assigned GPU indices
  - `MLFLOW_TRACKING_URI` — points to `http://172.17.0.1:5001` (Docker host IP)
  - For multi-node: `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`

**Logs:** streamed chunk-by-chunk from `container.logs(stream=True, follow=True)` into `{LOG_DIR}/{job_id}.log`.

**Cancellation:** calls `container.kill()` on the stored container reference.

---

### 5. GPU Inspector (`gpuflow/gpu/inspector.py`)

Wraps `pynvml` (with `nvidia-smi` fallback) to return per-GPU metrics:
- Index, name, total memory (MB), used memory (MB), utilization (%), availability

---

### 6. Database (`gpuflow/db/store.py`)

SQLite via `aiosqlite`. WAL journal mode. Single table:

```sql
CREATE TABLE jobs (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    status       TEXT NOT NULL,
    entrypoint   TEXT NOT NULL DEFAULT '',
    command      TEXT,
    requested_gpus   INTEGER NOT NULL,
    requested_nodes  INTEGER NOT NULL,
    assigned_gpus    TEXT NOT NULL DEFAULT '[]',   -- JSON array
    docker_image     TEXT NOT NULL,
    log_path         TEXT,
    error_message    TEXT,
    submitted_by     TEXT NOT NULL DEFAULT 'anonymous',
    created_at       TEXT NOT NULL
)
```

DB path defaults to `/tmp/gpuflow.db` (project filesystem is read-only on Lightning Studio).

---

### 7. MLflow Integration (`gpuflow/mlflow_server.py`, `gpuflow/api/routes/mlflow.py`)

**Server startup:** `mlflow server --host 0.0.0.0 --port 5001` launched as an `asyncio` subprocess on API startup. Backend and artifact store default to `./mlruns`.

**Container injection:** `MLFLOW_TRACKING_URI=http://172.17.0.1:5001` is injected into every training container so scripts can call `mlflow.log_metric()` without any configuration.

**Dashboard proxy:** All requests to `/mlflow/{path}` are reverse-proxied to `http://localhost:5001` via a persistent `httpx.AsyncClient`. This makes the MLflow UI accessible through the same port 8000 — required because Lightning Studio only exposes one port externally.

---

### 8. Debug Sessions (`gpuflow/debug/session_manager.py`)

Allows developers to launch an interactive container from any image available on the host and connect to it via VS Code.

**Session lifecycle:**
1. `POST /api/v1/debug/sessions` with `{image: "..."}` 
2. `SessionManager.create_session()` launches `docker run <image> sleep infinity` with:
   - All GPUs (`DeviceRequest(count=-1)`)
   - Workspace volume mounted
3. Returns `vscode_url` pointing to the existing code-server at port 30110 with `?folder=/workspace`
4. Returns `exec_cmd`: `docker exec -it gpuflow-debug-{id[:8]} bash` for terminal access
5. `DELETE /api/v1/debug/sessions/{id}` stops and removes the container

Sessions are held in memory. Dead containers are pruned on the next `create_session` call.

---

### 9. CLI (`gpuflow/cli/main.py`)

Built with Click. Reads `GPUFLOW_SERVER_URL` and `API_KEY` from environment variables (or `--server` / `--api-key` flags).

**Commands:**

```
gpuflow run --image <image> [--gpus N] [--nodes N] [--name NAME] [--user USER] CMD...

    --image   Required. Docker image to run in.
    --gpus    Number of GPUs to allocate (default 1).
    --nodes   Number of nodes (default 1).
    --name    Job name. Auto-generated from image tag if omitted.
    --user    Username for attribution in dashboard (default "anonymous").
    CMD       Any shell command. Passed verbatim as the container command.
              Quote the full string for commands with flags or cd paths.

    Examples:
      gpuflow run --image pytorch/pytorch:2.1.0 python train.py
      gpuflow run --image myrepo/env:v3 --gpus 2 --user alice torchrun train.py
      gpuflow run --image myrepo/env:v3 "cd /workspace/alice/proj && torchrun train.py"

gpuflow status [--status FILTER]   List all jobs with User column.
gpuflow logs <job_id> [-f]         Print or stream logs.
gpuflow cancel <job_id>            Cancel a queued or running job.
```

Sends `X-Submitted-By` header on job submit, which the API uses to set `submitted_by` (header takes precedence over the body field).

---

### 10. Dashboard (`dashboard/`)

Single-page app. Polls the API every 2 seconds. No build step — plain HTML/CSS/JS.

**Panels:**

| Panel | Data source | Update |
|-------|-------------|--------|
| GPU cards | `GET /api/v1/gpus` | Every 2s |
| Resource usage (donut + allocation bar) | Derived from jobs + GPUs | Every 2s |
| Jobs table | `GET /api/v1/jobs` | Every 2s |
| MLflow runs | `GET /api/v1/mlflow/runs` | Every 2s |
| Debug sessions | `GET /api/v1/debug/sessions` | Every 2s |
| Log viewer | `GET /api/v1/jobs/{id}/logs` | On demand |

**GPU gauges:** Inline SVG with `stroke-dasharray` / `stroke-dashoffset` for circular utilization and memory dials. 30-point sparkline history via Chart.js 4.4.0 (diff-and-update pattern to avoid canvas leaks on re-render).

**Resource panel:** Donut chart aggregating GPU memory (MB) per `submitted_by` across running jobs. GPU allocation bar shows per-device user assignment with deterministic HSL colors from a username hash.

**API key:** Stored in `localStorage` under key `gpuflow_api_key`. Set via "Set API Key" button. Sent as `X-API-Key` header on all fetch calls.

---

### 11. Landing Page (`landing/`, `docs/`)

Static marketing page at `/landing/`. Also published to GitHub Pages at `https://ajayrafa25.github.io/gpuflow/` from the `docs/` folder on `main`.

Content: full-dashboard hero screenshot, 4 command code blocks, 4 panel screenshots, 3-step workflow.

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `dev-change-me` | Auth key for all API requests |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `DB_PATH` | `./gpuflow.db` | SQLite file path |
| `LOG_DIR` | `./logs` | Job log directory |
| `DEFAULT_DOCKER_IMAGE` | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` | Fallback image if none specified |
| `SCHEDULER_POLL_INTERVAL` | `2.0` | Seconds between scheduler ticks |
| `MAX_GPUS_PER_JOB` | `16` | Upper bound on GPUs per job |
| `MLFLOW_PORT` | `5001` | MLflow server port |
| `MLFLOW_STORE_PATH` | `./mlruns` | MLflow backend + artifact root |
| `MLFLOW_CONTAINER_URI` | `http://172.17.0.1:5001` | Tracking URI injected into containers |
| `PUBLIC_HOST` | `localhost` | Hostname used in debug session VS Code URLs |

---

## Deployment

### Requirements
- Python 3.10+
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU(s)

### Install

```bash
git clone https://github.com/ajayrafa25/gpuflow
cd gpuflow
pip install -e .
cp .env.example .env   # set API_KEY
python -m uvicorn gpuflow.api.main:app --host 0.0.0.0 --port 8000
```

### Known constraints
- SQLite DB must be on a writable filesystem. On Lightning Studio the project directory is read-only; set `DB_PATH=/tmp/gpuflow.db` and `LOG_DIR=/tmp/gpuflow_logs`.
- `code-server` on Lightning Studio runs as a single instance on port 30110. Debug sessions reuse this instance rather than launching new ones.
- Docker host IP for MLflow container-to-host routing is `172.17.0.1` (default Docker bridge). Adjust `MLFLOW_CONTAINER_URI` if your Docker network differs.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| API server | FastAPI, uvicorn |
| Async DB | aiosqlite |
| Config | pydantic-settings |
| Docker | docker-py |
| GPU metrics | pynvml |
| HTTP client | httpx |
| MLflow | mlflow ≥ 2.13 |
| CLI | Click |
| CLI output | Rich |
| Dashboard charts | Chart.js 4.4.0 |
| Dashboard fonts | Inter, JetBrains Mono (Google Fonts) |

---

## Roadmap

| Feature | Version |
|---------|---------|
| Priority scheduling | v2 |
| Job resource limits (memory cap) | v2 |
| Multi-node distributed training (multi-machine) | v2 |
| Kubernetes backend | v3 |
| SaaS / hosted offering | v3 |
