# 📄 SPEC.md — GPUFlow v1

# 1. 🧭 Product Overview

## Product Name
GPUFlow

## Tagline
Simple GPU scheduling and distributed training for ML teams

## Problem Statement

ML teams struggle with:
- manual GPU allocation
- poor GPU utilization
- broken distributed training (NCCL issues)
- lack of visibility into running jobs

Existing solutions:
- Kubernetes → too complex  
- Slurm → powerful but hard to configure  

Need a simple, ML-native scheduler.

## Solution

GPUFlow provides:
- job submission CLI
- GPU-aware scheduler
- distributed training launcher
- minimal dashboard
- Docker-based execution

# 2. 🎯 Goals

## Primary Goals (MVP)
- Run ML jobs on shared GPUs
- Avoid GPU conflicts
- Support multi-GPU training
- Support multi-node training (basic)
- Provide job queue + logs
- Simple install

## Non-Goals (v1)
- full Kubernetes replacement
- model deployment
- feature store
- hyperparameter tuning
- autoscaling cloud infra

# 3. 🏗️ System Architecture

High-level:

CLI → API Server → Scheduler / Worker / GPU Inspector → Docker + Logs

# 4. 🧱 Core Components

## CLI
Commands:
gpuflow run train.py --gpus 2 --name exp1
gpuflow status
gpuflow logs <job_id>
gpuflow cancel <job_id>

## API Server
- FastAPI
- Handles jobs, GPUs, users

## Scheduler
- FIFO queue
- GPU allocation

## Worker
- executes jobs

## Runner
- launches Docker
- streams logs

## GPU Inspector
- pynvml / nvidia-smi

# 5. 🧩 Data Model

Job:
- id
- name
- status
- entrypoint
- command
- requested_gpus
- requested_nodes
- assigned_gpus
- docker_image
- log_path
- error_message
- created_at

# 6. ⚙️ Job Execution Flow

1. User submits job
2. Job queued
3. Scheduler assigns GPUs
4. Worker runs job
5. Docker container launched
6. Logs stored
7. Job completes

# 7. 🚀 Distributed Training

Single node:
torchrun --nproc_per_node=2 train.py

Multi-node:
- MASTER_ADDR
- MASTER_PORT
- WORLD_SIZE
- RANK

# 8. 🐳 Docker Execution

docker run --gpus all -e CUDA_VISIBLE_DEVICES=0,1 <image>

# 9. 📊 API Endpoints

POST /jobs
GET /jobs
GET /jobs/{id}
DELETE /jobs/{id}
GET /gpus

# 10. 🧪 Scheduler Logic

free_gpus = all - busy
if enough → assign else wait

# 11. 🖥️ UI

- dashboard
- jobs
- logs

# 12. 🔐 Security

- local
- API key

# 13. 📦 Deployment

install.sh
docker-compose

# 14. 📈 Metrics

- GPU utilization
- job success rate
- queue wait time

# 15. 🧱 Roadmap

v2:
- Kubernetes
- MLflow
- priority scheduling

v3:
- SaaS
- billing

# 16. ⚠️ Risks

- GPU contention
- NCCL issues
- storage bottlenecks

# 17. ✅ Acceptance Criteria

- job runs
- GPUs allocated correctly
- logs visible
- Docker isolation works
