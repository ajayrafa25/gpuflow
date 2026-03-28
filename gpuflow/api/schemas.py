from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from gpuflow.models.job import JobStatus


class JobSubmitRequest(BaseModel):
    name: str
    entrypoint: str
    command: Optional[str] = None
    requested_gpus: int = Field(default=1, ge=1)
    requested_nodes: int = Field(default=1, ge=1)
    docker_image: Optional[str] = None


class JobResponse(BaseModel):
    id: str
    name: str
    status: JobStatus
    entrypoint: str
    command: Optional[str]
    requested_gpus: int
    requested_nodes: int
    assigned_gpus: List[int]
    docker_image: str
    log_path: Optional[str]
    error_message: Optional[str]
    created_at: datetime


class GPUResponse(BaseModel):
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    utilization_pct: int
    is_available: bool
