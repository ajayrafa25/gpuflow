from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    status: JobStatus = JobStatus.QUEUED
    entrypoint: str
    command: Optional[str] = None
    requested_gpus: int = 1
    requested_nodes: int = 1
    assigned_gpus: List[int] = Field(default_factory=list)
    docker_image: str
    log_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def new(cls, create: "JobCreate", default_image: str) -> "Job":
        return cls(
            name=create.name,
            entrypoint=create.entrypoint,
            command=create.command,
            requested_gpus=create.requested_gpus,
            requested_nodes=create.requested_nodes,
            docker_image=create.docker_image or default_image,
        )


class JobCreate(BaseModel):
    name: str
    entrypoint: str
    command: Optional[str] = None
    requested_gpus: int = Field(default=1, ge=1)
    requested_nodes: int = Field(default=1, ge=1)
    docker_image: Optional[str] = None
