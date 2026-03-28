from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from gpuflow.api.auth import require_api_key
from gpuflow.api.schemas import JobResponse, JobSubmitRequest
from gpuflow.config import settings
from gpuflow.models.job import Job, JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def submit_job(
    request: Request,
    body: JobSubmitRequest,
    _: str = Depends(require_api_key),
):
    store = request.app.state.store
    job = Job.new(body, settings.DEFAULT_DOCKER_IMAGE)
    await store.create(job)
    return job


@router.get("", response_model=List[JobResponse])
async def list_jobs(
    request: Request,
    status_filter: Optional[str] = None,
    _: str = Depends(require_api_key),
):
    store = request.app.state.store
    job_status = None
    if status_filter:
        try:
            job_status = JobStatus(status_filter)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_filter}")
    return await store.list(status=job_status)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    request: Request,
    job_id: str,
    _: str = Depends(require_api_key),
):
    store = request.app.state.store
    job = await store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    request: Request,
    job_id: str,
    _: str = Depends(require_api_key),
):
    store = request.app.state.store
    runner = request.app.state.runner
    job = await store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(status_code=400, detail=f"Job already in terminal state: {job.status}")
    if job.status == JobStatus.RUNNING:
        runner.cancel(job_id)
    await store.update_status(job_id, JobStatus.CANCELLED)


@router.get("/{job_id}/logs")
async def stream_logs(
    request: Request,
    job_id: str,
    follow: bool = False,
    _: str = Depends(require_api_key),
):
    store = request.app.state.store
    job = await store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.log_path:
        raise HTTPException(status_code=404, detail="No logs available yet")

    log_file = Path(job.log_path)
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    async def generate():
        with open(log_file, "rb") as f:
            while True:
                chunk = f.read(4096)
                if chunk:
                    yield chunk
                elif follow and job.status == JobStatus.RUNNING:
                    await asyncio.sleep(0.5)
                else:
                    break

    return StreamingResponse(generate(), media_type="text/plain")
