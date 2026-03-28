from __future__ import annotations

import asyncio
import logging

from gpuflow.config import settings
from gpuflow.db.store import JobStore
from gpuflow.gpu.inspector import GPUInspector
from gpuflow.models.job import JobStatus
from gpuflow.worker.worker import Worker

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, store: JobStore, inspector: GPUInspector, worker: Worker):
        self._store = store
        self._inspector = inspector
        self._worker = worker

    async def run(self):
        logger.info("Scheduler started")
        while True:
            try:
                await self._tick()
            except Exception as exc:
                logger.exception("Scheduler tick error: %s", exc)
            await asyncio.sleep(settings.SCHEDULER_POLL_INTERVAL)

    async def _tick(self):
        queued = await self._store.list(status=JobStatus.QUEUED)
        if not queued:
            return

        running = await self._store.get_running_jobs()
        allocated: set[int] = set()
        for job in running:
            allocated.update(job.assigned_gpus)

        all_gpus = await self._inspector.get_all()
        total_gpu_indices = [g.index for g in all_gpus]
        free_indices = [i for i in total_gpu_indices if i not in allocated]

        for job in queued:
            if len(free_indices) < job.requested_gpus:
                break  # FIFO: wait until enough GPUs are free

            assigned = free_indices[: job.requested_gpus]
            free_indices = free_indices[job.requested_gpus :]

            await self._store.update_status(
                job.id, JobStatus.RUNNING, assigned_gpus=assigned
            )
            # Reload the job with updated assigned_gpus
            updated_job = await self._store.get(job.id)
            asyncio.create_task(self._worker.execute(updated_job))
            logger.info("Scheduled job %s on GPUs %s", job.id[:8], assigned)
