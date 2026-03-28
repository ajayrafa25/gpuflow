from __future__ import annotations

from gpuflow.db.store import JobStore
from gpuflow.models.job import Job, JobStatus
from gpuflow.runner.docker_runner import DockerRunner


class Worker:
    def __init__(self, store: JobStore, runner: DockerRunner):
        self._store = store
        self._runner = runner

    async def execute(self, job: Job):
        try:
            await self._runner.run(job, self._store)
        except Exception as exc:
            await self._store.update_status(
                job.id, JobStatus.FAILED, error_message=repr(exc)
            )
