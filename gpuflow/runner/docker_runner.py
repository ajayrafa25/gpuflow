from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, Optional

import docker
import docker.types

from gpuflow.config import settings
from gpuflow.models.job import Job, JobStatus


class DockerRunner:
    def __init__(self):
        self._client = docker.from_env()
        self._containers: Dict[str, object] = {}

    def _build_command(self, job: Job) -> str:
        if job.command:
            return job.command
        n = job.requested_gpus
        if n > 1 or job.requested_nodes > 1:
            return f"torchrun --nproc_per_node={n} {job.entrypoint}"
        return f"python {job.entrypoint}"

    def _build_env(self, job: Job) -> dict:
        env: dict = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in job.assigned_gpus),
            "MLFLOW_TRACKING_URI": settings.MLFLOW_CONTAINER_URI,
        }
        if job.requested_nodes > 1:
            env.update({
                "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
                "MASTER_PORT": "29500",
                "WORLD_SIZE": str(job.requested_gpus * job.requested_nodes),
                "RANK": "0",
            })
        return env

    def _run_blocking(self, job: Job, log_path: str):
        command = self._build_command(job)
        env = self._build_env(job)

        device_ids = [str(i) for i in job.assigned_gpus] if job.assigned_gpus else []
        device_requests = []
        if device_ids:
            device_requests = [
                docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])
            ]

        cwd = os.getcwd()
        volumes = {cwd: {"bind": "/workspace", "mode": "rw"}}

        container = self._client.containers.run(
            job.docker_image,
            command=f"bash -c '{command}'",
            environment=env,
            device_requests=device_requests,
            volumes=volumes,
            working_dir="/workspace",
            detach=True,
        )
        self._containers[job.id] = container

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as f:
            for chunk in container.logs(stream=True, follow=True):
                f.write(chunk)

        result = container.wait()
        container.remove(force=True)
        self._containers.pop(job.id, None)
        return result["StatusCode"]

    async def run(self, job: Job, store) -> None:
        log_path = str(Path(settings.LOG_DIR) / f"{job.id}.log")
        await store.update_status(job.id, JobStatus.RUNNING, log_path=log_path)

        loop = asyncio.get_running_loop()
        try:
            exit_code = await loop.run_in_executor(None, self._run_blocking, job, log_path)
            if exit_code == 0:
                await store.update_status(job.id, JobStatus.COMPLETED)
            else:
                await store.update_status(
                    job.id, JobStatus.FAILED,
                    error_message=f"Container exited with code {exit_code}"
                )
        except Exception as exc:
            self._containers.pop(job.id, None)
            await store.update_status(
                job.id, JobStatus.FAILED,
                error_message=str(exc)
            )

    def cancel(self, job_id: str):
        container = self._containers.get(job_id)
        if container:
            try:
                container.kill()
            except Exception:
                pass
            self._containers.pop(job_id, None)
