from __future__ import annotations

import json
from typing import List, Optional

import aiosqlite

from gpuflow.models.job import Job, JobStatus


class JobStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self):
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                entrypoint TEXT NOT NULL,
                command TEXT,
                requested_gpus INTEGER NOT NULL,
                requested_nodes INTEGER NOT NULL,
                assigned_gpus TEXT NOT NULL DEFAULT '[]',
                docker_image TEXT NOT NULL,
                log_path TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await self._conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()

    def _row_to_job(self, row: aiosqlite.Row) -> Job:
        d = dict(row)
        d["assigned_gpus"] = json.loads(d["assigned_gpus"])
        return Job(**d)

    async def create(self, job: Job) -> Job:
        await self._conn.execute(
            """INSERT INTO jobs
               (id, name, status, entrypoint, command, requested_gpus, requested_nodes,
                assigned_gpus, docker_image, log_path, error_message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id, job.name, job.status.value, job.entrypoint, job.command,
                job.requested_gpus, job.requested_nodes,
                json.dumps(job.assigned_gpus), job.docker_image,
                job.log_path, job.error_message,
                job.created_at.isoformat(),
            ),
        )
        await self._conn.commit()
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        return self._row_to_job(row) if row else None

    async def list(self, status: Optional[JobStatus] = None) -> List[Job]:
        if status:
            async with self._conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC", (status.value,)
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self._conn.execute("SELECT * FROM jobs ORDER BY created_at ASC") as cur:
                rows = await cur.fetchall()
        return [self._row_to_job(r) for r in rows]

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        assigned_gpus: Optional[List[int]] = None,
        log_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        fields = ["status = ?"]
        values: list = [status.value]

        if assigned_gpus is not None:
            fields.append("assigned_gpus = ?")
            values.append(json.dumps(assigned_gpus))
        if log_path is not None:
            fields.append("log_path = ?")
            values.append(log_path)
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)

        values.append(job_id)
        await self._conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values
        )
        await self._conn.commit()

    async def get_running_jobs(self) -> List[Job]:
        return await self.list(status=JobStatus.RUNNING)
