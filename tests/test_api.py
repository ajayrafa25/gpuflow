from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gpuflow.api.main import app
from gpuflow.gpu.inspector import GPUInfo


@pytest.fixture
def client(tmp_path):
    from gpuflow.db.store import JobStore
    from gpuflow.gpu.inspector import GPUInspector
    from gpuflow.runner.docker_runner import DockerRunner
    from gpuflow.worker.worker import Worker

    store = MagicMock(spec=JobStore)
    store.create = AsyncMock(side_effect=lambda j: j)
    store.list = AsyncMock(return_value=[])
    store.get = AsyncMock(return_value=None)
    store.update_status = AsyncMock()

    inspector = MagicMock(spec=GPUInspector)
    inspector.get_all = AsyncMock(return_value=[
        GPUInfo(index=0, name="Tesla T4", total_memory_mb=16384,
                used_memory_mb=1024, utilization_pct=5, is_available=True)
    ])

    runner = MagicMock(spec=DockerRunner)
    worker = MagicMock(spec=Worker)

    app.state.store = store
    app.state.inspector = inspector
    app.state.runner = runner
    app.state.worker = worker

    with patch("gpuflow.api.main.JobStore", return_value=store), \
         patch("gpuflow.api.main.GPUInspector", return_value=inspector), \
         patch("gpuflow.api.main.DockerRunner", return_value=runner), \
         patch("gpuflow.api.main.Worker", return_value=worker), \
         patch("gpuflow.api.main.Scheduler") as MockSched:
        sched_instance = MagicMock()
        sched_instance.run = AsyncMock()
        MockSched.return_value = sched_instance
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, store, inspector


API_KEY = "dev-change-me"
HEADERS = {"X-API-Key": API_KEY}


def test_missing_api_key(client):
    c, _, _ = client
    resp = c.post("/api/v1/jobs", json={"name": "x", "entrypoint": "x.py"})
    assert resp.status_code == 422  # missing header


def test_wrong_api_key(client):
    c, _, _ = client
    resp = c.get("/api/v1/jobs", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_submit_job(client):
    c, store, _ = client
    from gpuflow.models.job import Job, JobStatus
    from datetime import datetime, timezone
    fake_job = Job(
        name="test", entrypoint="train.py",
        docker_image="pytorch/pytorch:latest",
        created_at=datetime.now(timezone.utc),
    )
    store.create = AsyncMock(return_value=fake_job)

    resp = c.post("/api/v1/jobs", headers=HEADERS, json={
        "name": "test", "entrypoint": "train.py", "requested_gpus": 2
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test"
    assert data["status"] == "queued"


def test_list_gpus(client):
    c, _, inspector = client
    resp = c.get("/api/v1/gpus", headers=HEADERS)
    assert resp.status_code == 200
    gpus = resp.json()
    assert len(gpus) == 1
    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "Tesla T4"


def test_job_not_found(client):
    c, store, _ = client
    store.get = AsyncMock(return_value=None)
    resp = c.get("/api/v1/jobs/nonexistent", headers=HEADERS)
    assert resp.status_code == 404
