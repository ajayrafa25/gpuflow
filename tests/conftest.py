import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from gpuflow.db.store import JobStore
from gpuflow.gpu.inspector import GPUInfo, GPUInspector
from gpuflow.models.job import Job, JobCreate, JobStatus


@pytest_asyncio.fixture
async def store(tmp_path):
    s = JobStore(str(tmp_path / "test.db"))
    await s.init()
    yield s
    await s.close()


@pytest.fixture
def mock_inspector():
    inspector = MagicMock(spec=GPUInspector)
    gpus = [
        GPUInfo(index=i, name=f"Tesla T4 #{i}", total_memory_mb=16384,
                used_memory_mb=0, utilization_pct=0, is_available=True)
        for i in range(4)
    ]
    inspector.get_all = AsyncMock(return_value=gpus)
    inspector.get_free_gpu_indices = GPUInspector.get_free_gpu_indices.__get__(inspector)
    return inspector


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.cancel = MagicMock()
    return runner


def make_job(name="test", gpus=1, nodes=1):
    create = JobCreate(name=name, entrypoint="train.py", requested_gpus=gpus, requested_nodes=nodes)
    return Job.new(create, "pytorch/pytorch:latest")
