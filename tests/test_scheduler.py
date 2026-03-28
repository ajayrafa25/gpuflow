import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from gpuflow.models.job import JobStatus
from gpuflow.scheduler.scheduler import Scheduler
from gpuflow.worker.worker import Worker
from tests.conftest import make_job


@pytest_asyncio.fixture
async def scheduler(store, mock_inspector, mock_runner):
    worker = MagicMock(spec=Worker)
    worker.execute = AsyncMock()
    sched = Scheduler(store=store, inspector=mock_inspector, worker=worker)
    return sched, store, worker


@pytest.mark.asyncio
async def test_fifo_schedules_first_two_when_two_gpus_free(store, mock_inspector, mock_runner):
    # Only 2 GPUs available
    from gpuflow.gpu.inspector import GPUInfo
    mock_inspector.get_all = AsyncMock(return_value=[
        GPUInfo(index=0, name="T4", total_memory_mb=16384, used_memory_mb=0, utilization_pct=0, is_available=True),
        GPUInfo(index=1, name="T4", total_memory_mb=16384, used_memory_mb=0, utilization_pct=0, is_available=True),
    ])

    worker = MagicMock()
    worker.execute = AsyncMock()
    sched = Scheduler(store=store, inspector=mock_inspector, worker=worker)

    job1 = make_job("job1", gpus=1)
    job2 = make_job("job2", gpus=1)
    job3 = make_job("job3", gpus=1)
    for j in [job1, job2, job3]:
        await store.create(j)

    await sched._tick()

    j1 = await store.get(job1.id)
    j2 = await store.get(job2.id)
    j3 = await store.get(job3.id)

    assert j1.status == JobStatus.RUNNING
    assert j2.status == JobStatus.RUNNING
    assert j3.status == JobStatus.QUEUED


@pytest.mark.asyncio
async def test_fifo_blocks_on_large_job(store, mock_inspector, mock_runner):
    from gpuflow.gpu.inspector import GPUInfo
    mock_inspector.get_all = AsyncMock(return_value=[
        GPUInfo(index=0, name="T4", total_memory_mb=16384, used_memory_mb=0, utilization_pct=0, is_available=True),
    ])

    worker = MagicMock()
    worker.execute = AsyncMock()
    sched = Scheduler(store=store, inspector=mock_inspector, worker=worker)

    big_job = make_job("big", gpus=4)   # needs 4, only 1 available
    small_job = make_job("small", gpus=1)
    await store.create(big_job)
    await store.create(small_job)

    await sched._tick()

    bj = await store.get(big_job.id)
    sj = await store.get(small_job.id)

    # FIFO: big job blocks, small job behind it also waits
    assert bj.status == JobStatus.QUEUED
    assert sj.status == JobStatus.QUEUED


@pytest.mark.asyncio
async def test_no_double_allocation(store, mock_inspector, mock_runner):
    from gpuflow.gpu.inspector import GPUInfo
    mock_inspector.get_all = AsyncMock(return_value=[
        GPUInfo(index=0, name="T4", total_memory_mb=16384, used_memory_mb=0, utilization_pct=0, is_available=True),
        GPUInfo(index=1, name="T4", total_memory_mb=16384, used_memory_mb=0, utilization_pct=0, is_available=True),
    ])

    worker = MagicMock()
    worker.execute = AsyncMock()
    sched = Scheduler(store=store, inspector=mock_inspector, worker=worker)

    job1 = make_job("j1", gpus=1)
    await store.create(job1)
    await sched._tick()

    updated = await store.get(job1.id)
    assert updated.status == JobStatus.RUNNING
    assert len(updated.assigned_gpus) == 1
    gpu0 = updated.assigned_gpus[0]

    # Second tick: job1 now running, job2 should get the other GPU
    job2 = make_job("j2", gpus=1)
    await store.create(job2)
    await sched._tick()

    updated2 = await store.get(job2.id)
    assert updated2.status == JobStatus.RUNNING
    assert updated2.assigned_gpus[0] != gpu0
