from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from gpuflow.api.routes.gpus import router as gpus_router
from gpuflow.api.routes.jobs import router as jobs_router
from gpuflow.config import settings
from gpuflow.db.store import JobStore
from gpuflow.gpu.inspector import GPUInspector
from gpuflow.runner.docker_runner import DockerRunner
from gpuflow.scheduler.scheduler import Scheduler
from gpuflow.worker.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = JobStore(settings.DB_PATH)
    await store.init()

    inspector = GPUInspector()
    runner = DockerRunner()
    worker = Worker(store=store, runner=runner)
    scheduler = Scheduler(store=store, inspector=inspector, worker=worker)

    app.state.store = store
    app.state.inspector = inspector
    app.state.runner = runner
    app.state.worker = worker

    scheduler_task = asyncio.create_task(scheduler.run())
    logger.info("GPUFlow server started on %s:%s", settings.API_HOST, settings.API_PORT)

    yield

    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass
    await store.close()
    logger.info("GPUFlow server shut down")


app = FastAPI(title="GPUFlow", version="1.0.0", lifespan=lifespan)

app.include_router(jobs_router, prefix="/api/v1")
app.include_router(gpus_router, prefix="/api/v1")

_dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
if _dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard")
