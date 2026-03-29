from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from gpuflow.api.routes.debug import router as debug_router
from gpuflow.api.routes.gpus import router as gpus_router
from gpuflow.api.routes.jobs import router as jobs_router
from gpuflow.api.routes.mlflow import router as mlflow_router
from gpuflow.config import settings
from gpuflow.db.store import JobStore
from gpuflow.debug.session_manager import SessionManager
from gpuflow.gpu.inspector import GPUInspector
from gpuflow.mlflow_server import start as start_mlflow
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
    session_manager = SessionManager(workspace=os.getcwd())

    app.state.store = store
    app.state.inspector = inspector
    app.state.runner = runner
    app.state.worker = worker
    app.state.session_manager = session_manager

    scheduler_task = asyncio.create_task(scheduler.run())

    # Start MLflow tracking server
    mlflow_proc = await start_mlflow(
        port=settings.MLFLOW_PORT,
        store_path=settings.MLFLOW_STORE_PATH,
    )
    app.state.mlflow_proc = mlflow_proc

    logger.info("GPUFlow server started on %s:%s", settings.API_HOST, settings.API_PORT)

    yield

    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

    await session_manager.kill_all()

    mlflow_proc.terminate()
    try:
        await asyncio.wait_for(mlflow_proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        mlflow_proc.kill()

    await store.close()
    logger.info("GPUFlow server shut down")


app = FastAPI(title="GPUFlow", version="1.0.0", lifespan=lifespan)

app.include_router(jobs_router, prefix="/api/v1")
app.include_router(gpus_router, prefix="/api/v1")
app.include_router(mlflow_router, prefix="/api/v1")
app.include_router(debug_router, prefix="/api/v1")

_dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
if _dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")

_landing_path = Path(__file__).parent.parent.parent / "landing"
if _landing_path.exists():
    app.mount("/landing", StaticFiles(directory=str(_landing_path), html=True), name="landing")

_screenshots_path = Path(__file__).parent.parent.parent / "screenshots" / "live"
if _screenshots_path.exists():
    app.mount("/screenshots", StaticFiles(directory=str(_screenshots_path)), name="screenshots")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/landing")


_MLFLOW_UPSTREAM = f"http://localhost:{settings.MLFLOW_PORT}"
_mlflow_proxy_client = httpx.AsyncClient(base_url=_MLFLOW_UPSTREAM, follow_redirects=True, timeout=30)


@app.api_route("/mlflow/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"], include_in_schema=False)
async def mlflow_proxy(path: str, request: Request):
    url = f"/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    body = await request.body()
    resp = await _mlflow_proxy_client.request(request.method, url, headers=headers, content=body)
    # Rewrite absolute URLs in the response so static assets resolve correctly
    excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
    return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers, media_type=resp.headers.get("content-type"))
