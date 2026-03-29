from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

import httpx

from gpuflow.api.auth import require_api_key
from gpuflow.config import settings

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

_MLFLOW_BASE = f"http://localhost:{settings.MLFLOW_PORT}"


async def _mlflow_get(path: str):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_MLFLOW_BASE}{path}")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="MLflow server not available yet")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@router.get("/experiments")
async def list_experiments(_: str = Depends(require_api_key)):
    return await _mlflow_get("/api/2.0/mlflow/experiments/search?max_results=20")


@router.get("/runs")
async def recent_runs(limit: int = 10, _: str = Depends(require_api_key)):
    data = await _mlflow_get(
        f"/api/2.0/mlflow/runs/search?max_results={limit}"
    )
    return data
