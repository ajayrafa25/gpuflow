from __future__ import annotations

from typing import List

import docker
from fastapi import APIRouter, Depends, HTTPException, Request, status

from gpuflow.api.auth import require_api_key
from gpuflow.api.schemas import DebugSessionRequest, DebugSessionResponse

router = APIRouter(prefix="/debug", tags=["debug"])


def _session_to_response(s) -> DebugSessionResponse:
    return DebugSessionResponse(
        id=s.id,
        vscode_url=s.vscode_url,
        exec_cmd=s.exec_cmd,
        image=s.image,
        container_name=s.container_name,
        created_at=s.created_at,
    )


@router.get("/images")
async def list_images(_: str = Depends(require_api_key)):
    try:
        client = docker.from_env()
        images = client.images.list()
        result = []
        for img in images:
            tags = img.tags or [img.short_id]
            result.append({
                "id": img.short_id,
                "tags": tags,
                "size_mb": round(img.attrs.get("Size", 0) / (1024 * 1024), 1),
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=DebugSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: Request,
    body: DebugSessionRequest,
    _: str = Depends(require_api_key),
):
    manager = request.app.state.session_manager
    try:
        session = await manager.create_session(body.image)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _session_to_response(session)


@router.get("/sessions", response_model=List[DebugSessionResponse])
async def list_sessions(
    request: Request,
    _: str = Depends(require_api_key),
):
    manager = request.app.state.session_manager
    return [_session_to_response(s) for s in manager.list_sessions()]


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def kill_session(
    request: Request,
    session_id: str,
    _: str = Depends(require_api_key),
):
    manager = request.app.state.session_manager
    killed = await manager.kill_session(session_id)
    if not killed:
        raise HTTPException(status_code=404, detail="Session not found")
