from typing import List

from fastapi import APIRouter, Depends, Request

from gpuflow.api.auth import require_api_key
from gpuflow.api.schemas import GPUResponse

router = APIRouter(prefix="/gpus", tags=["gpus"])


@router.get("", response_model=List[GPUResponse])
async def list_gpus(
    request: Request,
    _: str = Depends(require_api_key),
):
    inspector = request.app.state.inspector
    gpus = await inspector.get_all()
    return [GPUResponse(**g.__dict__) for g in gpus]
