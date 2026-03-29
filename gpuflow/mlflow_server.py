from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def start(port: int, store_path: str = "./mlruns") -> asyncio.subprocess.Process:
    Path(store_path).mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--backend-store-uri", store_path,
        "--default-artifact-root", store_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    logger.info("MLflow tracking server started on port %s (pid %s)", port, proc.pid)
    return proc
