from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import docker
import docker.types

from gpuflow.config import settings

logger = logging.getLogger(__name__)

# The existing VS Code server in this environment
_VSCODE_PORT = 30110


@dataclass
class DebugSession:
    id: str
    image: str
    container_id: str
    container_name: str
    workspace: str
    created_at: datetime

    @property
    def vscode_url(self) -> str:
        """URL to open the workspace in the running VS Code server."""
        encoded = self.workspace.replace("/", "%2F")
        return f"http://{settings.PUBLIC_HOST}:{_VSCODE_PORT}/?folder={encoded}"

    @property
    def exec_cmd(self) -> str:
        return f"docker exec -it {self.container_name} bash"


class SessionManager:
    def __init__(self, workspace: str):
        self._workspace = workspace
        self._sessions: Dict[str, DebugSession] = {}
        self._client = docker.from_env()

    def _container_alive(self, container_id: str) -> bool:
        try:
            c = self._client.containers.get(container_id)
            return c.status == "running"
        except Exception:
            return False

    async def create_session(self, image: str) -> DebugSession:
        # Clean dead sessions
        dead = [sid for sid, s in self._sessions.items() if not self._container_alive(s.container_id)]
        for sid in dead:
            self._sessions.pop(sid)

        session_id = str(uuid.uuid4())
        container_name = f"gpuflow-debug-{session_id[:8]}"

        def _start():
            device_requests = []
            try:
                import pynvml
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                if count > 0:
                    device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
            except Exception:
                pass

            return self._client.containers.run(
                image,
                command="sleep infinity",
                name=container_name,
                volumes={self._workspace: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                device_requests=device_requests,
                detach=True,
                remove=False,
            )

        loop = asyncio.get_running_loop()
        container = await loop.run_in_executor(None, _start)

        session = DebugSession(
            id=session_id,
            image=image,
            container_id=container.id,
            container_name=container_name,
            workspace=self._workspace,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = session
        logger.info("Debug session %s started container %s from image %s", session_id[:8], container_name, image)
        return session

    def list_sessions(self) -> List[DebugSession]:
        return [s for s in self._sessions.values() if self._container_alive(s.container_id)]

    async def kill_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False

        def _stop():
            try:
                c = self._client.containers.get(session.container_id)
                c.stop(timeout=5)
                c.remove(force=True)
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _stop)
        self._sessions.pop(session_id, None)
        logger.info("Debug session %s killed", session_id[:8])
        return True

    async def kill_all(self):
        for session_id in list(self._sessions.keys()):
            await self.kill_session(session_id)
