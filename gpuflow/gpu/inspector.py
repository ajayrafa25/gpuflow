from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    utilization_pct: int
    is_available: bool


class GPUInspector:
    def __init__(self):
        self._use_pynvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._use_pynvml = True
        except Exception:
            self._pynvml = None

    async def get_all(self) -> List[GPUInfo]:
        loop = asyncio.get_event_loop()
        if self._use_pynvml:
            return await loop.run_in_executor(None, self._get_pynvml)
        return await loop.run_in_executor(None, self._get_nvidiasmi)

    def _get_pynvml(self) -> List[GPUInfo]:
        pynvml = self._pynvml
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                util = 0
            total_mb = mem.total // (1024 * 1024)
            used_mb = mem.used // (1024 * 1024)
            gpus.append(GPUInfo(
                index=i,
                name=name,
                total_memory_mb=total_mb,
                used_memory_mb=used_mb,
                utilization_pct=util,
                is_available=(used_mb < total_mb * 0.9),
            ))
        return gpus

    def _get_nvidiasmi(self) -> List[GPUInfo]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[0])
                name = parts[1]
                total_mb = int(parts[2])
                used_mb = int(parts[3])
                util = int(parts[4])
                gpus.append(GPUInfo(
                    index=idx,
                    name=name,
                    total_memory_mb=total_mb,
                    used_memory_mb=used_mb,
                    utilization_pct=util,
                    is_available=(used_mb < total_mb * 0.9),
                ))
            except (ValueError, IndexError):
                continue
        return gpus

    def get_free_gpu_indices(self, all_gpus: List[GPUInfo], allocated: Set[int]) -> List[int]:
        return [g.index for g in all_gpus if g.index not in allocated]
