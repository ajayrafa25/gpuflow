from unittest.mock import MagicMock, patch

import pytest

from gpuflow.gpu.inspector import GPUInspector


def test_nvidiasmi_fallback_parsing():
    inspector = GPUInspector.__new__(GPUInspector)
    inspector._use_pynvml = False

    csv_output = "0, Tesla T4, 15360, 512, 10\n1, Tesla T4, 15360, 256, 3\n"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=csv_output, returncode=0)
        gpus = inspector._get_nvidiasmi()

    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].name == "Tesla T4"
    assert gpus[0].total_memory_mb == 15360
    assert gpus[0].used_memory_mb == 512
    assert gpus[0].utilization_pct == 10
    assert gpus[1].index == 1


def test_nvidiasmi_not_found():
    inspector = GPUInspector.__new__(GPUInspector)
    inspector._use_pynvml = False

    with patch("subprocess.run", side_effect=FileNotFoundError):
        gpus = inspector._get_nvidiasmi()

    assert gpus == []


def test_free_gpu_indices():
    from gpuflow.gpu.inspector import GPUInfo
    inspector = GPUInspector.__new__(GPUInspector)

    gpus = [GPUInfo(index=i, name="T4", total_memory_mb=16384,
                    used_memory_mb=0, utilization_pct=0, is_available=True)
            for i in range(4)]

    free = inspector.get_free_gpu_indices(gpus, allocated={0, 2})
    assert free == [1, 3]
