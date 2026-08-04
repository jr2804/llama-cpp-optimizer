# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Detect system capabilities and output as JSON.

Usage:
    uv run detect-system.py

Outputs structured JSON with GPU, RAM, CPU, and storage information.
"""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class GpuInfo:
    """Information about a single GPU."""

    name: str
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    compute_cap: str = ""
    backend: str = ""


@dataclass
class SystemInfo:
    """Complete system capabilities snapshot."""

    platform: str = ""
    os_version: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    cpu_cores_logical: int = 0
    cpu_name: str = ""
    cpu_architecture: str = ""
    ram_speed_mhz: int = 0
    disk_free_gb: float = 0.0


def run_powershell(script: str) -> str:
    """Run a PowerShell command and return stdout."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def detect_gpu_nvidia(info: SystemInfo) -> None:
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                info.gpus.append(
                    GpuInfo(
                        name=parts[0],
                        vram_total_mb=int(float(parts[1])),
                        vram_free_mb=int(float(parts[2])),
                        compute_cap=parts[3],
                        backend="cuda",
                    )
                )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass


def detect_gpu_wmi(info: SystemInfo) -> None:
    """Detect GPUs via WMI (fallback, also catches AMD/Intel)."""
    if info.gpus:
        return  # already have NVIDIA info
    output = run_powershell(
        "Get-CimInstance Win32_VideoController | "
        'Select-Object Name, AdapterRAM | ConvertTo-Json -Compress'
    )
    if not output:
        return
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            data = [data]
        for g in data:
            name = g.get("Name", "Unknown GPU")
            vram = g.get("AdapterRAM", 0)
            if vram:
                vram_mb = vram // (1024 * 1024)
            else:
                vram_mb = 0
            backend = "unknown"
            name_lower = name.lower()
            if "nvidia" in name_lower:
                backend = "cuda"
            elif "amd" in name_lower or "radeon" in name_lower:
                backend = "vulkan"
            elif "intel" in name_lower:
                backend = "vulkan"
            info.gpus.append(
                GpuInfo(name=name, vram_total_mb=vram_mb, backend=backend)
            )
    except (json.JSONDecodeError, KeyError):
        pass


def detect_ram(info: SystemInfo) -> None:
    """Detect total and free RAM."""
    total = run_powershell(
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
    )
    free = run_powershell(
        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"
    )
    try:
        if total:
            info.ram_total_gb = round(int(total) / (1024**3), 1)
    except ValueError:
        pass
    try:
        if free:
            info.ram_free_gb = round(int(free) * 1024 / (1024**3), 1)
    except ValueError:
        pass


def detect_cpu(info: SystemInfo) -> None:
    """Detect CPU information."""
    cores = run_powershell("[Environment]::ProcessorCount")
    name = run_powershell("(Get-CimInstance Win32_Processor).Name")
    arch = run_powershell("(Get-CimInstance Win32_Processor).Architecture")
    speed = run_powershell(
        "Get-CimInstance Win32_PhysicalMemory | "
        "Select-Object -First 1 -ExpandProperty Speed"
    )

    try:
        if cores:
            info.cpu_cores_logical = int(cores)
    except ValueError:
        pass
    if name:
        info.cpu_name = name.strip()
    if arch:
        arch_map = {"0": "x86", "9": "x64", "12": "ARM64"}
        info.cpu_architecture = arch_map.get(arch.strip(), arch.strip())
    try:
        if speed:
            info.ram_speed_mhz = int(speed)
    except ValueError:
        pass


def detect_disk(info: SystemInfo) -> None:
    """Detect free disk space on C:."""
    free = run_powershell(
        "$d = Get-CimInstance Win32_LogicalDisk -Filter 'DeviceID=\"C:\"'; "
        "[math]::Round($d.FreeSpace / 1GB, 1)"
    )
    try:
        if free:
            info.disk_free_gb = float(free)
    except ValueError:
        pass


def main() -> None:
    """Detect system capabilities and print as JSON."""
    info = SystemInfo()
    info.platform = sys.platform
    info.os_version = platform.version()

    detect_gpu_nvidia(info)
    detect_gpu_wmi(info)
    detect_ram(info)
    detect_cpu(info)
    detect_disk(info)

    print(json.dumps(asdict(info), indent=2))


if __name__ == "__main__":
    main()
