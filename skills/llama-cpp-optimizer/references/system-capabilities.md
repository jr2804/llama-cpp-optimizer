# System Capabilities Detection

How to detect local system capabilities needed for llama.cpp parameter tuning.

## GPU Detection

### NVIDIA GPUs

```bash
# Basic info
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader

# Detailed info
nvidia-smi -q

# PowerShell: get GPU info
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion"
```

**Key values to extract:**

- GPU name (e.g., "NVIDIA GeForce RTX 3060 Laptop GPU")
- Total VRAM (e.g., 6144 MB = 6 GB)
- Free VRAM (e.g., 5120 MB)
- Compute capability (e.g., 8.6)

### AMD GPUs

```bash
# ROCm
rocm-smi --showmeminfo vram

# PowerShell
powershell -Command "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | Select-Object Name, AdapterRAM"
```

### Intel GPUs

```bash
# Intel GPU
powershell -Command "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*Intel*' } | Select-Object Name, AdapterRAM"
```

### Vulkan (all GPUs)

```bash
# If vulkaninfo is available
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|vulkan|memory"

# Or via llama.cpp itself
llama-bench --help 2>&1 | head -5  # shows loaded backends
```

## RAM Detection

```bash
# Total physical RAM (bytes)
powershell -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"

# Available RAM (bytes)
powershell -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024"

# Human-readable
powershell -Command "$total = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; '{0:N1} GB' -f ($total / 1GB)"
```

## CPU Detection

```bash
# Logical core count
powershell -Command "[Environment]::ProcessorCount"

# CPU name
powershell -Command "(Get-CimInstance Win32_Processor).Name"

# CPU architecture
powershell -Command "(Get-CimInstance Win32_Processor).Architecture"
# 0 = x86, 9 = x64, 12 = ARM64

# RAM speed (important for MoE CPU offloading)
powershell -Command "Get-CimInstance Win32_PhysicalMemory | Select-Object Speed, ConfiguredClockSpeed"
```

## Storage Detection

```bash
# Free disk space on C: (GB)
powershell -Command "$drive = Get-CimInstance Win32_LogicalDisk -Filter 'DeviceID=\"C:\"'; '{0:N1} GB free' -f ($drive.FreeSpace / 1GB)"

# Disk type (SSD vs HDD)
powershell -Command "Get-CimInstance Win32_LogicalDisk -Filter 'DeviceID=\"C:\"' | Select-Object DeviceID, Size, FileSystem"
```

## Unified Detection Script

Run this to get all system info at once:

```bash
powershell -Command "
Write-Host '=== GPU ==='
Get-CimInstance Win32_VideoController | ForEach-Object {
    Write-Host ('  {0}: {1:N0} MB' -f $_.Name, ($_.AdapterRAM / 1MB))
}
Write-Host '=== RAM ==='
$total = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024
Write-Host ('  Total: {0:N1} GB' -f ($total / 1GB))
Write-Host ('  Free:  {0:N1} GB' -f ($free / 1GB))
Write-Host '=== CPU ==='
Write-Host ('  Cores: {0}' -f [Environment]::ProcessorCount)
Write-Host ('  Name:  {0}' -f (Get-CimInstance Win32_Processor).Name)
"
```

## VRAM Estimation for KV Cache

The KV cache size depends on context size, model architecture, and cache type:

```text
KV_cache_bytes = 2 * num_layers * ctx_size * (num_kv_heads * head_dim) * bytes_per_value
```

Where `bytes_per_value` depends on cache type:

- `f16`: 2 bytes
- `q8_0`: 1 byte
- `q4_0`: 0.5 bytes

### Example: Qwen3.6-35B-A3B at 64K context

```text
num_layers = 40
num_kv_heads = 2
head_dim = 256
ctx_size = 64000

KV_cache_K = 40 * 64000 * (2 * 256) * 2 = 2,621,440,000 bytes ≈ 2.4 GB (f16)
KV_cache_V = same = 2.4 GB (f16)
Total KV cache = 4.8 GB (f16)

With q4_0: 40 * 64000 * (2 * 256) * 0.5 = 655,360,000 bytes ≈ 0.6 GB each
Total KV cache = 1.2 GB (q4_0)
```

## GPU Capability Matrix

| GPU | VRAM | Flash Attn | Recommended |
|-----|------|------------|-------------|
| NVIDIA GTX 1060 6GB | 6 GB | No (Pascal) | 3B-8B dense, MoE with CPU offload |
| NVIDIA RTX 3060 6GB | 6 GB | Yes (Ampere) | 3B-14B dense, 35B MoE with CPU offload |
| NVIDIA RTX 3060 Ti 8GB | 8 GB | Yes | 7B-14B dense, 35B MoE partial offload |
| NVIDIA RTX 3070 8GB | 8 GB | Yes | 7B-14B dense, 35B MoE partial offload |
| NVIDIA RTX 3080 10GB | 10 GB | Yes | 7B-30B dense, 35B MoE partial offload |
| NVIDIA RTX 3090 24GB | 24 GB | Yes | 30B-70B dense, 70B MoE partial offload |
| NVIDIA RTX 4090 24GB | 24 GB | Yes | 30B-70B dense, 70B MoE partial offload |
| AMD RX 6800 16GB | 16 GB | Yes (Vulkan) | 7B-30B dense, 35B MoE partial offload |
| AMD RX 7900 XTX 24GB | 24 GB | Yes (Vulkan) | 30B-70B dense, 70B MoE partial offload |
| Apple M1/M2/M3 (unified) | Shared | Yes (Metal) | Depends on total unified memory |
