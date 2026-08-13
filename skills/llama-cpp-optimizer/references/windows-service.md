# Windows Service Reference

Running `llama-server` as a Windows service using [Servy](https://github.com/aelassas/servy).

## Install Servy

```powershell
# Download latest release
# https://github.com/aelassas/servy/releases

# Or via Scoop
scoop bucket add extras
scoop install servy
```

CLI docs: <https://github.com/aelassas/servy/wiki/Servy-CLI>

## Create — via `mise x` (version-proof, recommended)

A hardcoded path to the versioned install dir (`b10361`) breaks silently on every mise update. Instead, point the service at `mise.exe` and let it resolve the current version:

```powershell
$mise = (Get-Command mise).Source   # resolves the WinGet symlink to mise.exe
```

### Required `--envVars` (LocalSystem can't see your user profile)

The service runs as LocalSystem by default, which resolves `~` to `C:\Windows\System32\config\systemprofile` — mise can't find your config, installs, or trust store. Pass them explicitly:

| Var | Value (adjust to your user) |
|-----|------------------------------|
| `MISE_DATA_DIR` | `C:\Users\<user>\AppData\Local\mise` |
| `MISE_CONFIG_DIR` | `C:\Users\<user>\.config\mise` |
| `MISE_CACHE_DIR` | `C:\Users\<user>\AppData\Local\Temp\mise` |
| `MISE_STATE_DIR` | `C:\Users\<user>\.local\state\mise` — **required**, else the project `mise.toml` is "not trusted" (trust store is per-user) |
| `USERPROFILE`, `HOME` | `C:\Users\<user>` |

### Use the `.exe` suffix

`mise x github:ggml-org/llama.cpp -- llama-server.exe ...` — without `.exe`, mise looks up a shim by that name and fails with `cannot find binary path` in a non-interactive env.

```powershell
# Router mode (presets.ini) — recommended: models load lazily, fast service start
servy-cli install -n llama-cpp `
  -p $mise `
  --displayName "llama.cpp Server" `
  -d "Local llama.cpp inference server (via mise)" `
  --startupDir "C:\PortableApps\llama-cpp" `
  "--params=x github:ggml-org/llama.cpp -- llama-server.exe --models-preset presets.ini --port 8001 --host 127.0.0.1 --log-disable" `
  "--envVars=MISE_DATA_DIR=C:\Users\<user>\AppData\Local\mise;MISE_CONFIG_DIR=C:\Users\<user>\.config\mise;MISE_CACHE_DIR=C:\Users\<user>\AppData\Local\Temp\mise;MISE_STATE_DIR=C:\Users\<user>\.local\state\mise;USERPROFILE=C:\Users\<user>;HOME=C:\Users\<user>" `
  --startupType Automatic
```

## Create — direct exe path (simpler, breaks on update)

Only if you accept re-installing the service after each mise update. Or for a single model directly (no presets.ini):

```powershell
# Single model directly (no presets.ini)
servy-cli install -n llama-cpp-granite `
  -p "C:\Users\jan.reimes\AppData\Local\mise\installs\github-ggml-org-llama-cpp\b10361\llama-server.exe" `
  --startupDir "C:\PortableApps\llama-cpp" `
  "--params=--model models/granite-4.1-3b-Q4_K_M.gguf --n-gpu-layers 99 --port 8001 --host 127.0.0.1 --log-disable" `
  --startupType Automatic
```

The current CLI is `servy-cli.exe` (the `create` verb was renamed to `install` in newer versions).

## Control

```powershell
servy-cli start -n llama-cpp
servy-cli stop -n llama-cpp
servy-cli restart -n llama-cpp
servy-cli status -n llama-cpp
# Or in the Services console (services.msc)
Get-Service llama-cpp
```

## Remove

```powershell
servy-cli uninstall -n llama-cpp
```

## Notes

- **Admin required**: `install`/`uninstall` need an elevated shell. Run the CLI from an elevated PowerShell, or wrap it in a `.ps1` and `Start-Process powershell -Verb RunAs`.
- **`--params` quoting**: pass the whole parameter string as one argument: `"--params=--models-preset presets.ini --port 8001 ..."`. Split args cause "The specified path is invalid." / "unknown option" errors.
- **`--envVars` format**: semicolon-separated `VAR=value` pairs, one argument.
- **Diagnose failures**: add `--stdout C:\PortableApps\llama-cpp\logs\llama-svc-stdout.log --stderr C:\PortableApps\llama-cpp\logs\llama-svc-stderr.log` to `install` (or the Application event log, provider `Servy`) — the stderr file shows the real mise error (untrusted config, missing binary, etc.).
- `--log-disable` suppresses console output (logs are otherwise lost without a console window).
- Use `--log-on-message` and `--log-on-event` to capture logs to the Windows Event Log if needed.
- The service runs under `LocalSystem` by default. For GPU access, this is sufficient on most setups. If you get CUDA/Vulkan errors, try `--cred-domain` or run under a user account with GPU access.
- Router mode (`--models-preset`) is recommended: models load lazily on first request, so the service starts fast even with multi-GB models.

CLI reference: <https://github.com/aelassas/servy/wiki/Servy-CLI>
Desktop app: <https://github.com/aelassas/servy/wiki/Servy-Desktop-App>
