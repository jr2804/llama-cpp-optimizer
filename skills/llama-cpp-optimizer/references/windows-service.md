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

## Choose your setup: hard-coded path (recommended) vs mise-managed (discouraged)

**For a system-wide service, use a hard-coded exe path. Do not wrap the service in `mise x`.**

The mise approach requires six environment variables just so LocalSystem can find
mise's data/cache/state dirs, breaks on trust-store issues, and adds a fragile
resolution layer between Servy and the binary. A stable portable folder with a
direct exe path has none of these problems — and llama.cpp releases are
infrequent enough that auto-update buys little.

Use mise only in **local project directories** where you are *not* registering a service:
mise pins the llama.cpp version per project via `mise.toml`, which is exactly what you want
for dev work — and exactly what you don't want for an always-on service.

| | hard-coded exe path | mise-managed (`mise x`) |
|---|---|---|
| Servy setup | simple: point at the exe directly, zero env vars | complex: `mise x` wrapper + 6 `--envVars` |
| Reliability | high — nothing between Servy and the exe | fragile: trust store, profile dirs, shim lookup can all break silently |
| Updates | manual (re-download, reinstall service) | auto (mise resolves latest) |
| Best for | **system-wide services** | local project dirs only (no service) |

## Create — hard-coded exe path (simpler, recommended for single machines)

Extract llama.cpp to a stable location (e.g. `C:\PortableApps\llama-cpp\`) and point Servy straight at the exe. No `mise x` wrapper, **no `--envVars` at all** — the LocalSystem profile problem disappears because there's no tool to find your user config.

```powershell
# Router mode (presets.ini) — recommended: models load lazily, fast service start
servy-cli install -n llama-cpp `
  -p "C:\PortableApps\llama-cpp\llama-server.exe" `
  --displayName "llama.cpp Server" `
  -d "Local llama.cpp inference server" `
  --startupDir "C:\PortableApps\llama-cpp" `
  "--params=--models-preset presets.ini --port 8001 --host 127.0.0.1 --log-disable" `
  --startupType Automatic
```

> **`presets.ini` is a llama.cpp file, not a Servy file.** Servy is only the
> service wrapper — it does not parse `presets.ini` and has no model-routing
> concept of its own. The INI file is read by `llama-server` itself via the
> `--models-preset` flag (see [server-tuning.md](server-tuning.md) for the
> format). When asked to "generate a prefix-ini for these models", produce a
> llama.cpp `--models-preset` INI — one file, one instance, one port, one
> section per model.

Update routine (after downloading a new release): copy the new `llama-server.exe` (and its `.dll`s) over the old ones, then `servy-cli restart -n llama-cpp`. No reinstall needed if you keep the same path.

```powershell
# Or: single model directly (no presets.ini)
servy-cli install -n llama-cpp-granite `
  -p "C:\PortableApps\llama-cpp\llama-server.exe" `
  --startupDir "C:\PortableApps\llama-cpp" `
  "--params=--model models/granite-4.1-3b-Q4_K_M.gguf --n-gpu-layers 99 --port 8001 --host 127.0.0.1 --log-disable" `
  --startupType Automatic
```

## Create — via `mise x` (discouraged — local project dirs only)

> **Not recommended for system-wide services.** Prefer the direct exe path above. This
> section is kept only for completeness / migrating existing setups. If your goal is a
> per-project pinned llama.cpp without a service, use plain `mise x` in the project shell
> instead of registering it with Servy.

A hardcoded path to the versioned install dir (`b10375`) breaks silently on every mise update. Instead, point the service at `mise.exe` and let it resolve the current version:

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
servy-cli install -n llama-cpp `
  -p $mise `
  --displayName "llama.cpp Server" `
  -d "Local llama.cpp inference server (via mise)" `
  --startupDir "C:\PortableApps\llama-cpp" `
  "--params=x github:ggml-org/llama.cpp -- llama-server.exe --models-preset presets.ini --port 8001 --host 127.0.0.1 --log-disable" `
  "--envVars=MISE_DATA_DIR=C:\Users\<user>\AppData\Local\mise;MISE_CONFIG_DIR=C:\Users\<user>\.config\mise;MISE_CACHE_DIR=C:\Users\<user>\AppData\Local\Temp\mise;MISE_STATE_DIR=C:\Users\<user>\.local\state\mise;USERPROFILE=C:\Users\<user>;HOME=C:\Users\<user>" `
  --startupType Automatic
```

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
