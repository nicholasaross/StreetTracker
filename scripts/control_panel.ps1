# Launch the StreetTracker operator control panel, detached.
#
# Hosting model: on-demand. This starts the panel in its own process so it
# outlives the launching terminal (closing this shell does NOT stop it); it
# ends at logout / reboot, or when you stop it (see below). It runs OUTSIDE any
# Claude Code helper tree, so it survives Claude app recycles and consumes zero
# tokens once running.
#
# Usage:
#   pwsh -NoProfile -File scripts/control_panel.ps1                 # 0.0.0.0:8095, opens browser
#   pwsh -NoProfile -File scripts/control_panel.ps1 -Port 8096
#   pwsh -NoProfile -File scripts/control_panel.ps1 -BindHost 127.0.0.1   # localhost only
#   pwsh -NoProfile -File scripts/control_panel.ps1 -NoBrowser
#
# Access model: binds 0.0.0.0 by default so the dashboards are viewable from any
# LAN device (phone / wall screen); control actions stay localhost-only,
# enforced server-side. Pass -BindHost 127.0.0.1 to keep it fully local.
#
# Stop it:  Get-Process | Where-Object { $_.Path -like '*uv*' -and ... }  -- or
# simply close the spawned window, or `taskkill /IM uv.exe` if it's the only uv.

param(
    [int]$Port = 8095,
    [string]$BindHost = "0.0.0.0",
    [string]$OutputRoot = "output",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot   # scripts/.. = repo root
Set-Location $repo

$panelArgs = @(
    "run", "streettracker", "control",
    "--output-root", $OutputRoot,
    "--host", $BindHost,
    "--port", $Port
)

Write-Host "[control] starting panel: uv $($panelArgs -join ' ')  (cwd=$repo)"
Start-Process -FilePath "uv" -ArgumentList $panelArgs -WorkingDirectory $repo -WindowStyle Minimized

Write-Host "[control] panel launching on http://localhost:$Port/  (bind=$BindHost, output-root=$OutputRoot)"
Write-Host "[control] control actions are localhost-only; dashboards are visible to $BindHost."

if (-not $NoBrowser) {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:$Port/"
}
