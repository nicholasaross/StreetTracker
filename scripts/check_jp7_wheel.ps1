#!/usr/bin/env pwsh
#Requires -Version 7.0
<#
.SYNOPSIS
  Gate-check for the StreetTracker JetPack 7.2 Orin upgrade: is the torch wheel
  the upgrade needs published on the jetson-ai-lab index yet?

.DESCRIPTION
  The single-Orin upgrade is blocked until pypi.jetson-ai-lab.io publishes a
  JetPack-7 / CUDA-13.x torch (+ torchvision) wheel built for CPython 3.12
  (cp312) on aarch64. The Orin's sm_87 GPU kernels live INSIDE the wheel, not in
  its filename, and the jetson-ai-lab index is sm_87-targeted by construction --
  so a matching cp312/aarch64 wheel there IS the publication signal. (Actual
  sm_87 execution can only be proven on the device.)

  A self-sanity probe hits a known-populated JP6 shelf first, so a "nothing
  found" on jp7 is distinguishable from a parser/site failure.

  Exits 0 when a matching torch+torchvision pair is found on one shelf, else 1
  -- so it can be wired into a scheduled re-check later. (Exit 1 = "not yet",
  not an error.)

.EXAMPLE
  pwsh -ExecutionPolicy Bypass -File scripts/check_jp7_wheel.ps1
#>
[CmdletBinding()]
param(
    [string]   $Base      = 'https://pypi.jetson-ai-lab.io',
    [string]   $JetPack   = 'jp7',
    [string]   $PyTag     = 'cp312',
    [string]   $PlatTag   = 'aarch64',
    [string[]] $CudaProbe = @('cu132','cu131','cu130','cu129','cu128'),
    [string[]] $Packages  = @('torch','torchvision'),
    [version]  $MinTorch  = '2.10'
)

function Get-Url([string]$Url) {
    try {
        $r = Invoke-WebRequest -Uri $Url -SkipHttpErrorCheck -TimeoutSec 20 `
                -MaximumRedirection 5 -Headers @{ 'User-Agent' = 'streettracker-jp7-check/1' }
        [pscustomobject]@{ Status = [int]$r.StatusCode; Content = [string]$r.Content }
    } catch {
        [pscustomobject]@{ Status = -1; Content = $_.Exception.Message }
    }
}

# Parse wheel filenames for a package out of a PEP 503 "simple" page.
function Get-Wheels([string]$Content, [string]$Pkg) {
    $pat = [regex]::Escape($Pkg) + '-[0-9][^"''<>\s]*\.whl'
    $names = [regex]::Matches($Content, $pat) | ForEach-Object { $_.Value } | Sort-Object -Unique
    foreach ($n in $names) {
        # {name}-{version}(-{build})?-{pytag}-{abitag}-{plat}.whl
        $parts = ($n -replace '\.whl$', '') -split '-'
        if ($parts.Count -lt 5) { continue }
        [pscustomobject]@{ File = $n; Version = $parts[1]; PyTag = $parts[-3]; Plat = $parts[-1] }
    }
}

function Test-MinVersion([string]$Version, [version]$Min) {
    $core = ([regex]::Match($Version, '^\d+(\.\d+){1,3}')).Value
    if (-not $core) { return $null }
    try { return ([version]$core -ge $Min) } catch { return $null }
}

Write-Host ""
Write-Host "== StreetTracker JP7.2 wheel gate-check ==" -ForegroundColor Cyan
Write-Host ("Index : {0}" -f $Base)
Write-Host ("Need  : '{0}' shelf with {1} + *{2}* wheels for [{3}] (torch >= {4})" -f `
    $JetPack, $PyTag, $PlatTag, ($Packages -join ', '), $MinTorch)
Write-Host ("Time  : {0}" -f (Get-Date).ToString('u'))
Write-Host ""

# --- 0. parser sanity against a known-populated JP6 shelf -------------------
$sanityN = 0; $sanityCu = $null; $sanityStatus = $null
foreach ($sc in @('cu129', 'cu126', 'cu128')) {
    $s = Get-Url "$Base/jp6/$sc/+simple/torch/"
    $sanityStatus = $s.Status
    if ($s.Status -eq 200) {
        $n = (Get-Wheels $s.Content 'torch' | Measure-Object).Count
        if ($n -gt 0) { $sanityN = $n; $sanityCu = $sc; break }
    }
}
if ($sanityN -gt 0) {
    Write-Host ("[sanity] jp6/{0} torch -> {1} wheel(s) parsed (parser OK)" -f $sanityCu, $sanityN) -ForegroundColor DarkGray
} else {
    Write-Host ("[sanity] WARN: no JP6 torch wheels parsed (last HTTP {0}). A 'not found' below may be a site/parser issue, not absence." -f $sanityStatus) -ForegroundColor Yellow
}
Write-Host ""

# --- 1. does a jp7 shelf exist at all? --------------------------------------
$rootListed = (Get-Url "$Base/").Content -match [regex]::Escape($JetPack)
$jp7Root    = Get-Url "$Base/$JetPack/"
Write-Host ("[discover] '{0}' in index root listing : {1}" -f $JetPack, $(if ($rootListed) { 'YES' } else { 'no' }))
Write-Host ("[discover] {0}/{1}/ -> HTTP {2}" -f $Base, $JetPack, $jp7Root.Status)

$discovered = @()
if ($jp7Root.Status -eq 200) {
    $discovered = [regex]::Matches($jp7Root.Content, 'cu\d{3}') | ForEach-Object { $_.Value } | Sort-Object -Unique
    if ($discovered) { Write-Host ("[discover] cu variants on {0}/ : {1}" -f $JetPack, ($discovered -join ', ')) }
}
$cudas = @($discovered + $CudaProbe) | Select-Object -Unique
Write-Host ("[discover] probing shelves : {0}" -f ($cudas -join ', '))
Write-Host ""

# --- 2. probe each shelf for the required wheels ----------------------------
$readyShelf = $null
foreach ($cu in $cudas) {
    $hits = @{}
    $anyPage = $false
    foreach ($pkg in $Packages) {
        $resp = Get-Url "$Base/$JetPack/$cu/+simple/$pkg/"
        if ($resp.Status -ne 200) { continue }
        $anyPage = $true
        $wheels = @(Get-Wheels $resp.Content $pkg)
        $match  = @($wheels | Where-Object { $_.PyTag -eq $PyTag -and $_.Plat -like "*$PlatTag*" })
        $hits[$pkg] = $match
        if ($match.Count -gt 0) {
            Write-Host ("[{0}/{1}] {2}: {3} matching wheel(s)" -f $JetPack, $cu, $pkg, $match.Count) -ForegroundColor Green
            foreach ($w in $match) {
                $flag = if ($pkg -eq 'torch' -and (Test-MinVersion $w.Version $MinTorch) -eq $false) { '   <-- below min!' } else { '' }
                Write-Host ("           - {0}{1}" -f $w.File, $flag)
            }
        } elseif ($wheels.Count -gt 0) {
            Write-Host ("[{0}/{1}] {2}: page exists, but no {3}/*{4}* wheel ({5} other wheel(s))" -f `
                $JetPack, $cu, $pkg, $PyTag, $PlatTag, $wheels.Count) -ForegroundColor Yellow
        } else {
            Write-Host ("[{0}/{1}] {2}: page exists, 0 wheels parsed" -f $JetPack, $cu, $pkg) -ForegroundColor Yellow
        }
    }
    if (-not $anyPage) {
        Write-Host ("[{0}/{1}] absent (no +simple package pages)" -f $JetPack, $cu) -ForegroundColor DarkGray
        continue
    }
    $complete = $true
    foreach ($pkg in $Packages) { if (-not ($hits[$pkg] -and $hits[$pkg].Count -gt 0)) { $complete = $false } }
    if ($complete -and -not $readyShelf) { $readyShelf = $cu }
}

# --- 3. verdict -------------------------------------------------------------
Write-Host ""
Write-Host "== VERDICT ==" -ForegroundColor Cyan
if ($readyShelf) {
    Write-Host ("READY: {0}/{1} carries {2}+{3} wheels for {4}." -f $JetPack, $readyShelf, $PyTag, $PlatTag, ($Packages -join ' + ')) -ForegroundColor Green
    Write-Host ("  Next: point pyproject.toml [[tool.uv.index]] at {0}/{1}/{2}/+simple/ ," -f $Base, $JetPack, $readyShelf)
    Write-Host  "        then proceed with the 5 JP7 edits (CLAUDE.md -> 'JetPack 7 upgrade plan')."
    Write-Host  "        NB: sm_87 execution is still only provable on the device itself."
    exit 0
} else {
    Write-Host ("NOT YET: no '{0}' shelf with {1}+{2} wheels for [{3}] found." -f $JetPack, $PyTag, $PlatTag, ($Packages -join ', ')) -ForegroundColor Yellow
    Write-Host  "  Action: stay on JP6 (the live service is safe via --no-sync). Re-run this later."
    Write-Host  "          If it stays empty long-term, the fallback is jetson-containers (Docker)."
    exit 1
}
