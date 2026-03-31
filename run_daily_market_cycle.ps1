[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$CaptureOnly,
    [switch]$ReportOnly,
    [switch]$SkipPostCapture
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Resolve-PythonExe {
    param([string]$RequestedPath)

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($RequestedPath) {
        $candidates.Add($RequestedPath)
    }

    $fallbacks = @(
        "C:\Users\fraun\AppData\Local\Programs\Python\Python314\python.exe",
        "C:\Users\fraun\AppData\Local\Programs\Python\Python313\python.exe"
    )
    foreach ($fallback in $fallbacks) {
        $candidates.Add($fallback)
    }

    try {
        $pythonCmd = Get-Command python -ErrorAction Stop
        if ($pythonCmd -and $pythonCmd.Source -and ($pythonCmd.Source -notmatch 'WindowsApps')) {
            $candidates.Add($pythonCmd.Source)
        }
    }
    catch {
    }


    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Python executable was not found. Pass -PythonExe or make sure python is on PATH."
}

function Invoke-Step {
    param(
        [string]$StepName,
        [string]$PythonPath,
        [string]$ScriptName
    )

    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $StepName"
    & $PythonPath $ScriptName
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

$pythonPath = Resolve-PythonExe -RequestedPath $PythonExe
$todayIso = Get-Date -Format "yyyy-MM-dd"
$shouldCapture = -not $ReportOnly
$shouldReport = -not $CaptureOnly

Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $pythonPath"
Write-Host "Cycle date: $todayIso"

if ($shouldCapture) {
    Invoke-Step -StepName "Pre-capture sportsbook lines" -PythonPath $pythonPath -ScriptName "capture_market_lines.py"
}

if ($shouldReport) {
    Invoke-Step -StepName "Build MLB daily report" -PythonPath $pythonPath -ScriptName "run_daily_mlb_report.py"
}

if ($shouldCapture -and $shouldReport -and -not $SkipPostCapture) {
    Invoke-Step -StepName "Post-report sportsbook refresh" -PythonPath $pythonPath -ScriptName "capture_market_lines.py"
}

if ($shouldReport) {
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Run opening-day readiness audit"
    & $pythonPath "opening_day_readiness.py" "--report-date" $todayIso
    if ($LASTEXITCODE -ne 0) {
        throw "Run opening-day readiness audit failed with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "Cycle complete."
Write-Host "Report:      $ProjectRoot\MLB_Report_$todayIso.txt"
Write-Host "Workbook:    $ProjectRoot\MLB_Season_To_Date_2026.xlsx"
Write-Host "Watchlist:   $ProjectRoot\Sportsbook_Watchlist_$todayIso.txt"
Write-Host "Readiness:   $ProjectRoot\OPENING_DAY_READINESS_$todayIso.txt"

