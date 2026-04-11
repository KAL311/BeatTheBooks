[CmdletBinding()]
param(
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$LogDir  = Join-Path $ProjectRoot "Maintenance"
$LogFile = Join-Path $LogDir "earn_back_$(Get-Date -Format 'yyyy-MM-dd').log"

function Resolve-PythonExe {
    param([string]$RequestedPath)
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($RequestedPath) { $candidates.Add($RequestedPath) }
    @(
        "C:\Users\fraun\AppData\Local\Programs\Python\Python314\python.exe",
        "C:\Users\fraun\AppData\Local\Programs\Python\Python313\python.exe"
    ) | ForEach-Object { $candidates.Add($_) }
    try {
        $pythonCmd = Get-Command python -ErrorAction Stop
        if ($pythonCmd.Source -notmatch 'WindowsApps') { $candidates.Add($pythonCmd.Source) }
    } catch {}
    foreach ($c in ($candidates | Select-Object -Unique)) {
        if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path }
    }
    throw "Python executable not found. Pass -PythonExe or ensure python is on PATH."
}

$pythonPath = Resolve-PythonExe -RequestedPath $PythonExe
$runDate    = Get-Date -Format "yyyy-MM-dd"

"[${runDate}] earn-back run starting" | Tee-Object -FilePath $LogFile
"Python: $pythonPath" | Tee-Object -FilePath $LogFile -Append
"Project: $ProjectRoot" | Tee-Object -FilePath $LogFile -Append

& $pythonPath "fit_regular_earn_back.py" 2>&1 | Tee-Object -FilePath $LogFile -Append

if ($LASTEXITCODE -ne 0) {
    "ERROR: fit_regular_earn_back.py exited with code $LASTEXITCODE" | Tee-Object -FilePath $LogFile -Append
    exit $LASTEXITCODE
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] earn-back run complete. Log: $LogFile" | Tee-Object -FilePath $LogFile -Append
