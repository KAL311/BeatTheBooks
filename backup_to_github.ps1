[CmdletBinding()]
param(
    [string]$Message = "",
    [switch]$SkipPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Require-GitRepo {
    $inside = (& git rev-parse --is-inside-work-tree 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $inside -ne "true") {
        throw "This folder is not an initialized git repository."
    }
}

function Get-CommitMessage {
    param([string]$RequestedMessage)

    if ($RequestedMessage -and $RequestedMessage.Trim()) {
        return $RequestedMessage.Trim()
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    return "Backup sync $timestamp"
}

Require-GitRepo

$statusBefore = (& git status --short).Trim()
if (-not $statusBefore) {
    Write-Host "No tracked backup changes to commit."
    exit 0
}

$commitMessage = Get-CommitMessage -RequestedMessage $Message

Write-Host "Project root: $ProjectRoot"
Write-Host "Commit message: $commitMessage"

& git add -A
if ($LASTEXITCODE -ne 0) {
    throw "git add failed."
}

$statusAfterAdd = (& git status --short).Trim()
if (-not $statusAfterAdd) {
    Write-Host "Nothing new to commit after staging."
    exit 0
}

& git commit -m $commitMessage
if ($LASTEXITCODE -ne 0) {
    throw "git commit failed."
}

if ($SkipPush) {
    Write-Host "Commit created locally. Push skipped."
    exit 0
}

& git push
if ($LASTEXITCODE -ne 0) {
    throw "git push failed."
}

Write-Host "GitHub backup sync complete."
