<#
.SYNOPSIS
VHR Satellite Segmentation - Docker Runner Script (Windows PowerShell)
Run complete pipeline without any manual Python/GDAL/CUDA environment setup.

.EXAMPLE
.\run.ps1 -Doctor
.\run.ps1 -Test
.\run.ps1 -Model segformer -Demo
.\run.ps1 --model segformer --demo
.\run.ps1 -Gpu -Model segformer -Demo
#>
param(
    [switch]$Gpu,
    [switch]$Doctor,
    [switch]$Test,
    [string]$Model,
    [switch]$Demo,
    [switch]$SkipEvaluation,
    [string]$Raster,
    [string]$Reference,
    [string]$WeightsDir,
    [string]$OutputDir,
    [string]$Device,
    [string]$VectorFormat,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

if ($Doctor) {
    docker compose run --rm doctor
    exit $LASTEXITCODE
}

if ($Test) {
    docker compose run --rm test
    exit $LASTEXITCODE
}

$service = if ($Gpu) { "gpu" } else { "cpu" }
$cmdArgs = @()

if ($Model) { $cmdArgs += @("--model", $Model) }
if ($Demo) { $cmdArgs += @("--demo") }
if ($SkipEvaluation) { $cmdArgs += @("--skip-evaluation") }
if ($Raster) { $cmdArgs += @("--raster", $Raster) }
if ($Reference) { $cmdArgs += @("--reference", $Reference) }
if ($WeightsDir) { $cmdArgs += @("--weights-dir", $WeightsDir) }
if ($OutputDir) { $cmdArgs += @("--output-dir", $OutputDir) }
if ($Device) { $cmdArgs += @("--device", $Device) }
if ($VectorFormat) { $cmdArgs += @("--vector-format", $VectorFormat) }

if ($null -ne $ArgsList -and $ArgsList.Count -gt 0) {
    foreach ($arg in $ArgsList) {
        # Convert single-dash named args (e.g. -help, -model) to double-dash (--help, --model)
        if ($arg -match "^-[a-zA-Z]" -and -not ($arg -match "^--")) {
            $cmdArgs += ("-" + $arg.ToLower())
        } else {
            $cmdArgs += $arg
        }
    }
}

if ($cmdArgs.Count -eq 0) {
    docker compose run --rm $service --help
    exit $LASTEXITCODE
}

docker compose run --rm $service @cmdArgs
exit $LASTEXITCODE
