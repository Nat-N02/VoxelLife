$exe = Join-Path $PSScriptRoot "voxel.exe"
$paramfile = Join-Path $PSScriptRoot "params.txt"
$metricsDir = Join-Path $PSScriptRoot "metrics"

if (!(Test-Path $metricsDir)) {
    New-Item -ItemType Directory -Path $metricsDir | Out-Null
}

$radii = @(2, 6, 10, 14, 18, 22) 
$repairs = @(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9) 
$decays = @(0.0, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002) 
$seeds = @(1,2,3,4)

$MAX_JOBS = 8
$script:jobs = New-Object System.Collections.ArrayList

function Wait-ForSlot {
    param($max)
    while ($script:jobs.Count -ge $max) {
        $done = Wait-Job -Job $script:jobs -Any
        Receive-Job $done | Out-Host
        $script:jobs.Remove($done) | Out-Null
    }
}

$runId = 0

foreach ($r in $radii) {
  foreach ($f in $repairs) {
    foreach ($w in $decays) {
      foreach ($s in $seeds) {

        $runId++
        Wait-ForSlot $MAX_JOBS

        # Build filename OUTSIDE the job
        $metricsFile = Join-Path $metricsDir ("metrics_{0}_r{1}_f{2}_w{3}_s{4}.csv" -f `
            $runId, $r, $f, $w, $s)

        Write-Host "QUEUE radius=$r repair_frac=$f W_decay=$w seed=$s"
        Write-Host "  -> $metricsFile"

        $job = Start-Job -ScriptBlock {
          param($exe, $paramfile, $r, $f, $w, $s, $metricsFile)

          & $exe `
            --params $paramfile `
            --set "sent_tail_radius=$r" `
            --set "repair_tail_frac=$f" `
            --set "W_decay=$w" `
            --seed $s `
            --metrics $metricsFile

        } -ArgumentList $exe, $paramfile, $r, $f, $w, $s, $metricsFile

        $script:jobs.Add($job) | Out-Null
      }
    }
  }
}

Wait-Job -Job $script:jobs | ForEach-Object {
    Receive-Job $_ | Out-Host
}
