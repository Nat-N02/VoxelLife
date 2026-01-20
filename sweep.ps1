$exe = Join-Path $PSScriptRoot "voxel.exe"
$paramfile = Join-Path $PSScriptRoot "params.txt"

$radii   = @(2,4,6,8,10,12)
$repairs = @(0.60,0.65,0.70,0.75,0.80,0.85)
$decays  = @(0.0, 0.0005, 0.001)
$seeds   = @(1,2)

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

foreach ($r in $radii) {
  foreach ($f in $repairs) {
    foreach ($w in $decays) {
      foreach ($s in $seeds) {

        Wait-ForSlot $MAX_JOBS

        Write-Host "QUEUE radius=$r repair_frac=$f W_decay=$w seed=$s"

        $job = Start-Job -ScriptBlock {
          param($exe, $paramfile, $r, $f, $w, $s)

          $metrics = "metrics_r${r}_f${f}_w${w}_s${s}.csv"

          & $exe `
            --params $paramfile `
            --set "sent_tail_radius=$r" `
            --set "repair_tail_frac=$f" `
            --set "W_decay=$w" `
            --seed $s `
            --metrics $metrics

        } -ArgumentList $exe, $paramfile, $r, $f, $w, $s

        $script:jobs.Add($job) | Out-Null
      }
    }
  }
}

Wait-Job -Job $script:jobs | ForEach-Object {
    Receive-Job $_ | Out-Host
}

