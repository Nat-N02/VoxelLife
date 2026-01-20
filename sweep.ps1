$exe = ".\voxel.exe"
$paramfile = "params.txt"

$radii   = @(2,4,6,8,10,12)
$repairs = @(0.60,0.65,0.70,0.75,0.80,0.85)
$decays  = @(0.0, 0.0005, 0.001)
$seeds   = @(1,2)

foreach ($r in $radii) {
  foreach ($f in $repairs) {
    foreach ($w in $decays) {
      foreach ($s in $seeds) {

        Write-Host "RUN radius=$r repair_frac=$f W_decay=$w seed=$s"

        & $exe `
          --params $paramfile `
          --set "sent_tail_radius=$r" `
          --set "repair_tail_frac=$f" `
          --set "W_decay=$w" `
          --seed $s

      }
    }
  }
}
