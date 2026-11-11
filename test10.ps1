# Run 10 known videos through the live API (like the jury) and compare with labels.
# Usage (in the forte folder):  powershell -ExecutionPolicy Bypass -File .\test10.ps1
$u = "https://bluff-humbly-refutable.ngrok-free.dev"
$k = "irT6KctFB9KCM_HrepsR0OkLA2saZpKsDBw3-2Wm8Y4"
$videos = [ordered]@{
  "X-26" = "violation"; "X-32" = "violation"; "X-40" = "violation"; "X-46" = "violation"; "X-4" = "violation"
  "X-10" = "no_violation"; "X-3" = "no_violation"; "X-12" = "no_violation"; "X-14" = "no_violation"; "X-45" = "no_violation"
}
$rows = foreach ($v in $videos.Keys) {
  $t0 = Get-Date
  $post = curl.exe -s -X POST "$u/api/v1/analyze" -H "X-API-Key: $k" -F "file=@data\videos\$v.mov"
  $id = ($post | ConvertFrom-Json).job_id
  if (-not $id) { Write-Host "$v  POST failed: $post"; continue }
  do {
    Start-Sleep 3
    $r = curl.exe -s "$u/api/v1/jobs/$id" -H "X-API-Key: $k" | ConvertFrom-Json
  } while ($r.status -eq "processing")
  $row = [pscustomobject]@{
    video    = $v
    expected = $videos[$v]
    got      = $r.verdict
    ok       = ($r.verdict -eq $videos[$v])
    status   = $r.status
    sec      = [int]((Get-Date) - $t0).TotalSeconds
    time     = ($r.violations.timecode_sec -join ", ")
    frames   = (($r.violations | ForEach-Object { $_.evidence_frames.Count }) -join ",")
    type     = ($r.violations.type -join "; ")
  }
  Write-Host ("{0,-5} expected={1,-12} got={2,-12} ok={3}  {4}s  {5}" -f $row.video, $row.expected, $row.got, $row.ok, $row.sec, $row.type)
  $row
}
$rows | Format-Table video, expected, got, ok, status, sec, time, frames, type -AutoSize
Write-Host ("Correct: {0} of {1}" -f ($rows | Where-Object ok).Count, $rows.Count)
