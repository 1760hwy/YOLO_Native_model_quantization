# Step 6: Fetch detection results from RDK X5
# Run in PowerShell:
#   cd C:\e\workspace\experiment\EI_yolo\step\step6
#   powershell -ExecutionPolicy Bypass -File .\fetch_results.ps1

$BOARD_USER    = "sunrise"
$BOARD_IP      = "192.168.0.106"
$BOARD_RESULTS = "/home/sunrise/step5/results"
$LOCAL_STEP6   = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================================"
Write-Host "Step 6: Fetch RDK X5 results"
Write-Host "  From: ${BOARD_USER}@${BOARD_IP}:${BOARD_RESULTS}"
Write-Host "  To:   ${LOCAL_STEP6}"
Write-Host "============================================================"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out_dir = Join-Path $LOCAL_STEP6 "results_$timestamp"
New-Item -ItemType Directory -Path $out_dir -Force | Out-Null

Write-Host ""
Write-Host "[Download] Fetching results from board..."
scp -r "${BOARD_USER}@${BOARD_IP}:${BOARD_RESULTS}/*" $out_dir

if ($LASTEXITCODE -eq 0) {
    $img_count = (Get-ChildItem "$out_dir\*.jpg" -ErrorAction SilentlyContinue).Count
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Done!"
    Write-Host "  Saved to: $out_dir"
    Write-Host "  Result images: $img_count"
    $report_path = "$out_dir\detection_report.json"
    if (Test-Path $report_path) {
        $report = Get-Content $report_path -Raw | ConvertFrom-Json
        Write-Host ""
        Write-Host "  --- Summary ---"
        Write-Host ("  Images:      " + $report.num_images)
        Write-Host ("  Detections:  " + $report.total_detections)
        Write-Host ("  BPU infer:   " + $report.avg_infer_ms + " ms (avg)")
        Write-Host ("  End-to-end:  " + $report.avg_total_ms + " ms (avg)")
        Write-Host ("  Est. FPS:    " + $report.fps_estimate)
    }
    Write-Host "============================================================"
} else {
    Write-Host "[ERROR] scp failed. Check:"
    Write-Host "  1. RDK X5 is on and IP is correct: ${BOARD_IP}"
    Write-Host "  2. Detection has finished (run_detect.sh completed)"
    Write-Host "  3. SSH works: ssh ${BOARD_USER}@${BOARD_IP}"
    exit 1
}
