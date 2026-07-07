param(
    [string]$DeviceName = "Integrated Camera",
    [string]$BoardAddress = "192.168.137.2",
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"
$ffmpeg = "C:\Users\Saturday\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
$target = "tcp://${BoardAddress}:${Port}?tcp_nodelay=1"

Write-Host "Camera: $DeviceName"
Write-Host "Target: $target"

while ($true) {
    & $ffmpeg -nostdin -hide_banner -loglevel warning `
        -f dshow -rtbufsize 128M -framerate 30 -video_size 1280x720 `
        -i "video=$DeviceName" -an `
        -vf "fps=25,format=yuv420p" `
        -c:v libx264 -pix_fmt yuv420p -profile:v high -level:v 3.1 `
        -preset veryfast -tune zerolatency -b:v 8M -maxrate 8M -bufsize 2M `
        -g 25 -keyint_min 25 -sc_threshold 0 -bf 0 -refs 1 `
        -x264-params "repeat-headers=1" `
        -mpegts_flags resend_headers -muxdelay 0 -muxpreload 0 `
        -f mpegts $target

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Publisher stopped normally; reconnecting in 1 second."
    } else {
        Write-Warning "Publisher exited with code $LASTEXITCODE; reconnecting in 1 second."
    }
    Start-Sleep -Seconds 1
}
