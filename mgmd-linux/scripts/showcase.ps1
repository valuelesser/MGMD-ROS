#requires -Version 7
# Launch the Docker ROS demo and its native Windows 1080p viewer together.
[CmdletBinding(PositionalBinding=$false)]
param(
    [ValidateRange(1024,65535)][int] $MasterPort = 11530,
    [Parameter(Position=0,ValueFromRemainingArguments)][string[]] $LaunchArgs
)
$ErrorActionPreference = 'Stop'
$null = Get-Command ffplay.exe
$docker = (Get-Command docker.exe).Source
$pidFile = '/tmp/mgmd-showcase-' + [guid]::NewGuid().ToString('N') + '.pid'
$frameRate = '30000/1001'
foreach ($option in $LaunchArgs) {
    if ($option -match '^rate:=(.+)$') { $frameRate = $Matches[1] }
}
$entry = @'
set -e
source "$HOME/mgmd-all-in-gpu/mgmd-linux/scripts/showcase_env.sh"
if python3 -c 'import rosgraph,sys; sys.exit(0 if rosgraph.is_master_online() else 1)'; then
    echo "ROS master already in use: $ROS_MASTER_URI" >&2
    exit 1
fi
exec roslaunch -p "$1" --pid "$2" mav_tracker_nodes showcase.launch rqt:=false "${@:3}"
'@
$start = [Diagnostics.ProcessStartInfo]::new($docker)
$start.UseShellExecute = $false
foreach ($item in (@('exec','-u','valuelesser','-e',"ROS_MASTER_URI=http://127.0.0.1:$MasterPort",
        '-e','ROS_IP=127.0.0.1','mgmd-ros','bash','-c',$entry.Replace("`r",''),'showcase',
        "$MasterPort",$pidFile) + $LaunchArgs)) {
    if ($null -ne $item) { $start.ArgumentList.Add($item) }
}
$playback = [Diagnostics.Process]::Start($start)
try {
    & "$PSScriptRoot/preview_ros_native.ps1" -MasterPort $MasterPort -FrameRate $frameRate -PlaybackProcess $playback
} finally {
    if (-not $playback.HasExited) {
        & $docker exec -u valuelesser mgmd-ros bash -c 'if test -s "$1"; then kill -INT "$(cat "$1")" 2>/dev/null || true; fi' showcase $pidFile
    }
    $playback.WaitForExit()
    $exitCode = $playback.ExitCode
    $playback.Dispose()
}
if ($exitCode) { throw "Docker ROS launch exited with code $exitCode" }
