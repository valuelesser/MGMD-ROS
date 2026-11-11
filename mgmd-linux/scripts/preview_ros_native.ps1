# Run on Windows while track_gpu.launch is publishing its 1080p preview.
param(
    [string] $Container = 'mgmd-ros',
    [int] $MasterPort = 11331
)

$ErrorActionPreference = 'Stop'
$ffplay = (Get-Command ffplay.exe).Source
$docker = (Get-Command docker.exe).Source
$subscriber = @'
import sys
import rospy
from sensor_msgs.msg import Image

def write_frame(frame):
    if (frame.width, frame.height, frame.encoding, frame.step) != (1920, 1080, 'bgr8', 5760):
        rospy.signal_shutdown('Expected packed 1920x1080 BGR preview')
        return
    try:
        sys.stdout.buffer.write(frame.data)
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        rospy.signal_shutdown('Windows player closed')

rospy.init_node('windows_native_preview', anonymous=True)
rospy.Subscriber('/mav/debug_image', Image, write_frame, queue_size=1, buff_size=2**24)
rospy.spin()
'@

$sourceInfo = [Diagnostics.ProcessStartInfo]::new($docker)
$sourceInfo.UseShellExecute = $false
$sourceInfo.CreateNoWindow = $true
$sourceInfo.RedirectStandardInput = $true
$sourceInfo.RedirectStandardOutput = $true
foreach ($arg in @('exec','-i','-u','valuelesser','-e',"ROS_MASTER_URI=http://127.0.0.1:$MasterPort",
        $Container,'bash','-lc','source /opt/ros/noetic/setup.bash; exec python3 -u -')) {
    $sourceInfo.ArgumentList.Add($arg)
}
$playerInfo = [Diagnostics.ProcessStartInfo]::new($ffplay)
$playerInfo.UseShellExecute = $false
$playerInfo.RedirectStandardInput = $true
foreach ($arg in @('-loglevel','warning','-fflags','nobuffer','-framedrop',
        '-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080',
        '-framerate','30','-x','1920','-y','1080','-autoexit',
        '-window_title','MGMD ROS 1080p native preview','-i','-')) {
    $playerInfo.ArgumentList.Add($arg)
}

$source = [Diagnostics.Process]::Start($sourceInfo)
$player = $null
try {
    $source.StandardInput.Write($subscriber)
    $source.StandardInput.Close()
    $player = [Diagnostics.Process]::Start($playerInfo)
    $copy = $source.StandardOutput.BaseStream.CopyToAsync($player.StandardInput.BaseStream)
    while (-not $copy.IsCompleted -and -not $player.WaitForExit(250)) {}
    if ($copy.IsCompleted -and -not $player.HasExited) {
        $copy.GetAwaiter().GetResult()
        $player.StandardInput.Close()
        $player.WaitForExit()
    }
}
finally {
    if (-not $source.HasExited) { $source.Kill() }
    if ($null -ne $player -and -not $player.HasExited) { $player.Kill() }
    $source.Dispose()
    if ($null -ne $player) { $player.Dispose() }
}
