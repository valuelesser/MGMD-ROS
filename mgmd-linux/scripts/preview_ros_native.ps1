# Run on Windows while a ROS node is publishing its BGR preview.
param(
    [string] $Container = 'mgmd-ros',
    [int] $MasterPort = 11331,
    [string] $FrameRate = '30',
    [ValidateRange(1, 4096)] [int] $Width = 1920,
    [ValidateRange(1, 4096)] [int] $Height = 1080
)

$ErrorActionPreference = 'Stop'
$ffplay = (Get-Command ffplay.exe).Source
$docker = (Get-Command docker.exe).Source
$subscriber = @'
import sys
import os
import rospy
from sensor_msgs.msg import Image

width, height = int(os.environ['MGMD_PREVIEW_WIDTH']), int(os.environ['MGMD_PREVIEW_HEIGHT'])

def write_frame(frame):
    if (frame.width, frame.height, frame.encoding, frame.step) != (width, height, 'bgr8', width*3):
        rospy.signal_shutdown('Unexpected BGR preview dimensions or stride')
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
        '-e',"MGMD_PREVIEW_WIDTH=$Width",'-e',"MGMD_PREVIEW_HEIGHT=$Height",
        $Container,'bash','-lc','source /opt/ros/noetic/setup.bash; exec python3 -u -')) {
    $sourceInfo.ArgumentList.Add($arg)
}
$playerInfo = [Diagnostics.ProcessStartInfo]::new($ffplay)
$playerInfo.UseShellExecute = $false
$playerInfo.RedirectStandardInput = $true
foreach ($arg in @('-loglevel','warning','-fflags','nobuffer','-framedrop',
        '-f','rawvideo','-pixel_format','bgr24','-video_size',"${Width}x${Height}",
        '-framerate',$FrameRate,'-x',[string]$Width,'-y',[string]$Height,'-autoexit',
        '-window_title',"MGMD ROS ${Height}p native preview",'-i','-')) {
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
