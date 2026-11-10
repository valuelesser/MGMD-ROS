#!/usr/bin/env bash
# Run inside mgmd-ros with WSLg relay, as the project user.
set -eo pipefail
run=$(realpath -m "${1:?new output directory required}")
encoder=${2:-nvenc}
case "$encoder" in nvenc|cpu) ;; *) exit 2 ;; esac
mkdir "$run"
source <(sed -n '1,/^cd "\$ROOT"$/p' /usr/local/bin/mgmd-ros-demo)
export DISPLAY=${DISPLAY:-127.0.0.1:47} QT_X11_NO_MITSHM=1
export ROS_MASTER_URI=http://127.0.0.1:11327 ROS_IP=127.0.0.1 ROS_HOME="$run/ros"
if timeout 3 rosnode list >/dev/null 2>&1; then
    echo 'ROS probe port is occupied' >&2
    exit 1
fi
timeout 5 xdpyinfo >/dev/null

core= viewer= launch=
cleanup() {
    if [[ -n "$launch" ]]; then
        kill -INT "$launch" 2>/dev/null || true
        wait "$launch" || true
        launch=
    fi
    if [[ -n "$viewer" ]]; then
        kill -TERM "$viewer" 2>/dev/null || true
        for _ in {1..5}; do
            kill -0 "$viewer" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$viewer" 2>/dev/null || true
        wait "$viewer" 2>/dev/null || true
        viewer=
    fi
    if [[ -n "$core" ]]; then
        kill -INT "$core" 2>/dev/null || true
        wait "$core" || true
        core=
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid roscore -p 11327 >"$run/core.log" 2>&1 </dev/null &
core=$!
for _ in {1..15}; do
    if timeout 2 rosnode list >/dev/null 2>&1; then break; fi
    kill -0 "$core"
    sleep 1
done
timeout 3 rosnode list
setsid rosrun rqt_image_view rqt_image_view /mav/debug_image >"$run/viewer.log" 2>&1 </dev/null &
viewer=$!
for _ in {1..15}; do
    xwininfo -root -tree | sed -n '/ImageView - rqt/p' >"$run/window.txt"
    if [[ -s "$run/window.txt" ]]; then break; fi
    kill -0 "$viewer"
    sleep 1
done
test -s "$run/window.txt"
setsid roslaunch mav_tracker_nodes track.launch video:=phantom30 width:=1920 height:=1080 \
    rate:=30 loop:=false limit:=600 device:=cuda d_min:=0.02 frame_delta_predict:=true \
    association_radius_factor:=2.0 visualize:=true "encoder:=$encoder" \
    "out_video:=$run/phantom30.mp4" >"$run/launch.log" 2>&1 </dev/null &
launch=$!
timeout 16 python3 - <<'PY' | tee "$run/image.txt"
import rospy
from sensor_msgs.msg import Image
rospy.init_node("g0_probe", anonymous=True)
for topic in ("/camera/image_raw", "/mav/debug_image"):
    frame = rospy.wait_for_message(topic, Image, timeout=7)
    print(topic, frame.width, frame.height, frame.encoding)
    assert (frame.width, frame.height, frame.encoding) == (1920, 1080, "bgr8")
PY
timeout 6 rostopic hz /mav/debug_image >"$run/debug_hz.txt" 2>&1 || test "$?" -eq 124
rostopic info /mav/debug_image | tee "$run/debug_topic.txt"
grep -q rqt_gui "$run/debug_topic.txt"
for _ in {1..35}; do
    if grep -q '共发布 600 帧' "$run/launch.log"; then break; fi
    kill -0 "$launch"
    sleep 1
done
grep '共发布 600 帧' "$run/launch.log"
cleanup
if grep -qE '处理异常|process has died|ffmpeg 退出码|Shutdown errors' "$run/launch.log"; then
    tail -30 "$run/launch.log"
    exit 1
fi
grep -E 'LAC provider=CUDAExecutionProvider.*LAD provider=CUDAExecutionProvider' "$run/launch.log"
grep -E '共处理 [1-9][0-9]* 帧' "$run/launch.log"
grep '视频已保存' "$run/launch.log"
ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=codec_name,width,height,nb_read_frames -of csv=p=0 \
    "$run/phantom30.mp4" | tee "$run/video.csv"
if [[ "$encoder" == nvenc ]]; then
    grep -q 'encoder=nvenc' "$run/launch.log"
    grep -Eq '^h264,1920,1080,[1-9][0-9]*$' "$run/video.csv"
fi
tail -4 "$run/debug_hz.txt"
