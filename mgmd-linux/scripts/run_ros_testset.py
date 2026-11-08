"""Full ROS file-source/detector/visualizer benchmark; run in ROS Python 3.8.

Source scripts/env_setup_ros38.sh and ros1_ws/devel/setup.bash before running.
Outputs raw timestamp joins, per-video metrics and a resumable suite summary.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rospy
from sensor_msgs.msg import Image
from mav_tracker_msgs.msg import DetectionArray
from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.metrics import Detection, evaluate, summary


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)


def run_video(video, args, config):
    out = args.out / video
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        saved = json.loads((out / 'result.json').read_text())
        if saved['configuration'] != config:
            raise RuntimeError('Resume configuration mismatch: ' + video)
        return saved
    gt = load_video_gt(video)
    end = max(gt)
    images, detections, timings = {}, {}, []
    lock = threading.Lock()
    last_image_at = [None]
    last_detection_at = [None]

    def image_cb(msg):
        with lock:
            images[msg.header.stamp.to_nsec()] = int(msg.header.seq)
            last_image_at[0] = time.monotonic()

    def det_cb(msg):
        with lock:
            stamp = msg.header.stamp.to_nsec()
            detections[stamp] = [[d.x1, d.y1, d.x2, d.y2,
                                  d.confidence, d.track_id] for d in msg.detections]
            timings.append([msg.mfe_ms, msg.tf_ms, msg.lac_ms, msg.lad_ms])
            last_detection_at[0] = time.monotonic()

    sub_img = rospy.Subscriber('/camera/image_raw', Image, image_cb,
                               queue_size=8, buff_size=32 * 1024 * 1024)
    sub_det = rospy.Subscriber('/mav/detections', DetectionArray, det_cb, queue_size=100)
    recording = args.out / (video + '.mp4')
    log_path = out / 'roslaunch.log'
    proc = None
    started = time.monotonic()
    try:
        with log_path.open('w') as log:
            proc = subprocess.Popen([
                'roslaunch', 'mav_tracker_nodes', 'track.launch',
                'video:=' + video, 'rate:=' + str(args.rate),
                'limit:=' + str(end), 'loop:=false',
                'width:=1920', 'height:=1080', 'device:=cuda',
                'visualize:=true', 'encoder:=nvenc',
                'out_video:=' + str(recording),
            ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1'))
            deadline = started + end / args.rate + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError('roslaunch exited: ' + video)
                now = time.monotonic()
                with lock:
                    complete = images and max(images.values()) >= end
                    quiet = last_detection_at[0] is not None and now - last_detection_at[0] > 4
                    image_quiet = last_image_at[0] is not None and now - last_image_at[0] > 12
                if complete and quiet and image_quiet:
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Incomplete source or stalled detector: ' + video)
    finally:
        stop(proc)
        sub_img.unregister()
        sub_det.unregister()
        write_json(out / 'capture.json', {
            'images_by_stamp': images, 'detections_by_stamp': detections,
            'module_ms': timings,
        })
    text = log_path.read_text(errors='replace')
    required = 'LAC provider=CUDAExecutionProvider  LAD provider=CUDAExecutionProvider'
    if required not in text:
        raise RuntimeError('CUDA provider check failed: ' + video)
    if 'encoder=nvenc' not in text or '[nvenc] ffmpeg 退出码' in text:
        raise RuntimeError('NVENC check failed: ' + video)
    if '处理异常' in text or 'Traceback (most recent call last)' in text:
        raise RuntimeError('ROS node error: ' + video)
    missing = set(detections) - set(images)
    if missing:
        raise RuntimeError('Detection stamps missing image join: ' + str(len(missing)))
    if set(images.values()) != set(range(1, end + 1)):
        raise RuntimeError('Capture did not receive every source frame: ' + video)
    mapped = {images[s]: [Detection(*d[:5]) for d in boxes]
              for s, boxes in detections.items()}
    m = summary(evaluate(mapped, gt))
    baseline = json.loads((args.offline / (video + '.json')).read_text())
    if baseline['frames_run'] != end - 2 or set(gt) != set(range(1, end + 1)):
        raise RuntimeError('Offline boundary adjustment needs manual audit: ' + video)
    old = baseline['metrics']
    extra_fn = len(gt[1].objects) + len(gt[end].objects)
    off = counts(old['tp'], old['fp'], old['fn'] + extra_fn)
    result = {
        'video': video, 'configuration': config, 'frames': end,
        'gt_objects': sum(len(g.objects) for g in gt.values()),
        'image_messages': len(images), 'detection_messages': len(detections),
        'frames_without_output': end - len(detections),
        'elapsed_s': time.monotonic() - started,
        'module_ms_mean': [sum(row[i] for row in timings) / len(timings)
                           for i in range(4)],
        'ros': m, 'offline_full_frames': off,
        'offline_original': baseline, 'recording': str(recording),
    }
    write_json(out / 'result.json', result)
    return result


def counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=2 * p * r / (p + r) if p + r else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rate', type=float, default=30)
    ap.add_argument('--offline', type=Path, default=C.LOGS / 'testset/final_d002')
    ap.add_argument('--videos', nargs='+', default=[C.video_id_to_name(i) for i in C.TEST_IDS])
    args = ap.parse_args()
    if args.rate <= 0:
        ap.error('rate must be positive')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    import rosnode
    conflicts = set(rosnode.get_node_names()) & {'/detector_node', '/video_source', '/visualize_node'}
    if conflicts:
        raise RuntimeError('Existing tracking nodes must be stopped: ' + str(conflicts))
    rospy.init_node('ros_testset_capture', anonymous=True, disable_signals=True)
    config = dict(rate=args.rate, resolution=[1920, 1080], device='cuda',
                  visualize=True, encoder='nvenc', d_min=0.02,
                  frame_delta_predict=True, association_radius_factor=2.0,
                  offline=str(args.offline.resolve()),
                  code_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], text=True).strip())
    per = []
    for i, video in enumerate(args.videos, 1):
        print('[%d/%d] START %s' % (i, len(args.videos), video), flush=True)
        row = run_video(video, args, config)
        per.append(row)
        ros = counts(*(sum(r['ros'][k] for r in per) for k in ('tp', 'fp', 'fn')))
        off = counts(*(sum(r['offline_full_frames'][k] for r in per) for k in ('tp', 'fp', 'fn')))
        write_json(args.out / 'summary.json', dict(
            configuration=config, completed=len(per), expected=len(args.videos),
            ros=ros, offline_full_frames=off, paper=dict(precision=.84, recall=.59, f1=.69),
            per_video=per))
        print('[%d/%d] DONE %s frames=%d processed=%d P=%.4f R=%.4f F1=%.4f' %
              (i, len(args.videos), video, row['frames'], row['detection_messages'],
               row['ros']['precision'], row['ros']['recall'], row['ros']['f1']), flush=True)


if __name__ == '__main__':
    main()
