"""Slow full-video ROS run versus an independent, matching ONNX offline loop."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import rospy
from sensor_msgs.msg import Image
from mav_tracker_msgs.msg import DetectionArray
from mgmd import config as C
from mgmd.annotations import load_video_gt
from mgmd.metrics import Detection, evaluate, summary
from run_ros_testset import stop, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='phantom70')
    ap.add_argument('--rate', type=float, default=2)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.rate <= 0:
        ap.error('rate must be positive')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    gt = load_video_gt(args.video)
    end = max(gt)
    import rosnode
    if set(rosnode.get_node_names()) & {'/detector_node', '/visualize_node', '/video_source'}:
        raise RuntimeError('Stop existing tracking nodes first')
    rospy.init_node('nodrop_comparison', anonymous=True, disable_signals=True)
    lock = threading.Lock()
    formal = [False]
    images, detections = {}, {}
    last_det = [time.monotonic()]
    warm_lad = [False]
    logs, procs = [], []

    def launch(label, command):
        log = (args.out / (label + '.log')).open('w')
        logs.append(log)
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        procs.append(p)
        return p

    def image_cb(msg):
        with lock:
            if formal[0]:
                images[msg.header.stamp.to_nsec()] = int(msg.header.seq)

    def det_cb(msg):
        with lock:
            last_det[0] = time.monotonic()
            if formal[0]:
                detections[msg.header.stamp.to_nsec()] = [
                    [d.x1, d.y1, d.x2, d.y2, d.confidence, d.track_id] for d in msg.detections]
            else:
                warm_lad[0] |= msg.lad_ms > 0

    si = rospy.Subscriber('/camera/image_raw', Image, image_cb, queue_size=8,
                          buff_size=32 * 1024 * 1024)
    sd = rospy.Subscriber('/mav/detections', DetectionArray, det_cb, queue_size=100)
    recording = args.out / (args.video + '.mp4')
    try:
        detector = launch('detector', ['rosrun', 'mav_tracker_nodes', 'detector_node.py', '_device:=cuda'])
        deadline = time.monotonic() + 90
        while '就绪，等待' not in (args.out / 'detector.log').read_text(errors='replace'):
            if detector.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Detector did not become ready')
            time.sleep(.5)
        source_args = ['rosrun', 'mav_tracker_nodes', 'video_source.py', '_source:=file',
                       '_path:=' + args.video, '_loop:=false', '_width:=1920', '_height:=1080']
        print('Warmup: 180 frames at 12 Hz; formal capture disabled', flush=True)
        warm = launch('warmup_source', source_args + ['_rate:=12', '_limit:=180'])
        warm.wait(timeout=90)
        while time.monotonic() - last_det[0] < 5:
            time.sleep(.5)
        if not warm_lad[0]:
            raise RuntimeError('Warmup did not exercise LAD')
        viz = launch('visualize', ['rosrun', 'mav_tracker_nodes', 'visualize_node.py',
                                   '_fps:=' + str(args.rate), '_out:=' + str(recording)])
        deadline = time.monotonic() + 30
        while '就绪' not in (args.out / 'visualize.log').read_text(errors='replace'):
            if viz.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Visualizer did not become ready')
            time.sleep(.2)
        with lock:
            formal[0] = True
        print('Formal source: %s %.2f Hz, expect %d images / %d centers' %
              (args.video, args.rate, end, end - 2), flush=True)
        # Read to EOF: the normal rate.sleep after the final publish lets its queue drain.
        source = launch('formal_source', source_args + ['_rate:=' + str(args.rate), '_limit:=0'])
        deadline = time.monotonic() + end / args.rate + 120
        progress = time.monotonic()
        while True:
            now = time.monotonic()
            if source.poll() is not None and now - last_det[0] > 5:
                if source.returncode != 0:
                    raise RuntimeError('Source exited with error')
                break
            if detector.poll() is not None or viz.poll() is not None or now > deadline:
                raise RuntimeError('Node exited or run timed out')
            if now - progress > 20:
                with lock:
                    print('images=%d centers=%d last=%s' %
                          (len(images), len(detections), max(images.values()) if images else None), flush=True)
                progress = now
            time.sleep(.5)
    finally:
        for p in reversed(procs):
            stop(p)
        si.unregister()
        sd.unregister()
        for log in logs:
            log.close()
        write_json(args.out / 'capture.json', dict(images_by_stamp=images, detections_by_stamp=detections))
    text = (args.out / 'detector.log').read_text(errors='replace')
    assert 'LAC provider=CUDAExecutionProvider  LAD provider=CUDAExecutionProvider' in text
    assert '处理异常' not in text and 'encoder=nvenc' in (args.out / 'visualize.log').read_text(errors='replace')
    assert not (set(detections) - set(images)), 'Timestamp join failed'
    mapped = {images[s]: [Detection(*d[:5]) for d in boxes] for s, boxes in detections.items()}
    audit = dict(missing_images=sorted(set(range(1, end + 1)) - set(images.values())),
                 missing_centers=sorted(set(range(2, end)) - set(mapped)),
                 extra_centers=sorted(set(mapped) - set(range(2, end))))
    write_json(args.out / 'coverage.json', audit)
    assert not any(audit.values()), 'Run has frame losses; lower source rate and repeat'
    ros_metrics = summary(evaluate(mapped, gt))
    print('ROS no-drop verified:', json.dumps(ros_metrics)[:350], flush=True)

    # Independent offline iteration with exactly the ROS tracking options.
    spec = importlib.util.spec_from_file_location('reference', Path(__file__).with_name('run_testset_onnx.py'))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    from mgmd_onnx.tf_fast import FastSortTracker
    from mgmd_onnx import LacOnnx, LadOnnx
    reference.SortTracker = lambda **kw: FastSortTracker(
        **kw, frame_delta_predict=True, association_radius_factor=2.0)
    reference._DUMP['dir'] = str(args.out / 'offline_detections')
    cfg = reference.MFEConfig()
    cfg.apply_recommended()
    lac, lad = LacOnnx(device='cuda'), LadOnnx(device='cuda')
    assert all(m.session.get_providers()[0] == 'CUDAExecutionProvider' for m in (lac, lad))
    print('Running matching offline ONNX reference', flush=True)
    offline = reference.run_video(args.video, cfg, lac, lad, d_mode='self', d_min=.02,
                                  lac_thresh=.5, lad_conf=.5, iou_thresh=.3, max_age=3,
                                  span=3, min_hits=1, limit=None)
    raw = pickle.loads((args.out / 'offline_detections' / (args.video + '.pkl')).read_bytes())
    differences = []
    for frame in range(2, end):
        a = sorted([[d.x1, d.y1, d.x2, d.y2, d.conf] for d in mapped[frame]])
        b = sorted([list(d) for d in raw.get(frame, [])])
        if len(a) != len(b) or (a and not np.allclose(a, b, rtol=1e-5, atol=1e-3)):
            differences.append(frame)
    result = dict(video=args.video, rate=args.rate, images=len(images), centers=len(mapped),
                  coverage=audit, warmup_frames=180, ros=ros_metrics, offline_matching=offline,
                  differing_frames_at_1e_3=differences,
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(args.out / 'result.json', result)
    print('COMPLETE matching offline:', offline['precision'], offline['recall'], offline['f1'],
          'different frames:', len(differences), flush=True)


if __name__ == '__main__':
    main()
