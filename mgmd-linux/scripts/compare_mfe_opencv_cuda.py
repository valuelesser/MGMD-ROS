"""Compare MFE image stages on frozen pixels and homographies."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from mgmd import mfe


FLAGS = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP


def config():
    cfg = mfe.MFEConfig().apply_recommended()
    cfg.validate()
    return cfg


def capture(out, videos, frames):
    cfg = config()
    out.mkdir()
    (out / "source.json").write_text(json.dumps({"opencv": cv2.__version__, "frames": frames}))
    for video in videos:
        folder = out / video.stem
        folder.mkdir()
        cap = cv2.VideoCapture(str(video))
        gray = []
        try:
            for i in range(frames):
                ok, image = cap.read()
                assert ok and image.shape == (1080, 1920, 3), (video, i)
                np.save(folder / f"bgr{i}.npy", image)
                gray.append(mfe.preprocess(image, cfg))
                if len(gray) < 3:
                    continue
                ref = {}
                masks = []
                for direction, source in (("p", gray[0]), ("n", gray[2])):
                    aligned, valid, matrix = mfe.align_to_reference(source, gray[1], cfg)
                    ref[f"aligned_{direction}"] = aligned
                    ref[f"valid_{direction}"] = valid
                    ref[f"H_{direction}"] = np.eye(3) if matrix is None else matrix
                    masks.append(mfe._binarize(cv2.absdiff(gray[1], aligned), cfg.threshold, valid, cfg))
                closed, boxes = mfe.postprocess(cv2.bitwise_or(*masks), cfg)
                np.savez(folder / f"ref{i - 1}.npz", **ref, gray=gray[1], closed=closed,
                         boxes=np.asarray(boxes, dtype=np.int32).reshape(-1, 4))
                gray.pop(0)
        finally:
            cap.release()


class Comparison:
    def __init__(self):
        import torch
        from mgmd_onnx import cuda_mfe
        from mgmd_onnx.cuda_frames import DevicePlane

        self.torch = torch
        self.device_plane = DevicePlane
        self.mfe = cuda_mfe
        self.cfg = config()
        self.stream = cv2.cuda.Stream.Null()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.median = cv2.cuda.createMedianFilter(cv2.CV_8UC1, 5)
        self.open = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, self.kernel, iterations=1)
        self.close = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, self.kernel, iterations=3)
        self.lk = cv2.cuda.SparsePyrLKOpticalFlow_create((15, 15), 3, 30)
        self.times = defaultdict(list)
        self.counts = Counter()
        self.maxima = Counter()
        self.lk_errors = []

    @staticmethod
    def upload(value):
        image = cv2.cuda.GpuMat()
        image.upload(np.ascontiguousarray(value))
        return image

    def borrow(self, image):
        width, height = image.size()
        pointer = image.cudaPtr()
        value = self.torch.as_tensor(self.device_plane(pointer, (height, width), image.step),
                                     device="cuda")
        assert value.data_ptr() == pointer
        return value

    def time(self, backend, stage, fn, *args, keep=True):
        start = perf_counter()
        value = fn(*args)
        if backend == "opencv_cuda":
            self.stream.waitForCompletion()
        elif backend in ("torch_cuda", "hybrid"):
            self.torch.cuda.current_stream().synchronize()
        if keep:
            self.times[f"{backend}.{stage}"].append((perf_counter() - start) * 1000)
        return value

    def diff(self, key, actual, expected):
        delta = np.abs(actual.astype(np.int16) - expected)
        self.counts[key] += int(np.count_nonzero(delta))
        self.maxima[key] = max(self.maxima[key], int(delta.max(initial=0)))

    def mask_cpu(self, center, aligned, valid):
        return mfe._binarize(cv2.absdiff(center, aligned), self.cfg.threshold, valid, self.cfg)

    def mask_cuda(self, center, aligned, valid):
        diff = cv2.cuda.absdiff(center, aligned)
        # OpenCV's scalar reduction returns the adaptive threshold to the host.
        threshold = self.cfg.threshold + int(cv2.cuda.sum(diff)[0] / (diff.size()[0] * diff.size()[1]))
        _, mask = cv2.cuda.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        masked = cv2.cuda.bitwise_and(mask, valid)
        return cv2.cuda.bitwise_and(self.median.apply(masked), valid)

    def compare_lk(self, previous, center, prev_gpu, center_gpu, keep):
        h, w = center.shape
        points = mfe.grid_keypoints(w, h, self.cfg.grid_cols, self.cfg.grid_rows)
        gpu_points = self.upload(points.reshape(1, -1, 2))
        ref, status, _ = self.time("opencv_cpu", "sparse_lk", lambda: cv2.calcOpticalFlowPyrLK(
            previous, center, points, None, winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)), keep=keep)
        result, flags, _ = self.time("opencv_cuda", "sparse_lk", self.lk.calc,
                                      prev_gpu, center_gpu, gpu_points, None, keep=keep)
        a, b = status.ravel().astype(bool), flags.download().ravel().astype(bool)
        self.counts["lk_points"] += len(a)
        self.counts["lk_status_changes"] += int(np.count_nonzero(a != b))
        error = np.linalg.norm(ref.reshape(-1, 2)[a & b] - result.download().reshape(-1, 2)[a & b], axis=1)
        self.lk_errors.extend(error.tolist())

    def compare_frame(self, folder, i):
        keep = i > 5
        ref = np.load(folder / f"ref{i}.npz")
        try:
            bgr = [np.load(folder / f"bgr{j}.npy") for j in (i - 1, i, i + 1)]
            host = [self.time("opencv_cpu", "gray", cv2.cvtColor, image, cv2.COLOR_BGR2GRAY, keep=keep)
                    for image in bgr]
            gpu = [self.time("opencv_cuda", "gray", cv2.cuda.cvtColor, self.upload(image),
                             cv2.COLOR_BGR2GRAY, keep=keep) for image in bgr]
            torch_gpu = [self.time("torch_cuda", "gray", self.mfe.bgr_to_gray,
                                   self.torch.from_numpy(image).cuda(), keep=keep) for image in bgr]
            for name, image in (("opencv_cpu", host[1]), ("opencv_cuda", gpu[1].download()),
                                ("torch_cuda", torch_gpu[1].cpu().numpy())):
                self.diff(f"{name}.gray_vs_5", image, ref["gray"])
            self.compare_lk(host[0], host[1], gpu[0], gpu[1], keep)
            masks = defaultdict(list)
            isolated_cuda = []
            fixed_center = self.upload(ref["gray"])
            for j, direction in ((0, "p"), (2, "n")):
                valid, matrix = ref[f"valid_{direction}"], ref[f"H_{direction}"]
                fixed_aligned = ref[f"aligned_{direction}"]
                size = (host[1].shape[1], host[1].shape[0])
                cpu_warp = self.time("opencv_cpu", "warp", lambda: cv2.warpPerspective(
                    host[j], matrix, size, flags=FLAGS), keep=keep)
                cuda_warp = self.time("opencv_cuda", "warp", lambda: cv2.cuda.warpPerspective(
                    gpu[j], matrix, size, flags=FLAGS), keep=keep)
                matrix_gpu = self.torch.from_numpy(matrix).cuda()
                torch_warp = self.time("torch_cuda", "warp", self.mfe.warp_inverse,
                                       torch_gpu[j], matrix_gpu, keep=keep)
                for name, image in (("opencv_cpu", cpu_warp), ("opencv_cuda", cuda_warp.download()),
                                    ("torch_cuda", torch_warp.cpu().numpy())):
                    self.diff(f"{name}.warp_vs_5", image, ref[f"aligned_{direction}"])
                    self.diff(f"{name}.warp_vs_4", image, cpu_warp)
                masks["opencv_cpu"].append(self.time("opencv_cpu", "direction_mask", self.mask_cpu,
                                                     host[1], cpu_warp, valid, keep=keep))
                masks["opencv_cuda"].append(self.time("opencv_cuda", "direction_mask", self.mask_cuda,
                                                      gpu[1], cuda_warp, self.upload(valid), keep=keep))
                mask, _ = self.time("torch_cuda", "direction_mask", self.mfe.threshold_direction,
                                    torch_gpu[1], torch_warp, self.torch.from_numpy(valid).cuda(),
                                    self.cfg, keep=keep)
                masks["torch_cuda"].append(mask)
                # The OpenCV allocation stays alive until this synchronized Torch call returns.
                hybrid_mask, _ = self.time("hybrid", "direction_mask", self.hybrid_mask,
                                           torch_gpu[1], cuda_warp,
                                           self.torch.from_numpy(valid).cuda(), keep=keep)
                masks["hybrid"].append(hybrid_mask)
                isolated = self.mask_cuda(fixed_center, self.upload(fixed_aligned), self.upload(valid))
                self.diff("opencv_cuda.direction_mask_fixed_vs_5", isolated.download(),
                          self.mask_cpu(ref["gray"], fixed_aligned, valid))
                isolated_cuda.append(isolated)
            cpu_motion = cv2.bitwise_or(*masks["opencv_cpu"])
            cuda_motion = cv2.cuda.bitwise_or(*masks["opencv_cuda"])
            torch_motion = masks["torch_cuda"][0] | masks["torch_cuda"][1]
            hybrid_motion = masks["hybrid"][0] | masks["hybrid"][1]
            cpu_closed = self.time("opencv_cpu", "morphology", lambda: cv2.morphologyEx(
                cv2.morphologyEx(cpu_motion, cv2.MORPH_OPEN, self.kernel),
                cv2.MORPH_CLOSE, self.kernel, iterations=3), keep=keep)
            cuda_closed = self.time("opencv_cuda", "morphology",
                                    lambda: self.close.apply(self.open.apply(cuda_motion)), keep=keep)
            torch_closed = self.time("torch_cuda", "morphology", self.mfe.morphology,
                                     torch_motion, 3, 1, 3, keep=keep)
            hybrid_closed = self.time("hybrid", "morphology", self.mfe.morphology,
                                      hybrid_motion, 3, 1, 3, keep=keep)
            isolated_closed = self.close.apply(self.open.apply(cv2.cuda.bitwise_or(*isolated_cuda)))
            self.diff("opencv_cuda.closed_fixed_vs_5", isolated_closed.download(), ref["closed"])
            oracle = replace(self.cfg, open_iters=0, close_iters=0)
            for name, closed in (("opencv_cpu", cpu_closed), ("opencv_cuda", cuda_closed.download()),
                                 ("torch_cuda", torch_closed.cpu().numpy().astype(np.uint8) * 255),
                                 ("hybrid", hybrid_closed.cpu().numpy().astype(np.uint8) * 255)):
                self.diff(f"{name}.closed_vs_5", closed, ref["closed"])
                self.diff(f"{name}.closed_vs_4", closed, cpu_closed)
                _, boxes = mfe.postprocess(closed, oracle)
                expected = {tuple(box) for box in ref["boxes"]}
                actual = set(boxes)
                self.counts[f"{name}.box_changed_frames"] += int(actual != expected)
                self.counts[f"{name}.boxes_added"] += len(actual - expected)
                self.counts[f"{name}.boxes_removed"] += len(expected - actual)

            # Isolate connected components on exactly the same binary mask.
            fixed = self.upload(cpu_closed)
            _, labels = self.time("opencv_cpu", "ccl_labels", lambda: cv2.connectedComponents(
                cpu_closed, connectivity=8), keep=keep)
            cuda_labels = self.time("opencv_cuda", "ccl_labels", lambda: cv2.cuda.connectedComponents(
                fixed, connectivity=8), keep=keep).download()
            fg = cpu_closed != 0
            pairs = np.unique(np.stack((labels[fg], cuda_labels[fg]), axis=1), axis=0)
            same = len(pairs) == len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1]))
            same = same and bool(np.array_equal(cuda_labels != 0, fg))
            self.counts["ccl_partition_changed_frames"] += int(not same)
        finally:
            ref.close()

    def hybrid_mask(self, center, aligned, valid):
        return self.mfe.threshold_direction(center, self.borrow(aligned), valid, self.cfg)

    def report(self):
        return {"counts": dict(self.counts), "max_abs": dict(self.maxima),
                "lk_common_point_error_px": {"samples": len(self.lk_errors),
                                             "p50": percentile(self.lk_errors, 50),
                                             "p95": percentile(self.lk_errors, 95)},
                "milliseconds": {name: {"samples": len(values), "p50": percentile(values, 50),
                                         "p95": percentile(values, 95)}
                                 for name, values in self.times.items()}}


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("capture")
    prepare.add_argument("fixtures", type=Path)
    prepare.add_argument("videos", type=Path, nargs="+")
    prepare.add_argument("--frames", type=int, default=50)
    compare = sub.add_parser("compare")
    compare.add_argument("fixtures", type=Path)
    compare.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        if args.frames < 8:
            parser.error("--frames must be at least 8")
        capture(args.fixtures, args.videos, args.frames)
        return
    source = json.loads((args.fixtures / "source.json").read_text())
    assert cv2.cuda.getCudaEnabledDeviceCount() > 0
    results = {}
    for folder in sorted(path for path in args.fixtures.iterdir() if path.is_dir()):
        test = Comparison()
        for i in range(1, source["frames"] - 1):
            test.compare_frame(folder, i)
        results[folder.name] = test.report()
        print(folder.name, "done", flush=True)
    args.out.write_text(json.dumps({"reference": source, "comparison_opencv": cv2.__version__,
                                   "videos": results}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
