"""Device image operations and CUDA candidate extraction for MFE."""

from contextlib import nullcontext

import torch
from torch.nn.functional import avg_pool2d, grid_sample, max_pool2d, pad


def bgr_to_gray(image):
    if not image.is_cuda or image.dtype != torch.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a CUDA uint8 HWC BGR image")
    b, g, r = image.int().unbind(dim=2)
    return ((3735 * b + 19235 * g + 9798 * r + 16384) >> 15).to(torch.uint8)


def perspective_coordinates(matrix, x, y):
    z = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    factor = torch.where(z != 0, 1 / z, 0)
    sx = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) * factor
    sy = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) * factor
    return sx, sy


def warp_inverse(source, matrix, return_valid=False):
    """Apply a destination-to-source homography with bilinear sampling."""
    h, w = source.shape
    x = torch.arange(w, dtype=torch.float64, device=source.device)[None, :]
    y = torch.arange(h, dtype=torch.float64, device=source.device)[:, None]
    sx, sy = perspective_coordinates(matrix, x, y)
    grid = torch.stack(((2 * sx + 1) / w - 1, (2 * sy + 1) / h - 1), dim=-1)
    warped = grid_sample(source[None, None].double(), grid[None], padding_mode="zeros",
                         align_corners=False)
    pixels = warped[0, 0].round().clamp_(0, 255).to(torch.uint8)
    if return_valid:
        valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
        return pixels, valid
    return pixels


def threshold_direction(center, aligned, valid, cfg):
    delta = (center.to(torch.int16) - aligned).abs()
    threshold = cfg.threshold
    if cfg.adaptive_thresh:
        threshold = threshold + torch.div(delta.sum(dtype=torch.int64), delta.numel(),
                                          rounding_mode="floor")
    valid = valid.bool()
    mask = (delta > threshold) & valid
    if cfg.median_blur >= 3:
        radius = cfg.median_blur // 2
        padded = pad(mask[None, None].float(), (radius,) * 4, mode="replicate")
        mask = (avg_pool2d(padded, cfg.median_blur, stride=1)[0, 0] > .5) & valid
    return mask, threshold


def combine_motion(previous, following, mode):
    if following is None or mode == "2frame":
        return previous
    if mode == "3frame_and":
        return previous & following
    return previous | following


def morphology(mask, size, open_iters, close_iters):
    value = mask[None, None].float()
    # For binary data max pooling is dilation. Negation gives erosion with a neutral border.
    for iterations, signs in ((open_iters, (-1, 1)), (close_iters, (1, -1))):
        for sign in signs:
            for _ in range(iterations):
                value = sign * max_pool2d(sign * value, size, stride=1, padding=size // 2)
    return value[0, 0].bool()


class CudaMfe:
    """Sparse CUDA LK, batched RANSAC, and GPU connected component statistics."""

    def __init__(self, cfg, width, height, hypotheses=256, warp_backend="torch",
                 default_stream_lock=None):
        import cv2
        from mgmd.mfe import grid_keypoints

        cfg.validate()
        if cfg.preprocess_blur or cfg.disp_filter:
            raise ValueError("CUDA MFE supports the recommended preprocessing profile")
        if not hasattr(cv2.cuda, "connectedComponents") or cv2.cuda.getCudaEnabledDeviceCount() < 1:
            raise RuntimeError("CUDA OpenCV with cudaoptflow and cudaimgproc is required")
        self.cv = cv2
        self.cfg = cfg
        self.stream = torch.cuda.current_stream()
        self.default_stream_lock = default_stream_lock
        self.cv_stream = (cv2.cuda.wrapStream(self.stream.cuda_stream)
                          if self.stream.cuda_stream else cv2.cuda.Stream.Null())
        if warp_backend not in ("torch", "opencv"):
            raise ValueError("warp_backend must be torch or opencv")
        self.warp_backend = warp_backend
        if warp_backend == "opencv":
            self.warp_x = torch.arange(width, dtype=torch.float32, device="cuda")[None, :]
            self.warp_y = torch.arange(height, dtype=torch.float32, device="cuda")[:, None]
        self.lk = cv2.cuda.SparsePyrLKOpticalFlow_create((15, 15), 3, 30)
        points = grid_keypoints(width, height, cfg.grid_cols, cfg.grid_rows)
        self.points = torch.as_tensor(points.reshape(1, -1, 2), device="cuda")
        self.scale = float(max(width, height))
        self.identity = torch.eye(3, dtype=torch.float64, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(0)
        self.sample = torch.rand((hypotheses, 4), generator=generator, device="cuda")
        self.pixel = torch.arange(width * height, dtype=torch.int32, device="cuda")
        self.width, self.height = width, height
        self._gray_cache = {}
        self.candidate_cv_stream = self.cv_stream

    def _mat(self, value, kind):
        height, width = value.shape[:2]
        return self.cv.cuda.createGpuMatFromCudaMemory(
            height, width, kind, value.data_ptr(), value.stride(0) * value.element_size())

    @staticmethod
    def _tensor(mat, dtype, channels=1):
        width, height = mat.size()
        size = int(dtype[-1])
        shape = (height, width) if channels == 1 else (height, width, channels)
        strides = (mat.step, size) if channels == 1 else (mat.step, channels * size, size)

        class BorrowedMat:
            __cuda_array_interface__ = {"version": 2, "shape": shape, "strides": strides,
                                        "typestr": dtype, "data": (mat.cudaPtr(), False)}

        return torch.as_tensor(BorrowedMat(), device="cuda")

    def _homography(self, tracked, status):
        src = self.points.reshape(-1, 2).double() / self.scale
        dst = tracked.reshape(-1, 2).double() / self.scale
        valid = status.reshape(-1).bool() & torch.isfinite(dst).all(dim=1)
        dst = torch.where(valid[:, None], dst, 0)
        x, y = dst.unbind(dim=1)
        u, v = src.unbind(dim=1)
        z, one = torch.zeros_like(x), torch.ones_like(x)
        system = torch.stack((torch.stack((x, y, one, z, z, z, -u*x, -u*y), dim=1),
                              torch.stack((z, z, z, x, y, one, -v*x, -v*y), dim=1)), dim=1)
        count = valid.sum()
        order = valid.to(torch.uint8).argsort(descending=True)
        index = order[(self.sample * count.clamp_min(1)).long()]
        a = system[index].reshape(-1, 8, 8)
        b = src[index].reshape(-1, 8, 1)
        parameters, info = torch.linalg.solve_ex(a, b)
        parameters = parameters.squeeze(-1)
        mapped = torch.einsum("nki,bi->bnk", system[:, :, :6], parameters[:, :6])
        denominator = 1 + parameters[:, 6:8] @ dst.T
        error = (mapped / denominator[..., None] - src).square().sum(dim=-1)
        inliers = (error <= (self.cfg.ransac_thresh / self.scale)**2) & valid
        inliers &= info[:, None] == 0
        best = inliers.sum(dim=1).argmax().reshape(1)
        keep = inliers.index_select(0, best).reshape(-1)
        aw = (system * keep[:, None, None]).reshape(-1, 8)
        bw = (src * keep[:, None]).reshape(-1, 1)
        refined, status = torch.linalg.solve_ex(aw.T @ aw, aw.T @ bw)
        h = torch.cat((refined.flatten(), one[:1])).reshape(3, 3)
        scale = h.new_tensor((self.scale, self.scale, 1))
        h = h * scale[:, None] / scale
        good = (keep.sum() >= self.cfg.min_keypoints) & (status == 0) & torch.isfinite(h).all()
        return torch.where(good, h, self.identity)

    def _warp(self, source, matrix):
        if self.warp_backend == "torch":
            return warp_inverse(source, matrix, return_valid=True)
        cv = self.cv
        sx, sy = perspective_coordinates(matrix.float(), self.warp_x, self.warp_y)
        valid = (sx >= 0) & (sx <= self.width - 1) & (sy >= 0) & (sy <= self.height - 1)
        output = torch.empty_like(source)
        dst = self._mat(output, cv.CV_8UC1)
        cv.cuda.remap(self._mat(source, cv.CV_8UC1),
                      self._mat(sx, cv.CV_32FC1), self._mat(sy, cv.CV_32FC1),
                      cv.INTER_LINEAR, dst=dst, borderMode=cv.BORDER_CONSTANT,
                      stream=self.candidate_cv_stream)
        if dst.cudaPtr() != output.data_ptr():
            raise RuntimeError("OpenCV remap replaced the shared CUDA output")
        return output, valid

    def _direction(self, source, center, stamp):
        h, tracked, status = self._estimate(source, center, stamp)
        return self._motion(source, center, h, stamp)

    def _estimate(self, source, center, stamp):
        cv = self.cv
        tracked, status, _ = self.lk.calc(
            self._mat(source, cv.CV_8UC1), self._mat(center, cv.CV_8UC1),
            self._mat(self.points, cv.CV_32FC2), None, stream=self.cv_stream)
        stamp("LK")
        h = self._homography(self._tensor(tracked, "<f4", 2), self._tensor(status, "|u1"))
        stamp("RANSAC")
        return h, tracked, status

    def _motion(self, source, center, h, stamp):
        aligned, valid = self._warp(source, h)
        # The Torch solver consumes OpenCV-owned buffers asynchronously.
        torch.cuda.current_stream().synchronize()
        stamp("warp")
        mask, _ = threshold_direction(center, aligned, valid, self.cfg)
        stamp("threshold_median")
        return mask

    def _boxes(self, mask, stamp):
        cv = self.cv
        pixels = mask.to(torch.uint8)
        stream = torch.cuda.current_stream()
        # OpenCV 4.12 CCL has no stream argument.
        if stream.cuda_stream:
            stream.synchronize()
        with self.default_stream_lock or nullcontext():
            labels_mat = cv.cuda.connectedComponents(self._mat(pixels, cv.CV_8UC1), connectivity=8)
            if stream.cuda_stream:
                torch.cuda.default_stream().synchronize()
        stamp("CCL")
        ids = mask.flatten().nonzero().flatten()
        labels = self._tensor(labels_mat, "<i4").reshape(-1)[ids].long()
        pixel = self.pixel[ids]
        size = mask.numel() + 1
        area = torch.zeros(size, dtype=torch.int32, device=labels.device)
        area.scatter_add_(0, labels, torch.ones_like(pixel))
        minimum = torch.full_like(area, size)
        minimum.scatter_reduce_(0, labels, pixel, reduce="amin")
        x, y = pixel % self.width, pixel // self.width
        x1, y1 = torch.full_like(area, self.width), torch.full_like(area, self.height)
        x2, y2 = torch.zeros_like(area), torch.zeros_like(area)
        for target, coord, op in ((x1, x, "amin"), (y1, y, "amin"),
                                  (x2, x, "amax"), (y2, y, "amax")):
            target.scatter_reduce_(0, labels, coord, reduce=op)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        cfg = self.cfg
        keep = area >= max(1, cfg.official_area_min or cfg.min_area)
        keep[0] = False
        if cfg.max_area:
            keep &= area <= cfg.max_area
        if cfg.aspect_filter:
            keep &= (w >= cfg.min_aspect * h) & (w <= cfg.max_aspect * h)
        ids = keep.nonzero().flatten()
        ids = ids[minimum[ids].argsort()]
        boxes = torch.stack((x1[ids], y1[ids], x2[ids] + 1, y2[ids] + 1), dim=1).double()
        scores = (w[ids] * h[ids]).double()
        stamp("region_stats")
        return boxes, scores

    def candidates(self, previous, center, following, profile=None, *, frame_ids=None):
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("CUDA MFE requires its creation stream")
        def stamp(name):
            if profile is not None:
                event = torch.cuda.Event(enable_timing=True)
                event.record(self.stream)
                profile.append((name, event))

        stamp("start")
        frames = self.gray_frames((previous, center, following), frame_ids)
        stamp("gray")
        return self.candidates_from_gray(frames, stamp)

    def candidates_from_gray(self, frames, stamp):
        a = self._direction(frames[0], frames[1], stamp)
        b = self._direction(frames[2], frames[1], stamp) if self.cfg.needs_next else None
        return self.combine_candidates(a, b, stamp)

    def gray_frames(self, images, frame_ids):
        if frame_ids is None:
            frames = [bgr_to_gray(frame) for frame in images]
        else:
            # The MFE worker owns this cache and consumes windows in source order.
            cache = {}
            for seq, image in zip(frame_ids, images):
                gray = self._gray_cache.get(seq)
                cache[seq] = bgr_to_gray(image) if gray is None else gray
            self._gray_cache = cache
            frames = [cache[seq] for seq in frame_ids]
        return frames

    def combine_candidates(self, a, b, stamp):
        motion = combine_motion(a, b, self.cfg.mode)
        motion = morphology(motion, self.cfg.morph_size, self.cfg.open_iters, self.cfg.close_iters)
        stamp("morphology")
        return self._boxes(motion, stamp)
