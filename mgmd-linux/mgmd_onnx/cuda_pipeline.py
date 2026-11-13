"""CUDA stages for the experimental ROS file playback path."""

from threading import Lock

import torch
from torch.nn.functional import grid_sample

from mgmd import config
from mgmd.lad import _inject_syspath, yolov5_dir
from mgmd.mfe import MFEConfig
from mgmd_onnx.cuda_mfe import CudaMfe
from mgmd_onnx.cuda_models import CudaModel
from mgmd_onnx.cuda_tf import CudaTf


def cuda_model(path, stream=None, default_stream_lock=None, enqueue_only=False):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider = "CUDAExecutionProvider"
    if stream is not None:
        provider = (provider, {"device_id": stream.device.index,
                               "user_compute_stream": str(stream.cuda_stream)})
    session = ort.InferenceSession(str(path), options, providers=[provider])
    session.disable_fallback()
    run_options = None
    if enqueue_only:
        run_options = ort.RunOptions()
        run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    return CudaModel(session, default_stream_lock, run_options)


class CudaPipeline:
    def __init__(self, library, width, height, lac_threshold=.5, lad_threshold=.5,
                 warp_backend="torch", mfe_stream=None, model_stream=None, lad_stream=None,
                 second_lad_stream=None, async_lad=False, tracker=None, shared_lock=None):
        # OpenCV CCL uses the default stream and cannot run during ORT capture.
        default_stream_lock = shared_lock
        if default_stream_lock is None and mfe_stream is not None and model_stream is not None:
            default_stream_lock = Lock()
        with torch.cuda.stream(mfe_stream):
            self.mfe = CudaMfe(MFEConfig().apply_recommended(), width, height,
                               warp_backend=warp_backend, default_stream_lock=default_stream_lock)
        self.tf = CudaTf(library) if tracker is None else tracker
        self.lac = cuda_model(config.WEIGHTS_DIR / "lac.onnx", model_stream, default_stream_lock,
                              enqueue_only=async_lad)
        self.lad = cuda_model(config.WEIGHTS_DIR / "yolov5s_GLAD-crop.onnx",
                              lad_stream if lad_stream is not None else model_stream,
                              default_stream_lock,
                              enqueue_only=async_lad or second_lad_stream is not None)
        self.second_lad = (cuda_model(config.WEIGHTS_DIR / "yolov5s_GLAD-crop.onnx",
                                     second_lad_stream, default_stream_lock, enqueue_only=True)
                           if second_lad_stream is not None else None)
        _inject_syspath(yolov5_dir())
        self.lac_threshold = lac_threshold
        self.lad_threshold = lad_threshold
        self.width, self.height = width, height
        self.lac_axis = (torch.arange(32, device="cuda") + .5) / 32
        self.lad_axis = torch.arange(320, device="cuda")

    def _classify(self, image, boxes):
        lo = (boxes[:, :2].trunc() - 2).clamp_min(0).float()
        hi = torch.minimum(boxes[:, 2:].round() + 2,
                           boxes.new_tensor((self.width, self.height))).float()
        size = (hi - lo).clamp_min(1)
        sample = lo[:, None, :] + self.lac_axis[None, :, None] * size[:, None, :] - .5
        sample = torch.maximum(lo[:, None, :], torch.minimum(sample, hi[:, None, :] - 1))
        grid_x = ((2 * sample[:, :, 0] + 1) / self.width - 1)[:, None, :].expand(-1, 32, -1)
        grid_y = ((2 * sample[:, :, 1] + 1) / self.height - 1)[:, :, None].expand(-1, -1, 32)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        crops = grid_sample(image.expand(len(boxes), -1, -1, -1), grid,
                            align_corners=False, padding_mode="border")
        inputs = crops.round().div_(255).contiguous()
        return self.lac.run(inputs)[0].double().softmax(dim=1)[:, 1]

    def _refine(self, image, box, model):
        from utils.general import non_max_suppression

        center = (box[:2] + box[2:]) * .5
        x = (center[0] - 160).round().long().clamp(0, self.width - 320)
        y = (center[1] - 160).round().long().clamp(0, self.height - 320)
        crop = image[:, :, y + self.lad_axis[:, None], x + self.lad_axis[None, :]]
        raw = model.run(crop.div(255).contiguous())[0]
        det = non_max_suppression(raw, self.lad_threshold, .45)[0]
        if det.shape[0] == 0:
            return det.new_empty((0, 5))
        coords = det[:, :4].round().clamp(0, 320)
        centers = (coords[:, :2] + coords[:, 2:]) * .5
        nearest = (centers - 160).square().sum(dim=1).argmin().reshape(1)
        coords = coords.index_select(0, nearest) + torch.stack((x, y, x, y))
        return torch.cat((coords, det.index_select(0, nearest)[:, 4:5]), dim=1)

    def step(self, previous, center, following, seq, mfe_profile=None, *, frame_ids=None):
        timing = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        timing[0].record()
        boxes, scores = self.mfe.candidates(previous, center, following, mfe_profile,
                                           frame_ids=frame_ids)
        timing[1].record()
        return self.finish(center, boxes, scores, seq, timing)

    def finish(self, center, boxes, scores, seq, timing=None):
        image, tracks, ids, status = self.track_classify(center, boxes, scores, seq, timing)
        output = self.refine_tracks(image, tracks, ids)
        if timing:
            timing[4].record()
        return output, status, len(boxes), timing

    def track_classify(self, center, boxes, scores, seq, timing=None):
        tracks, ids, status = self.track(boxes, scores, seq)
        if timing:
            timing[2].record()
        image, tracks, ids = self.classify_tracks(center, tracks, ids)
        if timing:
            timing[3].record()
        return image, tracks, ids, status

    def track(self, boxes, scores, seq):
        result = self.tf.step(boxes, scores, seq)
        rows = torch.arange(result.boxes.shape[0], device=boxes.device)
        valid = rows < result.count
        return result.boxes[valid], result.ids[valid], result.status

    def classify_tracks(self, center, tracks, ids):
        image = center.permute(2, 0, 1).flip(0).unsqueeze(0).float()
        if tracks.shape[0]:
            selected = self._classify(image, tracks) >= self.lac_threshold
            tracks, ids = tracks[selected], ids[selected]
        return image, tracks, ids

    def refine_tracks(self, image, tracks, ids, model=None):
        if model is None:
            model = self.lad
        detections = []
        for track, track_id in zip(tracks.unbind(), ids.unbind()):
            box = self._refine(image, track, model)
            if box.shape[0]:
                detections.append(torch.cat((box, track_id.reshape(1, 1)), dim=1))
        return torch.cat(detections) if detections else tracks.new_empty((0, 6))
