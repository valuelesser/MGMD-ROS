"""Cost of fully padded LAC and a fixed four-ROI LAD on one phantom30 frame."""
from contextlib import nullcontext
import json
from time import perf_counter

import onnx
from onnx import numpy_helper
import torch

from mgmd import config
from mgmd_onnx.cuda_opencv_decoder import OpenCvCudaDecoder
from mgmd_onnx.cuda_pipeline import cuda_model


def batch_model(path, batch):
    model = onnx.load(str(path))
    for info in [*model.graph.input, *model.graph.output]:
        info.type.tensor_type.shape.dim[0].dim_value = batch
    del model.graph.value_info[:]
    shape_names = {n.input[1] for n in model.graph.node if n.op_type == "Reshape"}
    shape_names.update(n.input[3] for n in model.graph.node if n.op_type == "Resize")
    for value in model.graph.initializer:
        if value.name in shape_names:
            shape = numpy_helper.to_array(value).copy()
            assert shape[0] == 1
            shape[0] = batch
            value.CopyFrom(numpy_helper.from_array(shape, value.name))
    onnx.checker.check_model(model)
    return model.SerializeToString()


def measure(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(12):
        fn()
    torch.cuda.synchronize()
    return round(1000 * (perf_counter() - start) / 12, 3)


@torch.inference_mode()
def main():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), OpenCvCudaDecoder("/data/videos/phantom30.mp4", 25) as decoder:
        image = decoder.read().image.permute(2, 0, 1).flip(0).float() / 255
        crops = torch.stack([image[:, y:y+320, x:x+320] for x, y in
                             [(0, 0), (600, 300), (800, 400), (1200, 550)]])
        lac = cuda_model(config.WEIGHTS_DIR / "lac.onnx", stream, nullcontext(), True)
        path = config.WEIGHTS_DIR / "yolov5s_GLAD-crop.onnx"
        lad = cuda_model(path, stream, nullcontext(), True)
        batched = cuda_model(batch_model(path, 4), stream, nullcontext(), True)
        for model in (lac, lad, batched):
            model.enqueue_stream = stream
        patches = crops[:, :, :32, :32].repeat(256, 1, 1, 1).contiguous()
        lac_scores = [lac.run(patches[:size])[0] for size in (112, 1024)]
        singles = lambda: torch.cat([lad.run(crop.unsqueeze(0))[0] for crop in crops])
        group = lambda: batched.run(crops)[0]
        delta = (singles() - group()).abs().amax(dim=(0, 1))
        print(json.dumps(dict(lac112_ms=measure(lambda: lac.run(patches[:112])),
                              lac1024_ms=measure(lambda: lac.run(patches)),
                              lac_max_logit_delta=float((lac_scores[0]-lac_scores[1][:112]).abs().max()),
                              lad1_ms=measure(lambda: lad.run(crops[:1])),
                              lad4_serial_ms=measure(singles), lad4_batch_ms=measure(group),
                              lad_raw_max_delta=delta.tolist())))
if __name__ == "__main__":
    main()
