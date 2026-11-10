"""Compare the ROS ONNX model wrappers' host I/O with preallocated CUDA I/O."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from mgmd_onnx import make_session


RTOL = 1e-5
ATOL = 1e-3


def probe(model, batches, out):
    reference = make_session(model, "cuda")
    assert reference.get_providers()[0] == "CUDAExecutionProvider"
    inp, = reference.get_inputs()
    input_name, input_type, input_shape = inp.name, inp.type, inp.shape
    names = [item.name for item in reference.get_outputs()]
    assert input_type == "tensor(float)"
    rng = np.random.default_rng(20261111)
    cases = []
    for batch in batches:
        shape = (batch, *input_shape[1:])
        assert all(isinstance(dim, int) and dim > 0 for dim in shape)
        assert not isinstance(input_shape[0], int) or input_shape[0] == batch
        for value in (np.zeros(shape, np.float32), rng.random(shape, dtype=np.float32)):
            cases.append((value, reference.run(names, {input_name: value})))
    del reference

    device = torch.cuda.current_device()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(out / model.stem)
    session = ort.InferenceSession(str(model), options, providers=[
        ("CUDAExecutionProvider", {"device_id": device})
    ])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    same_session_refs = [session.run(names, {input_name: value}) for value, _ in cases]

    results = []
    buffers = {}
    for (value, independent), expected in zip(cases, same_session_refs):
        shape = value.shape
        if shape not in buffers:
            x = torch.empty(shape, dtype=torch.float32, device=device)
            ys = [
                torch.empty(y.shape, dtype=torch.from_numpy(y).dtype, device=device)
                for y in expected
            ]
            binding = session.io_binding()
            binding.bind_input(input_name, "cuda", device, value.dtype, shape, x.data_ptr())
            for name, y, ref in zip(names, ys, expected):
                binding.bind_output(name, "cuda", device, ref.dtype, ref.shape, y.data_ptr())
            buffers[shape] = x, ys, binding
        x, ys, binding = buffers[shape]
        # The test upload and download are outside the profiled inference call.
        x.copy_(torch.from_numpy(value))
        torch.cuda.current_stream().synchronize()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        bound = binding.get_outputs()
        assert len(bound) == len(ys)
        differences = []
        for name, result, y, ref, separate in zip(names, bound, ys, expected, independent):
            assert result.device_name().lower() == "cuda"
            assert result.data_ptr() == y.data_ptr()
            assert tuple(result.shape()) == ref.shape
            actual = y.cpu().numpy()
            differences.append({"name": name, "shape": list(ref.shape),
                                "matches": bool(np.allclose(actual, ref, rtol=RTOL, atol=ATOL)),
                                "max_abs": float(np.max(np.abs(actual - ref))),
                                "independent_max_abs": float(np.max(np.abs(actual - separate)))})
        results.append({"shape": list(shape), "outputs": differences})

    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text())
    runs = [event for event in events if event.get("name") == "model_run"]
    assert len(runs) == 2 * len(cases)
    bound_runs = runs[len(cases):]
    nodes = [event for event in events if event.get("cat") == "Node"
             and event.get("name", "").endswith("_kernel_time")
             and any(run["ts"] <= event["ts"] < run["ts"] + run["dur"] for run in bound_runs)]
    providers = Counter(event.get("args", {}).get("provider", "unknown") for event in nodes)
    copies = [event["name"] for event in nodes if "memcpy" in event["name"].lower()
              or "memcpy" in event.get("args", {}).get("op_name", "").lower()]
    return {
        "model": model.name, "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "input": {"name": input_name, "type": input_type, "model_shape": input_shape},
        "cases": results, "kernel_providers": dict(providers), "ort_copy_nodes": copies,
        "ort_profile": profile.name,
        "passed": (bool(nodes) and set(providers) == {"CUDAExecutionProvider"} and not copies
                   and all(item["matches"] for case in results for item in case["outputs"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("out", type=Path, help="New directory for the ORT profiles and summary")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir()
    models = [
        probe(args.weights / "lac.onnx", (1, 4), out),
        probe(args.weights / "yolov5s_GLAD-crop.onnx", (1,), out),
    ]
    summary = {
        "torch": torch.__version__, "ort": ort.__version__,
        "gpu": torch.cuda.get_device_name(), "rtol": RTOL, "atol": ATOL,
        "models": models,
    }
    report = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (out / "summary.json").write_text(report)
    print(report, end="")
    return 0 if all(model["passed"] for model in models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
