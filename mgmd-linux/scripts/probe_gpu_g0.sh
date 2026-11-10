#!/usr/bin/env bash
# Run inside mgmd-ros as the project user. Arguments: input video and new output directory.
set -eo pipefail

video=$(realpath "${1:?input video required}")
run=$(realpath -m "${2:?new output directory required}")
test -r "$video"
mkdir "$run"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
weights=/home/gaoduan/tracking/data_raw/mgmd_weights
site=$(python3 -m site --user-site)
cuda=$(find "$site/nvidia" /usr/local/lib/python3.8/dist-packages/nvidia \
    -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd: || true)
export LD_LIBRARY_PATH="${cuda}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

failed=0
probe() {
    local name=$1
    shift
    if "$@" >"$run/$name.log" 2>&1; then
        printf '%s\tPASS\n' "$name" | tee -a "$run/status.tsv"
    else
        printf '%s\tFAIL\n' "$name" | tee -a "$run/status.tsv"
        tail -8 "$run/$name.log"
        failed=1
    fi
}

{
    cat /etc/os-release
    python3 --version
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
    ffmpeg -version
    sha256sum "$video" "$weights/lac.onnx" "$weights/yolov5s_GLAD-crop.onnx"
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate "$video"
    for path in mgmd mgmd_onnx ros1_ws/src/mav_tracker; do
        (cd "$root" && find "$path" -type f -not -path '*/__pycache__/*' -print0 |
            sort -z | xargs -0 -r sha256sum)
    done
} >"$run/environment.log" 2>&1

probe cuda python3 - "$weights" "$run" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
import torch

print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("cuDNN", torch.backends.cudnn.version(), "ORT", ort.__version__)
assert torch.ones(8, device="cuda").sum().item() == 8
for name in ("lac.onnx", "yolov5s_GLAD-crop.onnx"):
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(sys.argv[2]) / name)
    session = ort.InferenceSession(str(Path(sys.argv[1]) / name), options,
                                   providers=["CUDAExecutionProvider"])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    inp = session.get_inputs()[0]
    x = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)
    outputs = session.run(None, {inp.name: x})
    assert outputs and all(np.isfinite(y).all() for y in outputs)
    with open(session.end_profiling()) as trace:
        kernels = Counter(event["args"]["provider"] for event in json.load(trace)
                          if "provider" in event.get("args", {}))
    assert kernels["CUDAExecutionProvider"] > 0
    print(name, session.get_providers(), [y.shape for y in outputs], dict(kernels))
PY

decode() {
    timeout 35 ffmpeg -nostdin -xerror -loglevel verbose -hwaccel cuda \
        -hwaccel_output_format cuda -i "$video" \
        -map 0:v:0 -an -vsync 0 -frames:v 120 -f null -
}
probe nvdec decode
probe nvdec_device_frame grep -q 'pixfmt:cuda' "$run/nvdec.log"
probe nvenc timeout 35 ffmpeg -nostdin -xerror -loglevel error -i "$video" \
    -map 0:v:0 -an -vsync 0 -frames:v 120 -c:v h264_nvenc "$run/encode.mp4"
probe direct timeout 35 ffmpeg -nostdin -xerror -loglevel verbose -hwaccel cuda \
    -hwaccel_output_format cuda -i "$video" -map 0:v:0 -an -vsync 0 -frames:v 120 \
    -c:v h264_nvenc "$run/direct.mp4"
probe direct_device_frame grep -q 'pixfmt:cuda' "$run/direct.log"

validate_video() {
    local output=$1
    ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=codec_name,width,height,nb_read_frames -of csv=p=0 "$output" |
        tee /dev/stderr | grep -qx 'h264,1920,1080,120'
}
for output in encode direct; do
    if [[ -f "$run/$output.mp4" ]]; then
        probe "${output}_verify" validate_video "$run/$output.mp4"
    fi
done
exit "$failed"
