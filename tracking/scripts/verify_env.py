"""scripts/verify_env.py — 环境安装验证（SPEC §8.4 / I-01 验收）。

打印 Python 版本、关键依赖版本、JAX 设备或明确提示 numpy 回退，以及
OpenCV 构建摘要（是否启用 CUDA、线程数）。退出码 0=JAX GPU 可用，
3=仅 CPU/numpy 回退（不视为失败，仅提示）。
"""
from __future__ import annotations

import platform
import sys


def _version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({exc.__class__.__name__})"
    return getattr(mod, "__version__", "unknown")


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"numpy: {_version('numpy')}")
    print(f"cv2: {_version('cv2')}")
    print(f"yaml: {_version('yaml')}")
    print(f"pandas: {_version('pandas')}")
    print(f"matplotlib: {_version('matplotlib')}")
    print(f"pytest: {_version('pytest')}")

    jax_status = 3
    try:
        import jax

        devices = jax.devices()
        print(f"jax: {jax.__version__}  backend={jax.default_backend()}")
        print(f"jax.devices(): {devices}")
        if any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices):
            jax_status = 0
        else:
            print("NOTE: 未检测到 GPU 设备 → 使用 numpy 后端回退运行（算法正确性不受影响）")
    except Exception as exc:  # noqa: BLE001
        print(f"jax: MISSING ({exc.__class__.__name__}: {exc})")
        print("NOTE: JAX 不可用 → 使用 numpy 后端回退；单测无需 GPU")

    try:
        import cv2

        build = cv2.getBuildInformation()
        cuda_lines = [ln.strip() for ln in build.splitlines()
                      if "CUDA" in ln or "parallel" in ln.lower()]
        print("OpenCV build (CUDA/threads 摘要):")
        for ln in cuda_lines[:8]:
            print(f"  {ln}")
    except Exception:  # noqa: BLE001
        pass

    return jax_status if jax_status != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
