from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# IMPORTANT:
# ONNX Runtime GPU needs CUDA/cuDNN DLLs available on Windows.
# Our working WhisperX Torch environment already contains CUDA 12.8 DLLs,
# so expose its torch/lib directory before importing onnxruntime.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TORCH_LIB = (
    PROJECT_ROOT
    / ".venv-whisperx"
    / "Lib"
    / "site-packages"
    / "torch"
    / "lib"
)

if not TORCH_LIB.exists():
    raise RuntimeError(
        f"Could not find Torch CUDA DLL directory:\n{TORCH_LIB}"
    )

# Make DLLs visible to Windows.
os.environ["PATH"] = (
    str(TORCH_LIB)
    + os.pathsep
    + os.environ.get("PATH", "")
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(TORCH_LIB))


import cv2
import numpy as np
import onnxruntime as ort


VIDEO = PROJECT_ROOT / (
    "Leon Kennedy meets Grace Ashcroft  Nick Apostolides and "
    "Angela Sant'Albano Interview Each Other - FGS (1080p, h264).mp4"
)

MODEL = (
    PROJECT_ROOT
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


def main() -> None:
    print("=== YUNET GPU TEST ===")

    print("Torch CUDA DLL directory:")
    print(TORCH_LIB)

    print("\nONNX Runtime:")
    print(ort.__version__)

    available = ort.get_available_providers()

    print("\nAvailable providers:")
    print(available)

    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available."
        )

    # Explicitly request CUDA first.
    session = ort.InferenceSession(
        str(MODEL),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("\nSession providers:")
    print(session.get_providers())

    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "CUDAExecutionProvider was requested but "
            "the session fell back to CPU."
        )

    input_meta = session.get_inputs()[0]

    print("\nModel input:")
    print("name:", input_meta.name)
    print("shape:", input_meta.shape)
    print("type:", input_meta.type)

    shape = input_meta.shape

    # This model reports [1, 3, 640, 640].
    # Keep the test dynamic instead of assuming 320.
    if (
        len(shape) != 4
        or shape[0] != 1
        or shape[1] != 3
        or not isinstance(shape[2], int)
        or not isinstance(shape[3], int)
    ):
        raise RuntimeError(
            f"Unexpected YuNet input shape: {shape}"
        )

    input_h = shape[2]
    input_w = shape[3]

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO}"
        )

    # Test around the selected editorial moment.
    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        951.88 * 1000.0,
    )

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            "Could not read test frame."
        )

    print("\nFrame:")
    print("shape:", frame.shape)

    resized = cv2.resize(
        frame,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Model expects NCHW float32.
    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        rgb.astype(np.float32)
        / 255.0
    )

    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )

    tensor = np.expand_dims(
        tensor,
        axis=0,
    )

    print("\nTensor:")
    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)

    outputs = session.run(
        None,
        {
            input_meta.name: tensor,
        },
    )

    print("\nInference succeeded.")

    for i, output in enumerate(outputs):
        print(
            f"output[{i}]: "
            f"shape={output.shape}, "
            f"dtype={output.dtype}"
        )

    print("\nSUCCESS: YuNet ran through CUDAExecutionProvider.")


if __name__ == "__main__":
    main()