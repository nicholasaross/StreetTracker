"""Off-device post-processing: ALPR, recolor, make/model classification,
cross-session re-id.

Each module is standalone — it reads a closed session directory, runs a
per-record computation, and rewrites the JSONL + data.json + summary.html.
Pattern documented in NanoTracker's `scripts/recolor_session.py`.
"""

import sys as _sys

if _sys.platform == "win32":
    # PyTorch's Windows cu126 wheels bundle the CUDA 12.6 runtime + cuDNN 9
    # DLLs under `torch/lib/`. ONNX Runtime's CUDAExecutionProvider needs
    # those DLLs on Windows' loader search path, but it only inspects PATH
    # plus directories registered via `os.add_dll_directory`. Register
    # torch's lib dir before any analysis module triggers an ORT session
    # so the CUDA EP loads instead of falling back to CPU.
    import os as _os
    try:
        import torch as _torch
        _torch_lib = _os.path.join(_os.path.dirname(_torch.__file__), "lib")
        if _os.path.isdir(_torch_lib):
            _os.add_dll_directory(_torch_lib)
    except ImportError:
        pass
