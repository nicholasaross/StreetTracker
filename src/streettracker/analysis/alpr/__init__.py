"""Post-processing ALPR pass over a closed StreetTracker session.

Compares two end-to-end license-plate-reading pipelines against the
``*_main_*.jpg`` snaps in a session directory:

- ``bespoke``   — ``license_plate_detector.pt`` (Ultralytics YOLOv8) + EasyOCR.
- ``preferred`` — ``open-image-models`` (ONNX) + ``fast-plate-ocr`` (ONNX).
- (optional) ``ablation_bespokedet_fastocr`` — bespoke detector + fast OCR.
  Isolates whether the bespoke detector or the EasyOCR head is the weak
  link in the bespoke pipeline.

Dev-box only. The Orin Nano can run these too, but the comparison study
is post-processing — never on the live runtime path. Install with the
``alpr`` extra: ``uv sync --extra alpr``.

Outputs land sidecar to the session, so nothing in
``<session>_data.json`` / ``<session>_summary.html`` is modified.

Ported wholesale from NanoTracker's ``alpr/`` package.
"""
