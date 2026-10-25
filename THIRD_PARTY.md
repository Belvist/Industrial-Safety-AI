# Third-party components

## Policy

Record the repository and license for external code, models and datasets. Check model/checkpoint licensing separately from runtime-library licensing. Preserve required notices. Model weights and captured media are runtime assets, excluded from Git.

## Vendored source

`backend/app/inference/tpu_wire/` contains the unmodified wire-format package
`tpu_pycloud_format` 0.0.1 (protobuf message + AMQP header names) issued by the
LinQ TPU organizers for the hackathon. It is vendored because its published
wheel pins `numpy<2`, while this project uses NumPy 2.x. The package is pure
Python and only serializes tensors; it does not contain the TPU runtime itself.
Files: `__init__.py`, `_io.py`, `headers.py`, `message.proto`,
`message_pb2.py`. Keep this copy in sync with the organizers' wheel if the
wire format changes.

## Optional evidence encoder

When OpenCV cannot encode H.264, the application invokes an installed FFmpeg
binary with `libx264` to produce a browser-playable MP4. No binary is vendored.
FFmpeg licensing depends on build options (LGPL/GPL); libx264 is GPL-licensed.
Use the notices for the actual binary when packaging the desktop application.
Sources: https://ffmpeg.org/legal.html and https://www.videolan.org/developers/x264.html.

## Runtime libraries


No third-party application source code was pasted into this repository. Dependencies are installed with pip/npm; their distributions retain their license notices.

| Component | Source | License |
|---|---|---|
| FastAPI | https://github.com/fastapi/fastapi | MIT |
| Uvicorn | https://github.com/encode/uvicorn | BSD-3-Clause |
| Pydantic / pydantic-settings | https://github.com/pydantic | MIT |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause |
| OpenCV Python | https://github.com/opencv/opencv-python | Package MIT; OpenCV Apache-2.0; bundled dependencies have their own notices |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | MIT |
| React | https://github.com/facebook/react | MIT |
| Vite | https://github.com/vitejs/vite | MIT |
| TypeScript | https://github.com/microsoft/TypeScript | Apache-2.0 |
| Lucide icons | https://github.com/lucide-icons/lucide | ISC |

## Downloaded model artifacts

`backend/scripts/download_model.py` downloads pinned ONNX artifacts, verifies SHA-256 and never loads Python/Pickle checkpoints. Models are not redistributed in this repository. No training dataset is downloaded by the application.

### PPE detector

- Publisher: Hexmon; base architecture by Ultralytics, YOLOv8m.
- Source: https://huggingface.co/Hexmon/vyra-yolo-ppe-detection
- Revision: `08895b33d95d2587423ebe4f7c1b9c41beebd642`, artifact `best.onnx`.
- SHA-256: `b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246`.
- Model card declares **CC BY 4.0**, while the actual ONNX metadata declares **AGPL-3.0**, inherited from Ultralytics. Do not treat the card as permission to ignore upstream AGPL terms. License compatibility/redistribution or closed-source commercial use needs resolution with the publishers; this integration does not relicense their artifact.
- Publisher reports training on [Roboflow PPE Combined Model v4](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4), CC BY 4.0. This attribution is from the model card; no dataset is redistributed here.
- Original classes: Fall-Detected, Gloves, Goggles, Hardhat, Ladder, Mask, NO-Gloves, NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest.
- The application uses helmet/no_helmet/vest/no_vest, with its own letterboxing, output decoding and NMS integration. The artifact itself is unmodified.

### Person detector

- Architecture/author: Ultralytics YOLOv8n trained on COCO; ONNX mirror: https://huggingface.co/webml/yolov8n.
- Revision: `85bc8d7ab30d4065a41909d756c42819b67b4388`, artifact `onnx/yolov8n.onnx`.
- SHA-256: `190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d`.
- Embedded ONNX metadata: Ultralytics, **AGPL-3.0**, https://ultralytics.com/license.
- Only `person` detections are used. This reduces dependence on the PPE model's person recall in non-industrial scenes. Artifact is unmodified.

Upstream license text: https://github.com/ultralytics/ultralytics/blob/main/LICENSE

## Locally trained models (not third-party, not redistributed in git)

Selected explicitly via `MODEL_PATH`; SHA registry lives in
`backend/scripts/download_model.py`. Weights, datasets and footage stay local.

| Artifact | SHA-256 (prefix) | Trained from |
|---|---|---|
| `ppe-v4.onnx` | `688bbf4d…` | V3-clean train minus head_* (CC BY 4.0 sources, see `training/SOURCES.md` lineage), YOLO11m, 80 epochs, seed 17 |
| `ppe-v4cloud.onnx` | `f02ceb50…` | Above + reviewed hard-hat sample + hb1204/coco-full (all CC BY 4.0), YOLO11m, 80 epochs, seed 17 |

Training-data licenses were verified per source (see `training/V4_TRAIN_QA.md`);
`hf-vision/hardhat` (no license tag) was excluded. Ultralytics training code is
AGPL-3.0 — closed-source commercial use of derived weights needs a separate
license decision; this file does not resolve it.

## Development validation asset

A public Ultralytics `bus.jpg` sample (https://ultralytics.com/images/bus.jpg) was used for real-inference smoke testing. It and derived test videos are not committed or included in the product. Application tests generate their own synthetic frames.

## Interface typeface

Manrope by the Manrope Project authors (Mikhail Sharanda, Mirko Velimirovic),
SIL Open Font License 1.1: https://github.com/google/fonts/tree/main/ofl/manrope.
Unmodified variable WOFF2 subsets (Latin, Latin Extended, Cyrillic and Cyrillic
Extended) are self-hosted in `frontend/src/assets/fonts/`. The license is kept
beside the files and included in the production bundle as `MANROPE_LICENSE.txt`.
The interface makes no runtime requests to a font CDN.

## Licensed real-video acceptance material

Intel IoT DevKit / sample-videos contributors, `worker-zone-detection.mp4`,
CC BY 4.0. Exact revision, source URL and checksums: `docs/acceptance/source.json`;
license: `docs/acceptance/INTEL_SOURCE_LICENSE.txt`. Acceptance screenshots and
clips add demo boundaries, actual detector/radar overlays and timestamps.
These derivatives do not imply endorsement by Intel. Original runtime footage
is downloaded on demand and is not committed.
