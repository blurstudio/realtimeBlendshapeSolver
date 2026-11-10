"""Download the runtime dependencies into ./thirdparty.

    python -I scripts/fetch_dependencies.py

Fetches:
  thirdparty/model/face_blendshapes.tflite   extracted from MediaPipe's face_landmarker.task
  thirdparty/model/canonical_face_model.obj  MediaPipe canonical face (468 verts), for the test

Nothing downloaded is executed.
"""

import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
import ssl
import certifi


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRDPARTY = os.path.join(ROOT, "thirdparty")

TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
OBJ_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)


def download(url, dest):
    print("downloading", url)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def fetch_model(force):
    out_dir = os.path.join(THIRDPARTY, "model")
    os.makedirs(out_dir, exist_ok=True)
    model = os.path.join(out_dir, "face_blendshapes.tflite")
    if force or not os.path.isfile(model):
        with tempfile.TemporaryDirectory() as tmp:
            task = os.path.join(tmp, "face_landmarker.task")
            download(TASK_URL, task)
            # A .task file is a zip of the three face landmarker models
            with zipfile.ZipFile(task) as z, open(model, "wb") as f:
                f.write(z.read("face_blendshapes.tflite"))
    print("model:", model)

    obj = os.path.join(out_dir, "canonical_face_model.obj")
    if force or not os.path.isfile(obj):
        download(OBJ_URL, obj)
    print("obj:  ", obj)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()
    fetch_model(args.force)


if __name__ == "__main__":
    main()
