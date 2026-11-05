"""Download the runtime dependencies into ./thirdparty.

    python -I scripts/fetch_dependencies.py

Fetches:
  thirdparty/model/face_blendshapes.tflite   extracted from MediaPipe's face_landmarker.task
  thirdparty/model/canonical_face_model.obj  MediaPipe canonical face (468 verts), for the test
  thirdparty/litert/libLiteRt.dll            TFLite C API, extracted from Google's
                                             ai-edge-litert wheel (.so / .dylib elsewhere)

Only the Python standard library and pip are used. Nothing downloaded is executed.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

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
LITERT_VERSION = "2.3.0"


def download(url, dest):
    print("downloading", url)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
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


def fetch_litert(version, force):
    out_dir = os.path.join(THIRDPARTY, "litert")
    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(os.path.join(out_dir, "libLiteRt.*"))
    if existing and not force:
        print("lib:  ", existing[0])
        return

    with tempfile.TemporaryDirectory() as tmp:
        # Download only; the wheel is never installed or imported.
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", "ai-edge-litert==" + version,
            "--no-deps", "--only-binary=:all:", "-d", tmp,
        ])
        wheels = glob.glob(os.path.join(tmp, "*.whl"))
        if not wheels:
            sys.exit("pip did not download an ai-edge-litert wheel for this platform")
        with zipfile.ZipFile(wheels[0]) as z:
            names = z.namelist()
            libs = [
                n for n in names
                if os.path.basename(n).split(".")[0] == "libLiteRt"
                and n.rsplit(".", 1)[-1] in ("dll", "so", "dylib")
            ]
            if not libs:
                shared = [n for n in names if n.endswith((".dll", ".so", ".dylib"))]
                sys.exit("libLiteRt not found in %s. Shared libraries in the wheel:\n  %s"
                         % (os.path.basename(wheels[0]), "\n  ".join(shared)))
            for n in libs + [n for n in names if "/licenses/" in n or n.endswith("LICENSE")]:
                dest = os.path.join(out_dir, os.path.basename(n))
                with open(dest, "wb") as f:
                    f.write(z.read(n))
                print("lib:  ", dest)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--litert-version", default=LITERT_VERSION)
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()
    fetch_model(args.force)
    fetch_litert(args.litert_version, args.force)


if __name__ == "__main__":
    main()
