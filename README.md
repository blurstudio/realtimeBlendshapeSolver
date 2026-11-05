# MayaTfLiteArkitSolver

A Maya node, `tfliteArkitSolver`, that turns a MediaPipe face mesh (for example
an Alembic or geometry cache) into the 52 ARKit-style blendshape weights. It
runs MediaPipe's `face_blendshapes.tflite` model through the TensorFlow Lite C
API.

## Getting the dependencies

You need two files that are not in this repo:

| File | What it is | Where it comes from |
| --- | --- | --- |
| `face_blendshapes.tflite` | The blendshape model, about 1 MB | Inside MediaPipe's `face_landmarker.task` bundle |
| `libLiteRt.dll` | The TensorFlow Lite runtime (LiteRT), which exports the TFLite C API | Inside Google's `ai-edge-litert` Python wheel |

To fetch both, run:

```
python -I scripts/fetch_dependencies.py
```

This puts them in `thirdparty/`, along with `canonical_face_model.obj`, which
the test uses. Nothing it downloads is installed or executed. You can also get
the files by hand.

### Getting the model by hand

1. Download
   <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>.
2. A `.task` file is a zip. Open it with any zip tool (or rename it to `.zip`)
   and extract `face_blendshapes.tflite`. The bundle also holds
   `face_detector.tflite` and `face_landmarks_detector.tflite`, which this node
   doesn't use.

### Getting the TFLite library by hand

Google doesn't publish a standalone Windows download of the TFLite C library,
which is why it's hard to find. Its Python wheel does contain one:

```
pip download ai-edge-litert==2.3.0 --no-deps --only-binary=:all: -d litert_wheel
```

A `.whl` file is also a zip. Extract `ai_edge_litert/libLiteRt.dll` from it.
The DLL is self-contained: it needs only the MSVC runtime, which Maya already
ships.

Any library that exports the standard TFLite C API also works. For example,
you can build `tensorflowlite_c.dll` from `tensorflow/lite/c` with CMake
(<https://ai.google.dev/edge/litert/build/cmake>). The plugin loads the library
at runtime, so building the plugin needs no TFLite headers or `.lib` file.

## Building

Run `quick_compile.bat`, or:

```
meson setup build -Dmaya:maya_version=2026 --vsenv
meson compile -C build
meson install -C build
meson test -C build   # standalone model test, no Maya needed
```

`meson install` copies the plugin into `output_Maya<version>/`. If the model
and library are present in `thirdparty/`, it copies them next to the plugin
too.

## Runtime lookup

- **Library:** the plugin tries these in order:
  1. The path in the `MAYA_TFLITE_LIBRARY` environment variable.
  2. `libLiteRt.dll` or `tensorflowlite_c.dll` in the plugin's folder.
  3. The system library search path.

  If none load, the plugin fails to load and lists everything it tried.
- **Model:** the `modelPath` attribute. If it's empty, the node uses
  `face_blendshapes.tflite` in the plugin's folder.

## Node: `tfliteArkitSolver`

### Inputs

| Plug | Type | Notes |
| --- | --- | --- |
| `inMesh` | mesh | Must have exactly 478 vertices in MediaPipe order: 468 face mesh vertices plus 10 iris vertices. The iris points are part of the model's input. |
| `modelPath` | string (file) | Path to `face_blendshapes.tflite`. Empty means next to the plugin. Set it again to retry after a failed load. |
| `inputSpace` | enum | `yUp` (default): Maya / `canonical_face_model.obj` orientation. `mediaPipeImage`: raw MediaPipe landmarks (+Y down). |
| `inputMatrix` | matrix | Applied to the points before they're projected onto XY. Connect a camera's `worldInverseMatrix` (together with the mesh's `worldMesh`) to solve from that camera's view. |
| `landmarkScale` | double2 | Scales X and Y. Only the aspect ratio matters. For raw normalized MediaPipe landmarks, set it to (image width, image height). |
| `swapLeftRight` | bool | Swaps every `*Left`/`*Right` output pair, for mirrored "selfie" footage. |

### Outputs

- `valid` (bool): false when the mesh is wrong, nothing is connected, or the
  model failed to load or run.
- 52 double plugs named exactly like the ARKit / MediaPipe shapes:
  `neutral`, `browDownLeft`, `browDownRight`, `browInnerUp`, ...,
  `noseSneerRight`. Values are in [0, 1].
- When `valid` is false, every output is 0.

`Left` and `Right` refer to the subject's sides, not the viewer's.

### How the solve works

The model sees only the X and Y of 146 selected landmarks, as they appear in
image space.

- **Position and scale:** the model normalizes these itself, so the mesh's
  position and overall size don't affect the result.
- **Aspect ratio and head rotation:** both do affect the result. A face turned
  away from the projection axis looks different to the model, as it would in
  footage.
- **Head pose and depth:** these aren't outputs. The model doesn't use depth,
  and MediaPipe computes head pose separately.

## Licenses

The model and LiteRT are Apache-2.0 (Google). The fetch script copies the
LiteRT license into `thirdparty/litert/`.
