# MayaTfLiteArkitSolver

A Maya node, `tfliteArkitSolver`, that turns a MediaPipe face mesh (for example
an Alembic or geometry cache) into the 52 ARKit-style blendshape weights. It
runs MediaPipe's `face_blendshapes.tflite` model.

TensorFlow Lite is not used. The model is a small MLP-Mixer, and
[src/blendshapeModel.cpp](src/blendshapeModel.cpp) runs it directly with
[Eigen](https://eigen.tuxfamily.org). It reads the weights from the `.tflite`
file and checks the graph's shapes, so only this model loads. The test checks
its scores against TensorFlow Lite's (they match to within about 1e-6). One
inference takes about 4 ms on one core.

## Getting the dependencies

- **The model:** `face_blendshapes.tflite`, about 1 MB, from MediaPipe's
  `face_landmarker.task` bundle. Run `python -I scripts/fetch_dependencies.py`,
  which puts it in `thirdparty/model/` along with `canonical_face_model.obj`,
  which the test uses. Nothing it downloads is installed or executed.
- **Eigen:** header-only. Meson uses a system Eigen if it finds one. Otherwise
  it downloads it through [subprojects/eigen.wrap](subprojects/eigen.wrap) the
  first time you run `meson setup`.

### Getting the model by hand

1. Download
   <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>.
2. A `.task` file is a zip. Open it with any zip tool (or rename it to `.zip`)
   and extract `face_blendshapes.tflite`. The bundle also holds
   `face_detector.tflite` and `face_landmarks_detector.tflite`, which this node
   doesn't use.

## Building

Run `quick_compile.bat`, or:

```
meson setup build -Dmaya:maya_version=2026 --vsenv
meson compile -C build
meson install -C build
meson test -C build   # standalone model test, no Maya needed
```

`meson install` copies the plugin into `output_Maya<version>/`. It also copies
the model, if it's present in `thirdparty/`, next to the plugin.

## Runtime lookup

The model comes from the `modelPath` attribute. If it's empty, the node uses
`face_blendshapes.tflite` in the plugin's folder.

## Node: `tfliteArkitSolver`

### Inputs

| Plug | Type | Notes |
| --- | --- | --- |
| `inMesh` | mesh | MediaPipe vertex order, with either 478 vertices (468 face mesh vertices plus 10 iris vertices) or 468 (no iris tracking). See [Meshes without irises](#meshes-without-irises). |
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

### Meshes without irises

The model's input includes the 10 iris landmarks. For a 468-vertex mesh, the
node synthesizes them from each eye's corners, looking straight ahead. This
means:

- **Gaze is not tracked.** `eyeLook*` stay near their resting values, at
  most about 0.2 on the canonical face.
- **Blink reads higher at rest.** The model partly judges `eyeBlink*` from
  where the lids sit relative to the iris. On the canonical face it reads
  about 0.4 with the eyes open, and closing one eye still clearly drives its
  own `eyeBlink*`.

## Verification mask

[scripts/arkit_verification_mask.py](scripts/arkit_verification_mask.py)
builds a mask that shows what the solved weights mean. It fits per-actor
blendshapes from a calibration cache, then builds the canonical face (or any
mesh in MediaPipe vertex order) with those shapes. The shapes go in a
blendShape node that the solver drives.

```python
import arkit_verification_mask as avm
avm.build_verification_mask('actorRest', 'calibrationCache')       # fit + build
avm.key_head_follow('arkitMask_head', 'actorRest', 'otherShotCache')  # reuse on a shot
```

The fit removes head pose with a Procrustes fit on stable landmarks. It then
solves `delta ≈ offset + Σ weight_k · shape_k` over all calibration frames by
ridge least squares. A side penalty keeps `*Left` shapes off the right side of
the face and the reverse, so pairs that usually fire together still come
apart on a wink. The calibration should exercise every shape. The script lists
the shapes it barely saw.

Test it, without the solver plugin, using `mayapy tests/test_verification_mask.py`.

## Licenses

The model is Apache-2.0 (Google). Eigen is MPL-2.0. It's used unmodified, as
headers.
