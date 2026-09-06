# Thor3D — controlled single-object edits in AI2-THOR / RoboTHOR

Render a scene, change exactly one object (size, rotation, position, or remove it),
and re-render from the identical camera so the two images differ **only** where
that object is.

The point is the same as `../Blender3D`: never composite in 2D. The object's
shadow, its occlusion of what is behind it, its perspective foreshortening, and
the light it bounces onto nearby surfaces are all recomputed by one renderer, so
the edited object genuinely belongs in the scene. The difference here is that
the scenes are AI2-THOR's photo-realistic indoor rooms with ~80 annotated objects
each, rather than assets you assemble yourself.

|  | |
|---|---|
| **Input** | a scene name (`FloorPlan1`, `FloorPlan_Train1_1`, …) and a camera pose |
| **Edit** | one object: uniform scale, rotation, translation, or removal |
| **Output** | RGB, depth, normals, two kinds of edge map, instance masks, a change mask, and a QA report proving the background held still |

---

## 1. Setup

Five steps, in order, starting from a fresh checkout of `Thor3D/`. Steps 1, 2, 4
and 5 are always needed. Step 3 applies to any container image that ships CUDA
without a graphics stack — common on GPU clusters, and the cause of nearly every
"no Vulkan device" failure below.

### Step 1 — Python environment

```bash
cd Thor3D
conda create -y -n thor3d python=3.10
conda activate thor3d
pip install -r requirements.txt

# vulkaninfo is a binary, not a pip package; xorg-libxext is needed in step 3
conda install -y -c conda-forge vulkan-tools xorg-libxext
```

`requirements.txt` asks for `opencv-python-headless`. If plain `opencv-python`
ends up installed alongside it, the non-headless one wins and `import cv2` fails
on a missing `libGL.so.1` — *after* Unity has already rendered a frame, which
makes it look like a rendering bug. `pip uninstall -y opencv-python` if both are
present.

### Step 2 — Unity build

The first run downloads AI2-THOR's Unity build (~800 MB zipped, 1.1 GB unpacked)
into `~/.ai2thor`. Fetch it up front so the first render can't time out:

```bash
python -c "
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
Controller(platform=CloudRendering, download_only=True)"
```

It lands in `~/.ai2thor/releases/thor-CloudRendering-<commit>/`, where `<commit>`
is the build pinned by the installed `ai2thor` (5.0.0 →
`f0825767cd50d69f666c7f282e54abfe58f1e917`). The path is hardcoded, so to keep
the build off your home partition, symlink `~/.ai2thor` elsewhere *before*
downloading.

Rendering uses AI2-THOR's `CloudRendering` platform — headless Vulkan, no X
server required.

### Step 3 — Graphics libraries, if the container lacks them

Run `vulkaninfo --summary` first. If it lists your NVIDIA GPU under `Devices:`,
skip to step 4. If it lists only `llvmpipe` (Mesa's CPU rasterizer) or nothing,
read on.

A GPU container image often ships CUDA and nothing else. The NVIDIA driver
libraries get mounted in by the container runtime, but the userspace they depend
on is absent, and the failure surfaces far from its cause:

```
RuntimeError: Could not find a Vulkan device corresponding to the CUDA device
with UUID <uuid>.
```

That is AI2-THOR reporting that `vulkaninfo` showed it no NVIDIA device. CUDA
works throughout — `nvidia-smi` and `cuInit` are fine — because only the
*graphics* path is broken. Two libraries are usually missing, and neither needs
root to supply:

- **`libXext.so.6`**, a hard `DT_NEEDED` of `libGLX_nvidia.so.0`. Without it the
  loader cannot open the ICD at all and logs `Failed to CreateInstance in ICD`.
- **libglvnd** (`libGL.so.1`, `libEGL.so.1`, `libGLdispatch.so.0`,
  `libGLX.so.0`, `libOpenGL.so.0`). `libGLX_nvidia.so.0` is a GLVND *vendor*
  library and refuses to initialize without the dispatch layer even when it is
  being used purely as a Vulkan ICD — `vk_icdNegotiateLoaderICDInterfaceVersion`
  returns `-3` (`VK_ERROR_INITIALIZATION_FAILED`) and every entry point comes
  back NULL. This one is easy to misdiagnose: no file access fails, and the
  driver never touches `/dev/nvidia*`, so `strace` shows nothing obviously wrong.

Stage both into one directory and put it on `LD_LIBRARY_PATH`:

```bash
conda create -y -p /tmp/glvnd -c conda-forge \
    libglvnd-cos7-x86_64 libglvnd-glx-cos7-x86_64 \
    libglvnd-egl-cos7-x86_64 libglvnd-opengl-cos7-x86_64
mkdir -p ~/.local/vulkanfix/lib
cp -P /tmp/glvnd/x86_64-conda-linux-gnu/sysroot/usr/lib64/lib{GL,EGL,GLX,GLdispatch,OpenGL}.so* \
      ~/.local/vulkanfix/lib/
cp -P $CONDA_PREFIX/lib/libXext.so.6* ~/.local/vulkanfix/lib/   # conda install -c conda-forge xorg-libxext

export LD_LIBRARY_PATH=~/.local/vulkanfix/lib
vulkaninfo --summary        # should now list your NVIDIA device
```

Make it automatic so every shell inherits it, rather than exporting by hand:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/{activate,deactivate}.d

cat > $CONDA_PREFIX/etc/conda/activate.d/thor3d_vulkan.sh <<'EOF'
export _THOR3D_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="$HOME/.local/vulkanfix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/thor3d_vulkan.sh <<'EOF'
if [ -n "${_THOR3D_OLD_LD_LIBRARY_PATH+x}" ]; then
    if [ -z "$_THOR3D_OLD_LD_LIBRARY_PATH" ]; then unset LD_LIBRARY_PATH
    else export LD_LIBRARY_PATH="$_THOR3D_OLD_LD_LIBRARY_PATH"; fi
    unset _THOR3D_OLD_LD_LIBRARY_PATH
fi
EOF
```

`activate.d` runs only at activation, so re-activate before testing:
`conda deactivate && conda activate thor3d`.

### Step 4 — GPU index

Find which CUDA index has a working Vulkan device, because it is often not 0:

```bash
nvidia-smi -L                                             # CUDA index -> GPU-<uuid>
vulkaninfo --summary | grep -E "^GPU[0-9]|deviceUUID"     # Vulkan index -> deviceUUID
```

Match the UUIDs. The CUDA index whose UUID appears in the `vulkaninfo` output is
the one to pass as `--gpu`. A GPU listed by `nvidia-smi` but absent from
`vulkaninfo` cannot render — a faulty card, or one the container exposes for
compute only.

If *every* GPU matches, AI2-THOR builds the mapping itself and there is nothing
to do. If any GPU is unmatched it raises regardless of which index you asked for,
because it insists on mapping all of them:

```
RuntimeError: Could not find a Vulkan device corresponding to the CUDA device
with UUID <uuid>.
```

Write the map by hand to bypass that. It is the only thing AI2-THOR's
`-force-device-index` flag is derived from:

```bash
echo '{"1": 0}' > ~/.ai2thor/cuda-vulkan-mapping.json   # CUDA 1 -> Vulkan 0
```

Then pass `--gpu 1` — the CUDA index, the key on the left — to every script and
to the web app. An index missing from the map fails immediately with
`KeyError: <index>`, which is the good case; the bad case is a map pointing at a
CPU device, which renders silently and slowly (see the CPU section below).

This file is a cache AI2-THOR never revalidates, so rewrite it whenever the set
of visible GPUs changes.

### Step 5 — Verify

```bash
conda deactivate && conda activate thor3d
vulkaninfo --summary | grep -E "^GPU[0-9]|deviceName"   # expect your NVIDIA device
python scripts/capture_scene.py --scene FloorPlan1 --out /tmp/thor3d-check \
    --list-objects --gpu 1
```

The last command prints the editable objects in `FloorPlan1` and exits. On a GPU
it takes a few seconds; if it instead hangs for 100 s and dies with
`TimeoutError`, rendering fell back to the CPU — recheck steps 3 and 4.

### Fallback: CPU-only rendering

As a last resort, with no usable Vulkan GPU at all, everything runs on Mesa's
`llvmpipe` software rasterizer — correct output, `far_background_unchanged`
included, about 15× slower.

Unity refuses a `deviceType=CPU` adapter on its own, logging
`Selected physical device (nil)` before segfaulting, but accepts it when named
explicitly. Point the map at llvmpipe's index and cap quality at `High`:

```bash
echo '{"0": 0}' > ~/.ai2thor/cuda-vulkan-mapping.json
python scripts/capture_scene.py --scene FloorPlan1 --look-at Fridge \
    --out out/fridge --gpu 0 --quality High
```

`Very High` and `Ultra` (the scripts' default) hang indefinitely in `Initialize`
under llvmpipe once the depth, normals, and instance-segmentation passes are on;
a smaller image does not rescue them, as a 300×300 `Ultra` capture hangs exactly
as a 512×512 one does.

| quality | startup | reset | frame |
|---|---|---|---|
| `Low` | 41 s | 4.1 s | 0.12 s |
| `Medium` | 47 s | 8.7 s | 0.21 s |
| `High` | 54 s | 11.0 s | 0.31 s |
| `Very High`, `Ultra` | — | — | hangs |

At 512×512 / `High` a capture takes 1 m 46 s and an edit pair 1 m 19 s, against
7.6 s and 5.2 s for the same runs on a GPU at `Ultra`. Nearly all of it is fixed
overhead — ~54 s of Unity startup, ~11 s per `reset(scene)` — while each extra
frame costs only ~0.31 s, so budget batch sweeps per scene rather than per image.

---

## 2. Quick start

Every command below passes `--gpu 1`, the index from step 4. Substitute your own;
`--gpu -1` lets Unity choose, which is only correct when the auto-mapping works.

```bash
conda activate thor3d
cd Thor3D

# 1. Capture a scene, auto-framing the object you intend to edit
python scripts/capture_scene.py --scene FloorPlan1 --look-at Fridge \
    --out out/fridge --gpu 1

# 2. Scale that object and re-render from the same camera
python scripts/edit_and_render.py --scene-spec out/fridge/scene.json \
    --object-type Fridge --scale 1.3 --out out/fridge/scale13 --gpu 1

# 3. Or rotate it instead
python scripts/edit_and_render.py --scene-spec out/fridge/scene.json \
    --object-type Fridge --rotate-y 35 --out out/fridge/rot35 --gpu 1
```

Don't know what's in a scene?

```bash
python scripts/capture_scene.py --scene FloorPlan1 --out /tmp/x --list-objects --gpu 1
```

### Batch

```bash
python scripts/make_pairs.py --scene-set ithor-kitchen --num-scenes 20 \
    --objects-per-scene 3 --scales 0.7 1.4 --rotations 45 90 --out data/pairs
```

Writes `data/pairs/<scene>/<edit-slug>/` per pair plus a top-level `index.json`.
Scene sets: `ithor`, `ithor-kitchen`, `ithor-living`, `robothor`,
`robothor-train`, `robothor-val`, or a comma-separated list of scene names.

### Interactive web app

```bash
conda activate thor3d
cd Thor3D
python webapp/server.py --gpu 1                                  # http://127.0.0.1:8000
python webapp/server.py --gpu 1 --port 8080 --width 640 --height 640
```

`--gpu` defaults to `0` here as in the scripts, so it must be passed explicitly
whenever 0 is not the right index — otherwise startup dies with `KeyError: 0`
against a hand-written map. Wait for `starting AI2-THOR ...` to be followed by
Flask's serving line before connecting; the first launch loads a scene.

The server binds `127.0.0.1` only, has no authentication, and can write files, so
reach it from your laptop over a tunnel rather than by binding publicly:

```bash
ssh -N -L 8080:127.0.0.1:8080 <user>@<gpu-host>    # then open http://127.0.0.1:8080
```

Pick a scene and an object, then drag sliders for uniform scale, yaw/pitch/roll,
and X/Y/Z offset. The original and edited renders sit side by side, with a
"difference" toggle that overlays the change. **Save pair** writes a full
`original/ edited/ diff.png changed_mask.png pair_report.json` set through the
same offline path `make_pairs.py` uses, so UI output and batch output are
identical in format.

Roughly 55 ms per update (~18 fps), fast enough to drag a slider and watch the
render follow. That works because the scene stays loaded and edits are applied
incrementally to the live Unity process rather than reloading the scene each
time (~265 ms). `ScaleObject` multiplies the current scale, so an absolute
slider value is reached by sending the ratio against the scale already applied —
verified drift-free to 2e-7 over 41 steps. Sliders always send absolute state,
so a dropped intermediate event cannot accumulate error.

Needs `flask`, which `requirements.txt` already installs.

### Python API

```python
from thor3d import ThorRenderer, ObjectEdit, save_pair

with ThorRenderer(width=512, height=512) as r:
    spec = r.capture_spec("FloorPlan1")             # snapshot the scene
    spec.save("scene.json")                          # replayable later

    edit = ObjectEdit(object_type="Fridge", scale=1.3)
    original, edited = r.render_pair(spec, edit)

    target = edited.metadata["edit"]["original"]["objectId"]
    report = save_pair(original, edited, "out/", target)
    assert report["far_background_unchanged"]
```

---

## 3. How the background is held fixed

Both images in a pair go through the *same* setup sequence, so nothing can
diverge between them:

1. `reset(scene)` — reloads the scene deterministically, restoring every
   object's default pose.
2. Optional `InitialRandomSpawn(seed)` — replayed with the recorded seed if the
   capture used a randomized layout.
3. `SetObjectPoses(placeStationary=True)` — replays the exact recorded pose of
   every movable object and pins them kinematic. A safety net on top of step 1.
4. `Teleport(position, rotation, horizon, standing)` — restores the camera
   bit-for-bit. `snapToGrid=False` at construction, otherwise THOR quantises the
   agent's position and the pose is not reproducible.
5. *(edited render only)* apply the one edit.

The edited object is then pinned with `TeleportObject(..., forceKinematic=True)`
so it cannot fall, settle, or roll after being changed — the edit stays a clean
controlled variable. **Physics is deliberately frozen**; a scaled-up object may
interpenetrate the surface it rests on rather than resting naturally on it.

### Auxiliary maps

Each render writes these next to `rgb.png`, for both the original and the edited half:

| file | what it is |
|---|---|
| `depth.npy` | float32, **metres**, straight from THOR — the authoritative copy |
| `depth.png` | same data as uint16 millimetres, for viewing and image-only tools (round-trips to within 1 mm) |
| `normals.png` | surface normals from THOR's normals pass |
| `edges_canny.png` | Canny edges on the RGB — the usual conditioning signal |
| `edges_geometric.png` | edges from geometry alone: depth discontinuities ∪ normal creases |
| `target_mask.png` | the edited object's instance mask |

Disable any of them with `--no-depth`, `--no-normals`, `--no-edges`.

**Two edge maps, because they answer different questions.** `edges_canny` follows
texture and lighting, so a patterned wall lights up even though it is flat.
`edges_geometric` sees only shape: depth jumps catch occlusion boundaries, and
normal creases catch corners where two surfaces meet without any depth jump.

**One of them is exactly reproducible and the other is not.** On the untouched
background of a pair, `edges_geometric` differs by **0 pixels**, while
`edges_canny` differs by ~80 out of ~120k (0.07%). The reason is that Canny
thresholds a render that is not bit-deterministic: pixels sitting exactly on the
threshold flip when anti-aliasing noise moves them by a level. Two things reduce
it — both halves of a pair use one shared threshold pair derived from the
original (recorded as `canny_thresholds` in `pair_report.json`), and a 3×3
Gaussian pre-blur roughly halves the flicker — but it cannot reach zero. If your
training signal needs edges that are bit-identical wherever the image is
unchanged, use `edges_geometric`.

### Verifying it worked

Every pair gets a `pair_report.json` splitting the image into three zones:

| zone | meaning |
|---|---|
| **changed** | union of the object's mask before and after, dilated 3 px |
| **near** | a 60 px halo around it — the object's shadow and bounced light |
| **far** | everything else — **must** be unchanged |

`far_background_unchanged: true` is the assertion that matters. Nonzero drift in
the *near* zone is not a bug: it is the object's shadow correctly following the
edit, and it's exactly the effect 2D compositing cannot produce.

Note that THOR's renderer is not bit-deterministic — two renders of an identical
scene differ by ~2 levels of 8-bit anti-aliasing noise — so "unchanged" means
"within `--tolerance` (default 4)", not an exact match.

---

## 4. Limits worth knowing

These are properties of AI2-THOR 5.0.0, verified against the actual build rather
than the docs.

**Scaling is uniform only.** `ScaleObject` takes a single `scale` float. There is
no non-uniform / per-axis stretch in the API, so "make the chair taller but not
wider" is not expressible. (The second `ScaleObject` overload's `x, y` are screen
coordinates for picking a target, not axis scales.)

**Edits need `forceAction=True`.** Without it `ScaleObject` fails with
`NullReferenceException: Target object not found within the scene` for any
object the agent could not physically reach. The renderer always passes it.

**RoboTHOR scenes use the default agent, not the LoCoBot.** The LoCoBot's
controller is a restricted subclass that rejects `ScaleObject`,
`TeleportObject`, *and* `SetObjectPoses` as invalid actions — no object editing
is possible under it at all. Thor3D therefore loads RoboTHOR scenes with
`agentMode="default"` (identical scene geometry) while keeping RoboTHOR's 60°
field of view. The camera sits at a different height than the published RoboTHOR
frames as a result. `--agent-mode locobot` is accepted for capture-only work and
raises a clear error if you then try to edit.

**`PausePhysicsAutoSim` is not used.** In 5.0.0 it leaves `ScaleObject` with
stale collider bounds: a 1.3× scale reports a 4.2× x-extent while y scales
correctly. Objects are frozen by making them kinematic instead, which is
equivalent and correct. The renderer logs a warning if a measured bounding-box
ratio disagrees with the requested scale by more than 5%.

**Some objects crash the Unity build when scaled.** `ScaleObject` on a
`Painting` kills the process outright; the same happens for scattered individual
assets (one `Drawer` in `FloorPlan_Train2_1` crashes while `Drawer` in
`FloorPlan1` is fine), so it is not predictable from the object type alone.
Two mitigations: `pick_editable_objects` skips the known-bad types by default,
and `make_pairs.py` detects a dead controller, calls `ThorRenderer.restart()`,
and carries on with the rest of the sweep instead of losing the run.

**Light sources change the whole image.** Scaling a `DeskLamp` or `FloorLamp`
alters the room's illumination, so `far_background_unchanged` will be `false`
and legitimately so. Filter these out with `--object-types` if you need pairs
whose difference is strictly local.

**Object identity across renders.** `--object-type` picks the instance covering
the most pixels in the current frame; use `--object-id` (from `--list-objects`)
when a scene has several of the same type and you need a specific one.

---

## 5. Layout

```
thor3d/
  spec.py       CameraSpec / ObjectEdit / SceneSpec — the serializable descriptions
  renderer.py   ThorRenderer — controller lifecycle, camera lock, edit application
  viewpoint.py  find_camera_for_object — searches navigable poses for a good framing
  io_utils.py   saving renders, change masks, and the background QA report
scripts/
  capture_scene.py    render an original + write scene.json
  edit_and_render.py  apply one edit, re-render, write the pair
  make_pairs.py       batch sweeps over scenes × objects × factors
webapp/
  server.py           Flask app; one Unity process, incremental live edits
  static/index.html   the UI (vanilla JS, no build step)
```

The viewpoint search aims at each object's **bounding-box centre**, not its
transform pivot (a fridge's pivot sits on the floor, so aiming at it tilts the
camera down past the object), and scores candidates by how close the object is
to covering a target fraction of the frame — maximizing pixel area instead just
walks the camera into the object's face until no background is left.
