#!/usr/bin/env python
"""Interactive web UI for editing one object and re-rendering the scene live.

Run it, open the printed URL, pick a scene and an object, then drag sliders to
change its size, rotation, and position. The camera and the rest of the scene
stay locked, so the preview differs from the original only where the object is.

    python webapp/server.py                 # http://127.0.0.1:8000
    python webapp/server.py --port 8080 --width 640 --height 640

Remote GPU box? Forward the port rather than exposing the server:

    ssh -N -L 8000:127.0.0.1:8000 user@host

Why this is fast enough to feel live: the scene is loaded once and edits are
applied incrementally to the live Unity process (~35 ms/frame) instead of
reloading the scene per change (~265 ms). ``ScaleObject`` is multiplicative and
drift-free, so an absolute scale is reached by applying the ratio against the
scale currently in effect; rotation and position are absolute teleports.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thor3d import ObjectEdit, SceneSpec, ThorRenderer, save_pair
from thor3d.renderer import INSTANT
from thor3d.spec import default_agent_mode, default_fov
from thor3d.viewpoint import find_camera_candidates, pick_editable_objects

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=None)


def png_data_uri(rgb: np.ndarray) -> str:
    """Encode a frame for the browser.

    PNG rather than JPEG because the UI's difference view compares these pixels
    directly and lossy artefacts would swamp the real change. compress_level=1
    roughly halves encode time for a modest size increase, which is the right
    trade over localhost.
    """
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Session:
    """The live scene, the object under edit, and the edits currently applied.

    A single Unity process backs the whole app, so every mutation holds
    ``lock``. Requests are cheap (tens of ms) and the UI keeps one in flight, so
    serialising them is simpler than pooling processes and fast enough.
    """

    def __init__(self, width: int, height: int, gpu: Optional[int], quality: str):
        self.lock = threading.Lock()
        self.width, self.height = width, height
        self.renderer = ThorRenderer(
            width=width, height=height, gpu_device=gpu, quality=quality,
            # The live preview only needs RGB, but Save writes depth/normals/edges,
            # so these passes must be on -- they are requested at Initialize time
            # and cannot be switched on just for the save.
            render_depth=True,
            render_instance_segmentation=True,
            render_normals=True,
        )
        self.scene: Optional[str] = None
        self.spec: Optional[SceneSpec] = None
        self.cameras: List[Any] = []
        self.camera_index = 0
        self.object_id: Optional[str] = None
        self.original_pose: Optional[Dict[str, Any]] = None
        self.original_size: Optional[Dict[str, float]] = None
        self.applied_scale = 1.0
        self.original_rgb: Optional[np.ndarray] = None

    # -------------------------------------------------------------- scene load

    def load_scene(self, scene: str) -> Dict[str, Any]:
        agent_mode = default_agent_mode(scene)
        self.renderer.controller.reset(
            scene=scene,
            agentMode=agent_mode,
            fieldOfView=default_fov(scene),
            renderInstanceSegmentation=True,
            snapToGrid=False,
        )
        self.scene = scene
        self.spec = None
        self.object_id = None
        objects = pick_editable_objects(self.renderer.controller.last_event.metadata["objects"])
        return {
            "scene": scene,
            "objects": sorted(
                (
                    {
                        "objectId": o["objectId"],
                        "objectType": o["objectType"],
                        "pickupable": bool(o.get("pickupable")),
                    }
                    for o in objects
                ),
                key=lambda d: (d["objectType"], d["objectId"]),
            ),
        }

    def select_object(self, object_id: str) -> Dict[str, Any]:
        """Frame the camera on the object and render the untouched baseline."""
        assert self.scene
        self.cameras = find_camera_candidates(self.renderer, self.scene, object_id)
        if not self.cameras:
            raise ValueError(
                f"{object_id} is not visible from any reachable position in {self.scene}"
            )
        self.camera_index = 0
        self.object_id = object_id
        return self._rebase()

    def cycle_camera(self, step: int) -> Dict[str, Any]:
        if not self.cameras:
            raise ValueError("no object selected")
        self.camera_index = (self.camera_index + step) % len(self.cameras)
        return self._rebase()

    def _rebase(self) -> Dict[str, Any]:
        """Reload the scene at the current camera and cache the unedited frame.

        The baseline must be re-rendered whenever the camera moves, since the
        'original' half of the comparison has to share the edited half's view.
        """
        assert self.scene and self.object_id
        camera = self.cameras[self.camera_index]
        self.spec = self.renderer.capture_spec(self.scene, camera=camera)
        self.renderer.reset_scene(self.spec)
        self.applied_scale = 1.0

        obj = self._live_object()
        self.original_pose = {
            "position": dict(obj["position"]),
            "rotation": dict(obj["rotation"]),
        }
        self.original_size = dict(obj["axisAlignedBoundingBox"]["size"])

        event = self.renderer.controller.step("Pass")
        self.original_rgb = np.asarray(event.frame)
        return {
            "image": png_data_uri(self.original_rgb),
            "objectId": self.object_id,
            "objectType": obj["objectType"],
            "original": {"pose": self.original_pose, "size": self.original_size},
            "camera": {
                "index": self.camera_index,
                "count": len(self.cameras),
                "position": camera.position,
                "yaw": camera.rotation["y"],
                "horizon": camera.horizon,
            },
        }

    def _live_object(self) -> Dict[str, Any]:
        for o in self.renderer.controller.last_event.metadata["objects"]:
            if o["objectId"] == self.object_id:
                return o
        raise KeyError(f"{self.object_id} vanished from the scene")

    # ------------------------------------------------------------------- edits

    def apply(self, scale: float, rot: Dict[str, float], pos: Dict[str, float]) -> Dict[str, Any]:
        """Drive the object to an absolute state described relative to baseline.

        ``scale`` is absolute (1.0 = untouched); ``rot`` and ``pos`` are offsets
        from the object's recorded original pose. Every call fully determines
        the result, so dropped intermediate slider events cannot accumulate
        error.
        """
        assert self.object_id and self.original_pose
        t0 = time.time()
        c = self.renderer.controller

        if abs(scale - self.applied_scale) > 1e-6:
            # ScaleObject multiplies the current scale, so ask for the ratio.
            ratio = scale / self.applied_scale
            ev = c.step(
                action="ScaleObject", objectId=self.object_id, scale=ratio,
                scaleOverSeconds=INSTANT, forceAction=True,
            )
            if not ev.metadata["lastActionSuccess"]:
                raise RuntimeError(ev.metadata.get("errorMessage", "ScaleObject failed"))
            self.applied_scale = scale

        target_pos = {
            k: self.original_pose["position"][k] + float(pos.get(k, 0.0)) for k in "xyz"
        }
        target_rot = {
            k: (self.original_pose["rotation"][k] + float(rot.get(k, 0.0))) % 360.0 for k in "xyz"
        }
        ev = c.step(
            action="TeleportObject", objectId=self.object_id,
            position=target_pos, rotation=target_rot,
            forceAction=True, forceKinematic=True,
        )
        if not ev.metadata["lastActionSuccess"]:
            raise RuntimeError(ev.metadata.get("errorMessage", "TeleportObject failed"))

        event = c.step("Pass")
        rgb = np.asarray(event.frame)
        obj = self._live_object()
        return {
            "image": png_data_uri(rgb),
            "size": dict(obj["axisAlignedBoundingBox"]["size"]),
            "position": dict(obj["position"]),
            "rotation": dict(obj["rotation"]),
            "ms": round((time.time() - t0) * 1000),
        }

    def save(self, out_dir: str, scale: float, rot, pos) -> Dict[str, Any]:
        """Write a full original/edited pair through the offline pipeline.

        Deliberately re-renders from a clean reset rather than reusing the live
        preview, so what lands on disk goes through the exact same path as
        make_pairs.py output.
        """
        assert self.spec and self.object_id
        edit_kwargs: Dict[str, Any] = {"object_id": self.object_id}
        if abs(scale - 1.0) > 1e-6:
            edit_kwargs["scale"] = scale
        if any(abs(v) > 1e-6 for v in rot.values()):
            edit_kwargs["rotation_delta"] = dict(rot)
        if any(abs(v) > 1e-6 for v in pos.values()):
            edit_kwargs["position_delta"] = dict(pos)
        if len(edit_kwargs) == 1:
            raise ValueError("nothing to save: no edit applied yet")

        original, edited = self.renderer.render_pair(self.spec, ObjectEdit(**edit_kwargs))
        report = save_pair(original, edited, out_dir, target_object_id=self.object_id,
                           save_depth=True, save_normals=True, save_edges=True)
        self.spec.save(os.path.join(out_dir, "scene.json"))

        # render_pair left the scene in the edited state; restore the live
        # session so the UI's sliders still describe what is on screen.
        self.renderer.reset_scene(self.spec)
        self.applied_scale = 1.0
        self.apply(scale, rot, pos)
        return {"out_dir": out_dir, "report": report}


SESSION: Optional[Session] = None
OUT_ROOT = "out/webapp"


def _err(e: Exception, code: int = 400):
    return jsonify({"error": f"{type(e).__name__}: {e}"}), code


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/scenes")
def api_scenes():
    c = SESSION.renderer.controller
    return jsonify({"ithor": list(c.ithor_scenes()), "robothor": list(c.robothor_scenes())})


@app.route("/api/scene", methods=["POST"])
def api_scene():
    scene = request.json["scene"]
    try:
        with SESSION.lock:
            return jsonify(SESSION.load_scene(scene))
    except Exception as e:
        return _err(e)


@app.route("/api/object", methods=["POST"])
def api_object():
    try:
        with SESSION.lock:
            return jsonify(SESSION.select_object(request.json["objectId"]))
    except Exception as e:
        return _err(e)


@app.route("/api/camera", methods=["POST"])
def api_camera():
    try:
        with SESSION.lock:
            return jsonify(SESSION.cycle_camera(int(request.json.get("step", 1))))
    except Exception as e:
        return _err(e)


@app.route("/api/edit", methods=["POST"])
def api_edit():
    d = request.json
    try:
        with SESSION.lock:
            return jsonify(
                SESSION.apply(
                    float(d.get("scale", 1.0)),
                    d.get("rotation", {}) or {},
                    d.get("position", {}) or {},
                )
            )
    except Exception as e:
        return _err(e)


@app.route("/api/save", methods=["POST"])
def api_save():
    d = request.json
    # Confine writes to OUT_ROOT: the name is user input and must not be able
    # to escape via .. or an absolute path.
    name = os.path.basename(str(d.get("name") or "").strip()) or time.strftime("pair_%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, name)
    try:
        with SESSION.lock:
            return jsonify(
                SESSION.save(
                    out_dir,
                    float(d.get("scale", 1.0)),
                    d.get("rotation", {}) or {},
                    d.get("position", {}) or {},
                )
            )
    except Exception as e:
        return _err(e)


def main() -> int:
    global SESSION, OUT_ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="default is localhost only; use SSH port forwarding for remote access")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--quality", default="Ultra")
    ap.add_argument("--scene", default="FloorPlan1", help="scene to preload")
    ap.add_argument("--out-root", default="out/webapp", help="where the Save button writes")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"warning: binding {args.host} exposes this server on the network; "
              "it has no authentication and can write files.", file=sys.stderr)

    OUT_ROOT = args.out_root
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("starting AI2-THOR ...", flush=True)
    SESSION = Session(args.width, args.height, None if args.gpu < 0 else args.gpu, args.quality)
    SESSION.load_scene(args.scene)
    print(f"\n  ready -> http://{args.host}:{args.port}\n", flush=True)

    # Unity is a child process that outlives an unclean exit. An orphan keeps
    # holding GPU memory and makes the *next* launch fail with an Initialize
    # timeout, so shut it down on signals as well as on normal return.
    def shutdown(signum, _frame):
        print(f"\nsignal {signum}: stopping AI2-THOR", flush=True)
        SESSION.renderer.stop()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        SESSION.renderer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
