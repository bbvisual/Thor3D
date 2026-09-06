#!/usr/bin/env python
"""Capture a scene: render the original image and save a replayable scene spec.

The scene spec written here is the anchor for everything downstream --
edit_and_render.py replays it exactly so the edited image shares this image's
camera and background.

    # default agent spawn
    python scripts/capture_scene.py --scene FloorPlan1 --out out/fp1

    # auto-frame a specific object so it is actually worth editing
    python scripts/capture_scene.py --scene FloorPlan1 --look-at Fridge --out out/fp1_fridge

    # RoboTHOR
    python scripts/capture_scene.py --scene FloorPlan_Train1_1 --look-at Television --out out/rt1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thor3d import ObjectEdit, ThorRenderer, save_render
from thor3d.spec import default_agent_mode, default_fov
from thor3d.viewpoint import find_camera_for_object, pick_editable_objects


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="e.g. FloorPlan1 or FloorPlan_Train1_1")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--look-at", help="objectType or objectId to frame the camera on")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--fov", type=float, default=None,
                    help="default: 90 for iTHOR, 60 for RoboTHOR/LoCoBot")
    ap.add_argument("--agent-mode", default=None,
                    help="default | locobot | arm. Defaults to 'default' for every scene: "
                         "the LoCoBot controller cannot perform object edits at all.")
    ap.add_argument("--gpu", type=int, default=0, help="-1 to let Unity choose")
    ap.add_argument("--quality", default="Ultra")
    ap.add_argument("--randomize-spawn", action="store_true", help="shuffle pickupables before capture")
    ap.add_argument("--seed", type=int, default=None, help="required with --randomize-spawn")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-normals", action="store_true", help="skip normals.png")
    ap.add_argument("--no-edges", action="store_true", help="skip edge maps")
    ap.add_argument("--list-objects", action="store_true", help="print editable objects and exit")
    args = ap.parse_args()

    if args.randomize_spawn and args.seed is None:
        ap.error("--randomize-spawn needs --seed so the layout can be replayed")

    agent_mode = args.agent_mode or default_agent_mode(args.scene)
    fov = args.fov if args.fov is not None else default_fov(args.scene)

    with ThorRenderer(
        width=args.width,
        height=args.height,
        gpu_device=None if args.gpu < 0 else args.gpu,
        quality=args.quality,
        render_depth=not args.no_depth,
        render_normals=not args.no_normals,
    ) as r:
        camera = None
        if args.look_at:
            r.controller.reset(
                scene=args.scene,
                agentMode=agent_mode,
                renderInstanceSegmentation=True,
                snapToGrid=False,
            )
            objs = r.controller.last_event.metadata["objects"]
            match = next((o for o in objs if o["objectId"] == args.look_at), None)
            if match is None:
                match = next((o for o in objs if o["objectType"] == args.look_at), None)
            if match is None:
                print(f"error: no object {args.look_at!r} in {args.scene}", file=sys.stderr)
                print("available types:", sorted({o["objectType"] for o in objs}), file=sys.stderr)
                return 2
            camera = find_camera_for_object(
                r, args.scene, match["objectId"], agent_mode=agent_mode
            )
            if camera is None:
                print(
                    f"error: {match['objectId']} is not visible from any reachable position",
                    file=sys.stderr,
                )
                return 3
            print(f"framed {match['objectId']} from {camera.position} yaw={camera.rotation['y']:.1f}")

        if camera is not None:
            camera.field_of_view = fov
            camera.width, camera.height = args.width, args.height

        spec = r.capture_spec(
            args.scene,
            camera=camera,
            randomize_spawn=args.randomize_spawn,
            seed=args.seed,
            agent_mode=agent_mode,
        )
        spec.camera.field_of_view = fov

        if args.list_objects:
            for o in pick_editable_objects(r.objects()):
                s = o["axisAlignedBoundingBox"]["size"]
                print(
                    f"{o['objectId']:<48} {o['objectType']:<18} "
                    f"size=({s['x']:.2f},{s['y']:.2f},{s['z']:.2f}) "
                    f"pickupable={bool(o.get('pickupable'))} visible={o['visible']}"
                )
            return 0

        os.makedirs(args.out, exist_ok=True)
        spec.save(os.path.join(args.out, "scene.json"))

        target_id = None
        if args.look_at:
            target_id = r.resolve_object(
                ObjectEdit(object_id=args.look_at)
                if "|" in args.look_at
                else ObjectEdit(object_type=args.look_at)
            )["objectId"]

        result = r.render(spec)
        written = save_render(
            result,
            os.path.join(args.out, "original"),
            target_object_id=target_id,
            save_depth=not args.no_depth,
            save_normals=not args.no_normals,
            save_edges=not args.no_edges,
        )

    print(json.dumps({"scene_spec": os.path.join(args.out, "scene.json"), **written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
