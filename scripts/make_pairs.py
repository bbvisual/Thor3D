#!/usr/bin/env python
"""Batch-generate (original, edited) pairs across scenes, objects, and factors.

One Unity process is reused for the whole run, and each pair re-derives its
scene from scratch, so a failure on one object cannot corrupt later pairs.

    # 20 iTHOR kitchens, 3 objects each, scale and rotation sweeps
    python scripts/make_pairs.py --scene-set ithor-kitchen --num-scenes 20 \
        --objects-per-scene 3 --scales 0.7 1.4 --rotations 45 90 --out data/pairs

    # RoboTHOR, specific object types only
    python scripts/make_pairs.py --scene-set robothor-train --num-scenes 5 \
        --object-types Television Chair --scales 1.5 --out data/robothor_pairs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thor3d import ObjectEdit, ThorRenderer, save_pair
from thor3d.spec import default_agent_mode, vec3
from thor3d.viewpoint import find_camera_for_object, pick_editable_objects


def resolve_scenes(renderer, scene_set: str, num_scenes: int, seed: int):
    c = renderer.controller
    if scene_set == "ithor":
        scenes = c.ithor_scenes()
    elif scene_set == "ithor-kitchen":
        scenes = c.ithor_scenes(include_kitchens=True, include_living_rooms=False,
                                include_bedrooms=False, include_bathrooms=False)
    elif scene_set == "ithor-living":
        scenes = c.ithor_scenes(include_kitchens=False, include_living_rooms=True,
                                include_bedrooms=False, include_bathrooms=False)
    elif scene_set == "robothor":
        scenes = c.robothor_scenes()
    elif scene_set == "robothor-train":
        scenes = c.robothor_scenes(include_train=True, include_val=False)
    elif scene_set == "robothor-val":
        scenes = c.robothor_scenes(include_train=False, include_val=True)
    else:
        scenes = [s.strip() for s in scene_set.split(",") if s.strip()]
    scenes = list(scenes)
    random.Random(seed).shuffle(scenes)
    return scenes[:num_scenes] if num_scenes > 0 else scenes


def edits_for(object_id: str, scales, rotations, translations):
    """The factor sweep applied to one object: one edit per factor level."""
    out = []
    for s in scales:
        out.append(ObjectEdit(object_id=object_id, scale=s))
    for deg in rotations:
        out.append(ObjectEdit(object_id=object_id, rotation_delta=vec3(y=deg)))
    for dx, dy, dz in translations:
        out.append(ObjectEdit(object_id=object_id, position_delta=vec3(dx, dy, dz)))
    return out


def _controller_died(exc: Exception) -> bool:
    """Whether this failure means the Unity process is gone rather than the action being rejected."""
    msg = str(exc).lower()
    return any(
        s in msg
        for s in ("write to closed file", "broken pipe", "connection reset", "not running", "closed")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-set", default="ithor-kitchen",
                    help="ithor | ithor-kitchen | ithor-living | robothor | robothor-train | "
                         "robothor-val | comma-separated scene names")
    ap.add_argument("--num-scenes", type=int, default=5, help="0 for all")
    ap.add_argument("--objects-per-scene", type=int, default=3)
    ap.add_argument("--object-types", nargs="*", default=None, help="restrict to these types")
    ap.add_argument("--pickupable-only", action="store_true")
    ap.add_argument("--scales", type=float, nargs="*", default=[0.7, 1.4])
    ap.add_argument("--rotations", type=float, nargs="*", default=[45.0])
    ap.add_argument("--translations", type=float, nargs="*", default=[],
                    help="flat list of dx dy dz triplets, e.g. 0.2 0 0 -0.2 0 0")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--quality", default="Ultra")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-normals", action="store_true", help="skip normals.png")
    ap.add_argument("--no-edges", action="store_true", help="skip edge maps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=int, default=4)
    args = ap.parse_args()

    if len(args.translations) % 3:
        ap.error("--translations needs a multiple of 3 values (dx dy dz per offset)")
    translations = [tuple(args.translations[i:i + 3]) for i in range(0, len(args.translations), 3)]
    if not (args.scales or args.rotations or translations):
        ap.error("nothing to sweep: pass --scales and/or --rotations and/or --translations")

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    index, n_ok, n_fail = [], 0, 0

    with ThorRenderer(
        width=args.width,
        height=args.height,
        gpu_device=None if args.gpu < 0 else args.gpu,
        quality=args.quality,
        render_depth=not args.no_depth,
        render_normals=not args.no_normals,
    ) as r:
        scenes = resolve_scenes(r, args.scene_set, args.num_scenes, args.seed)
        print(f"{len(scenes)} scenes: {scenes}")

        for scene in scenes:
            try:
                agent_mode = default_agent_mode(scene)
                r.controller.reset(
                    scene=scene,
                    agentMode=agent_mode,
                    renderInstanceSegmentation=True,
                    snapToGrid=False,
                )
                pool = pick_editable_objects(
                    r.controller.last_event.metadata["objects"],
                    require_pickupable=args.pickupable_only,
                )
                if args.object_types:
                    wanted = set(args.object_types)
                    pool = [o for o in pool if o["objectType"] in wanted]
                rng.shuffle(pool)
            except Exception:
                print(f"[{scene}] scene load failed:\n{traceback.format_exc()}", file=sys.stderr)
                n_fail += 1
                continue

            chosen = 0
            for obj in pool:
                if chosen >= args.objects_per_scene:
                    break
                object_id = obj["objectId"]
                try:
                    camera = find_camera_for_object(r, scene, object_id, agent_mode=agent_mode)
                except Exception:
                    camera = None
                if camera is None:
                    continue  # never visible from a navigable spot; not a usable target

                spec = r.capture_spec(scene, camera=camera, agent_mode=agent_mode)
                chosen += 1

                for edit in edits_for(object_id, args.scales, args.rotations, translations):
                    pair_dir = os.path.join(args.out, scene, edit.slug())
                    try:
                        original, edited = r.render_pair(spec, edit)
                        record = edited.metadata.get("edit")
                        if record is None:
                            raise RuntimeError("edit resolved to no object")
                        report = save_pair(
                            original, edited, pair_dir,
                            target_object_id=record["original"]["objectId"],
                            save_depth=not args.no_depth,
                            save_normals=not args.no_normals,
                            save_edges=not args.no_edges,
                            tolerance=args.tolerance,
                        )
                        spec.save(os.path.join(pair_dir, "scene.json"))
                        index.append({
                            "dir": os.path.relpath(pair_dir, args.out),
                            "scene": scene,
                            "object_id": object_id,
                            "object_type": obj["objectType"],
                            "edit": edit.to_dict(),
                            "changed_fraction": report["changed_fraction"],
                            "far_background_unchanged": report["far_background_unchanged"],
                        })
                        n_ok += 1
                        print(f"[ok]   {scene} {edit.slug()} "
                              f"changed={report['changed_fraction']:.4f} "
                              f"far_bg_ok={report['far_background_unchanged']}")
                    except Exception as e:
                        n_fail += 1
                        print(f"[fail] {scene} {edit.slug()}: {str(e)[:200]}", file=sys.stderr)
                        # Some actions (ScaleObject on a Painting) kill the
                        # build. Relaunch so the rest of the sweep still runs.
                        if _controller_died(e):
                            r.restart()
                            break

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n{n_ok} pairs written, {n_fail} failed -> {os.path.join(args.out, 'index.json')}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
