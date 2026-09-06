#!/usr/bin/env python
"""Apply one edit to a captured scene and re-render it from the same camera.

Writes original/ and edited/ image sets, a changed-region mask, a diff image,
and a QA report confirming the background did not move.

    python scripts/edit_and_render.py --scene-spec out/fp1_fridge/scene.json \
        --object-type Fridge --scale 1.35 --out out/fp1_fridge/scale135

    python scripts/edit_and_render.py --scene-spec out/fp1_fridge/scene.json \
        --object-type Fridge --rotate-y 40 --out out/fp1_fridge/rot40

    # scene spec is optional; without one the scene's default spawn is used
    python scripts/edit_and_render.py --scene FloorPlan1 --object-type Apple \
        --scale 2.0 --translate 0 0.1 0 --out out/apple_big
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thor3d import ObjectEdit, SceneSpec, ThorRenderer, save_pair
from thor3d.spec import vec3


def build_edit(args) -> ObjectEdit:
    kwargs = {}
    if args.object_id:
        kwargs["object_id"] = args.object_id
    else:
        kwargs["object_type"] = args.object_type

    if args.scale is not None:
        kwargs["scale"] = args.scale
    if args.remove:
        kwargs["remove"] = True

    if args.rotation is not None:
        kwargs["rotation"] = vec3(*args.rotation)
    elif any(v is not None for v in (args.rotate_x, args.rotate_y, args.rotate_z)):
        kwargs["rotation_delta"] = vec3(args.rotate_x or 0, args.rotate_y or 0, args.rotate_z or 0)

    if args.position is not None:
        kwargs["position"] = vec3(*args.position)
    elif args.translate is not None:
        kwargs["position_delta"] = vec3(*args.translate)

    return ObjectEdit(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("scene source")
    src.add_argument("--scene-spec", help="scene.json from capture_scene.py")
    src.add_argument("--scene", help="scene name, if no --scene-spec")

    tgt = ap.add_argument_group("target object")
    g = tgt.add_mutually_exclusive_group(required=True)
    g.add_argument("--object-id", help="exact THOR objectId")
    g.add_argument("--object-type", help="objectType; picks the most visible instance")

    ed = ap.add_argument_group("edit (combine freely)")
    ed.add_argument("--scale", type=float, help="uniform scale factor, e.g. 1.5")
    ed.add_argument("--rotate-x", type=float, help="degrees to add to current rotation")
    ed.add_argument("--rotate-y", type=float)
    ed.add_argument("--rotate-z", type=float)
    ed.add_argument("--rotation", type=float, nargs=3, metavar=("X", "Y", "Z"), help="absolute euler degrees")
    ed.add_argument("--translate", type=float, nargs=3, metavar=("DX", "DY", "DZ"), help="offset in metres")
    ed.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="absolute world position")
    ed.add_argument("--remove", action="store_true", help="delete the object")

    out = ap.add_argument_group("output")
    out.add_argument("--out", required=True)
    out.add_argument("--width", type=int, default=512)
    out.add_argument("--height", type=int, default=512)
    out.add_argument("--gpu", type=int, default=0)
    out.add_argument("--quality", default="Ultra")
    out.add_argument("--no-depth", action="store_true")
    out.add_argument("--no-normals", action="store_true", help="skip normals.png")
    out.add_argument("--no-edges", action="store_true", help="skip edge maps")
    out.add_argument("--tolerance", type=int, default=4, help="max per-channel background drift allowed")

    args = ap.parse_args()
    if not args.scene_spec and not args.scene:
        ap.error("pass --scene-spec or --scene")

    edit = build_edit(args)
    if edit.is_noop:
        ap.error("no edit requested; pass at least one of --scale/--rotate-*/--translate/--remove")

    with ThorRenderer(
        width=args.width,
        height=args.height,
        gpu_device=None if args.gpu < 0 else args.gpu,
        quality=args.quality,
        render_depth=not args.no_depth,
        render_normals=not args.no_normals,
    ) as r:
        if args.scene_spec:
            spec = SceneSpec.load(args.scene_spec)
            # The spec's own resolution wins so the pair matches the captured original.
            args.width, args.height = spec.camera.width, spec.camera.height
        else:
            spec = r.capture_spec(args.scene)

        original, edited = r.render_pair(spec, edit)

        record = edited.metadata.get("edit")
        if record is None:
            print("error: edit did not resolve to any object", file=sys.stderr)
            return 4
        target_id = record["original"]["objectId"]

        report = save_pair(
            original,
            edited,
            args.out,
            target_object_id=target_id,
            save_depth=not args.no_depth,
            save_normals=not args.no_normals,
            save_edges=not args.no_edges,
            tolerance=args.tolerance,
        )

    print(json.dumps(report, indent=2))
    if report["near_zone_changed"]:
        print(
            f"note: {report['near_pixels_over_tolerance']} px changed in the halo around the "
            "object -- its shadow and bounced light following the edit. This is correct 3D "
            "behaviour and is what 2D compositing cannot reproduce.",
            file=sys.stderr,
        )
    if not report["far_background_unchanged"]:
        print(
            f"WARNING: {report['far_pixels_over_tolerance']} px changed far from the edited "
            "object. Something moved that should not have -- inspect diff.png.",
            file=sys.stderr,
        )
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
