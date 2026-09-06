"""Thor3D -- controlled single-object edits re-rendered in AI2-THOR / RoboTHOR.

Given a scene and a viewpoint, render the scene twice: once as-is, and once
with exactly one object scaled, rotated, moved, or removed. Everything else --
camera, lighting, and every other object -- is restored identically, so the two
images differ only where the edit is.

    from thor3d import ThorRenderer, ObjectEdit, save_pair

    with ThorRenderer(width=512, height=512) as r:
        spec = r.capture_spec("FloorPlan1")
        orig, edited = r.render_pair(spec, ObjectEdit(object_type="Fridge", scale=1.4))
        save_pair(orig, edited, "out/", edited.metadata["edit"]["original"]["objectId"])
"""

from .io_utils import (
    background_report,
    canny_edges,
    canny_thresholds,
    depth_to_png16,
    geometric_edges,
    save_pair,
    save_render,
)
from .renderer import RenderResult, ThorRenderer
from .spec import CameraSpec, ObjectEdit, SceneSpec, poses_from_metadata, vec3

__all__ = [
    "ThorRenderer",
    "RenderResult",
    "SceneSpec",
    "CameraSpec",
    "ObjectEdit",
    "poses_from_metadata",
    "vec3",
    "save_render",
    "save_pair",
    "background_report",
    "canny_edges",
    "canny_thresholds",
    "geometric_edges",
    "depth_to_png16",
]
