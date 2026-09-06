"""Finding a camera pose that actually looks at the object you want to edit.

A scene's default agent spawn usually has the target object off-screen or
occluded, which makes for a useless before/after pair. This module searches the
navigable positions for the viewpoint where the target covers the most pixels.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .spec import CameraSpec, supports_standing

# Fallback eye height above the agent's floor position, used only if the build
# does not report cameraPosition. The real offset differs per agent mode
# (~+0.675 m for the default agent, ~-0.03 m for the LoCoBot).
_DEFAULT_EYE_OFFSET = 0.675


def _eye_offset(event) -> float:
    """Camera height above the agent's feet, read from the live build."""
    cam = event.metadata.get("cameraPosition")
    if not cam:
        return _DEFAULT_EYE_OFFSET
    return float(cam["y"]) - float(event.metadata["agent"]["position"]["y"])


def _aim_point(obj: Dict[str, Any]) -> Dict[str, float]:
    """Where to point the camera: the object's visual centre.

    Not ``obj["position"]`` -- a THOR object's transform pivot often sits at its
    base (a fridge's pivot is on the floor), so aiming at it tilts the camera
    down past the object.
    """
    return dict(obj["axisAlignedBoundingBox"]["center"])


def _yaw_towards(frm: Dict[str, float], to: Dict[str, float]) -> float:
    """Unity yaw (degrees, clockwise from +z) pointing from one point at another."""
    return math.degrees(math.atan2(to["x"] - frm["x"], to["z"] - frm["z"])) % 360.0


def _horizon_towards(frm: Dict[str, float], to: Dict[str, float], eye_height: float) -> float:
    """Unity camera horizon: positive looks down, negative looks up."""
    ground = math.hypot(to["x"] - frm["x"], to["z"] - frm["z"])
    if ground < 1e-6:
        return 0.0
    return math.degrees(math.atan2((frm["y"] + eye_height) - to["y"], ground))


def _touches_border(mask, margin: int = 2) -> bool:
    """True if the object runs off the edge of the frame."""
    return bool(
        mask[:margin, :].any() or mask[-margin:, :].any()
        or mask[:, :margin].any() or mask[:, -margin:].any()
    )


def _frame_score(coverage: float, clipped: bool, target_fraction: float) -> float:
    """Higher is better. Rewards a target-sized, fully-visible object.

    Plain "maximize pixel area" is the wrong objective: it walks the camera
    right up to the object until it fills the frame and there is no background
    left to hold constant. Scoring distance from a target coverage in log space
    treats "twice too big" and "half too small" as equally bad, and clipping is
    penalised hard so the edited object still fits on screen after being scaled
    up.
    """
    score = -abs(math.log(coverage / target_fraction))
    if clipped:
        score -= 10.0
    return score


def find_camera_for_object(
    renderer,
    scene: str,
    object_id: str,
    min_distance: float = 0.8,
    max_distance: float = 4.5,
    max_candidates: int = 40,
    horizon_choices: Tuple[float, ...] = (0.0,),
    min_pixel_fraction: float = 0.0015,
    target_pixel_fraction: float = 0.12,
    max_pixel_fraction: float = 0.45,
    agent_mode: Optional[str] = None,
) -> Optional[CameraSpec]:
    """Return the viewpoint that best frames ``object_id``.

    Candidate positions are the scene's reachable positions within
    ``[min_distance, max_distance]`` of the object; each is aimed directly at
    the object rather than sampling yaw blindly, so one render per candidate is
    enough.

    Args:
        renderer: A live :class:`~thor3d.renderer.ThorRenderer`.
        horizon_choices: Extra pitch offsets (degrees) to try on top of the
            computed aim. Widen this if the object sits very high or low.
        min_pixel_fraction: Reject viewpoints where the object covers less than
            this fraction of the frame. The default is low enough to admit small
            objects like an apple, which can never reach the target fraction.
        agent_mode: Passed through to ``reset`` so the search sees the same
            camera height the final render will use.
        target_pixel_fraction: Preferred on-screen size of the object. The
            default leaves plenty of background and room for the object to grow
            when scaled up.
        max_pixel_fraction: Reject viewpoints where the object dominates the
            frame, since those leave almost no background to hold fixed.

    Returns:
        The best :class:`CameraSpec`, or ``None`` if the object was never
        acceptably visible from anywhere navigable.
    """
    ranked = find_camera_candidates(
        renderer, scene, object_id,
        min_distance=min_distance, max_distance=max_distance,
        max_candidates=max_candidates, horizon_choices=horizon_choices,
        min_pixel_fraction=min_pixel_fraction,
        target_pixel_fraction=target_pixel_fraction,
        max_pixel_fraction=max_pixel_fraction,
        agent_mode=agent_mode,
    )
    return ranked[0] if ranked else None


def find_camera_candidates(
    renderer,
    scene: str,
    object_id: str,
    min_distance: float = 0.8,
    max_distance: float = 4.5,
    max_candidates: int = 40,
    horizon_choices: Tuple[float, ...] = (0.0,),
    min_pixel_fraction: float = 0.0015,
    target_pixel_fraction: float = 0.12,
    max_pixel_fraction: float = 0.45,
    agent_mode: Optional[str] = None,
) -> List[CameraSpec]:
    """Every acceptable viewpoint of ``object_id``, best-framed first.

    Same search as :func:`find_camera_for_object`, but returns the whole ranked
    list so a UI can offer "next viewpoint" instead of only the single best.
    """
    c = renderer.controller
    reset_kwargs = dict(
        scene=scene,
        width=renderer.width,
        height=renderer.height,
        renderInstanceSegmentation=True,
        snapToGrid=False,
    )
    if agent_mode:
        reset_kwargs["agentMode"] = agent_mode
    c.reset(**reset_kwargs)

    event = c.step(action="GetReachablePositions")
    if not event.metadata["lastActionSuccess"]:
        raise RuntimeError("GetReachablePositions failed: " + event.metadata.get("errorMessage", ""))
    positions: List[Dict[str, float]] = event.metadata["actionReturn"]

    target = next((o for o in event.metadata["objects"] if o["objectId"] == object_id), None)
    if target is None:
        raise KeyError(f"{object_id!r} not present in {scene}")
    obj_pos = _aim_point(target)
    eye_offset = _eye_offset(event)

    scored = sorted(
        (
            (math.dist((p["x"], p["z"]), (obj_pos["x"], obj_pos["z"])), p)
            for p in positions
        ),
        key=lambda t: t[0],
    )
    candidates = [p for d, p in scored if min_distance <= d <= max_distance][:max_candidates]
    if not candidates:
        candidates = [p for _, p in scored[:max_candidates]]

    n_pixels = renderer.width * renderer.height
    scored_cameras: List[Tuple[float, CameraSpec]] = []

    for pos in candidates:
        yaw = _yaw_towards(pos, obj_pos)
        base_horizon = _horizon_towards(pos, obj_pos, eye_offset)
        for extra in horizon_choices:
            horizon = max(-30.0, min(60.0, base_horizon + extra))
            tp = dict(
                action="Teleport",
                position=pos,
                rotation={"x": 0.0, "y": yaw, "z": 0.0},
                horizon=horizon,
                forceAction=True,
            )
            if supports_standing(agent_mode or "default"):
                tp["standing"] = True
            ev = c.step(**tp)
            if not ev.metadata["lastActionSuccess"]:
                continue
            masks = renderer._instance_masks(ev)
            if object_id not in masks:
                continue
            mask = masks[object_id]
            coverage = float(mask.sum()) / n_pixels
            if not (min_pixel_fraction <= coverage <= max_pixel_fraction):
                continue

            score = _frame_score(coverage, _touches_border(mask), target_pixel_fraction)
            scored_cameras.append(
                (
                    score,
                    CameraSpec(
                        mode="agent",
                        position={k: float(pos[k]) for k in ("x", "y", "z")},
                        rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
                        horizon=float(horizon),
                        standing=True,
                        field_of_view=float(ev.metadata.get("fov", 90.0)),
                        width=renderer.width,
                        height=renderer.height,
                    ),
                )
            )

    scored_cameras.sort(key=lambda t: t[0], reverse=True)
    return [cam for _, cam in scored_cameras]


# Structural geometry that is not a meaningful "object" to edit.
STRUCTURAL_TYPES = ("Floor", "Wall", "Ceiling", "Room", "Doorway", "Doorframe")

# Types excluded from batch runs because editing them is unsafe or meaningless.
# Painting is the important one: ScaleObject on it *crashes the Unity build*
# outright in 5.0.0, taking the whole process down. The rest are flat,
# wall-mounted, or fixed fixtures that look wrong when scaled or rotated.
UNSAFE_TO_EDIT_TYPES = (
    "Painting",
    "Poster",
    "Mirror",
    "Window",
    "Blinds",
    "Curtains",
    "ShowerCurtain",
    "ShowerGlass",
    "ShowerDoor",
    "LightSwitch",
    "Door",
)


def pick_editable_objects(
    objects: List[Dict[str, Any]],
    require_pickupable: bool = False,
    exclude_types: Optional[Tuple[str, ...]] = None,
) -> List[Dict[str, Any]]:
    """Objects that are safe and sensible edit targets.

    ``ScaleObject`` and ``TeleportObject`` work on structural objects too
    (counters, fridges), so pickupables are not required by default.

    Pass ``exclude_types=()`` to disable filtering -- but note that this
    re-admits ``Painting``, which crashes the renderer when scaled.
    """
    if exclude_types is None:
        exclude_types = STRUCTURAL_TYPES + UNSAFE_TO_EDIT_TYPES
    out = []
    for o in objects:
        if o["objectType"] in exclude_types:
            continue
        if require_pickupable and not o.get("pickupable"):
            continue
        out.append(o)
    return out
