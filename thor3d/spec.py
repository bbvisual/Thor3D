"""Serializable descriptions of a camera, a scene, and an object edit.

Everything the pipeline needs to reproduce a render is captured here so that the
"before" and "after" images are guaranteed to come from an identical setup apart
from the single edit under test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

Vec3 = Dict[str, float]


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return {"x": float(x), "y": float(y), "z": float(z)}


def _round_vec(v: Vec3, nd: int = 6) -> Vec3:
    return {k: round(float(v[k]), nd) for k in ("x", "y", "z")}


@dataclass
class CameraSpec:
    """A fully determined viewpoint.

    ``mode="agent"`` renders from the agent's ego camera, which is what the
    stock iTHOR/RoboTHOR frames are. ``mode="thirdparty"`` adds a free-floating
    camera not bound to agent navigability, useful when you want a viewpoint the
    agent cannot legally stand in.
    """

    mode: str = "agent"
    position: Vec3 = field(default_factory=vec3)
    rotation: Vec3 = field(default_factory=vec3)
    horizon: float = 0.0
    standing: bool = True
    field_of_view: float = 90.0
    width: int = 512
    height: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraSpec":
        return cls(**d)


@dataclass
class ObjectEdit:
    """One controlled modification of a single object.

    Exactly one object is touched. Fields left as ``None`` are not modified, so
    a scale-only edit leaves rotation and position bit-identical to the source
    scene.

    Attributes:
        object_id: THOR ``objectId`` (e.g. ``"Apple|-00.47|+01.15|+00.48"``).
            Prefer this over ``object_type`` when a scene holds several
            instances of the same type.
        object_type: Alternative selector; resolves to the instance of that type
            closest to the camera centre. Ignored when ``object_id`` is set.
        scale: Uniform scale factor. THOR's ``ScaleObject`` is uniform-only --
            see the README for why non-uniform stretching is not available.
        rotation / rotation_delta: Absolute Euler angles in degrees, or a
            per-axis offset applied to the object's current rotation.
        position / position_delta: Absolute world position, or an offset in
            metres.
        remove: Delete the object from the scene entirely.
    """

    object_id: Optional[str] = None
    object_type: Optional[str] = None
    scale: Optional[float] = None
    rotation: Optional[Vec3] = None
    rotation_delta: Optional[Vec3] = None
    position: Optional[Vec3] = None
    position_delta: Optional[Vec3] = None
    remove: bool = False

    def __post_init__(self) -> None:
        if not self.object_id and not self.object_type:
            raise ValueError("ObjectEdit needs either object_id or object_type")
        if self.rotation and self.rotation_delta:
            raise ValueError("set rotation or rotation_delta, not both")
        if self.position and self.position_delta:
            raise ValueError("set position or position_delta, not both")
        if self.scale is not None and self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    @property
    def is_noop(self) -> bool:
        return not (
            self.remove
            or self.scale is not None
            or self.rotation
            or self.rotation_delta
            or self.position
            or self.position_delta
        )

    def slug(self) -> str:
        """Short filesystem-safe tag describing the edit, for output dirnames."""
        target = (self.object_id or self.object_type or "obj").split("|")[0]
        parts = [target]
        if self.remove:
            parts.append("removed")
        if self.scale is not None:
            parts.append(f"scale{self.scale:g}".replace(".", "p"))
        rot = self.rotation or self.rotation_delta
        if rot:
            tag = "rot" if self.rotation else "drot"
            parts.append(f"{tag}{rot.get('y', 0):g}".replace(".", "p").replace("-", "m"))
        pos = self.position or self.position_delta
        if pos:
            tag = "pos" if self.position else "dpos"
            parts.append(
                f"{tag}{pos.get('x', 0):g}_{pos.get('z', 0):g}".replace(".", "p").replace("-", "m")
            )
        return "-".join(parts) or f"{target}-noop"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v is not False}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectEdit":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SceneSpec:
    """A complete, replayable description of the *unedited* scene.

    ``object_poses`` is a snapshot of every movable object taken at capture
    time. Replaying it before each render is what makes the background pixel
    -stable across the before/after pair, even if the scene was randomized.
    """

    scene: str
    camera: CameraSpec
    object_poses: List[Dict[str, Any]] = field(default_factory=list)
    seed: Optional[int] = None
    randomize_spawn: bool = False
    agent_mode: str = "default"
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "camera": self.camera.to_dict(),
            "object_poses": self.object_poses,
            "seed": self.seed,
            "randomize_spawn": self.randomize_spawn,
            "agent_mode": self.agent_mode,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneSpec":
        return cls(
            scene=d["scene"],
            camera=CameraSpec.from_dict(d["camera"]),
            object_poses=d.get("object_poses", []),
            seed=d.get("seed"),
            randomize_spawn=d.get("randomize_spawn", False),
            agent_mode=d.get("agent_mode", "default"),
            notes=d.get("notes", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SceneSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def is_robothor(scene: str) -> bool:
    """RoboTHOR scenes are named ``FloorPlan_Train1_1`` / ``FloorPlan_Val1_1``."""
    return scene.startswith("FloorPlan_")


def default_agent_mode(scene: str) -> str:
    """Always the default agent, including for RoboTHOR scenes.

    RoboTHOR's published frames come from the LoCoBot, but its controller is a
    restricted subclass that rejects ``ScaleObject``, ``TeleportObject``, and
    ``SetObjectPoses`` as invalid actions -- so no object edits are possible
    under it at all. The default agent loads the identical scene geometry and
    supports every edit, so it is what this pipeline uses; the only difference
    is camera height, and :func:`default_fov` still matches RoboTHOR's framing.
    """
    return "default"


def default_fov(scene: str) -> float:
    """Field of view matching each dataset's published frames."""
    return 60.0 if is_robothor(scene) else 90.0


def supports_standing(agent_mode: str) -> bool:
    """Whether ``Teleport`` accepts a ``standing`` argument for this agent.

    The LoCoBot has a fixed-height camera and its Teleport overload rejects
    ``standing`` outright rather than ignoring it.
    """
    return agent_mode != "locobot"


def supports_object_edits(agent_mode: str) -> bool:
    """Whether this agent's controller exposes the object-manipulation actions."""
    return agent_mode != "locobot"


def poses_from_metadata(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a ``SetObjectPoses`` payload from ``event.metadata['objects']``.

    Only pickupable/moveable objects are included: THOR rejects the payload if
    it contains structural objects, and those never move anyway.
    """
    poses = []
    for o in objects:
        if not (o.get("pickupable") or o.get("moveable")):
            continue
        poses.append(
            {
                "objectName": o["name"],
                "position": _round_vec(o["position"]),
                "rotation": _round_vec(o["rotation"]),
            }
        )
    return poses
