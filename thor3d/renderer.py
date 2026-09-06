"""Headless AI2-THOR renderer with a locked camera and a frozen background.

The contract this module implements: for a given :class:`SceneSpec`, two calls
to :meth:`ThorRenderer.render` -- one with no edit and one with an
:class:`ObjectEdit` -- produce images that differ only where the edited object
is. Camera pose, field of view, lighting, and every other object's pose are
restored identically before each render.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .spec import (
    CameraSpec,
    ObjectEdit,
    SceneSpec,
    default_agent_mode,
    default_fov,
    poses_from_metadata,
    supports_object_edits,
    supports_standing,
)

log = logging.getLogger("thor3d")

# ai2thor nags whenever a RoboTHOR scene is loaded without the LoCoBot. Using
# the default agent there is a deliberate choice (see spec.default_agent_mode --
# the LoCoBot cannot edit objects at all), so the reminder is just noise.
warnings.filterwarnings("ignore", message=".*RoboTHOR scene without using the standard LoCoBot.*")

# ScaleObject/TeleportObject animate over time by default; 0 makes them instant
# so no physics frames elapse between the edit and the render.
INSTANT = 0.0


class RenderResult:
    """One rendered frame plus its ground truth."""

    def __init__(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        instance_masks: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        normals: Optional[np.ndarray] = None,
    ):
        self.rgb = rgb
        self.depth = depth              # float32 metres, or None
        self.normals = normals          # uint8 RGB(A) encoded surface normals, or None
        self.instance_masks = instance_masks
        self.metadata = metadata

    def mask_for(self, object_id: str) -> np.ndarray:
        """Boolean mask of one object, empty if it is not visible."""
        m = self.instance_masks.get(object_id)
        if m is None:
            return np.zeros(self.rgb.shape[:2], dtype=bool)
        return m.astype(bool)


class ThorRenderer:
    """Owns a single AI2-THOR controller and renders scenes through it.

    Reusing one controller across many renders is far faster than restarting
    Unity per image, and ``reset()`` restores the stock scene layout each time.

    Args:
        gpu_device: Which GPU to render on. Requires ``vulkaninfo`` on PATH
            (``conda install -c conda-forge vulkan-tools``). Pass ``None`` to
            let Unity choose.
        platform: Defaults to headless ``CloudRendering``. Pass ``"osmesa"`` to
            force software rendering, or a real platform class.
        quality: Unity quality preset. "Ultra" gives the best shadows and
            reflections; "Very Low" is fastest.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        gpu_device: Optional[int] = 0,
        quality: str = "Ultra",
        render_depth: bool = True,
        render_instance_segmentation: bool = True,
        render_normals: bool = True,
        platform: Any = None,
        **controller_kwargs: Any,
    ):
        import ai2thor.controller

        if platform is None:
            from ai2thor.platform import CloudRendering

            platform = CloudRendering

        self.width = width
        self.height = height
        self.render_depth = render_depth
        self.render_instance_segmentation = render_instance_segmentation
        self.render_normals = render_normals
        self._current_scene: Optional[str] = None
        # Flipped off permanently the first time the build rejects the action,
        # so a batch run warns once instead of once per render.
        self._set_object_poses_supported = True

        # Kept so restart() can relaunch Unity identically after a crash.
        self._controller_args = dict(
            platform=platform,
            gpu_device=gpu_device,
            width=width,
            height=height,
            quality=quality,
            renderDepthImage=render_depth,
            renderInstanceSegmentation=render_instance_segmentation,
            renderNormalsImage=render_normals,
            # Free-form agent placement: without this, Teleport snaps the agent
            # to the navigation grid and the camera pose is not reproducible.
            snapToGrid=False,
            **controller_kwargs,
        )
        self.controller = ai2thor.controller.Controller(**self._controller_args)

    # ------------------------------------------------------------------ setup

    def _step(self, action: str, **kwargs: Any):
        event = self.controller.step(action=action, **kwargs)
        if not event.metadata["lastActionSuccess"]:
            raise RuntimeError(
                f"THOR action {action} failed: {event.metadata.get('errorMessage', '')}"
            )
        return event

    def reset_scene(self, spec: SceneSpec):
        """Load the scene and restore the exact recorded state.

        This is the step that guarantees the background is shared: the same
        sequence runs before the original render and before the edited one.
        """
        event = self.controller.reset(
            scene=spec.scene,
            width=spec.camera.width,
            height=spec.camera.height,
            fieldOfView=spec.camera.field_of_view,
            agentMode=spec.agent_mode,
            renderDepthImage=self.render_depth,
            renderInstanceSegmentation=self.render_instance_segmentation,
            renderNormalsImage=self.render_normals,
            snapToGrid=False,
        )
        self._current_scene = spec.scene

        if spec.randomize_spawn and spec.seed is not None:
            event = self._step("InitialRandomSpawn", randomSeed=spec.seed)

        # Replay the recorded layout. placeStationary keeps objects kinematic so
        # they do not drift while the rest of the setup happens.
        #
        # This is only a safety net: reset() already restores the stock layout
        # deterministically, and a randomized layout is reproduced by replaying
        # InitialRandomSpawn with the same seed above. So when the action is
        # unavailable -- notably under agentMode="locobot", where the build
        # rejects it as an invalid action -- skipping it is harmless.
        if spec.object_poses and self._set_object_poses_supported:
            try:
                event = self.controller.step(
                    action="SetObjectPoses",
                    objectPoses=spec.object_poses,
                    placeStationary=True,
                )
                ok = event.metadata["lastActionSuccess"]
                err = event.metadata.get("errorMessage", "")
            except ValueError as e:  # unknown action for this agent
                ok, err = False, str(e)
            if not ok:
                self._set_object_poses_supported = False
                log.warning(
                    "SetObjectPoses unavailable (%s); using the scene's own layout, "
                    "which reset() already restores deterministically.",
                    err.strip()[:160],
                )

        # Deliberately no PausePhysicsAutoSim here: in ai2thor 5.0.0 it leaves
        # ScaleObject with stale collider bounds (the object's reported x/z size
        # inflates by an arbitrary factor while y scales correctly). Objects are
        # instead pinned by making them kinematic -- placeStationary above for
        # the background, forceKinematic in _apply_edit for the target -- which
        # achieves the same frozen result without the bug.
        return self._apply_camera(spec.camera, spec.agent_mode)

    def _apply_camera(self, cam: CameraSpec, agent_mode: str = "default"):
        """Place the viewpoint. Identical inputs give an identical view."""
        if cam.mode == "agent":
            kwargs = dict(
                position=cam.position,
                rotation=cam.rotation,
                horizon=cam.horizon,
                forceAction=True,
            )
            if supports_standing(agent_mode):
                kwargs["standing"] = cam.standing
            return self._step("Teleport", **kwargs)
        if cam.mode == "thirdparty":
            # Park the agent's body out of frame, then add the free camera.
            # Third-party cameras are cleared by reset(), so this re-adds it.
            return self._step(
                "AddThirdPartyCamera",
                position=cam.position,
                rotation=cam.rotation,
                fieldOfView=cam.field_of_view,
            )
        raise ValueError(f"unknown camera mode {cam.mode!r}")

    # ------------------------------------------------------------- inspection

    def objects(self) -> List[Dict[str, Any]]:
        return self.controller.last_event.metadata["objects"]

    def capture_spec(
        self,
        scene: str,
        camera: Optional[CameraSpec] = None,
        randomize_spawn: bool = False,
        seed: Optional[int] = None,
        agent_mode: Optional[str] = None,
    ) -> SceneSpec:
        """Load a scene and snapshot it into a replayable :class:`SceneSpec`.

        With no ``camera``, the agent's default spawn pose is recorded. With no
        ``agent_mode``, RoboTHOR scenes use the LoCoBot and iTHOR scenes the
        default agent, matching each dataset's published frames.
        """
        agent_mode = agent_mode or default_agent_mode(scene)
        self.controller.reset(
            scene=scene,
            width=self.width,
            height=self.height,
            agentMode=agent_mode,
            fieldOfView=camera.field_of_view if camera else default_fov(scene),
            renderDepthImage=self.render_depth,
            renderInstanceSegmentation=self.render_instance_segmentation,
            renderNormalsImage=self.render_normals,
            snapToGrid=False,
        )
        self._current_scene = scene

        if randomize_spawn and seed is not None:
            self._step("InitialRandomSpawn", randomSeed=seed)

        if camera is not None:
            self._apply_camera(camera, agent_mode)
        event = self.controller.step(action="Pass")
        agent = event.metadata["agent"]

        if camera is None:
            camera = CameraSpec(
                mode="agent",
                position=dict(agent["position"]),
                rotation=dict(agent["rotation"]),
                horizon=float(agent["cameraHorizon"]),
                standing=bool(agent.get("isStanding", True)),
                field_of_view=float(event.metadata.get("fov", 90.0)),
                width=self.width,
                height=self.height,
            )

        return SceneSpec(
            scene=scene,
            camera=camera,
            object_poses=poses_from_metadata(event.metadata["objects"]),
            seed=seed,
            randomize_spawn=randomize_spawn,
            agent_mode=agent_mode,
        )

    def resolve_object(self, edit: ObjectEdit, event=None) -> Dict[str, Any]:
        """Turn an edit's selector into a concrete object from live metadata.

        An ``object_type`` selector picks the instance covering the most pixels
        in the current frame, so batch jobs target whatever is actually on
        screen rather than an occluded duplicate in another room.
        """
        event = event or self.controller.last_event
        objects = event.metadata["objects"]

        if edit.object_id:
            for o in objects:
                if o["objectId"] == edit.object_id:
                    return o
            raise KeyError(f"object_id {edit.object_id!r} not in scene {self._current_scene}")

        candidates = [o for o in objects if o["objectType"] == edit.object_type]
        if not candidates:
            raise KeyError(f"no {edit.object_type!r} in scene {self._current_scene}")

        masks = self._instance_masks(event)
        if masks:
            visible = [(int(masks[o["objectId"]].sum()), o) for o in candidates if o["objectId"] in masks]
            if visible:
                return max(visible, key=lambda t: t[0])[1]
        return min(candidates, key=lambda o: o.get("distance", 1e9))

    # ------------------------------------------------------------------ edits

    def _apply_edit(self, edit: ObjectEdit, agent_mode: str = "default") -> Dict[str, Any]:
        """Apply one edit and return a record of what was requested vs achieved."""
        if not supports_object_edits(agent_mode):
            raise RuntimeError(
                f"agentMode={agent_mode!r} cannot edit objects: its controller rejects "
                "ScaleObject/TeleportObject as invalid actions. Use agent_mode='default', "
                "which loads the same scene geometry."
            )
        target = self.resolve_object(edit)
        object_id = target["objectId"]
        original = {
            "objectId": object_id,
            "name": target["name"],
            "objectType": target["objectType"],
            "position": dict(target["position"]),
            "rotation": dict(target["rotation"]),
            "size": dict(target["axisAlignedBoundingBox"]["size"]),
        }
        record: Dict[str, Any] = {"requested": edit.to_dict(), "original": original}

        if edit.remove:
            self._step("DisableObject", objectId=object_id)
            record["achieved"] = {"removed": True}
            return record

        if edit.scale is not None:
            # forceAction bypasses the visibility/interactability gate; without
            # it ScaleObject only works on objects the agent could reach.
            self._step(
                "ScaleObject",
                objectId=object_id,
                scale=edit.scale,
                scaleOverSeconds=INSTANT,
                forceAction=True,
            )

        target_position = dict(original["position"])
        if edit.position:
            target_position = dict(edit.position)
        elif edit.position_delta:
            target_position = {
                k: original["position"][k] + float(edit.position_delta.get(k, 0.0))
                for k in ("x", "y", "z")
            }

        target_rotation = dict(original["rotation"])
        if edit.rotation:
            target_rotation = dict(edit.rotation)
        elif edit.rotation_delta:
            target_rotation = {
                k: (original["rotation"][k] + float(edit.rotation_delta.get(k, 0.0))) % 360.0
                for k in ("x", "y", "z")
            }

        # Always teleport, even for a scale-only edit: forceKinematic pins the
        # object so it cannot settle or drop, which is what keeps the edit a
        # clean controlled variable. forceAction ignores collision checks so a
        # scaled-up object stays exactly where it was asked to be.
        self._step(
            "TeleportObject",
            objectId=object_id,
            position=target_position,
            rotation=target_rotation,
            forceAction=True,
            forceKinematic=True,
        )

        after = self.resolve_object(ObjectEdit(object_id=object_id))
        record["achieved"] = {
            "position": dict(after["position"]),
            "rotation": dict(after["rotation"]),
            "size": dict(after["axisAlignedBoundingBox"]["size"]),
        }
        record["requested_pose"] = {"position": target_position, "rotation": target_rotation}

        if edit.scale is not None:
            record["achieved_scale_ratio"] = self._scale_ratio(
                original["size"], record["achieved"]["size"], edit.scale
            )
        return record

    @staticmethod
    def _scale_ratio(before: Dict[str, float], after: Dict[str, float], requested: float) -> Dict[str, float]:
        """Measured size ratio per axis, as a check that the scale really landed.

        An object whose bounding box is dominated by an articulated part (an
        open fridge door, a swung cabinet) can report a ratio that differs from
        ``requested`` even when the scale applied correctly, so this is logged
        rather than enforced.
        """
        ratio = {k: (after[k] / before[k] if before[k] > 1e-6 else float("nan")) for k in ("x", "y", "z")}
        worst = max(abs(v - requested) for v in ratio.values() if v == v)
        if worst > 0.05 * requested:
            log.warning(
                "requested scale %.3f but measured bbox ratio %s -- check the render",
                requested,
                {k: round(v, 3) for k, v in ratio.items()},
            )
        return ratio

    # ----------------------------------------------------------------- render

    def _instance_masks(self, event) -> Dict[str, np.ndarray]:
        if not self.render_instance_segmentation:
            return {}
        try:
            return dict(event.instance_masks)
        except Exception:  # segmentation not produced for this frame
            return {}

    def _frames(self, event, camera_mode: str):
        """Pull (rgb, depth, normals, instance masks) for the active camera."""
        if camera_mode == "thirdparty":
            rgb = event.third_party_camera_frames[0]
            depth = event.third_party_depth_frames[0] if self.render_depth else None
            normals = (
                event.third_party_normals_frames[0]
                if self.render_normals and event.third_party_normals_frames
                else None
            )
            masks = (
                dict(event.third_party_instance_masks[0])
                if self.render_instance_segmentation and event.third_party_instance_masks
                else {}
            )
            return rgb, depth, normals, masks
        return (
            event.frame,
            event.depth_frame if self.render_depth else None,
            getattr(event, "normals_frame", None) if self.render_normals else None,
            self._instance_masks(event),
        )

    def render(self, spec: SceneSpec, edit: Optional[ObjectEdit] = None) -> RenderResult:
        """Render ``spec``, optionally with one object edited.

        The scene is reset and restored from scratch on every call, so results
        do not depend on what was rendered before.
        """
        self.reset_scene(spec)

        edit_record = None
        if edit is not None and not edit.is_noop:
            edit_record = self._apply_edit(edit, spec.agent_mode)

        event = self.controller.step(action="Pass")
        rgb, depth, normals, masks = self._frames(event, spec.camera.mode)

        metadata = {
            "scene": spec.scene,
            "camera": spec.camera.to_dict(),
            "agent": event.metadata["agent"],
            "fov": event.metadata.get("fov"),
            "edit": edit_record,
            "objects": [
                {
                    "objectId": o["objectId"],
                    "name": o["name"],
                    "objectType": o["objectType"],
                    "position": o["position"],
                    "rotation": o["rotation"],
                    "size": o["axisAlignedBoundingBox"]["size"],
                    "visible": o["visible"],
                }
                for o in event.metadata["objects"]
            ],
        }
        return RenderResult(
            np.asarray(rgb), depth, masks, metadata,
            normals=np.asarray(normals) if normals is not None else None,
        )

    def render_pair(
        self, spec: SceneSpec, edit: ObjectEdit
    ) -> Tuple[RenderResult, RenderResult]:
        """Render the unedited scene and the edited scene under one setup."""
        return self.render(spec, edit=None), self.render(spec, edit=edit)

    # ---------------------------------------------------------------- cleanup

    def stop(self) -> None:
        try:
            self.controller.stop()
        except Exception:
            pass

    def restart(self) -> None:
        """Tear down and relaunch Unity with the same settings.

        Some actions can kill the build outright -- ``ScaleObject`` on a
        ``Painting`` is a known one -- after which every later step fails with
        "write to closed file". Batch jobs call this to recover instead of
        losing the rest of the run.
        """
        log.warning("restarting the AI2-THOR controller")
        self.stop()
        import ai2thor.controller

        self.controller = ai2thor.controller.Controller(**self._controller_args)
        self._current_scene = None
        self._set_object_poses_supported = True

    def __enter__(self) -> "ThorRenderer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
