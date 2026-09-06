"""Writing render results to disk and checking that the background held still."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .renderer import RenderResult


def save_render(
    result: RenderResult,
    out_dir: str,
    prefix: str = "",
    target_object_id: Optional[str] = None,
    save_depth: bool = True,
    save_normals: bool = True,
    save_edges: bool = True,
    canny: Optional[tuple] = None,
    canny_blur: int = 3,
    depth_scale: float = 1000.0,
) -> Dict[str, str]:
    """Write rgb / depth / normals / edges / target mask / metadata for one render.

    Args:
        canny: Canny thresholds to use for ``edges_canny.png``. Callers writing
            a *pair* must pass the same tuple for both halves -- see
            :func:`canny_edges`. If ``None``, thresholds are derived from this
            image alone, which is fine for a standalone render.
    """
    os.makedirs(out_dir, exist_ok=True)
    p = (lambda name: os.path.join(out_dir, f"{prefix}{name}")) if prefix else (
        lambda name: os.path.join(out_dir, name)
    )
    written: Dict[str, str] = {}

    Image.fromarray(result.rgb).save(p("rgb.png"))
    written["rgb"] = p("rgb.png")

    if save_depth and result.depth is not None:
        # .npy is the authoritative metric copy; the PNG is for viewing and for
        # tools that only speak images.
        np.save(p("depth.npy"), result.depth.astype(np.float32))
        written["depth"] = p("depth.npy")
        Image.fromarray(depth_to_png16(result.depth, depth_scale)).save(p("depth.png"))
        written["depth_png"] = p("depth.png")

    if save_normals and result.normals is not None:
        Image.fromarray(np.asarray(result.normals)[..., :3]).save(p("normals.png"))
        written["normals"] = p("normals.png")

    if save_edges:
        thresholds = canny or canny_thresholds(result.rgb)
        _save_bool(canny_edges(result.rgb, thresholds, canny_blur), p("edges_canny.png"))
        written["edges_canny"] = p("edges_canny.png")
        if result.depth is not None:
            _save_bool(
                geometric_edges(result.depth, result.normals), p("edges_geometric.png")
            )
            written["edges_geometric"] = p("edges_geometric.png")

    if target_object_id:
        mask = result.mask_for(target_object_id)
        Image.fromarray((mask * 255).astype(np.uint8)).save(p("target_mask.png"))
        written["target_mask"] = p("target_mask.png")

    with open(p("metadata.json"), "w") as f:
        json.dump(result.metadata, f, indent=2)
    written["metadata"] = p("metadata.json")
    return written


# --------------------------------------------------------------- derived maps

def depth_to_png16(depth: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """Quantise metric depth to a 16-bit image, in millimetres by default.

    The float32 .npy stays the authoritative copy; this is the viewable/portable
    form. 65535 mm caps at 65.5 m, far beyond any THOR room.
    """
    return np.clip(depth * scale, 0, 65535).astype(np.uint16)


def canny_thresholds(rgb: np.ndarray, sigma: float = 0.33) -> tuple:
    """Median-based automatic Canny thresholds for one image.

    Derive these from the *original* image and reuse the same numbers for the
    edited one -- see :func:`canny_edges`.
    """
    v = float(np.median(rgb))
    return (int(max(0, (1.0 - sigma) * v)), int(min(255, (1.0 + sigma) * v)))


def canny_edges(rgb: np.ndarray, thresholds: tuple, blur: int = 3) -> np.ndarray:
    """Canny appearance edges, using *externally supplied* thresholds.

    Thresholds are a parameter rather than computed per-image on purpose. Auto
    thresholds derived per-image would differ slightly between the original and
    the edited render (the edit changes the image median), which would flip
    edge pixels all over the untouched background and destroy the very property
    this pipeline exists to guarantee. Both halves of a pair must use one set.

    ``blur`` is a Gaussian pre-filter (odd kernel size, 0 to disable). It is the
    conventional first step of Canny and matters more than usual here: THOR's
    renderer is not bit-deterministic, and without it the anti-aliasing noise
    flips roughly twice as many threshold-straddling background pixels between
    the two halves of a pair. It reduces that flicker but does not remove it --
    use :func:`geometric_edges` when the background edges must match exactly.
    """
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur and blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    return cv2.Canny(gray, thresholds[0], thresholds[1]) > 0


def geometric_edges(
    depth: np.ndarray,
    normals: Optional[np.ndarray] = None,
    depth_rel_threshold: float = 0.02,
    normal_dot_threshold: float = 0.9,
) -> np.ndarray:
    """Structural edges from geometry alone -- no texture, no lighting.

    Two complementary cues: depth discontinuities catch occlusion boundaries
    (where one surface ends and a farther one begins), and normal
    discontinuities catch creases between surfaces that touch (a wall corner,
    the lid of a box), which depth alone misses because depth is continuous
    across them.

    The depth gradient is normalised by depth so a threshold means "a 2% jump
    relative to the distance", making it scale-invariant: a step at 4 m is
    judged the same way as the same relative step at 1 m.
    """
    gy, gx = np.gradient(depth.astype(np.float32))
    rel = np.hypot(gx, gy) / np.maximum(depth, 1e-3)
    edges = rel > depth_rel_threshold

    if normals is not None:
        n = normals[..., :3].astype(np.float32) / 127.5 - 1.0
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
        crease = np.zeros(depth.shape, dtype=bool)
        dot_x = (n[:, 1:] * n[:, :-1]).sum(-1)
        dot_y = (n[1:, :] * n[:-1, :]).sum(-1)
        crease[:, 1:] |= dot_x < normal_dot_threshold
        crease[1:, :] |= dot_y < normal_dot_threshold
        edges |= crease

    return edges


def _save_bool(mask: np.ndarray, path: str) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def background_report(
    original: RenderResult,
    edited: RenderResult,
    target_object_id: str,
    dilate: int = 3,
    tolerance: int = 4,
    shadow_radius: int = 60,
) -> Dict[str, Any]:
    """Quantify what changed outside the edited object.

    Two different things get measured, because they mean different things:

    ``near`` -- an annulus of ``shadow_radius`` pixels around the object. Real
    changes here are expected and desirable: a rescaled object casts a
    different shadow and bounces light differently. This is precisely what 2D
    compositing cannot produce, so nonzero drift here is a feature.

    ``far`` -- everything beyond that. This should be untouched, and nonzero
    drift here means something actually moved that should not have.

    THOR's renderer is not bit-deterministic (repeated renders of an identical
    scene differ by ~2 levels of 8-bit anti-aliasing noise), so "unchanged"
    means "differs by no more than ``tolerance``" rather than an exact match.
    """
    before = original.mask_for(target_object_id)
    after = edited.mask_for(target_object_id)
    changed = before | after
    if dilate > 0:
        changed = _dilate(changed, dilate)

    diff = np.abs(original.rgb.astype(np.int16) - edited.rgb.astype(np.int16)).max(axis=2)

    near_zone = _dilate(changed, shadow_radius) & ~changed
    far_zone = ~_dilate(changed, shadow_radius)

    def stats(zone: np.ndarray, name: str) -> Dict[str, Any]:
        vals = diff[zone]
        if not vals.size:
            return {f"{name}_pixels": 0, f"{name}_max_abs_diff": 0,
                    f"{name}_mean_abs_diff": 0.0, f"{name}_pixels_over_tolerance": 0}
        over = int((vals > tolerance).sum())
        return {
            f"{name}_pixels": int(vals.size),
            f"{name}_max_abs_diff": int(vals.max()),
            f"{name}_mean_abs_diff": float(vals.mean()),
            f"{name}_pixels_over_tolerance": over,
            f"{name}_fraction_over_tolerance": float(over / vals.size),
        }

    out: Dict[str, Any] = {
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "tolerance": tolerance,
        "shadow_radius": shadow_radius,
    }
    out.update(stats(near_zone, "near"))
    out.update(stats(far_zone, "far"))
    # The one assertion that must hold: nothing outside the object's
    # neighbourhood moved. Drift in the near zone is the object's own shadow.
    out["far_background_unchanged"] = bool(out["far_pixels_over_tolerance"] == 0)
    out["near_zone_changed"] = bool(out.get("near_pixels_over_tolerance", 0) > 0)
    return out


def save_pair(
    original: RenderResult,
    edited: RenderResult,
    out_dir: str,
    target_object_id: str,
    save_depth: bool = True,
    tolerance: int = 4,
    shadow_radius: int = 60,
    save_normals: bool = True,
    save_edges: bool = True,
    canny_blur: int = 3,
    depth_scale: float = 1000.0,
) -> Dict[str, Any]:
    """Write an (original, edited) pair plus the change mask and a QA report."""
    os.makedirs(out_dir, exist_ok=True)

    # One threshold pair, derived from the original, applied to both halves --
    # otherwise the edit shifts the image median and edges flicker across the
    # untouched background.
    thresholds = canny_thresholds(original.rgb) if save_edges else None
    common = dict(target_object_id=target_object_id, save_depth=save_depth,
                  save_normals=save_normals, save_edges=save_edges,
                  canny=thresholds, canny_blur=canny_blur, depth_scale=depth_scale)
    save_render(original, os.path.join(out_dir, "original"), **common)
    save_render(edited, os.path.join(out_dir, "edited"), **common)

    changed = original.mask_for(target_object_id) | edited.mask_for(target_object_id)
    Image.fromarray((changed * 255).astype(np.uint8)).save(os.path.join(out_dir, "changed_mask.png"))

    diff = np.abs(original.rgb.astype(np.int16) - edited.rgb.astype(np.int16)).max(axis=2)
    Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8)).save(
        os.path.join(out_dir, "diff.png")
    )

    report = background_report(
        original, edited, target_object_id, tolerance=tolerance, shadow_radius=shadow_radius
    )
    if save_edges:
        report["canny_thresholds"] = list(thresholds)
    if save_depth and original.depth is not None:
        report["depth_units"] = "metres (depth.npy); depth.png is uint16 at 1/%g m" % depth_scale
        report["depth_scale"] = depth_scale
    report["edit"] = edited.metadata.get("edit")
    report["scene"] = edited.metadata.get("scene")
    with open(os.path.join(out_dir, "pair_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by ``radius`` pixels.

    Uses OpenCV when available; the pure-numpy fallback is O(radius) passes and
    gets slow at the radii used for the shadow annulus.
    """
    if radius <= 0:
        return mask
    try:
        import cv2

        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    except ImportError:
        pass

    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2]
            | padded[1:-1, 2:] | padded[1:-1, 1:-1]
        )
    return out
