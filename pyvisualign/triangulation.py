"""VisuAlign-compatible marker triangulation and plane warping.

Ports the incremental Delaunay triangulation and barycentric transform from
VisuAlign's data/Slice.java and nonlin/Triangle.java.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np


class Triangle:
    """Single triangle in display (marker columns 2/3) space with warp to atlas coords."""

    def __init__(self, a: int, b: int, c: int, points: List[List[float]]):
        p = sorted([a, b, c])
        self.a, self.b, self.c = p
        A, B, C = points[self.a], points[self.b], points[self.c]
        ax, ay = A[2], A[3]
        bx, by = B[2], B[3]
        cx, cy = C[2], C[3]
        self.minx = min(ax, bx, cx)
        self.maxx = max(ax, bx, cx)
        self.miny = min(ay, by, cy)
        self.maxy = max(ay, by, cy)
        self.A, self.B, self.C = A, B, C
        m = np.array([[bx - ax, cx - ax, ax], [by - ay, cy - ay, ay], [0.0, 0.0, 1.0]], dtype=float)
        if abs(np.linalg.det(m)) < 1e-15:
            self.decomp = None
        else:
            self.decomp = np.linalg.inv(m)
        a2 = (bx - cx) ** 2 + (by - cy) ** 2
        b2 = (ax - cx) ** 2 + (ay - cy) ** 2
        c2 = (ax - bx) ** 2 + (ay - by) ** 2
        fa = a2 * (b2 + c2 - a2)
        fb = b2 * (c2 + a2 - b2)
        fc = c2 * (a2 + b2 - c2)
        self.den = fa + fb + fc
        self.Mdenx = fa * ax + fb * bx + fc * cx
        self.Mdeny = fa * ay + fb * by + fc * cy
        self.r2den = (ax * self.den - self.Mdenx) ** 2 + (ay * self.den - self.Mdeny) ** 2

    def incirc(self, x: float, y: float) -> bool:
        return (x * self.den - self.Mdenx) ** 2 + (y * self.den - self.Mdeny) ** 2 < self.r2den

    def intri(self, x: float, y: float) -> Optional[np.ndarray]:
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return None
        if self.decomp is None:
            return None
        uv1 = self.decomp @ np.array([x, y, 1], dtype=float)
        if uv1[0] < 0 or uv1[0] > 1 or uv1[1] < 0 or uv1[1] > 1 or uv1[0] + uv1[1] > 1:
            return None
        return uv1

    def transform(self, x: float, y: float) -> Optional[np.ndarray]:
        uv1 = self.intri(x, y)
        if uv1 is None:
            return None
        A, B, C = self.A, self.B, self.C
        return np.array([
            A[0] + (B[0] - A[0]) * uv1[0] + (C[0] - A[0]) * uv1[1],
            A[1] + (B[1] - A[1]) * uv1[0] + (C[1] - A[1]) * uv1[1],
        ])


def _marker(x: float, y: float, nx: Optional[float] = None, ny: Optional[float] = None) -> List[float]:
    if nx is None:
        return [x, y, x, y]
    return [x, y, nx, ny]


def triangulate(markers: np.ndarray, width: float, height: float) -> Tuple[List[List[float]], List[Triangle]]:
    """Build VisuAlign's triangulation mesh including corner anchor points."""
    markers_list = [list(m) for m in markers]
    trimarkers: List[List[float]] = [
        _marker(-width * 0.1, -height * 0.1),
        _marker(width * 1.1, -height * 0.1),
        _marker(-width * 0.1, height * 1.1),
        _marker(width * 1.1, height * 1.1),
    ]
    triangles: List[Triangle] = [
        Triangle(0, 1, 2, trimarkers),
        Triangle(1, 2, 3, trimarkers),
    ]
    n = len(markers_list) + 4
    edges = np.zeros((n, n), dtype=np.int8)
    edges[0, 1] = edges[0, 2] = edges[1, 2] = edges[1, 3] = edges[2, 3] = 2

    for m in markers_list:
        x, y = m[2], m[3]
        found = False
        remove: List[Triangle] = []
        for tri in triangles:
            if found or tri.intri(x, y) is not None:
                found = True
            if tri.incirc(x, y):
                remove.append(tri)
        if found:
            for tri in remove:
                edges[tri.a, tri.b] -= 1
                edges[tri.a, tri.c] -= 1
                edges[tri.b, tri.c] -= 1
            for tri in remove:
                triangles.remove(tri)
            trimarkers.append(m)
            new_idx = len(trimarkers) - 1
            newtris: List[Triangle] = []
            for i in range(n):
                for j in range(n):
                    if edges[i, j] == 1:
                        tri = Triangle(i, j, new_idx, trimarkers)
                        if tri.decomp is not None:
                            newtris.append(tri)
            triangles.extend(newtris)
            for tri in newtris:
                edges[tri.a, tri.b] += 1
                edges[tri.a, tri.c] += 1
                edges[tri.b, tri.c] += 1

    return trimarkers, triangles


def warp_overlay(
    overlay: np.ndarray,
    triangles: List[Triangle],
    slice_width: float,
    slice_height: float,
) -> np.ndarray:
    """Backward-map overlay through triangulation (matches VisuAlign export)."""
    h, w = overlay.shape
    out = np.zeros((h, w), dtype=overlay.dtype)
    # Precompute display-space coordinates for each output pixel and flatten
    ys, xs = np.indices((h, w))
    fx = xs * slice_width / w
    fy = ys * slice_height / h
    flat_fx = fx.ravel()
    flat_fy = fy.ravel()
    out_flat = out.ravel()
    filled = np.zeros(out_flat.shape, dtype=bool)

    for tri in triangles:
        if tri.decomp is None:
            continue
        # Bounding-box filter to reduce points to test
        bbox = (flat_fx >= tri.minx) & (flat_fx <= tri.maxx) & (flat_fy >= tri.miny) & (flat_fy <= tri.maxy)
        to_check = bbox & (~filled)
        if not np.any(to_check):
            continue
        idxs = np.nonzero(to_check)[0]
        pts = np.vstack((flat_fx[idxs], flat_fy[idxs], np.ones(idxs.shape[0], dtype=float)))
        uv = tri.decomp @ pts
        u = uv[0, :]
        v = uv[1, :]
        inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & (u + v <= 1)
        if not np.any(inside):
            continue
        valid_idxs = idxs[inside]
        u = u[inside]
        v = v[inside]
        A, B, C = tri.A, tri.B, tri.C
        tx = A[0] + (B[0] - A[0]) * u + (C[0] - A[0]) * v
        ty = A[1] + (B[1] - A[1]) * u + (C[1] - A[1]) * v
        src_x = (tx * w / slice_width).astype(np.int64)
        src_y = (ty * h / slice_height).astype(np.int64)
        valid_bounds = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
        if not np.any(valid_bounds):
            continue
        final_idxs = valid_idxs[valid_bounds]
        sx = src_x[valid_bounds]
        sy = src_y[valid_bounds]
        out_flat[final_idxs] = overlay[sy, sx]
        filled[final_idxs] = True

    return out_flat.reshape(h, w)
