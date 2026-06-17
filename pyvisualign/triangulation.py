"""VisuAlign-compatible marker triangulation and plane warping.

Ports the incremental Delaunay triangulation and barycentric transform from
VisuAlign's data/Slice.java and nonlin/Triangle.java.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np


class _Matrix3:
    def __init__(self, a11, a21, a31, a12, a22, a32, a13, a23, a33):
        self.m = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], dtype=float)

    def inverse(self) -> Optional["_Matrix3"]:
        if abs(np.linalg.det(self.m)) < 1e-15:
            return None
        inv = np.linalg.inv(self.m)
        flat = inv.T.flatten()
        return _Matrix3(*flat)

    def rowmul(self, row: Sequence[float]) -> np.ndarray:
        return self.m @ np.array(row, dtype=float)


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
        self.decomp = _Matrix3(bx - ax, by - ay, 0, cx - ax, cy - ay, 0, ax, ay, 1).inverse()
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
        uv1 = self.decomp.rowmul([x, y, 1])
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
    for y in range(h):
        fy = y * slice_height / h
        for x in range(w):
            fx = x * slice_width / w
            for tri in triangles:
                t = tri.transform(fx, fy)
                if t is not None:
                    xx = int(t[0] * w / slice_width)
                    yy = int(t[1] * h / slice_height)
                    if 0 <= xx < w and 0 <= yy < h:
                        out[y, x] = overlay[yy, xx]
                    break
    return out
