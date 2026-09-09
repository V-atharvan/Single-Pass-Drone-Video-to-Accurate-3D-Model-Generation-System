"""
Georeferencing Rigid Body Transformation Pipeline — TASK-052.

Transforms local 3D reconstruction coordinates into geographic coordinate systems:
  1. Sim3 7-parameter similarity transformation (scale s, rotation R, translation t)
     aligning local reconstruction camera trajectory with GPS/RTK ground references
     using the closed-form Umeyama algorithm.
  2. Exact WGS84 (EPSG:4326), ECEF (EPSG:4978), and Local East-North-Up (ENU)
     geodetic coordinate conversions.
  3. Computes 3D geographic bounding box (`min_lat`, `max_lat`, `min_lon`, `max_lon`,
     `min_alt`, `max_alt`) and PostGIS-compatible GeoJSON polygons.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

logger = logging.getLogger("geospatial.georeferencer")

# WGS84 Ellipsoid Constants
_WGS84_A = 6378137.0                   # semi-major axis (meters)
_WGS84_F = 1.0 / 298.257223563         # flattening
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F) # semi-minor axis
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F**2 # first eccentricity squared
_WGS84_E_PRIME2 = (_WGS84_A**2 - _WGS84_B**2) / _WGS84_B**2 # second eccentricity squared


@dataclass
class GeographicBoundingBox:
    """3D WGS84 Geographic Bounding Box."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_alt_m: float
    max_alt_m: float

    @property
    def center_lat(self) -> float:
        return 0.5 * (self.min_lat + self.max_lat)

    @property
    def center_lon(self) -> float:
        return 0.5 * (self.min_lon + self.max_lon)

    @property
    def center_alt(self) -> float:
        return 0.5 * (self.min_alt_m + self.max_alt_m)

    def to_geojson_polygon(self) -> Dict[str, Any]:
        """Returns GeoJSON Polygon geometry compatible with PostGIS."""
        coords = [
            [self.min_lon, self.min_lat],
            [self.max_lon, self.min_lat],
            [self.max_lon, self.max_lat],
            [self.min_lon, self.max_lat],
            [self.min_lon, self.min_lat],  # Closed ring
        ]
        return {
            "type": "Polygon",
            "coordinates": [coords],
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Sim3Transform:
    """
    7-parameter similarity transformation.
    X_target = s * R @ X_src + t
    """

    scale: float
    R: np.ndarray                      # (3, 3) rotation matrix
    t: np.ndarray                      # (3,) translation vector
    origin_wgs84: Tuple[float, float, float] # (lat, lon, alt_m)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Applies Sim3 transformation to an (N, 3) point array."""
        if len(points) == 0:
            return points.copy()
        return (self.scale * (points @ self.R.T)) + self.t.flatten()


def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Converts WGS84 (lat, lon, alt) to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_phi**2)

    X = (N + alt_m) * cos_phi * cos_lam
    Y = (N + alt_m) * cos_phi * sin_lam
    Z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_phi

    return float(X), float(Y), float(Z)


def ecef_to_wgs84(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """Converts ECEF X, Y, Z to WGS84 (lat_deg, lon_deg, alt_m) using Bowring's method."""
    p = math.sqrt(X**2 + Y**2)
    if p < 1e-6:
        # Near poles
        lat = 90.0 if Z >= 0 else -90.0
        lon = 0.0
        alt = abs(Z) - _WGS84_B
        return lat, lon, float(alt)

    theta = math.atan2(Z * _WGS84_A, p * _WGS84_B)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    phi = math.atan2(
        Z + _WGS84_E_PRIME2 * _WGS84_B * sin_theta**3,
        p - _WGS84_E2 * _WGS84_A * cos_theta**3,
    )
    lam = math.atan2(Y, X)

    sin_phi = math.sin(phi)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_phi**2)
    alt = p / math.cos(phi) - N

    return float(math.degrees(phi)), float(math.degrees(lam)), float(alt)


def wgs84_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    origin_wgs84: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Converts WGS84 point to local East-North-Up (ENU) coordinates relative to origin."""
    x, y, z = wgs84_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = wgs84_to_ecef(origin_wgs84[0], origin_wgs84[1], origin_wgs84[2])

    dx = x - x0
    dy = y - y0
    dz = z - z0

    phi0 = math.radians(origin_wgs84[0])
    lam0 = math.radians(origin_wgs84[1])

    sin_phi = math.sin(phi0)
    cos_phi = math.cos(phi0)
    sin_lam = math.sin(lam0)
    cos_lam = math.cos(lam0)

    east = -sin_lam * dx + cos_lam * dy
    north = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    up = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return float(east), float(north), float(up)


def enu_to_wgs84(
    east: float,
    north: float,
    up: float,
    origin_wgs84: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Converts local ENU coordinates back to WGS84 (lat, lon, alt)."""
    x0, y0, z0 = wgs84_to_ecef(origin_wgs84[0], origin_wgs84[1], origin_wgs84[2])

    phi0 = math.radians(origin_wgs84[0])
    lam0 = math.radians(origin_wgs84[1])

    sin_phi = math.sin(phi0)
    cos_phi = math.cos(phi0)
    sin_lam = math.sin(lam0)
    cos_lam = math.cos(lam0)

    # Inverse rotation matrix (transpose)
    dx = -sin_lam * east - sin_phi * cos_lam * north + cos_phi * cos_lam * up
    dy = cos_lam * east - sin_phi * sin_lam * north + cos_phi * sin_lam * up
    dz = cos_phi * north + sin_phi * up

    return ecef_to_wgs84(x0 + dx, y0 + dy, z0 + dz)


def estimate_sim3_umeyama(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes optimal similarity transformation (scale, R, t) aligning src_points to dst_points.
    Minimizes: sum || s * R * src_i + t - dst_i ||^2 using Umeyama SVD.
    """
    n, dim = src_points.shape
    if n < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    src_var = np.mean(np.sum(src_centered**2, axis=1))

    # Covariance matrix H
    H = (dst_centered.T @ src_centered) / n

    U, S, Vt = np.linalg.svd(H)

    # Ensure right-handed coordinate system
    d = np.linalg.det(U) * np.linalg.det(Vt)
    S_mat = np.eye(dim, dtype=np.float64)
    if d < 0:
        S_mat[-1, -1] = -1.0

    R = U @ S_mat @ Vt
    scale = (np.trace(S_mat @ np.diag(S))) / max(1e-8, src_var)
    t = dst_mean - scale * (R @ src_mean)

    return float(scale), R.astype(np.float64), t.astype(np.float64)


class Georeferencer:
    """
    Aligns local 3D models with real-world geospatial telemetry and transforms coordinates.
    """

    def align_to_gps(
        self,
        local_camera_positions: np.ndarray,
        gps_wgs84_coords: Sequence[Tuple[float, float, float]],
    ) -> Sim3Transform:
        """
        Estimates Sim3 transformation aligning local reconstruction trajectory with GPS track.
        """
        n = min(len(local_camera_positions), len(gps_wgs84_coords))
        if n < 3:
            origin = gps_wgs84_coords[0] if gps_wgs84_coords else (0.0, 0.0, 0.0)
            return Sim3Transform(1.0, np.eye(3), np.zeros(3), origin)

        origin = gps_wgs84_coords[0]

        # Convert GPS coordinates to local ENU relative to first point
        enu_dst = np.array([
            wgs84_to_enu(lat, lon, alt, origin)
            for lat, lon, alt in gps_wgs84_coords[:n]
        ], dtype=np.float64)

        src = local_camera_positions[:n].astype(np.float64)

        scale, R, t = estimate_sim3_umeyama(src, enu_dst)

        return Sim3Transform(
            scale=scale,
            R=R,
            t=t,
            origin_wgs84=origin,
        )

    def compute_geographic_bbox(
        self,
        local_points: np.ndarray,
        sim3: Sim3Transform,
    ) -> GeographicBoundingBox:
        """
        Transforms points to ENU and converts bounding vertices to WGS84 bounding box.
        """
        if len(local_points) == 0:
            lat, lon, alt = sim3.origin_wgs84
            return GeographicBoundingBox(lat, lat, lon, lon, alt, alt)

        enu_pts = sim3.transform_points(local_points)

        e_min, n_min, u_min = np.min(enu_pts, axis=0)
        e_max, n_max, u_max = np.max(enu_pts, axis=0)

        # Convert corners to WGS84
        c1 = enu_to_wgs84(e_min, n_min, u_min, sim3.origin_wgs84)
        c2 = enu_to_wgs84(e_max, n_max, u_max, sim3.origin_wgs84)

        min_lat = min(c1[0], c2[0])
        max_lat = max(c1[0], c2[0])
        min_lon = min(c1[1], c2[1])
        max_lon = max(c1[1], c2[1])
        min_alt = min(c1[2], c2[2])
        max_alt = max(c1[2], c2[2])

        return GeographicBoundingBox(
            min_lat=float(min_lat),
            max_lat=float(max_lat),
            min_lon=float(min_lon),
            max_lon=float(max_lon),
            min_alt_m=float(min_alt),
            max_alt_m=float(max_alt),
        )
