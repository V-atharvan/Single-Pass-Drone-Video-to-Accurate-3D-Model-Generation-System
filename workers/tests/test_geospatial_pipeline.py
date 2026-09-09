"""
Unit Tests for Phase 9: Georeferencing, Accuracy Assessment, DEM/DSM, 3D Tiling, and Job Finalization.
Covers TASK-052 through TASK-056.
"""
from __future__ import annotations

import json
import math
import struct
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from packages.schemas.python.single_pass_schemas import (
    AccuracyReport,
    ConfidenceLevel,
    PositioningMode,
    ScaleSource,
)
from workers.geospatial.accuracy_evaluator import (
    BENCHMARK_HORIZONTAL_RMSE_TARGET_M,
    BENCHMARK_VERTICAL_RMSE_TARGET_M,
    AccuracyEvaluator,
    MeasurementUncertaintyMetrics,
)
from workers.geospatial.dem_rasterizer import (
    NODATA_VALUE,
    DemRasterizer,
    encode_geotiff_float32,
)
from workers.geospatial.georeferencer import (
    GeographicBoundingBox,
    Georeferencer,
    Sim3Transform,
    ecef_to_wgs84,
    enu_to_wgs84,
    estimate_sim3_umeyama,
    wgs84_to_ecef,
    wgs84_to_enu,
)
from workers.geospatial.tileset_generator import (
    TilesetGenerator,
    compute_geographic_region_radians,
)


# ─── TASK-052: Georeferencer ───────────────────────────────────────────────

class TestWGS84ECEFConversion:
    """Validate WGS84 ↔ ECEF round-trips to 1mm precision."""

    def test_equator_prime_meridian(self):
        x, y, z = wgs84_to_ecef(0.0, 0.0, 0.0)
        lat, lon, alt = ecef_to_wgs84(x, y, z)
        assert abs(lat) < 1e-6
        assert abs(lon) < 1e-6
        assert abs(alt) < 0.01  # 1cm

    def test_north_pole(self):
        x, y, z = wgs84_to_ecef(90.0, 0.0, 0.0)
        lat, lon, alt = ecef_to_wgs84(x, y, z)
        assert abs(lat - 90.0) < 1e-4

    def test_arbitrary_point_round_trip(self):
        lat_in, lon_in, alt_in = 28.6139, 77.2090, 250.0  # New Delhi
        x, y, z = wgs84_to_ecef(lat_in, lon_in, alt_in)
        lat_out, lon_out, alt_out = ecef_to_wgs84(x, y, z)
        assert abs(lat_out - lat_in) < 1e-7
        assert abs(lon_out - lon_in) < 1e-7
        assert abs(alt_out - alt_in) < 0.001  # 1mm

    def test_southern_hemisphere_round_trip(self):
        lat_in, lon_in, alt_in = -33.8688, 151.2093, 30.0  # Sydney
        x, y, z = wgs84_to_ecef(lat_in, lon_in, alt_in)
        lat_out, lon_out, alt_out = ecef_to_wgs84(x, y, z)
        assert abs(lat_out - lat_in) < 1e-7
        assert abs(lon_out - lon_in) < 1e-7
        assert abs(alt_out - alt_in) < 0.001


class TestENUConversion:
    """Validate WGS84 ↔ Local ENU conversions."""

    def test_enu_origin_is_zero(self):
        origin = (28.6139, 77.2090, 0.0)
        e, n, u = wgs84_to_enu(28.6139, 77.2090, 0.0, origin)
        assert abs(e) < 0.01
        assert abs(n) < 0.01
        assert abs(u) < 0.01

    def test_enu_east_direction(self):
        origin = (0.0, 0.0, 0.0)
        # Moving ~1m east at the equator = ~9e-6 degrees longitude
        e, n, u = wgs84_to_enu(0.0, 9e-6, 0.0, origin)
        assert e > 0.5
        assert abs(n) < 0.5

    def test_enu_up_direction(self):
        origin = (28.0, 77.0, 0.0)
        e, n, u = wgs84_to_enu(28.0, 77.0, 100.0, origin)
        assert u > 95.0

    def test_enu_round_trip(self):
        origin = (20.0, 80.0, 100.0)
        target = (20.001, 80.001, 200.0)
        e, n, u = wgs84_to_enu(*target, origin)
        lat_r, lon_r, alt_r = enu_to_wgs84(e, n, u, origin)
        assert abs(lat_r - target[0]) < 1e-6
        assert abs(lon_r - target[1]) < 1e-6
        assert abs(alt_r - target[2]) < 0.05


class TestUmeyamaSim3:
    """Validate closed-form Umeyama Sim3 alignment."""

    def test_identity_alignment(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        s, R, t = estimate_sim3_umeyama(pts, pts)
        assert abs(s - 1.0) < 0.01
        assert np.allclose(R, np.eye(3), atol=0.01)
        assert np.allclose(t, [0, 0, 0], atol=0.01)

    def test_translation_only(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        offset = np.array([5.0, -3.0, 10.0])
        dst = src + offset
        s, R, t = estimate_sim3_umeyama(src, dst)
        assert abs(s - 1.0) < 0.05
        assert np.allclose(t, offset, atol=0.1)

    def test_scale_only(self):
        src = np.random.default_rng(42).random((10, 3))
        scale_true = 2.5
        dst = src * scale_true
        s, R, t = estimate_sim3_umeyama(src, dst)
        assert abs(s - scale_true) < 0.15

    def test_sparse_input_fallback(self):
        src = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        dst = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64)
        s, R, t = estimate_sim3_umeyama(src, dst)
        assert isinstance(s, float)


class TestGeoreferencerAlign:
    """Validate Georeferencer.align_to_gps and compute_geographic_bbox."""

    def test_georeferencer_align_minimal(self):
        g = Georeferencer()
        local_pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]], dtype=np.float64)
        gps_coords = [(28.0, 77.0, 100.0), (28.0001, 77.0, 100.0), (28.0, 77.0001, 100.0), (28.0001, 77.0001, 100.0)]
        sim3 = g.align_to_gps(local_pts, gps_coords)
        assert sim3.scale > 0.0
        assert sim3.R.shape == (3, 3)
        assert sim3.origin_wgs84 == gps_coords[0]

    def test_compute_geographic_bbox(self):
        g = Georeferencer()
        gps_coords = [(28.0, 77.0, 100.0), (28.01, 77.0, 100.0), (28.0, 77.01, 100.0), (28.01, 77.01, 100.0)]
        local_pts = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [100, 100, 0]], dtype=np.float64)
        sim3 = g.align_to_gps(local_pts, gps_coords)
        bbox = g.compute_geographic_bbox(local_pts, sim3)
        assert bbox.min_lat < bbox.max_lat
        assert bbox.min_lon < bbox.max_lon

    def test_geojson_polygon_closed(self):
        bbox = GeographicBoundingBox(min_lat=10.0, max_lat=11.0, min_lon=20.0, max_lon=21.0, min_alt_m=0.0, max_alt_m=100.0)
        geo = bbox.to_geojson_polygon()
        coords = geo["coordinates"][0]
        assert coords[0] == coords[-1], "GeoJSON polygon must be closed"


# ─── TASK-053: Accuracy Evaluator ─────────────────────────────────────────

class TestAbsoluteRMSE:
    """Validate horizontal and vertical RMSE formulas."""

    def test_zero_error(self):
        pts = np.array([[0, 0, 0], [10, 5, 2], [3, 7, 1]], dtype=np.float64)
        ev = AccuracyEvaluator()
        result = ev.compute_absolute_rmse(pts, pts)
        assert result.horizontal_rmse_m < 1e-9
        assert result.vertical_rmse_m < 1e-9
        assert result.sample_count == 3

    def test_known_error(self):
        est = np.array([[0, 0, 0]], dtype=np.float64)
        ref = np.array([[0.3, 0.4, 0.6]], dtype=np.float64)  # h_err = 0.5, v_err = 0.6
        ev = AccuracyEvaluator()
        result = ev.compute_absolute_rmse(est, ref)
        assert abs(result.horizontal_rmse_m - 0.5) < 1e-6
        assert abs(result.vertical_rmse_m - 0.6) < 1e-6

    def test_empty_input(self):
        ev = AccuracyEvaluator()
        result = ev.compute_absolute_rmse(np.empty((0, 3)), np.empty((0, 3)))
        assert result.sample_count == 0
        assert result.is_benchmark_horizontal_achieved is True

    def test_benchmark_flags(self):
        ev = AccuracyEvaluator()
        est = np.array([[0, 0, 0]], dtype=np.float64)
        ref_good = np.array([[0.2, 0.1, 0.4]], dtype=np.float64)
        ref_bad = np.array([[1.0, 1.5, 2.0]], dtype=np.float64)
        r_good = ev.compute_absolute_rmse(est, ref_good)
        r_bad = ev.compute_absolute_rmse(est, ref_bad)
        assert r_good.is_benchmark_horizontal_achieved is True
        assert r_bad.is_benchmark_horizontal_achieved is False


class TestPositioningMode:
    """Validate positioning mode determination."""

    def test_rtk_mode(self):
        ev = AccuracyEvaluator()
        mode, scale = ev.determine_positioning_mode(has_rtk=True)
        assert mode == PositioningMode.RTK
        assert scale == ScaleSource.RTK

    def test_visual_only_mode(self):
        ev = AccuracyEvaluator()
        mode, scale = ev.determine_positioning_mode(is_visual_only=True)
        assert mode == PositioningMode.VISUAL_ONLY
        assert scale == ScaleSource.VISUAL_ONLY

    def test_degraded_gps_mode(self):
        ev = AccuracyEvaluator()
        mode, scale = ev.determine_positioning_mode(gps_hdop=5.5, has_baro=True)
        assert mode == PositioningMode.GPS_DEGRADED
        assert scale == ScaleSource.BAROMETRIC

    def test_gps_imu_mode(self):
        ev = AccuracyEvaluator()
        mode, scale = ev.determine_positioning_mode(has_imu=True, has_baro=False)
        assert mode == PositioningMode.GPS_IMU
        assert scale == ScaleSource.GPS_BASELINE


class TestMeasurementUncertainty:
    """Validate measurement error propagation."""

    def test_high_confidence_within_benchmark(self):
        ev = AccuracyEvaluator()
        result = ev.propagate_measurement_uncertainty(
            ConfidenceLevel.HIGH, ConfidenceLevel.HIGH,
            observed_pct=90.0, inferred_pct=5.0
        )
        assert result.estimated_measurement_error_pct <= 2.5
        assert result.requires_measurement_warning is False

    def test_inferred_surface_triggers_warning(self):
        ev = AccuracyEvaluator()
        result = ev.propagate_measurement_uncertainty(
            ConfidenceLevel.MEDIUM, ConfidenceLevel.MEDIUM,
            observed_pct=60.0, inferred_pct=25.0
        )
        assert result.requires_measurement_warning is True


class TestFullEvaluate:
    """Integration test: full evaluate() produces a valid AccuracyReport."""

    def test_evaluate_gps_imu_mode(self):
        ev = AccuracyEvaluator()
        rng = np.random.default_rng(42)
        est = rng.random((20, 3)) * 100
        noise = rng.standard_normal((20, 3)) * 0.3
        ref = est + noise

        result = ev.evaluate(
            estimated_positions_enu=est,
            reference_positions_enu=ref,
            observed_surface_pct=85.0,
            partially_observed_pct=10.0,
            ai_inferred_pct=5.0,
        )
        report = result.report
        assert 0.0 <= report.overall_quality_score <= 100.0
        assert report.horizontal_rmse_meters >= 0.0
        assert report.positioning_mode == PositioningMode.GPS_IMU
        assert report.scale_source == ScaleSource.GPS_BASELINE
        assert isinstance(report.estimated_horizontal_uncertainty_m, float)

    def test_accuracy_report_never_no_fabricated_positioning_mode(self):
        ev = AccuracyEvaluator()
        est = np.zeros((5, 3))
        ref = np.ones((5, 3)) * 0.1
        result = ev.evaluate(est, ref, has_rtk=True)
        assert result.report.positioning_mode == PositioningMode.RTK
        assert result.report.scale_source == ScaleSource.RTK

    def test_to_dict_has_accuracy_notice(self):
        ev = AccuracyEvaluator()
        est = np.zeros((3, 3))
        ref = np.array([[0.1, 0.1, 0.2], [0.2, 0.1, 0.3], [0.05, 0.05, 0.1]])
        result = ev.evaluate(est, ref)
        d = result.to_dict()
        assert "accuracy_notice" in d
        assert "MEASURED" in d["accuracy_notice"]


# ─── TASK-054: DEM / DSM GeoTIFF ──────────────────────────────────────────

class TestGeoTIFFBinaryWriter:
    """Validate the pure-Python GeoTIFF writer header and data structure."""

    def test_tiff_magic_bytes(self):
        raster = np.full((4, 4), 100.0, dtype=np.float32)
        tiff_bytes = encode_geotiff_float32(raster, 77.0, 28.0, 0.1, -0.1, epsg_code=4326)
        # 'II' = little-endian, 0x002A = TIFF magic
        assert tiff_bytes[:2] == b"II"
        assert tiff_bytes[2:4] == b"\x2a\x00"

    def test_tiff_contains_elevation_values(self):
        elevation = 250.75
        raster = np.full((2, 2), elevation, dtype=np.float32)
        tiff_bytes = encode_geotiff_float32(raster, 0.0, 0.0, 1.0, -1.0)
        # Verify elevation value appears in binary
        packed = struct.pack("<f", elevation)
        assert packed in tiff_bytes

    def test_tiff_nodata_value(self):
        raster = np.array([[100.0, NODATA_VALUE], [200.0, NODATA_VALUE]], dtype=np.float32)
        tiff_bytes = encode_geotiff_float32(raster, 0.0, 0.0, 1.0, -1.0)
        assert len(tiff_bytes) > 8


class TestDemRasterizer:
    """Validate DemRasterizer.rasterize() and generate_dem_and_dsm()."""

    def _make_terrain_points(self, n: int = 100, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        z = rng.uniform(10, 30, n)
        return np.column_stack([x, y, z])

    def test_dem_shape(self):
        pts = self._make_terrain_points()
        dr = DemRasterizer(default_resolution_m=5.0)
        grid, ox, oy, px, py = dr.rasterize(pts, is_dem=True, fill_voids=False)
        assert grid.ndim == 2
        assert grid.shape[0] > 0 and grid.shape[1] > 0

    def test_dem_has_valid_elevations(self):
        pts = self._make_terrain_points()
        dr = DemRasterizer(default_resolution_m=5.0)
        grid, _, _, _, _ = dr.rasterize(pts, is_dem=True, fill_voids=True)
        valid = grid[grid != NODATA_VALUE]
        assert len(valid) > 0
        assert np.all(valid >= 5.0)

    def test_dsm_higher_than_dem(self):
        """DSM should generally have equal or higher values than DEM for the same region."""
        pts = self._make_terrain_points()
        dr = DemRasterizer(default_resolution_m=5.0)
        dem, ox, oy, px, py = dr.rasterize(pts, is_dem=True, fill_voids=True)
        dsm, _, _, _, _ = dr.rasterize(pts, is_dem=False, fill_voids=True)
        valid_mask = (dem != NODATA_VALUE) & (dsm != NODATA_VALUE)
        assert np.all(dsm[valid_mask] >= dem[valid_mask] - 0.1)

    def test_generate_dem_and_dsm_files(self):
        pts = self._make_terrain_points()
        dr = DemRasterizer(default_resolution_m=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            dem_meta, dsm_meta = dr.generate_dem_and_dsm(pts, output_dir=tmp, resolution_m=5.0)
            assert Path(dem_meta.file_path).exists()
            assert Path(dsm_meta.file_path).exists()
            dem_bytes = Path(dem_meta.file_path).read_bytes()
            assert dem_bytes[:2] == b"II"
            assert dem_meta.raster_type == "DEM"
            assert dsm_meta.raster_type == "DSM"
            assert dem_meta.valid_pixel_count > 0

    def test_empty_points_fallback(self):
        dr = DemRasterizer()
        pts = np.empty((0, 3))
        grid, *_ = dr.rasterize(pts, is_dem=True)
        assert grid.shape == (10, 10)


# ─── TASK-055: OGC 3D Tiles ───────────────────────────────────────────────

class TestTilesetGenerator:
    """Validate OGC 3D Tiles 1.1 tileset.json conformance."""

    def _make_mesh(self, n_verts: int = 120, n_faces: int = 40):
        from workers.mesh.mesh_reconstructor import TriangleMesh
        rng = np.random.default_rng(99)
        verts = rng.random((n_verts, 3)) * 100.0
        faces = rng.integers(0, n_verts, (n_faces, 3))
        normals = np.zeros_like(verts)
        normals[:, 2] = 1.0
        mesh = TriangleMesh(vertices=verts, faces=faces, vertex_normals=normals)
        return mesh

    def test_tileset_json_asset_version(self):
        from workers.geospatial.georeferencer import Sim3Transform
        mesh = self._make_mesh()
        sim3 = Sim3Transform(scale=1.0, R=np.eye(3), t=np.zeros(3), origin_wgs84=(28.0, 77.0, 100.0))
        gen = TilesetGenerator()
        with tempfile.TemporaryDirectory() as tmp:
            meta = gen.generate_tileset(mesh, sim3, output_dir=tmp)
            tileset = json.loads(Path(meta.tileset_json_path).read_text())
            assert tileset["asset"]["version"] == "1.1"

    def test_tileset_json_has_root_bounding_region(self):
        from workers.geospatial.georeferencer import Sim3Transform
        mesh = self._make_mesh()
        sim3 = Sim3Transform(scale=1.0, R=np.eye(3), t=np.zeros(3), origin_wgs84=(20.0, 78.0, 50.0))
        gen = TilesetGenerator()
        with tempfile.TemporaryDirectory() as tmp:
            meta = gen.generate_tileset(mesh, sim3, output_dir=tmp)
            tileset = json.loads(Path(meta.tileset_json_path).read_text())
            bv = tileset["root"]["boundingVolume"]
            assert "region" in bv
            region = bv["region"]
            assert len(region) == 6  # [west, south, east, north, min_h, max_h]

    def test_tile_glb_files_written(self):
        from workers.geospatial.georeferencer import Sim3Transform
        mesh = self._make_mesh()
        sim3 = Sim3Transform(scale=1.0, R=np.eye(3), t=np.zeros(3), origin_wgs84=(28.0, 77.0, 100.0))
        gen = TilesetGenerator()
        with tempfile.TemporaryDirectory() as tmp:
            meta = gen.generate_tileset(mesh, sim3, output_dir=tmp)
            root_glb = Path(tmp) / "tile_root.glb"
            assert root_glb.exists()
            assert meta.tile_count >= 1

    def test_geographic_region_radians(self):
        origin = (28.0, 77.0, 0.0)
        region = compute_geographic_region_radians([0, 0, 0], [100, 100, 50], origin)
        assert len(region) == 6
        west, south, east, north, min_h, max_h = region
        assert west < east
        assert south < north


# ─── TASK-056: Job Finalizer ──────────────────────────────────────────────

class TestJobFinalizer:
    """Validate JobFinalizer output contracts and PostGIS polygon format."""

    @pytest.mark.asyncio
    async def test_finalize_without_session(self):
        from apps.api.src.db.models import AssetType
        from workers.orchestrator.job_finalizer import AssetRegistrationInput, JobFinalizer

        job_id = uuid.uuid4()
        project_id = uuid.uuid4()
        bbox = GeographicBoundingBox(min_lat=28.0, max_lat=28.1, min_lon=77.0, max_lon=77.1, min_alt_m=100.0, max_alt_m=250.0)
        report = AccuracyReport(
            overall_quality_score=88.5,
            horizontal_rmse_meters=0.32,
            vertical_rmse_meters=0.48,
            coverage_percent=94.0,
            dynamic_contamination_percent=1.5,
            observed_surface_percent=83.0,
            partially_observed_percent=11.0,
            ai_inferred_percent=6.0,
        )
        assets = [
            AssetRegistrationInput(AssetType.GLB, "projects/p1/models/m1/model.glb", 5_000_000),
            AssetRegistrationInput(AssetType.DEM_TIF, "projects/p1/models/m1/dem.tif", 200_000),
        ]
        finalizer = JobFinalizer()
        result = await finalizer.finalize_job(
            job_id=job_id,
            project_id=project_id,
            bbox=bbox,
            accuracy_report=report,
            assets=assets,
            session=None,
        )
        assert result.status == "COMPLETED"
        assert result.asset_count == 2
        assert result.job_id == job_id

    def test_postgis_wkt_polygon_closed(self):
        from workers.orchestrator.job_finalizer import JobFinalizer
        bbox = GeographicBoundingBox(min_lat=10.0, max_lat=11.0, min_lon=20.0, max_lon=21.0, min_alt_m=0.0, max_alt_m=100.0)
        finalizer = JobFinalizer()
        wkt = finalizer.format_postgis_wkt_polygon(bbox)
        assert wkt.startswith("SRID=4326;POLYGON((")
        assert wkt.endswith("))")
        # Closed ring: first and last coordinate pair must match
        inner = wkt[len("SRID=4326;POLYGON(("):-2]
        points = [p.strip() for p in inner.split(",")]
        assert points[0] == points[-1]
