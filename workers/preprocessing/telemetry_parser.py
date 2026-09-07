"""
Flight Telemetry, GPS, Barometric Altitude Extractor & Time Synchronizer – TASK-020.
Parses DJI SRT subtitles, CSV, and JSON flight metadata.
Extracts georeferenced coordinates, barometric altitude, detects RTK/PPK corrections,
and interpolates trajectory fixes to synchronize with video frames.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union
from uuid import UUID

import numpy as np

# Ensure packages are resolvable
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
_APPS_API_DIR = _ROOT_DIR / "apps" / "api"
if str(_APPS_API_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_API_DIR))

from geoalchemy2.elements import WKTElement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models import Flight, FlightTelemetry

logger = logging.getLogger("preprocessing.telemetry_parser")


@dataclass
class ParsedTelemetryPoint:
    """Individual flight telemetry fix with optional barometric and RTK metrics."""

    timestamp_offset: float
    latitude: float
    longitude: float
    altitude_msl: float
    altitude_relative: Optional[float] = None
    altitude_barometric_m: Optional[float] = None
    heading: Optional[float] = None
    speed: Optional[float] = None
    gimbal_pitch: Optional[float] = None
    gimbal_roll: Optional[float] = None
    gimbal_yaw: Optional[float] = None
    hdop: Optional[float] = None
    vdop: Optional[float] = None
    has_rtk: bool = False
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFlightTelemetry:
    """Cohesive trajectory dataset with metadata indicators."""

    records: list[ParsedTelemetryPoint]
    has_rtk_corrections: bool
    has_barometric_altitude: bool
    duration_seconds: float
    sample_rate_hz: float


# ---------------------------------------------------------------------------
# SRT Timestamp Parsing Helper
# ---------------------------------------------------------------------------

def _parse_srt_timestamp_to_seconds(ts_str: str) -> float:
    """Converts '00:01:23,456' or '00:01:23.456' into elapsed seconds."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    return 0.0


# ---------------------------------------------------------------------------
# DJI SRT Parser
# ---------------------------------------------------------------------------

def parse_dji_srt(srt_content: str) -> ParsedFlightTelemetry:
    """
    Parses DJI subtitle files (both legacy comma/space separated and bracketed key-value formats).
    Extracts latitude, longitude, altitude, barometric altitude, heading, and RTK indicators.
    """
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    records: list[ParsedTelemetryPoint] = []
    has_rtk_global = False
    has_baro_global = False

    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})")

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        # Find timing line
        start_seconds = None
        text_lines: list[str] = []
        for line in lines:
            time_match = time_pattern.search(line)
            if time_match:
                start_seconds = _parse_srt_timestamp_to_seconds(time_match.group(1))
            elif not line.isdigit():
                text_lines.append(line)

        if start_seconds is None or not text_lines:
            continue

        text_content = " ".join(text_lines)

        # 1. Bracketed format: [latitude: 37.774929] [longitude: -122.419416] [altitude: 125.400] ...
        lat = None
        lon = None
        alt_msl = None
        alt_rel = None
        baro_alt = None
        heading = None
        speed = None
        g_pitch = None
        g_roll = None
        g_yaw = None
        is_rtk = False

        # Bracketed regex searches
        lat_m = re.search(r"\[latitude\s*:\s*([+-]?\d+\.?\d*)\]", text_content, re.IGNORECASE)
        lon_m = re.search(r"\[longitude\s*:\s*([+-]?\d+\.?\d*)\]", text_content, re.IGNORECASE)
        alt_m = re.search(r"\[altitude\s*:\s*([+-]?\d+\.?\d*)\]", text_content, re.IGNORECASE)
        rel_alt_m = re.search(r"\[rel_alt\s*:\s*([+-]?\d+\.?\d*)\]", text_content, re.IGNORECASE)
        baro_m = re.search(r"\[barometer\s*:\s*([+-]?\d+\.?\d*)\]", text_content, re.IGNORECASE)
        heading_m = re.search(r"\[(?:heading|yaw)\s*:\s*([+-]?\d+\.?\d*)\]", text_content, re.IGNORECASE)
        rtk_m = re.search(r"\[rtk_flag\s*:\s*([1-9]\d*)\]", text_content, re.IGNORECASE)

        if lat_m and lon_m:
            lat = float(lat_m.group(1))
            lon = float(lon_m.group(1))
            alt_msl = float(alt_m.group(1)) if alt_m else 0.0
            alt_rel = float(rel_alt_m.group(1)) if rel_alt_m else None
            if baro_m:
                baro_alt = float(baro_m.group(1))
                has_baro_global = True
            if heading_m:
                heading = float(heading_m.group(1)) % 360.0
            if rtk_m:
                is_rtk = True
                has_rtk_global = True

        else:
            # 2. Legacy DJI format: GPS(lon, lat, count) or GPS (lat, lon, count) | BAROMETER: 45.2m | RTK (1)
            gps_match = re.search(r"GPS\s*\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*(?:,\s*(\d+))?\)", text_content)
            if gps_match:
                v1 = float(gps_match.group(1))
                v2 = float(gps_match.group(2))
                # Latitude is within [-90, 90], Longitude is within [-180, 180]
                if abs(v1) <= 90.0 and abs(v2) <= 180.0:
                    lat, lon = v1, v2
                else:
                    lon, lat = v1, v2

            # BAROMETER: 45.2m
            baro_legacy = re.search(r"BAROMETER\s*:\s*([+-]?\d+\.?\d*)", text_content, re.IGNORECASE)
            if baro_legacy:
                baro_alt = float(baro_legacy.group(1))
                has_baro_global = True

            # Height H: 12.5m
            height_match = re.search(r"\bH\s*:?\s*([+-]?\d+\.?\d*)m", text_content)
            if height_match:
                alt_msl = float(height_match.group(1))

            # RTK indicator
            if re.search(r"RTK\s*\(\s*[1-9]\s*\)", text_content, re.IGNORECASE):
                is_rtk = True
                has_rtk_global = True

            # Speed H.S: 5.2m/s
            speed_match = re.search(r"H\.S\s*:?\s*([+-]?\d+\.?\d*)", text_content)
            if speed_match:
                speed = abs(float(speed_match.group(1)))

        if lat is not None and lon is not None and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            records.append(
                ParsedTelemetryPoint(
                    timestamp_offset=round(start_seconds, 3),
                    latitude=lat,
                    longitude=lon,
                    altitude_msl=alt_msl if alt_msl is not None else 0.0,
                    altitude_relative=alt_rel,
                    altitude_barometric_m=baro_alt,
                    heading=heading,
                    speed=speed,
                    gimbal_pitch=g_pitch,
                    gimbal_roll=g_roll,
                    gimbal_yaw=g_yaw,
                    has_rtk=is_rtk,
                    raw_data={"text": text_content[:200]},
                )
            )

    # Sort records by timestamp
    records.sort(key=lambda r: r.timestamp_offset)
    duration = records[-1].timestamp_offset - records[0].timestamp_offset if records else 0.0
    rate = len(records) / max(1.0, duration) if duration > 0 else 1.0

    return ParsedFlightTelemetry(
        records=records,
        has_rtk_corrections=has_rtk_global,
        has_barometric_altitude=has_baro_global,
        duration_seconds=round(duration, 2),
        sample_rate_hz=round(rate, 2),
    )


# ---------------------------------------------------------------------------
# CSV Telemetry Parser
# ---------------------------------------------------------------------------

def parse_csv_telemetry(csv_content: str) -> ParsedFlightTelemetry:
    """Parses tabular CSV telemetry files with header auto-detection."""
    lines = [l for l in csv_content.strip().splitlines() if l.strip()]
    if not lines:
        return ParsedFlightTelemetry([], False, False, 0.0, 0.0)

    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        return ParsedFlightTelemetry([], False, False, 0.0, 0.0)

    # Standardize column mapping
    col_map: dict[str, str] = {}
    for col in reader.fieldnames:
        c_low = col.lower().strip()
        if c_low in ("lat", "latitude", "gps_lat", "osd_latitude"):
            col_map["latitude"] = col
        elif c_low in ("lon", "lng", "longitude", "gps_lon", "osd_longitude"):
            col_map["longitude"] = col
        elif c_low in ("alt", "altitude", "altitude_msl", "height_msl", "gps_alt"):
            col_map["altitude_msl"] = col
        elif c_low in ("rel_alt", "altitude_rel", "relative_altitude", "height"):
            col_map["altitude_relative"] = col
        elif c_low in ("baro_alt", "barometer", "barometric_altitude", "altitude_barometric_m", "baro"):
            col_map["altitude_barometric_m"] = col
        elif c_low in ("heading", "yaw", "compass", "flight_yaw"):
            col_map["heading"] = col
        elif c_low in ("speed", "speed_mps", "groundspeed", "h_speed"):
            col_map["speed"] = col
        elif c_low in ("timestamp", "time", "timestamp_offset", "time_sec", "offset"):
            col_map["timestamp"] = col
        elif c_low in ("rtk", "rtk_status", "rtk_flag", "is_rtk", "rtk_health"):
            col_map["rtk"] = col
        elif c_low in ("hdop",):
            col_map["hdop"] = col
        elif c_low in ("vdop",):
            col_map["vdop"] = col

    if "latitude" not in col_map or "longitude" not in col_map:
        raise ValueError("CSV telemetry missing required 'latitude' and 'longitude' columns.")

    records: list[ParsedTelemetryPoint] = []
    has_rtk_global = False
    has_baro_global = False
    idx = 0

    for row in reader:
        try:
            lat = float(row[col_map["latitude"]])
            lon = float(row[col_map["longitude"]])
            alt_msl = float(row[col_map["altitude_msl"]]) if "altitude_msl" in col_map and row[col_map["altitude_msl"]] else 0.0
            alt_rel = float(row[col_map["altitude_relative"]]) if "altitude_relative" in col_map and row[col_map["altitude_relative"]] else None

            baro_alt = None
            if "altitude_barometric_m" in col_map and row[col_map["altitude_barometric_m"]]:
                baro_alt = float(row[col_map["altitude_barometric_m"]])
                has_baro_global = True

            heading = float(row[col_map["heading"]]) % 360.0 if "heading" in col_map and row[col_map["heading"]] else None
            speed = abs(float(row[col_map["speed"]])) if "speed" in col_map and row[col_map["speed"]] else None

            if "timestamp" in col_map and row[col_map["timestamp"]]:
                ts = float(row[col_map["timestamp"]])
            else:
                ts = float(idx)  # Fallback 1 Hz

            is_rtk = False
            if "rtk" in col_map and row[col_map["rtk"]]:
                val = str(row[col_map["rtk"]]).strip().lower()
                if val in ("1", "true", "yes", "fixed", "rtk_fixed", "float"):
                    is_rtk = True
                    has_rtk_global = True

            hdop = float(row[col_map["hdop"]]) if "hdop" in col_map and row[col_map["hdop"]] else None
            vdop = float(row[col_map["vdop"]]) if "vdop" in col_map and row[col_map["vdop"]] else None

            records.append(
                ParsedTelemetryPoint(
                    timestamp_offset=round(ts, 3),
                    latitude=lat,
                    longitude=lon,
                    altitude_msl=alt_msl,
                    altitude_relative=alt_rel,
                    altitude_barometric_m=baro_alt,
                    heading=heading,
                    speed=speed,
                    hdop=hdop,
                    vdop=vdop,
                    has_rtk=is_rtk,
                )
            )
            idx += 1
        except (ValueError, TypeError):
            continue

    records.sort(key=lambda r: r.timestamp_offset)
    duration = records[-1].timestamp_offset - records[0].timestamp_offset if records else 0.0
    rate = len(records) / max(1.0, duration) if duration > 0 else 1.0

    return ParsedFlightTelemetry(
        records=records,
        has_rtk_corrections=has_rtk_global,
        has_barometric_altitude=has_baro_global,
        duration_seconds=round(duration, 2),
        sample_rate_hz=round(rate, 2),
    )


# ---------------------------------------------------------------------------
# JSON Telemetry Parser
# ---------------------------------------------------------------------------

def parse_json_telemetry(json_content: Union[str, list, dict]) -> ParsedFlightTelemetry:
    """Parses JSON array or dictionary wrapper of telemetry points."""
    data = json.loads(json_content) if isinstance(json_content, str) else json_content

    if isinstance(data, dict):
        # Extract list from common envelope keys
        points = data.get("telemetry") or data.get("records") or data.get("points") or []
    elif isinstance(data, list):
        points = data
    else:
        raise ValueError("Invalid JSON telemetry structure: expected list or object containing array.")

    records: list[ParsedTelemetryPoint] = []
    has_rtk_global = False
    has_baro_global = False

    for idx, pt in enumerate(points):
        lat = pt.get("latitude") or pt.get("lat")
        lon = pt.get("longitude") or pt.get("lon") or pt.get("lng")
        if lat is None or lon is None:
            continue

        alt_msl = pt.get("altitude_msl") or pt.get("altitude") or pt.get("alt") or 0.0
        alt_rel = pt.get("altitude_relative") or pt.get("rel_alt")
        baro_alt = pt.get("altitude_barometric_m") or pt.get("baro_alt") or pt.get("baroAlt") or pt.get("barometer")
        if baro_alt is not None:
            has_baro_global = True

        heading = pt.get("heading") or pt.get("yaw")
        if heading is not None:
            heading = float(heading) % 360.0

        speed = pt.get("speed_mps") or pt.get("speed")
        ts = pt.get("timestamp_offset") or pt.get("timestamp") or float(idx)

        is_rtk = bool(pt.get("has_rtk") or pt.get("rtk") or pt.get("rtk_status") in ("FIXED", "FLOAT", 1))
        if is_rtk:
            has_rtk_global = True

        records.append(
            ParsedTelemetryPoint(
                timestamp_offset=round(float(ts), 3),
                latitude=float(lat),
                longitude=float(lon),
                altitude_msl=float(alt_msl),
                altitude_relative=float(alt_rel) if alt_rel is not None else None,
                altitude_barometric_m=float(baro_alt) if baro_alt is not None else None,
                heading=heading,
                speed=float(speed) if speed is not None else None,
                hdop=float(pt.get("hdop")) if pt.get("hdop") is not None else None,
                vdop=float(pt.get("vdop")) if pt.get("vdop") is not None else None,
                has_rtk=is_rtk,
            )
        )

    records.sort(key=lambda r: r.timestamp_offset)
    duration = records[-1].timestamp_offset - records[0].timestamp_offset if records else 0.0
    rate = len(records) / max(1.0, duration) if duration > 0 else 1.0

    return ParsedFlightTelemetry(
        records=records,
        has_rtk_corrections=has_rtk_global,
        has_barometric_altitude=has_baro_global,
        duration_seconds=round(duration, 2),
        sample_rate_hz=round(rate, 2),
    )


# ---------------------------------------------------------------------------
# Telemetry Interpolator
# ---------------------------------------------------------------------------

def interpolate_telemetry_at_timestamps(
    records: list[ParsedTelemetryPoint],
    target_timestamps: list[float],
) -> list[ParsedTelemetryPoint]:
    """
    Interpolates flight trajectory coordinates at target frame timestamps.
    Performs linear interpolation on coordinates, altitudes, and speeds,
    and angular unwrapped interpolation on compass heading.
    """
    if not records:
        return []

    if len(records) == 1:
        # Replicate single fix across all target timestamps
        single = records[0]
        return [
            ParsedTelemetryPoint(
                timestamp_offset=t,
                latitude=single.latitude,
                longitude=single.longitude,
                altitude_msl=single.altitude_msl,
                altitude_relative=single.altitude_relative,
                altitude_barometric_m=single.altitude_barometric_m,
                heading=single.heading,
                speed=single.speed,
                has_rtk=single.has_rtk,
            )
            for t in target_timestamps
        ]

    src_times = np.array([r.timestamp_offset for r in records])
    lats = np.array([r.latitude for r in records])
    lons = np.array([r.longitude for r in records])
    alt_msls = np.array([r.altitude_msl for r in records])

    has_baro = any(r.altitude_barometric_m is not None for r in records)
    if has_baro:
        valid_baro = [r.altitude_barometric_m if r.altitude_barometric_m is not None else 0.0 for r in records]
        baro_interp = np.interp(target_timestamps, src_times, valid_baro)
    else:
        baro_interp = None

    interp_lats = np.interp(target_timestamps, src_times, lats)
    interp_lons = np.interp(target_timestamps, src_times, lons)
    interp_alts = np.interp(target_timestamps, src_times, alt_msls)

    # Angular heading interpolation with phase unwrapping
    headings = [r.heading if r.heading is not None else 0.0 for r in records]
    unwrapped_rad = np.unwrap(np.radians(headings))
    interp_rad = np.interp(target_timestamps, src_times, unwrapped_rad)
    interp_headings = np.degrees(interp_rad) % 360.0

    interpolated_points: list[ParsedTelemetryPoint] = []
    for idx, t in enumerate(target_timestamps):
        interpolated_points.append(
            ParsedTelemetryPoint(
                timestamp_offset=round(float(t), 3),
                latitude=round(float(interp_lats[idx]), 7),
                longitude=round(float(interp_lons[idx]), 7),
                altitude_msl=round(float(interp_alts[idx]), 2),
                altitude_barometric_m=round(float(baro_interp[idx]), 2) if baro_interp is not None else None,
                heading=round(float(interp_headings[idx]), 2),
                has_rtk=records[0].has_rtk,
            )
        )

    return interpolated_points


# ---------------------------------------------------------------------------
# Database Persistence Helper
# ---------------------------------------------------------------------------

async def persist_flight_telemetry(
    flight_id: UUID,
    telemetry: ParsedFlightTelemetry,
    session: AsyncSession,
) -> int:
    """
    Inserts georeferenced flight_telemetry rows with 3D PostGIS Point geometry (POINTZ),
    and updates flight record flags for has_barometric_altitude and has_rtk_corrections.
    """
    # 1. Update flight flags
    flight_result = await session.execute(select(Flight).where(Flight.id == flight_id))
    flight = flight_result.scalar_one_or_none()
    if flight is not None:
        flight.has_barometric_altitude = telemetry.has_barometric_altitude
        flight.has_rtk_corrections = telemetry.has_rtk_corrections

    # 2. Insert telemetry records
    rows_added = 0
    for pt in telemetry.records:
        wkt_geom = f"POINT Z ({pt.longitude} {pt.latitude} {pt.altitude_msl})"
        row = FlightTelemetry(
            flight_id=flight_id,
            timestamp_offset=pt.timestamp_offset,
            location=WKTElement(wkt_geom, srid=4326),
            altitude_msl=pt.altitude_msl,
            altitude_barometric_m=pt.altitude_barometric_m,
            heading=pt.heading,
            speed=pt.speed,
            raw_json=pt.raw_data if pt.raw_data else None,
        )
        session.add(row)
        rows_added += 1

    await session.commit()
    logger.info("Persisted %d telemetry records for flight %s (RTK: %s, Baro: %s)", rows_added, flight_id, telemetry.has_rtk_corrections, telemetry.has_barometric_altitude)
    return rows_added
