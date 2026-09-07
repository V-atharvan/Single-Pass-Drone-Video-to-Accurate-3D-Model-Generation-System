"""
Unit and integration tests for Telemetry, Barometric Altitude & Time Synchronizer – TASK-020.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.api.src.db.models import Flight, FlightTelemetry
from workers.preprocessing.telemetry_parser import (
    ParsedFlightTelemetry,
    ParsedTelemetryPoint,
    interpolate_telemetry_at_timestamps,
    parse_csv_telemetry,
    parse_dji_srt,
    parse_json_telemetry,
    persist_flight_telemetry,
)

DJI_SRT_BRACKETED_SAMPLE = """
1
00:00:00,000 --> 00:00:01,000
HOME(122.4194,37.7749) 2026-09-08 12:00:00
GPS(122.4194, 37.7749, 15) BAROMETER:45.20 H: 45.20m, H.S: 5.2m/s, V.S: 0.1m/s, D: 120.0m
[latitude: 37.774929] [longitude: -122.419416] [altitude: 125.400] [rel_alt: 45.200] [barometer: 45.200] [heading: 180.500] [rtk_flag: 1]

2
00:00:01,000 --> 00:00:02,000
[latitude: 37.775029] [longitude: -122.419316] [altitude: 126.800] [rel_alt: 46.600] [barometer: 46.600] [heading: 182.000] [rtk_flag: 1]
"""

DJI_SRT_LEGACY_NO_BARO_SAMPLE = """
1
00:00:00,000 --> 00:00:01,000
GPS (37.774929, -122.419416, 12) H 12.5m | 2026-09-08 02:00:00

2
00:00:01,000 --> 00:00:02,000
GPS (37.775029, -122.419316, 12) H 13.0m | 2026-09-08 02:00:01
"""

CSV_SAMPLE_WITH_BARO_AND_RTK = """timestamp,latitude,longitude,altitude_msl,baro_alt,heading,speed,rtk
0.0,37.774929,-122.419416,125.4,45.2,180.5,5.2,FIXED
1.0,37.775029,-122.419316,126.8,46.6,182.0,5.4,FIXED
2.0,37.775129,-122.419216,128.2,48.0,183.5,5.1,FIXED
"""

JSON_SAMPLE = """[
    {"timestamp": 0.0, "latitude": 37.774929, "longitude": -122.419416, "altitude_msl": 125.4, "baro_alt": 45.2, "heading": 180.0, "speed": 5.0, "has_rtk": true},
    {"timestamp": 1.0, "latitude": 37.775029, "longitude": -122.419316, "altitude_msl": 126.8, "baro_alt": 46.6, "heading": 185.0, "speed": 5.2, "has_rtk": true}
]"""


def test_parse_dji_srt_with_baro_and_rtk():
    """Verify DJI SRT bracketed format correctly extracts barometric altitude and RTK flags."""
    parsed = parse_dji_srt(DJI_SRT_BRACKETED_SAMPLE)

    assert len(parsed.records) == 2
    assert parsed.has_barometric_altitude is True
    assert parsed.has_rtk_corrections is True

    pt1 = parsed.records[0]
    assert pt1.timestamp_offset == 0.0
    assert pt1.latitude == 37.774929
    assert pt1.longitude == -122.419416
    assert pt1.altitude_msl == 125.4
    assert pt1.altitude_barometric_m == 45.2
    assert pt1.heading == 180.5
    assert pt1.has_rtk is True


def test_parse_dji_srt_legacy_no_baro():
    """Verify legacy DJI SRT without barometer leaves altitude_barometric_m null and has_baro False."""
    parsed = parse_dji_srt(DJI_SRT_LEGACY_NO_BARO_SAMPLE)

    assert len(parsed.records) == 2
    assert parsed.has_barometric_altitude is False
    assert parsed.has_rtk_corrections is False

    pt1 = parsed.records[0]
    assert pt1.latitude == 37.774929
    assert pt1.longitude == -122.419416
    assert pt1.altitude_msl == 12.5
    assert pt1.altitude_barometric_m is None
    assert pt1.has_rtk is False


def test_parse_csv_telemetry():
    """Verify CSV telemetry parsing with barometric altitude and RTK status."""
    parsed = parse_csv_telemetry(CSV_SAMPLE_WITH_BARO_AND_RTK)

    assert len(parsed.records) == 3
    assert parsed.has_barometric_altitude is True
    assert parsed.has_rtk_corrections is True

    pt2 = parsed.records[1]
    assert pt2.timestamp_offset == 1.0
    assert pt2.latitude == 37.775029
    assert pt2.altitude_barometric_m == 46.6
    assert pt2.speed == 5.4


def test_parse_json_telemetry():
    """Verify JSON array parsing."""
    parsed = parse_json_telemetry(JSON_SAMPLE)

    assert len(parsed.records) == 2
    assert parsed.has_barometric_altitude is True
    assert parsed.has_rtk_corrections is True
    assert parsed.records[0].altitude_msl == 125.4


def test_interpolate_telemetry_at_timestamps():
    """Verify trajectory interpolation aligns with video frame timestamps."""
    records = [
        ParsedTelemetryPoint(
            timestamp_offset=0.0,
            latitude=37.0,
            longitude=-122.0,
            altitude_msl=100.0,
            altitude_barometric_m=20.0,
            heading=350.0,
        ),
        ParsedTelemetryPoint(
            timestamp_offset=2.0,
            latitude=37.0002,
            longitude=-122.0002,
            altitude_msl=104.0,
            altitude_barometric_m=24.0,
            heading=10.0,  # Crosses 360/0 degree meridian
        ),
    ]

    target_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    interpolated = interpolate_telemetry_at_timestamps(records, target_times)

    assert len(interpolated) == 5
    # Midpoint at t=1.0s
    mid = interpolated[2]
    assert mid.timestamp_offset == 1.0
    assert mid.latitude == pytest.approx(37.0001, abs=1e-5)
    assert mid.longitude == pytest.approx(-122.0001, abs=1e-5)
    assert mid.altitude_msl == pytest.approx(102.0, abs=0.1)
    assert mid.altitude_barometric_m == pytest.approx(22.0, abs=0.1)
    # Heading crosses 360 -> from 350 to 10 is +20 degrees total; midpoint should be 0 or 360 deg
    assert mid.heading == pytest.approx(0.0, abs=1.0) or mid.heading == pytest.approx(360.0, abs=1.0)


@pytest.mark.asyncio
async def test_persist_flight_telemetry():
    """Verify persist_flight_telemetry sets flight flags and adds FlightTelemetry records."""
    flight_id = uuid.uuid4()
    mock_flight = Flight(
        id=flight_id,
        project_id=uuid.uuid4(),
        original_filename="flight.mp4",
    )

    records = [
        ParsedTelemetryPoint(
            timestamp_offset=0.0,
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=120.0,
            altitude_barometric_m=45.0,
            heading=180.0,
            speed=5.0,
        ),
        ParsedTelemetryPoint(
            timestamp_offset=1.0,
            latitude=37.7750,
            longitude=-122.4193,
            altitude_msl=121.0,
            altitude_barometric_m=46.0,
            heading=182.0,
            speed=5.2,
        ),
    ]

    telemetry = ParsedFlightTelemetry(
        records=records,
        has_rtk_corrections=True,
        has_barometric_altitude=True,
        duration_seconds=1.0,
        sample_rate_hz=1.0,
    )

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_flight
    mock_session.execute = AsyncMock(return_value=mock_result)

    count = await persist_flight_telemetry(flight_id, telemetry, mock_session)

    assert count == 2
    assert mock_flight.has_barometric_altitude is True
    assert mock_flight.has_rtk_corrections is True
    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()
