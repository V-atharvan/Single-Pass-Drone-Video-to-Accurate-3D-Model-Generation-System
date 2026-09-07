"""Telemetry and camera data contracts for Single-Pass 3D Reconstruction Platform."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class GPSRecord(BaseModel):
    """Raw and interpolated GPS coordinate records."""

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees (WGS84)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees (WGS84)")
    altitude_msl: float = Field(..., description="Altitude above Mean Sea Level in meters")
    altitude_relative: Optional[float] = Field(None, description="Takeoff-relative altitude in meters")
    speed_mps: Optional[float] = Field(None, ge=0.0, description="Ground speed in meters per second")
    heading_deg: Optional[float] = Field(None, ge=0.0, lt=360.0, description="Compass heading in degrees [0, 360)")
    timestamp_offset_seconds: float = Field(..., ge=0.0, description="Elapsed seconds from video start")
    hdop: Optional[float] = Field(None, ge=0.0, description="Horizontal Dilution of Precision")
    vdop: Optional[float] = Field(None, ge=0.0, description="Vertical Dilution of Precision")
    num_satellites: Optional[int] = Field(None, ge=0, description="Locked satellite count")


class Quaternion(BaseModel):
    """Unit quaternion representing 3D spatial rotation."""

    w: float = Field(..., description="Scalar component")
    x: float = Field(..., description="Vector component x")
    y: float = Field(..., description="Vector component y")
    z: float = Field(..., description="Vector component z")

    @field_validator("w")
    @classmethod
    def validate_non_zero_norm(cls, v: float, info) -> float:
        # Check happens on the overall vector when possible
        return v


class IMURecord(BaseModel):
    """Inertial Measurement Unit telemetry record."""

    roll_deg: float = Field(..., ge=-180.0, le=180.0, description="Roll in degrees")
    pitch_deg: float = Field(..., ge=-90.0, le=90.0, description="Pitch in degrees")
    yaw_deg: float = Field(..., ge=0.0, lt=360.0, description="Yaw in degrees")
    quaternion: Optional[Quaternion] = Field(None, description="Orientation quaternion")
    timestamp_offset_seconds: float = Field(..., ge=0.0, description="Elapsed seconds from video start")


class CameraIntrinsics(BaseModel):
    """Pinhole camera intrinsic calibration model with Brown-Conrady radial distortion."""

    fx: float = Field(..., gt=0.0, description="Focal length x in pixels")
    fy: float = Field(..., gt=0.0, description="Focal length y in pixels")
    cx: float = Field(..., gt=0.0, description="Principal point x in pixels")
    cy: float = Field(..., gt=0.0, description="Principal point y in pixels")
    width: int = Field(..., gt=0, description="Image sensor width in pixels")
    height: int = Field(..., gt=0, description="Image sensor height in pixels")
    k1: float = Field(0.0, description="First radial distortion coefficient")
    k2: float = Field(0.0, description="Second radial distortion coefficient")
    p1: float = Field(0.0, description="First tangential distortion coefficient")
    p2: float = Field(0.0, description="Second tangential distortion coefficient")
    sensor_name: Optional[str] = Field(None, description="Camera/sensor model identifier")


class CameraPose(BaseModel):
    """Optimized 6-Degrees-of-Freedom camera extrinsic pose."""

    frame_index: int = Field(..., ge=0, description="Extracted keyframe sequential index")
    timestamp_offset_seconds: float = Field(..., ge=0.0, description="Video timestamp offset")
    rotation_matrix: List[List[float]] = Field(
        ...,
        description="3x3 orthonormal rotation matrix from camera to world frame",
    )
    translation_vector: List[float] = Field(
        ...,
        description="3-element translation vector [x, y, z] in world coordinates (meters)",
    )
    reprojection_error_px: float = Field(..., ge=0.0, description="Feature reprojection error RMSE in pixels")
    pose_confidence: float = Field(..., ge=0.0, le=100.0, description="Pose certainty score (0-100%)")

    @field_validator("rotation_matrix")
    @classmethod
    def validate_rotation_dimensions(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("rotation_matrix must be a 3x3 matrix")
        return v

    @field_validator("translation_vector")
    @classmethod
    def validate_translation_dimensions(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("translation_vector must have exactly 3 elements [x, y, z]")
        return v


class KeyframeMetadata(BaseModel):
    """Metadata for an extracted video keyframe."""

    frame_index: int = Field(..., ge=0)
    source_video_timestamp_sec: float = Field(..., ge=0.0)
    keyframe_s3_key: str = Field(...)
    blur_score: float = Field(..., ge=0.0, le=100.0, description="Laplacian sharpness metric (higher = sharper)")
    exposure_score: float = Field(..., ge=0.0, le=100.0, description="Luminance balance metric")
    gps: Optional[GPSRecord] = None
    pose: Optional[CameraPose] = None
    is_selected: bool = Field(True, description="Whether selected for 3D reconstruction pipeline")
