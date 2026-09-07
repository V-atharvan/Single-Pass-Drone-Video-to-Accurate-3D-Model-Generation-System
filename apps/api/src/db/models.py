"""
ORM Models – Single-Pass 3D Reconstruction Platform
All tables from TASK-005 of the project_todo_list.md.
"""
from __future__ import annotations

import uuid
import enum
from datetime import datetime
from typing import Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class UserRole(str, enum.Enum):
    ORG_ADMIN = "ORG_ADMIN"
    PROJECT_MANAGER = "PROJECT_MANAGER"
    OPERATOR = "OPERATOR"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"


class FlightStatus(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    READY = "READY"
    FAILED = "FAILED"


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    EXTRACTING_FRAMES = "EXTRACTING_FRAMES"
    ESTIMATING_POSE = "ESTIMATING_POSE"
    ESTIMATING_DEPTH = "ESTIMATING_DEPTH"
    SEGMENTING = "SEGMENTING"
    FUSING = "FUSING"
    RECONSTRUCTING = "RECONSTRUCTING"
    TEXTURING = "TEXTURING"
    GEOREFERENCING = "GEOREFERENCING"
    QUALITY_CHECK = "QUALITY_CHECK"
    GENERATING_TILES = "GENERATING_TILES"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class QualityPreset(str, enum.Enum):
    LOW = "LOW"
    BALANCED = "BALANCED"
    HIGH = "HIGH"


class ModelStatus(str, enum.Enum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class AssetType(str, enum.Enum):
    GLB = "GLB"
    OBJ = "OBJ"
    PLY = "PLY"
    LAS = "LAS"
    TILES_3D = "TILES_3D"
    DEM_TIF = "DEM_TIF"


class MeasurementType(str, enum.Enum):
    DISTANCE = "DISTANCE"
    HEIGHT = "HEIGHT"
    AREA = "AREA"
    VOLUME = "VOLUME"


# ---------------------------------------------------------------------------
# ORM Tables
# ---------------------------------------------------------------------------

class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship("User", back_populates="organization")
    projects: Mapped[list["Project"]] = relationship("Project", back_populates="organization")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("auth0_sub", name="uq_users_auth0_sub"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, name="user_role"), nullable=False, default=UserRole.VIEWER)
    auth0_sub: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")
    measurements: Mapped[list["Measurement"]] = relationship("Measurement", back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    location_name: Mapped[Optional[str]] = mapped_column(String(255))
    crs: Mapped[str] = mapped_column(String(64), nullable=False, default="EPSG:4326")
    bbox = mapped_column(Geometry("POLYGON", srid=4326))
    tags: Mapped[Optional[list]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    organization: Mapped["Organization"] = relationship("Organization", back_populates="projects")
    flights: Mapped[list["Flight"]] = relationship("Flight", back_populates="project")


class Flight(Base):
    __tablename__ = "flights"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    s3_video_key: Mapped[Optional[str]] = mapped_column(String(1024))
    s3_telemetry_key: Mapped[Optional[str]] = mapped_column(String(1024))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    fps: Mapped[Optional[float]] = mapped_column(Float)
    resolution_width: Mapped[Optional[int]] = mapped_column(Integer)
    resolution_height: Mapped[Optional[int]] = mapped_column(Integer)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[FlightStatus] = mapped_column(Enum(FlightStatus, name="flight_status"), nullable=False, default=FlightStatus.PENDING)
    video_quality_report_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    has_barometric_altitude: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, nullable=True)
    has_rtk_corrections: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped["Project"] = relationship("Project", back_populates="flights")
    telemetry: Mapped[list["FlightTelemetry"]] = relationship("FlightTelemetry", back_populates="flight")
    reconstruction_jobs: Mapped[list["ReconstructionJob"]] = relationship("ReconstructionJob", back_populates="flight")


class FlightTelemetry(Base):
    __tablename__ = "flight_telemetry"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flight_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("flights.id", ondelete="CASCADE"), nullable=False)
    timestamp_offset: Mapped[float] = mapped_column(Float, nullable=False)
    location = mapped_column(Geometry("POINTZ", srid=4326))
    altitude_msl: Mapped[Optional[float]] = mapped_column(Float)
    altitude_barometric_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    heading: Mapped[Optional[float]] = mapped_column(Float)
    speed: Mapped[Optional[float]] = mapped_column(Float)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB)

    flight: Mapped["Flight"] = relationship("Flight", back_populates="telemetry")


class ReconstructionJob(Base):
    __tablename__ = "reconstruction_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flight_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("flights.id", ondelete="CASCADE"), nullable=False)
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, name="job_status"), nullable=False, default=JobStatus.QUEUED)
    quality_preset: Mapped[QualityPreset] = mapped_column(Enum(QualityPreset, name="quality_preset"), nullable=False, default=QualityPreset.BALANCED)
    current_stage: Mapped[Optional[str]] = mapped_column(String(64))
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    flight: Mapped["Flight"] = relationship("Flight", back_populates="reconstruction_jobs")
    model: Mapped[Optional["Model"]] = relationship("Model", back_populates="job", uselist=False)


class Model(Base):
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reconstruction_jobs.id", ondelete="CASCADE"), nullable=False, unique=True)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    crs: Mapped[str] = mapped_column(String(64), nullable=False, default="EPSG:4326")
    bbox = mapped_column(Geometry("POLYGON", srid=4326))
    center_lat: Mapped[Optional[float]] = mapped_column(Float)
    center_lon: Mapped[Optional[float]] = mapped_column(Float)
    min_altitude: Mapped[Optional[float]] = mapped_column(Float)
    max_altitude: Mapped[Optional[float]] = mapped_column(Float)
    coverage_percent: Mapped[Optional[float]] = mapped_column(Float)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    position_rmse: Mapped[Optional[float]] = mapped_column(Float)
    vertical_rmse: Mapped[Optional[float]] = mapped_column(Float)
    s3_prefix: Mapped[Optional[str]] = mapped_column(String(1024))
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus, name="model_status"), nullable=False, default=ModelStatus.PROCESSING)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped["ReconstructionJob"] = relationship("ReconstructionJob", back_populates="model")
    assets: Mapped[list["ModelAsset"]] = relationship("ModelAsset", back_populates="model")
    measurements: Mapped[list["Measurement"]] = relationship("Measurement", back_populates="model")


class ModelAsset(Base):
    __tablename__ = "model_assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    asset_type: Mapped[AssetType] = mapped_column(Enum(AssetType, name="asset_type"), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    lod_level: Mapped[Optional[int]] = mapped_column(Integer)

    model: Mapped["Model"] = relationship("Model", back_populates="assets")


class Measurement(Base):
    __tablename__ = "measurements"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    type: Mapped[MeasurementType] = mapped_column(Enum(MeasurementType, name="measurement_type"), nullable=False)
    geometry_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    value: Mapped[Optional[float]] = mapped_column(Float)
    error_margin: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model: Mapped["Model"] = relationship("Model", back_populates="measurements")
    user: Mapped["User"] = relationship("User", back_populates="measurements")
