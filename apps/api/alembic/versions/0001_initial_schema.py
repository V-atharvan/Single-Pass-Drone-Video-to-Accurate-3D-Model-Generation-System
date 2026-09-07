"""Initial schema: organizations, users, projects, flights, telemetry,
reconstruction_jobs, models, model_assets, measurements with PostGIS extensions.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-08
"""
from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable required PostgreSQL extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

    # -----------------------------------------------------------------------
    # Enumerations
    # -----------------------------------------------------------------------
    op.execute("CREATE TYPE user_role AS ENUM ('ORG_ADMIN', 'PROJECT_MANAGER', 'OPERATOR', 'ANALYST', 'VIEWER');")
    op.execute("CREATE TYPE flight_status AS ENUM ('PENDING', 'UPLOADING', 'READY', 'FAILED');")
    op.execute(
        "CREATE TYPE job_status AS ENUM ("
        "'QUEUED', 'VALIDATING', 'EXTRACTING_FRAMES', 'ESTIMATING_POSE', "
        "'ESTIMATING_DEPTH', 'SEGMENTING', 'FUSING', 'RECONSTRUCTING', "
        "'TEXTURING', 'GEOREFERENCING', 'QUALITY_CHECK', 'GENERATING_TILES', "
        "'COMPLETED', 'FAILED', 'CANCELLED');"
    )
    op.execute("CREATE TYPE quality_preset AS ENUM ('LOW', 'BALANCED', 'HIGH');")
    op.execute("CREATE TYPE model_status AS ENUM ('PROCESSING', 'READY', 'FAILED');")
    op.execute("CREATE TYPE asset_type AS ENUM ('GLB', 'OBJ', 'PLY', 'LAS', 'TILES_3D', 'DEM_TIF');")
    op.execute("CREATE TYPE measurement_type AS ENUM ('DISTANCE', 'HEIGHT', 'AREA', 'VOLUME');")

    # -----------------------------------------------------------------------
    # organizations
    # -----------------------------------------------------------------------
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # -----------------------------------------------------------------------
    # users
    # -----------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("full_name", sa.String(255)),
        sa.Column("role", sa.Enum("ORG_ADMIN", "PROJECT_MANAGER", "OPERATOR", "ANALYST", "VIEWER", name="user_role"), nullable=False, server_default="VIEWER"),
        sa.Column("auth0_sub", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("auth0_sub", name="uq_users_auth0_sub"),
    )
    op.create_index("ix_users_org_id", "users", ["org_id"])

    # -----------------------------------------------------------------------
    # projects
    # -----------------------------------------------------------------------
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("location_name", sa.String(255)),
        sa.Column("crs", sa.String(64), nullable=False, server_default="EPSG:4326"),
        sa.Column("bbox", geoalchemy2.types.Geometry("POLYGON", srid=4326)),
        sa.Column("tags", postgresql.JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_projects_org_id", "projects", ["org_id"])
    op.create_index("ix_projects_bbox", "projects", ["bbox"], postgresql_using="gist")

    # -----------------------------------------------------------------------
    # flights
    # -----------------------------------------------------------------------
    op.create_table(
        "flights",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("s3_video_key", sa.String(1024)),
        sa.Column("s3_telemetry_key", sa.String(1024)),
        sa.Column("duration_seconds", sa.Float),
        sa.Column("fps", sa.Float),
        sa.Column("resolution_width", sa.Integer),
        sa.Column("resolution_height", sa.Integer),
        sa.Column("total_frames", sa.Integer),
        sa.Column("status", sa.Enum("PENDING", "UPLOADING", "READY", "FAILED", name="flight_status"), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_flights_project_id", "flights", ["project_id"])

    # -----------------------------------------------------------------------
    # flight_telemetry
    # -----------------------------------------------------------------------
    op.create_table(
        "flight_telemetry",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("flight_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("flights.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp_offset", sa.Float, nullable=False),
        sa.Column("location", geoalchemy2.types.Geometry("POINTZ", srid=4326)),
        sa.Column("altitude_msl", sa.Float),
        sa.Column("heading", sa.Float),
        sa.Column("speed", sa.Float),
        sa.Column("raw_json", postgresql.JSONB),
    )
    op.create_index("ix_flight_telemetry_flight_id", "flight_telemetry", ["flight_id"])
    op.create_index("ix_flight_telemetry_location", "flight_telemetry", ["location"], postgresql_using="gist")

    # -----------------------------------------------------------------------
    # reconstruction_jobs
    # -----------------------------------------------------------------------
    op.create_table(
        "reconstruction_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("flight_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("flights.id", ondelete="CASCADE"), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.Enum(
            "QUEUED", "VALIDATING", "EXTRACTING_FRAMES", "ESTIMATING_POSE",
            "ESTIMATING_DEPTH", "SEGMENTING", "FUSING", "RECONSTRUCTING",
            "TEXTURING", "GEOREFERENCING", "QUALITY_CHECK", "GENERATING_TILES",
            "COMPLETED", "FAILED", "CANCELLED", name="job_status"
        ), nullable=False, server_default="QUEUED"),
        sa.Column("quality_preset", sa.Enum("LOW", "BALANCED", "HIGH", name="quality_preset"), nullable=False, server_default="BALANCED"),
        sa.Column("current_stage", sa.String(64)),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_reconstruction_jobs_flight_id", "reconstruction_jobs", ["flight_id"])
    op.create_index("ix_reconstruction_jobs_org_id", "reconstruction_jobs", ["org_id"])
    op.create_index("ix_reconstruction_jobs_status", "reconstruction_jobs", ["status"])

    # -----------------------------------------------------------------------
    # models
    # -----------------------------------------------------------------------
    op.create_table(
        "models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("reconstruction_jobs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("crs", sa.String(64), nullable=False, server_default="EPSG:4326"),
        sa.Column("bbox", geoalchemy2.types.Geometry("POLYGON", srid=4326)),
        sa.Column("center_lat", sa.Float),
        sa.Column("center_lon", sa.Float),
        sa.Column("min_altitude", sa.Float),
        sa.Column("max_altitude", sa.Float),
        sa.Column("coverage_percent", sa.Float),
        sa.Column("confidence_score", sa.Float),
        sa.Column("position_rmse", sa.Float),
        sa.Column("vertical_rmse", sa.Float),
        sa.Column("s3_prefix", sa.String(1024)),
        sa.Column("status", sa.Enum("PROCESSING", "READY", "FAILED", name="model_status"), nullable=False, server_default="PROCESSING"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_models_project_id", "models", ["project_id"])
    op.create_index("ix_models_bbox", "models", ["bbox"], postgresql_using="gist")

    # -----------------------------------------------------------------------
    # model_assets
    # -----------------------------------------------------------------------
    op.create_table(
        "model_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("asset_type", sa.Enum("GLB", "OBJ", "PLY", "LAS", "TILES_3D", "DEM_TIF", name="asset_type"), nullable=False),
        sa.Column("s3_key", sa.String(1024), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger),
        sa.Column("lod_level", sa.Integer),
    )
    op.create_index("ix_model_assets_model_id", "model_assets", ["model_id"])

    # -----------------------------------------------------------------------
    # measurements
    # -----------------------------------------------------------------------
    op.create_table(
        "measurements",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("type", sa.Enum("DISTANCE", "HEIGHT", "AREA", "VOLUME", name="measurement_type"), nullable=False),
        sa.Column("geometry_json", postgresql.JSONB),
        sa.Column("value", sa.Float),
        sa.Column("error_margin", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_measurements_model_id", "measurements", ["model_id"])
    op.create_index("ix_measurements_user_id", "measurements", ["user_id"])


def downgrade() -> None:
    # Drop tables in reverse dependency order
    op.drop_table("measurements")
    op.drop_table("model_assets")
    op.drop_table("models")
    op.drop_table("reconstruction_jobs")
    op.drop_table("flight_telemetry")
    op.drop_table("flights")
    op.drop_table("projects")
    op.drop_table("users")
    op.drop_table("organizations")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS measurement_type;")
    op.execute("DROP TYPE IF EXISTS asset_type;")
    op.execute("DROP TYPE IF EXISTS model_status;")
    op.execute("DROP TYPE IF EXISTS quality_preset;")
    op.execute("DROP TYPE IF EXISTS job_status;")
    op.execute("DROP TYPE IF EXISTS flight_status;")
    op.execute("DROP TYPE IF EXISTS user_role;")
