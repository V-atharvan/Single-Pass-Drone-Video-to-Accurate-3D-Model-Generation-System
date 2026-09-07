# Backend API (`apps/api`)

Control plane API for the Single-Pass 3D Reconstruction Platform.

## Stack
- **Framework:** FastAPI + Python 3.12+
- **Database:** PostgreSQL 18 + PostGIS
- **ORM / Migrations:** SQLAlchemy 2.0 + Alembic
- **Caching & State:** Redis 7
- **Object Storage:** Amazon S3 / MinIO
- **Identity & RBAC:** Auth0 RS256 JWT validation
