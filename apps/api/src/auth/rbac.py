"""
Multi-tenant RBAC authorization guards.
Enforces resource ownership hierarchy: Organization → Project → Flight → Model → Asset.

Role hierarchy (highest → lowest):
  ORG_ADMIN > PROJECT_MANAGER > OPERATOR > ANALYST > VIEWER
"""
from __future__ import annotations

import enum
from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.base import get_session
from src.db.models import User, Project, Flight, Model, ReconstructionJob, UserRole
from src.auth.jwt import get_current_user


# ---------------------------------------------------------------------------
# Role precedence
# ---------------------------------------------------------------------------

_ROLE_RANK: dict[UserRole, int] = {
    UserRole.ORG_ADMIN: 5,
    UserRole.PROJECT_MANAGER: 4,
    UserRole.OPERATOR: 3,
    UserRole.ANALYST: 2,
    UserRole.VIEWER: 1,
}


def _has_min_role(user_role: UserRole, min_role: UserRole) -> bool:
    return _ROLE_RANK.get(user_role, 0) >= _ROLE_RANK.get(min_role, 99)


# ---------------------------------------------------------------------------
# Resolve current DB user from JWT claims
# ---------------------------------------------------------------------------

async def resolve_db_user(
    claims: dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> User:
    """
    Map Auth0 JWT claims to a local User record.
    Creates the user on first login if not found (just-in-time provisioning).
    """
    auth0_sub: str = claims["sub"]
    result = await session.execute(select(User).where(User.auth0_sub == auth0_sub))
    db_user = result.scalar_one_or_none()

    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not provisioned. Please complete onboarding.",
        )
    return db_user


# ---------------------------------------------------------------------------
# Resource access dependencies
# ---------------------------------------------------------------------------

def require_project_access(min_role: UserRole = UserRole.VIEWER):
    """
    Returns a FastAPI dependency that:
      - Resolves the current DB user.
      - Verifies the project belongs to the user's organization.
      - Checks the user has at least `min_role`.
    """
    async def _check(
        project_id: UUID,
        current_user: User = Depends(resolve_db_user),
        session: AsyncSession = Depends(get_session),
    ) -> Project:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")

        if project.org_id != current_user.org_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: cross-organization resource.")

        if not _has_min_role(current_user.role, min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Requires at least {min_role.value}.",
            )
        return project

    return _check


def require_flight_access(min_role: UserRole = UserRole.VIEWER):
    """Dependency verifying flight access via project → org ownership chain."""
    async def _check(
        flight_id: UUID,
        current_user: User = Depends(resolve_db_user),
        session: AsyncSession = Depends(get_session),
    ) -> Flight:
        result = await session.execute(
            select(Flight).join(Project, Flight.project_id == Project.id).where(Flight.id == flight_id)
        )
        flight = result.scalar_one_or_none()

        if flight is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flight not found.")

        # Load parent project to check org
        proj_result = await session.execute(select(Project).where(Project.id == flight.project_id))
        project = proj_result.scalar_one_or_none()

        if project is None or project.org_id != current_user.org_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: cross-organization resource.")

        if not _has_min_role(current_user.role, min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Requires at least {min_role.value}.",
            )
        return flight

    return _check


def require_model_access(min_role: UserRole = UserRole.VIEWER):
    """Dependency verifying model access via project → org ownership chain."""
    async def _check(
        model_id: UUID,
        current_user: User = Depends(resolve_db_user),
        session: AsyncSession = Depends(get_session),
    ) -> Model:
        result = await session.execute(select(Model).where(Model.id == model_id))
        model = result.scalar_one_or_none()

        if model is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")

        proj_result = await session.execute(select(Project).where(Project.id == model.project_id))
        project = proj_result.scalar_one_or_none()

        if project is None or project.org_id != current_user.org_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: cross-organization resource.")

        if not _has_min_role(current_user.role, min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Requires at least {min_role.value}.",
            )
        return model

    return _check


def require_job_access(min_role: UserRole = UserRole.VIEWER):
    """Dependency verifying reconstruction job access via org ownership chain."""
    async def _check(
        job_id: UUID,
        current_user: User = Depends(resolve_db_user),
        session: AsyncSession = Depends(get_session),
    ) -> ReconstructionJob:
        result = await session.execute(select(ReconstructionJob).where(ReconstructionJob.id == job_id))
        job = result.scalar_one_or_none()

        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reconstruction job not found.")

        if job.org_id != current_user.org_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: cross-organization resource.")

        if not _has_min_role(current_user.role, min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Requires at least {min_role.value}.",
            )
        return job

    return _check


# ---------------------------------------------------------------------------
# Pre-bound dependency singletons for route declarations and testing
# ---------------------------------------------------------------------------
require_operator_project_access = require_project_access(min_role=UserRole.OPERATOR)
require_viewer_project_access = require_project_access(min_role=UserRole.VIEWER)
require_operator_flight_access = require_flight_access(min_role=UserRole.OPERATOR)
require_viewer_flight_access = require_flight_access(min_role=UserRole.VIEWER)
require_operator_job_access = require_job_access(min_role=UserRole.OPERATOR)
require_viewer_job_access = require_job_access(min_role=UserRole.VIEWER)

