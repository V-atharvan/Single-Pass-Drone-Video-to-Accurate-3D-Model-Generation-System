"""
Projects CRUD API Endpoints – TASK-013
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.base import get_session
from src.db.models import Flight, Model, Project, User, UserRole
from src.auth.rbac import resolve_db_user

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ProjectCreateIn(BaseModel):
    name: str
    description: Optional[str] = None
    location_name: Optional[str] = None
    crs: str = "EPSG:4326"
    tags: Optional[list[str]] = None


class ProjectUpdateIn(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None


class ProjectOut(BaseModel):
    id: UUID
    org_id: UUID
    name: str
    description: Optional[str]
    location_name: Optional[str]
    crs: str
    tags: Optional[list[str]]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectDetailOut(ProjectOut):
    flight_count: int = 0
    model_count: int = 0
    avg_confidence_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED, summary="Create project")
async def create_project(
    body: ProjectCreateIn,
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> Project:
    project = Project(
        org_id=current_user.org_id,
        name=body.name,
        description=body.description,
        location_name=body.location_name,
        crs=body.crs,
        tags=body.tags,
    )
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut], summary="List projects")
async def list_projects(
    q: Optional[str] = Query(None, description="Search query on name or description"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> list[Project]:
    stmt = select(Project).where(Project.org_id == current_user.org_id)
    if q:
        stmt = stmt.where(Project.name.ilike(f"%{q}%"))
    stmt = stmt.order_by(Project.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())


@router.get("/{project_id}", response_model=ProjectDetailOut, summary="Get project details")
async def get_project(
    project_id: UUID,
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    if project.org_id != current_user.org_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    # Aggregate counts
    flight_count_result = await session.execute(
        select(func.count()).where(Flight.project_id == project_id)
    )
    flight_count = flight_count_result.scalar_one()

    model_count_result = await session.execute(
        select(func.count()).where(Model.project_id == project_id)
    )
    model_count = model_count_result.scalar_one()

    avg_conf_result = await session.execute(
        select(func.avg(Model.confidence_score)).where(Model.project_id == project_id)
    )
    avg_conf = avg_conf_result.scalar_one()

    return {
        **{col.name: getattr(project, col.name) for col in project.__table__.columns},
        "flight_count": flight_count,
        "model_count": model_count,
        "avg_confidence_score": float(avg_conf) if avg_conf is not None else None,
    }


@router.patch("/{project_id}", response_model=ProjectOut, summary="Update project metadata")
async def update_project(
    project_id: UUID,
    body: ProjectUpdateIn,
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> Project:
    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    if project.org_id != current_user.org_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    if body.name is not None:
        project.name = body.name
    if body.description is not None:
        project.description = body.description
    if body.tags is not None:
        project.tags = body.tags

    await session.commit()
    await session.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete project")
async def delete_project(
    project_id: UUID,
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    # Requires at least PROJECT_MANAGER role
    if current_user.role not in (UserRole.ORG_ADMIN, UserRole.PROJECT_MANAGER):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role to delete projects.")

    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    if project.org_id != current_user.org_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    await session.delete(project)
    await session.commit()
