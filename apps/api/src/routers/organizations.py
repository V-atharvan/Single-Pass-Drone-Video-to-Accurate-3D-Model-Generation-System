"""
Organizations and Users Management Endpoints – TASK-012
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.base import get_session
from src.db.models import Organization, User, UserRole
from src.auth.rbac import resolve_db_user

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class OrganizationOut(BaseModel):
    id: UUID
    name: str

    model_config = {"from_attributes": True}


class MemberOut(BaseModel):
    id: UUID
    email: str
    full_name: str | None
    role: UserRole

    model_config = {"from_attributes": True}


class UpdateRoleIn(BaseModel):
    role: UserRole


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/me", response_model=OrganizationOut, summary="Get caller's organization")
async def get_my_organization(
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> Organization:
    result = await session.execute(select(Organization).where(Organization.id == current_user.org_id))
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return org


@router.get("/me/members", response_model=list[MemberOut], summary="List organization members")
async def list_members(
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> list[User]:
    result = await session.execute(select(User).where(User.org_id == current_user.org_id))
    return list(result.scalars().all())


@router.patch("/me/members/{user_id}", response_model=MemberOut, summary="Update a member's role (Admin only)")
async def update_member_role(
    user_id: UUID,
    body: UpdateRoleIn,
    current_user: User = Depends(resolve_db_user),
    session: AsyncSession = Depends(get_session),
) -> User:
    # Only ORG_ADMIN can modify roles
    if current_user.role != UserRole.ORG_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only ORG_ADMIN can modify member roles.")

    result = await session.execute(
        select(User).where(User.id == user_id, User.org_id == current_user.org_id)
    )
    target_user = result.scalar_one_or_none()

    if target_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found in your organization.")

    target_user.role = body.role
    await session.commit()
    await session.refresh(target_user)
    return target_user
