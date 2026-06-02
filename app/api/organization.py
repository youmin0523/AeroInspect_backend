# =============================================
# app/api/organization.py
# 역할: 조직(회사) 관리 REST API
#       - GET    /organizations/my          → 내 조직 정보
#       - GET    /organizations/members     → 같은 조직 멤버 목록 (채팅 팀원 목록)
#       - POST   /organizations             → 조직 생성 (B2B 자동 매칭 or 수동)
#       - POST   /organizations/members/invite → 멤버 초대
#       - PATCH  /organizations/members/{user_id} → 멤버 정보 수정
#       - DELETE /organizations/members/{user_id} → 멤버 제거
#
# 설계:
#   - biz_number 자동 매칭: 회원가입 시 B2B 사업자등록번호가 기존 Organization 과
#     일치하면 자동으로 OrganizationMember 생성 (auth.py 회원가입 로직에서 호출)
#   - 관리자 초대: admin/owner 가 이메일로 초대 → invited 상태로 멤버 생성
# =============================================

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user, get_current_org_member, require_role, require_superadmin
from app.models.department import Department
from app.models.organization import Organization, OrganizationMember
from app.models.user import User
from app.schemas.organization import (
    AssignMemberRequest,
    JoinByCodeRequest,
    OrganizationCreate,
    OrganizationResponse,
    OrgMemberResponse,
    OrgMemberListResponse,
    InviteMemberRequest,
    UnaffiliatedUserResponse,
    UpdateMemberRequest,
)
from app.services.notification_service import notification_service

router = APIRouter()


# ── 헬퍼: 현재 사용자의 조직 조회 ─────────────
async def _get_user_org(db: AsyncSession, user_id: UUID):
    """현재 사용자가 소속된 Organization + OrganizationMember 반환"""
    result = await db.execute(
        select(OrganizationMember, Organization)
        .join(Organization, Organization.id == OrganizationMember.organization_id)
        .where(OrganizationMember.user_id == user_id)
        .where(OrganizationMember.status == "active")
    )
    row = result.first()
    if not row:
        return None, None
    return row[0], row[1]  # member, org


@router.get("/my", response_model=OrganizationResponse)
async def get_my_organization(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 소속 조직 정보 조회"""
    member, org = await _get_user_org(db, current_user.id)
    if not org:
        raise HTTPException(status_code=404, detail="소속된 조직이 없습니다.")

    count = await db.scalar(
        select(func.count(OrganizationMember.id))
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.status == "active")
    )

    return OrganizationResponse(
        id=org.id, name=org.name, biz_number=org.biz_number,
        invite_code=org.invite_code, invite_code_expires_at=org.invite_code_expires_at,
        member_count=count, created_at=org.created_at,
    )


@router.get("/members", response_model=OrgMemberListResponse)
async def list_organization_members(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    같은 조직의 멤버 목록 조회 — 메신저 '새 대화' 팀원 목록에서 사용.
    active + invited 상태 멤버 모두 반환.
    """
    member, org = await _get_user_org(db, current_user.id)
    if not org:
        raise HTTPException(status_code=404, detail="소속된 조직이 없습니다.")

    result = await db.execute(
        select(OrganizationMember, User)
        .join(User, User.id == OrganizationMember.user_id)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.status.in_(["active", "invited"]))
        .order_by(OrganizationMember.role, User.name)
    )

    members = []
    for om, user in result:
        members.append(OrgMemberResponse(
            user_id=user.id,
            name=user.name,
            email=user.email,
            phone=user.phone,
            initials=user.name[:2].upper() if user.name else "??",
            role=om.role,
            department=om.department,
            position=om.position,
            status=om.status,
            started_at=om.started_at,
            ended_at=om.ended_at,
        ))

    active_count = sum(1 for m in members if m.status == "active")

    return OrgMemberListResponse(
        organization=OrganizationResponse(
            id=org.id, name=org.name, biz_number=org.biz_number,
            invite_code=org.invite_code, invite_code_expires_at=org.invite_code_expires_at,
            member_count=active_count, created_at=org.created_at,
        ),
        members=members,
        total=len(members),
    )


@router.post("", response_model=OrganizationResponse, status_code=201)
async def create_organization(
    payload: OrganizationCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    조직 생성.
    biz_number 가 이미 존재하면 409 Conflict (기존 조직에 가입 유도).
    생성자는 자동으로 owner 권한 부여.
    """
    # 사업자등록번호 중복 체크
    if payload.biz_number:
        existing = await db.scalar(
            select(Organization.id).where(Organization.biz_number == payload.biz_number)
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="해당 사업자등록번호로 등록된 조직이 이미 존재합니다.",
            )

    org = Organization(name=payload.name, biz_number=payload.biz_number)
    db.add(org)
    await db.flush()

    # 생성자를 owner 로 등록
    db.add(OrganizationMember(
        organization_id=org.id,
        user_id=current_user.id,
        role="owner",
        status="active",
    ))
    await db.flush()

    return OrganizationResponse(
        id=org.id, name=org.name, biz_number=org.biz_number,
        invite_code=org.invite_code, invite_code_expires_at=org.invite_code_expires_at,
        member_count=1, created_at=org.created_at,
    )


@router.post("/members/invite", response_model=OrgMemberResponse, status_code=201)
async def invite_member(
    payload: InviteMemberRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    멤버 초대 (admin/owner 전용).
    이메일로 사용자를 찾아 invited 상태로 조직에 추가.
    """
    # 권한 확인
    my_member, org = await _get_user_org(db, current_user.id)
    if not org:
        raise HTTPException(status_code=404, detail="소속된 조직이 없습니다.")
    if my_member.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="멤버 초대 권한이 없습니다. (admin 이상)")

    # 초대할 사용자 찾기
    target_user = await db.scalar(select(User).where(User.email == payload.email))
    if not target_user:
        raise HTTPException(status_code=404, detail="해당 이메일의 사용자를 찾을 수 없습니다.")

    # 이미 소속 여부 확인
    existing = await db.scalar(
        select(OrganizationMember.id)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.user_id == target_user.id)
    )
    if existing:
        raise HTTPException(status_code=409, detail="이미 조직에 소속된 사용자입니다.")

    new_member = OrganizationMember(
        organization_id=org.id,
        user_id=target_user.id,
        role=payload.role,
        department=payload.department,
        position=payload.position,
        status="active",
    )
    db.add(new_member)
    await db.flush()

    return OrgMemberResponse(
        user_id=target_user.id,
        name=target_user.name,
        email=target_user.email,
        phone=target_user.phone,
        initials=target_user.name[:2].upper() if target_user.name else "??",
        role=new_member.role,
        department=new_member.department,
        position=new_member.position,
        status=new_member.status,
        started_at=new_member.started_at,
        ended_at=new_member.ended_at,
    )


@router.patch("/members/{user_id}", response_model=OrgMemberResponse)
async def update_member(
    user_id: UUID,
    payload: UpdateMemberRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """멤버 부서/직위/권한/상태 수정 (admin/owner 전용)"""
    my_member, org = await _get_user_org(db, current_user.id)
    if not org:
        raise HTTPException(status_code=404, detail="소속된 조직이 없습니다.")
    if my_member.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="수정 권한이 없습니다.")

    result = await db.execute(
        select(OrganizationMember, User)
        .join(User, User.id == OrganizationMember.user_id)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.user_id == user_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="해당 멤버를 찾을 수 없습니다.")

    target_member, target_user = row
    if payload.role is not None:
        target_member.role = payload.role
    if payload.department is not None:
        target_member.department = payload.department
    if payload.position is not None:
        target_member.position = payload.position
    if payload.started_at is not None:
        target_member.started_at = payload.started_at
    if payload.ended_at is not None:
        target_member.ended_at = payload.ended_at
        # 퇴사일이 과거이면 자동으로 비활성 처리
        from datetime import datetime, timezone
        if payload.ended_at <= datetime.now(timezone.utc):
            target_member.status = "deactivated"
    if payload.status is not None:
        target_member.status = payload.status
    await db.flush()

    return OrgMemberResponse(
        user_id=target_user.id,
        name=target_user.name,
        email=target_user.email,
        phone=target_user.phone,
        initials=target_user.name[:2].upper() if target_user.name else "??",
        role=target_member.role,
        department=target_member.department,
        position=target_member.position,
        status=target_member.status,
        started_at=target_member.started_at,
        ended_at=target_member.ended_at,
    )


@router.delete("/members/{user_id}", status_code=204)
async def remove_member(
    user_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """멤버 제거 (admin/owner 전용, owner 자신은 제거 불가)"""
    my_member, org = await _get_user_org(db, current_user.id)
    if not org:
        raise HTTPException(status_code=404, detail="소속된 조직이 없습니다.")
    if my_member.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="멤버 제거 권한이 없습니다.")

    result = await db.execute(
        select(OrganizationMember)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.user_id == user_id)
    )
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="해당 멤버를 찾을 수 없습니다.")
    if target.role == "owner":
        raise HTTPException(status_code=400, detail="조직 소유자는 제거할 수 없습니다.")

    await db.delete(target)
    await db.flush()


# ── 미소속 사용자 목록 (admin/owner 전용) ────
@router.get("/unaffiliated-users", response_model=list[UnaffiliatedUserResponse])
async def list_unaffiliated_users(
    org_tuple=Depends(require_role("owner", "admin")),
    db: AsyncSession = Depends(get_db),
):
    """소속 조직이 없는 사용자 목록 (admin/owner 전용)"""
    affiliated_subq = (
        select(OrganizationMember.user_id)
        .where(OrganizationMember.status == "active")
    ).subquery()

    result = await db.execute(
        select(User)
        .where(User.id.notin_(select(affiliated_subq.c.user_id)))
        .order_by(User.created_at.desc())
    )
    users = result.scalars().all()
    return [UnaffiliatedUserResponse.model_validate(u) for u in users]


# ── 미소속 사용자 조직 배정 (admin/owner 또는 superadmin) ─
@router.post("/members/assign", response_model=OrgMemberResponse, status_code=201)
async def assign_member(
    payload: AssignMemberRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    미소속 사용자를 조직에 배정.
    - 슈퍼어드민: organization_id 를 지정하여 아무 조직에 배정 가능
    - 일반 admin/owner: 자신의 조직에만 배정 (organization_id 무시)
    """
    # 배정 대상 조직 결정
    if current_user.is_superadmin and payload.organization_id:
        # 슈퍼어드민이 특정 조직을 지정
        target_org = await db.scalar(select(Organization).where(Organization.id == payload.organization_id))
        if not target_org:
            raise HTTPException(status_code=404, detail="해당 조직을 찾을 수 없습니다.")
        org = target_org
    else:
        # 일반 admin/owner → 자기 조직
        member_row, org = await _get_user_org(db, current_user.id)
        if not org:
            raise HTTPException(status_code=403, detail="소속된 조직이 없습니다.")
        if member_row.role not in ("owner", "admin"):
            raise HTTPException(status_code=403, detail="멤버 배정 권한이 없습니다. (admin 이상)")

    # 배정 대상 사용자 확인
    target_user = await db.scalar(select(User).where(User.id == payload.user_id))
    if not target_user:
        raise HTTPException(status_code=404, detail="해당 사용자를 찾을 수 없습니다.")

    # 이미 소속 여부 확인
    existing = await db.scalar(
        select(OrganizationMember.id)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.user_id == payload.user_id)
    )
    if existing:
        raise HTTPException(status_code=409, detail="이미 조직에 소속된 사용자입니다.")

    new_member = OrganizationMember(
        organization_id=org.id,
        user_id=payload.user_id,
        role=payload.role,
        department=payload.department,
        position=payload.position,
        status="active",
    )
    db.add(new_member)
    await db.flush()

    # 배정된 본인에게 알림
    await notification_service.create(
        db=db,
        user_id=target_user.id,
        category="team",
        title=f"{org.name} 조직에 배정되었습니다",
        message=f"역할: {new_member.role}" + (f" / 부서: {new_member.department}" if new_member.department else ""),
        metadata={"organization_id": str(org.id), "role": new_member.role},
    )

    return OrgMemberResponse(
        user_id=target_user.id,
        name=target_user.name,
        email=target_user.email,
        phone=target_user.phone,
        initials=target_user.name[:2].upper() if target_user.name else "??",
        role=new_member.role,
        department=new_member.department,
        position=new_member.position,
        status=new_member.status,
        started_at=new_member.started_at,
        ended_at=new_member.ended_at,
    )


# ── 초대코드로 조직 가입 ─────────────────────
@router.post("/join", response_model=OrganizationResponse)
async def join_by_invite_code(
    payload: JoinByCodeRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """초대 코드로 조직 가입 (미소속 사용자용)"""
    org = await db.scalar(
        select(Organization).where(Organization.invite_code == payload.invite_code.upper())
    )
    if not org:
        raise HTTPException(status_code=404, detail="유효하지 않은 초대 코드입니다.")

    # 초대코드 만료 여부 확인
    if org.is_invite_code_expired():
        raise HTTPException(status_code=410, detail="초대 코드가 만료되었습니다. 관리자에게 새 코드를 요청하세요.")

    # 이미 소속 여부 확인
    existing = await db.scalar(
        select(OrganizationMember)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.user_id == current_user.id)
    )
    if existing:
        if existing.status == "invited":
            # invited 상태 → 초대코드로 가입 시 active로 전환
            existing.status = "active"
            await db.flush()
        elif existing.status == "active":
            raise HTTPException(status_code=409, detail="이미 해당 조직에 소속되어 있습니다.")
        else:
            # deactivated 등 → 관리자에게 문의
            raise HTTPException(status_code=403, detail="비활성 상태입니다. 관리자에게 문의하세요.")
    else:
        db.add(OrganizationMember(
            organization_id=org.id,
            user_id=current_user.id,
            role="member",
            status="active",
        ))
        await db.flush()

    # 그 조직의 owner/admin들에게 신규 가입 알림 (본인 제외)
    admin_rows = await db.execute(
        select(OrganizationMember.user_id)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.role.in_(("owner", "admin")))
        .where(OrganizationMember.status == "active")
    )
    admin_user_ids = [uid for (uid,) in admin_rows.all() if uid != current_user.id]
    if admin_user_ids:
        await notification_service.create_for_many(
            db=db,
            user_ids=admin_user_ids,
            category="team",
            title=f"{current_user.name or '신규 멤버'}님이 가입했습니다",
            message=f"{org.name} 조직에 초대 코드로 합류했습니다.",
            metadata={
                "organization_id": str(org.id),
                "new_member_id": str(current_user.id),
            },
        )

    count = await db.scalar(
        select(func.count(OrganizationMember.id))
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.status == "active")
    )

    return OrganizationResponse(
        id=org.id, name=org.name, biz_number=org.biz_number,
        invite_code=org.invite_code, invite_code_expires_at=org.invite_code_expires_at,
        member_count=count, created_at=org.created_at,
    )


# ── 초대코드 재생성 (관리자/소유자 전용) ──────────
@router.post("/invite-code/regenerate", response_model=OrganizationResponse)
async def regenerate_invite_code(
    org_tuple=Depends(require_role("owner", "admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    초대 코드 재생성 (admin/owner 전용).
    퇴사자 보안, 만료 시 갱신 등의 용도.
    새 코드 발급 + 만료일 30일 연장.
    """
    user, member, org = org_tuple

    org.regenerate_invite_code()
    await db.flush()

    count = await db.scalar(
        select(func.count(OrganizationMember.id))
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.status == "active")
    )

    return OrganizationResponse(
        id=org.id, name=org.name, biz_number=org.biz_number,
        invite_code=org.invite_code, invite_code_expires_at=org.invite_code_expires_at,
        member_count=count, created_at=org.created_at,
    )


# ══════════════════════════════════════════════
# 슈퍼어드민 전용: 전체 사용자 목록
# ══════════════════════════════════════════════

class _AllUserResponse(BaseModel):
    id: UUID
    name: str
    email: str
    phone: str
    account_type: str
    is_superadmin: bool
    created_at: object
    organization_name: str | None = None
    role: str | None = None
    department: str | None = None
    position: str | None = None
    status: str | None = None


@router.get("/admin/all-users", response_model=list[_AllUserResponse])
async def list_all_users(
    search: str = Query(None, description="이름/이메일/전화번호 검색"),
    superadmin=Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    """플랫폼 전체 사용자 목록 (슈퍼어드민 전용). 조직·역할 정보 포함."""
    from sqlalchemy.orm import aliased
    OrgMem = aliased(OrganizationMember)
    Org = aliased(Organization)

    query = (
        select(User, OrgMem, Org)
        .outerjoin(OrgMem, (OrgMem.user_id == User.id) & (OrgMem.status == "active"))
        .outerjoin(Org, Org.id == OrgMem.organization_id)
        .order_by(User.created_at.desc())
    )

    if search:
        query = query.where(
            User.name.ilike(f"%{search}%")
            | User.email.ilike(f"%{search}%")
            | User.phone.ilike(f"%{search}%")
        )

    result = await db.execute(query)
    rows = result.all()

    return [
        _AllUserResponse(
            id=u.id, name=u.name, email=u.email, phone=u.phone,
            account_type=u.account_type, is_superadmin=u.is_superadmin,
            created_at=u.created_at,
            organization_name=org.name if org else None,
            role=om.role if om else None,
            department=om.department if om else None,
            position=om.position if om else None,
            status=om.status if om else None,
        )
        for u, om, org in rows
    ]


# ══════════════════════════════════════════════
# 부서 CRUD — 조직별 부서 관리
# 각 조직이 자체 부서명을 departments 테이블에서 관리.
# 조직 admin/owner가 추가/삭제/이름변경 가능.
# ══════════════════════════════════════════════

class _DeptRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class _DeptResponse(BaseModel):
    id: UUID
    organization_id: UUID
    name: str
    created_at: object


@router.get("/departments", response_model=list[_DeptResponse])
async def list_departments(
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """현재 조직의 부서 목록"""
    user, member, org = org_tuple
    result = await db.execute(
        select(Department).where(Department.organization_id == org.id).order_by(Department.name)
    )
    return [_DeptResponse(id=d.id, organization_id=d.organization_id, name=d.name, created_at=d.created_at) for d in result.scalars()]


@router.post("/departments", response_model=_DeptResponse, status_code=201)
async def create_department(
    payload: _DeptRequest,
    org_tuple=Depends(require_role("owner", "admin")),
    db: AsyncSession = Depends(get_db),
):
    """부서 추가 (admin/owner 전용)"""
    user, member, org = org_tuple
    existing = await db.scalar(
        select(Department.id).where(Department.organization_id == org.id, Department.name == payload.name)
    )
    if existing:
        raise HTTPException(status_code=409, detail="이미 존재하는 부서명입니다.")
    dept = Department(organization_id=org.id, name=payload.name)
    db.add(dept)
    await db.flush()
    return _DeptResponse(id=dept.id, organization_id=dept.organization_id, name=dept.name, created_at=dept.created_at)


@router.patch("/departments/{dept_id}", response_model=_DeptResponse)
async def update_department(
    dept_id: UUID,
    payload: _DeptRequest,
    org_tuple=Depends(require_role("owner", "admin")),
    db: AsyncSession = Depends(get_db),
):
    """부서 이름 변경 (admin/owner 전용)"""
    user, member, org = org_tuple
    dept = await db.scalar(select(Department).where(Department.id == dept_id, Department.organization_id == org.id))
    if not dept:
        raise HTTPException(status_code=404, detail="부서를 찾을 수 없습니다.")
    dept.name = payload.name
    await db.flush()
    return _DeptResponse(id=dept.id, organization_id=dept.organization_id, name=dept.name, created_at=dept.created_at)


@router.delete("/departments/{dept_id}", status_code=204)
async def delete_department(
    dept_id: UUID,
    org_tuple=Depends(require_role("owner", "admin")),
    db: AsyncSession = Depends(get_db),
):
    """부서 삭제 (admin/owner 전용)"""
    user, member, org = org_tuple
    dept = await db.scalar(select(Department).where(Department.id == dept_id, Department.organization_id == org.id))
    if not dept:
        raise HTTPException(status_code=404, detail="부서를 찾을 수 없습니다.")
    await db.delete(dept)
    await db.flush()


# ══════════════════════════════════════════════
# 슈퍼어드민 전용: 전체 조직 목록 + 조직별 부서 조회
# ══════════════════════════════════════════════

class _OrgListItem(BaseModel):
    id: UUID
    name: str
    biz_number: str | None = None
    member_count: int = 0

@router.get("/admin/all-orgs", response_model=list[_OrgListItem])
async def list_all_organizations(
    superadmin=Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    """플랫폼 전체 조직 목록 (슈퍼어드민 전용). 멤버 수 포함."""
    result = await db.execute(
        select(
            Organization,
            func.count(OrganizationMember.id).label("cnt"),
        )
        .outerjoin(OrganizationMember, (OrganizationMember.organization_id == Organization.id) & (OrganizationMember.status == "active"))
        .group_by(Organization.id)
        .order_by(Organization.name)
    )
    return [
        _OrgListItem(id=org.id, name=org.name, biz_number=org.biz_number, member_count=cnt)
        for org, cnt in result.all()
    ]


@router.get("/admin/orgs/{org_id}/departments", response_model=list[_DeptResponse])
async def list_org_departments(
    org_id: UUID,
    superadmin=Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    """특정 조직의 부서 목록 (슈퍼어드민 전용)."""
    result = await db.execute(
        select(Department).where(Department.organization_id == org_id).order_by(Department.name)
    )
    return [
        _DeptResponse(id=d.id, organization_id=d.organization_id, name=d.name, created_at=d.created_at)
        for d in result.scalars()
    ]
