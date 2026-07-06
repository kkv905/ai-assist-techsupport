"""Зависимости и роуты административного API."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from app.admin.schemas import (
    AdminUserOut,
    BroadcastClaimIn,
    BroadcastCompleteIn,
    BroadcastDeliveryOut,
    BroadcastIn,
    BroadcastOut,
    StatsOut,
)
from app.admin.service import AdminService
from app.chat.deps import DbSessionDep
from app.core.config import Settings, get_settings

router = APIRouter(prefix="/chats/admin", tags=["chat-admin"])


async def require_admin(
    x_admin_token: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    """Проверяет заголовок `X-Admin-Token` для административного доступа."""

    if x_admin_token != settings.admin_token.get_secret_value():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Недостаточно прав.")


async def get_admin_service(session: DbSessionDep) -> AdminService:
    """Создает сервис административных запросов поверх PostgreSQL-сессии."""

    if session is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Admin API требует PostgreSQL.")
    return AdminService(session=session)


AdminServiceDep = Annotated[AdminService, Depends(get_admin_service)]
AdminAuthDep = Annotated[None, Depends(require_admin)]


@router.get("/stats", response_model=StatsOut)
async def get_stats(_: AdminAuthDep, admin_service: AdminServiceDep) -> StatsOut:
    """Возвращает агрегированную статистику backend за последние 24 часа."""

    return await admin_service.get_stats()


@router.get("/users", response_model=list[AdminUserOut])
async def list_users(
    _: AdminAuthDep,
    admin_service: AdminServiceDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[AdminUserOut]:
    """Возвращает последних пользователей с краткой статистикой по чатам."""

    return await admin_service.list_users(limit=limit)


@router.post("/broadcast", response_model=BroadcastOut, status_code=status.HTTP_201_CREATED)
async def create_broadcast(
    payload: BroadcastIn,
    _: AdminAuthDep,
    admin_service: AdminServiceDep,
) -> BroadcastOut:
    """Создает отложенное задание на массовую рассылку."""

    return await admin_service.create_broadcast(payload.message, payload.interface_filter)


@router.post("/broadcasts/claim", response_model=list[BroadcastDeliveryOut])
async def claim_broadcasts(
    payload: BroadcastClaimIn,
    _: AdminAuthDep,
    admin_service: AdminServiceDep,
) -> list[BroadcastDeliveryOut]:
    """Выдает боту пакет pending-рассылок и помечает их как взятые в работу."""

    return await admin_service.claim_broadcasts(payload.interface_filter, limit=payload.limit)


@router.post("/broadcasts/{broadcast_id}/complete", response_model=dict[str, str])
async def complete_broadcast(
    broadcast_id: UUID,
    payload: BroadcastCompleteIn,
    _: AdminAuthDep,
    admin_service: AdminServiceDep,
) -> dict[str, str]:
    """Фиксирует итог обработки задания фоновой рассылки."""

    await admin_service.complete_broadcast(broadcast_id, payload.status, payload.error)
    return {"status": "ok"}
