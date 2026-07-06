"""Сервис административных операций backend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.schemas import AdminUserOut, BroadcastDeliveryOut, BroadcastOut, StatsOut
from app.chat.repositories.pg_models import BroadcastQueueRow, ChatMessageRow, ChatRow, MessageFeedbackRow, ModerationEventRow


@dataclass(slots=True)
class AdminService:
    """Выполняет административные запросы к PostgreSQL."""

    session: AsyncSession

    async def get_stats(self) -> StatsOut:
        """Собирает базовую operational-статистику за последние 24 часа."""

        since = datetime.now(UTC) - timedelta(hours=24)

        total_messages = int(
            await self.session.scalar(
                select(func.count(ChatMessageRow.id)).where(
                    ChatMessageRow.created_at >= since,
                    ChatMessageRow.deleted_at.is_(None),
                )
            )
            or 0
        )
        active_users = int(
            await self.session.scalar(
                select(func.count(func.distinct(ChatRow.owner_external_id)))
                .select_from(ChatMessageRow)
                .join(ChatRow, ChatRow.id == ChatMessageRow.chat_id)
                .where(ChatMessageRow.created_at >= since, ChatMessageRow.deleted_at.is_(None))
            )
            or 0
        )
        avg_latency_ms = float(
            await self.session.scalar(
                select(func.avg(ChatMessageRow.latency_ms)).where(
                    ChatMessageRow.created_at >= since,
                    ChatMessageRow.deleted_at.is_(None),
                    ChatMessageRow.role == "assistant",
                    ChatMessageRow.latency_ms.is_not(None),
                )
            )
            or 0.0
        )
        blocked_inputs = int(
            await self.session.scalar(
                select(func.count(ModerationEventRow.id)).where(
                    ModerationEventRow.created_at >= since,
                    ModerationEventRow.direction == "input",
                    ModerationEventRow.allowed.is_(False),
                )
            )
            or 0
        )
        allowed_inputs = int(
            await self.session.scalar(
                select(func.count(ChatMessageRow.id)).where(
                    ChatMessageRow.created_at >= since,
                    ChatMessageRow.deleted_at.is_(None),
                    ChatMessageRow.role == "user",
                )
            )
            or 0
        )
        total_feedback = int(
            await self.session.scalar(
                select(func.count(MessageFeedbackRow.id)).where(MessageFeedbackRow.created_at >= since)
            )
            or 0
        )
        up_feedback = int(
            await self.session.scalar(
                select(func.count(MessageFeedbackRow.id)).where(
                    MessageFeedbackRow.created_at >= since,
                    MessageFeedbackRow.value == "up",
                )
            )
            or 0
        )

        moderation_block_rate = blocked_inputs / (blocked_inputs + allowed_inputs) if (blocked_inputs + allowed_inputs) else 0.0
        feedback_up_ratio = up_feedback / total_feedback if total_feedback else 0.0
        return StatsOut(
            total_messages=total_messages,
            active_users=active_users,
            avg_latency_ms=round(avg_latency_ms, 2),
            moderation_block_rate=round(moderation_block_rate, 4),
            feedback_up_ratio=round(feedback_up_ratio, 4),
        )

    async def list_users(self, limit: int = 50) -> list[AdminUserOut]:
        """Возвращает последних пользователей с числом чатов и временем активности."""

        last_seen = func.max(func.coalesce(ChatMessageRow.created_at, ChatRow.created_at))
        query = (
            select(
                ChatRow.owner_external_id,
                ChatRow.interface,
                func.count(func.distinct(ChatRow.id)).label("chats_count"),
                last_seen.label("last_seen_at"),
            )
            .select_from(ChatRow)
            .join(ChatMessageRow, ChatMessageRow.chat_id == ChatRow.id, isouter=True)
            .group_by(ChatRow.owner_external_id, ChatRow.interface)
            .order_by(last_seen.desc())
            .limit(limit)
        )
        rows = (await self.session.execute(query)).all()
        return [
            AdminUserOut(
                owner_external_id=row.owner_external_id,
                interface=row.interface,
                chats_count=row.chats_count,
                last_seen_at=row.last_seen_at,
            )
            for row in rows
            if row.last_seen_at is not None
        ]

    async def create_broadcast(self, message: str, interface_filter: str) -> BroadcastOut:
        """Ставит задание рассылки в очередь."""

        row = BroadcastQueueRow(message=message, interface=interface_filter, status="pending")
        self.session.add(row)
        await self.session.commit()
        await self.session.refresh(row)
        return BroadcastOut(broadcast_id=row.id, status=row.status)

    async def claim_broadcasts(self, interface_filter: str, limit: int = 10) -> list[BroadcastDeliveryOut]:
        """Выбирает pending-рассылки, помечает их processing и подбирает получателей."""

        query = (
            select(BroadcastQueueRow)
            .where(BroadcastQueueRow.interface == interface_filter, BroadcastQueueRow.status == "pending")
            .order_by(BroadcastQueueRow.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        rows = (await self.session.scalars(query)).all()
        if not rows:
            return []

        claimed_at = datetime.now(UTC)
        deliveries: list[BroadcastDeliveryOut] = []
        recipients_query = select(func.distinct(ChatRow.owner_external_id)).where(ChatRow.interface == interface_filter)
        recipient_ids = [str(value) for value in (await self.session.scalars(recipients_query)).all()]
        for row in rows:
            row.status = "processing"
            row.claimed_at = claimed_at
            deliveries.append(
                BroadcastDeliveryOut(
                    broadcast_id=row.id,
                    message=row.message,
                    interface=row.interface,
                    recipient_ids=recipient_ids,
                )
            )
        await self.session.commit()
        return deliveries

    async def complete_broadcast(self, broadcast_id: UUID, status: str, error: str | None = None) -> None:
        """Фиксирует завершение фоновой рассылки."""

        query = (
            update(BroadcastQueueRow)
            .where(BroadcastQueueRow.id == broadcast_id)
            .values(
                status=status,
                error=error,
                completed_at=datetime.now(UTC),
            )
        )
        await self.session.execute(query)
        await self.session.commit()
