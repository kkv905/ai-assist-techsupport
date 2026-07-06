"""add moderation admin feedback tables

Revision ID: 20260706_01_admin_feedback
Revises: 20260705_01_chat_media_refs
Create Date: 2026-07-06 14:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260706_01_admin_feedback"
down_revision = "20260705_01_chat_media_refs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Добавляет таблицы feedback, модерации, очереди рассылок и колонку latency."""

    op.add_column("chat_messages", sa.Column("latency_ms", sa.Integer(), nullable=True))
    op.create_table(
        "message_feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("owner_external_id", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_external_id", "message_id", name="uq_message_feedback_owner_message"),
    )
    op.create_table(
        "moderation_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chat_id", sa.Uuid(), nullable=True),
        sa.Column("owner_external_id", sa.Text(), nullable=True),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("allowed", sa.Boolean(), nullable=False),
        sa.Column("categories", sa.JSON(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("blocked_by", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "broadcast_queue",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("interface", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Удаляет новые таблицы и колонку latency."""

    op.drop_table("broadcast_queue")
    op.drop_table("moderation_events")
    op.drop_table("message_feedback")
    op.drop_column("chat_messages", "latency_ms")
