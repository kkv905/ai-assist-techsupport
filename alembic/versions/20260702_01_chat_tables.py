"""chat tables

Revision ID: 20260702_01_chat_tables
Revises: 
Create Date: 2026-07-02 16:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260702_01_chat_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Создает таблицы чатов и сообщений."""

    op.create_table(
        "chats",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_external_id", sa.Text(), nullable=False),
        sa.Column("interface", sa.Text(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chat_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_messages_chat_created",
        "chat_messages",
        ["chat_id", sa.text("created_at DESC")],
        unique=False,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    """Удаляет таблицы чатов и сообщений."""

    op.drop_index("ix_chat_messages_chat_created", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_table("chats")
