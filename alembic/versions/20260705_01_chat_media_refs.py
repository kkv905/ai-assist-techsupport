"""add media refs to chat messages

Revision ID: 20260705_01_chat_media_refs
Revises: 20260702_01_chat_tables
Create Date: 2026-07-05 18:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260705_01_chat_media_refs"
down_revision = "20260702_01_chat_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Добавляет JSON-колонку для сохранения мультимодальных частей сообщения."""

    op.add_column("chat_messages", sa.Column("media_refs", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Удаляет колонку мультимодальных ссылок из сообщений чата."""

    op.drop_column("chat_messages", "media_refs")
