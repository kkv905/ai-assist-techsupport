"""Состояния FSM для Telegram-бота."""

from aiogram.fsm.state import State, StatesGroup


class AskFlow(StatesGroup):
    """Описывает шаги сценария уточняющего вопроса."""

    waiting_for_topic = State()
    waiting_for_question = State()
    confirming = State()
