from enum import Enum, auto
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, func, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from food_access_model.database.database import Base
from food_access_model.database.mixin import CRUDMixin

class Status(Enum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

class Session(Base, CRUDMixin):
    __tablename__ = 'sessions'
    id: Mapped[int] = mapped_column(primary_key=True)
    current_step: Mapped[int] = mapped_column(nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    status: Mapped[Status] = mapped_column(SQLEnum(Status), default=Status.RUNNING, nullable=False)
    # steps: Mapped[list['SessionStep']] = relationship('SessionStep', back_populates='session')

class SessionStep(Base, CRUDMixin):
    __tablename__ = 'session_steps'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey('sessions.id'), nullable=False)
    household_id: Mapped[int] = mapped_column(ForeignKey('households.id'), nullable=False)
    step: Mapped[int] = mapped_column(default=0, nullable=False)
    mfai: Mapped[float] = mapped_column(nullable=False)
    color: Mapped[str] = mapped_column(nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # session: Mapped[Session] = relationship('Session', back_populates='steps')

class SessionSnapshot(Base, CRUDMixin):
    __tablename__ = 'session_snapshots'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey('sessions.id'), nullable=False)
    step: Mapped[int] = mapped_column(default=0, nullable=False)
    households_state: Mapped[dict] = mapped_column(JSONB, nullable=False) # {household_id: {mfai: float, color: str}}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # session: Mapped[Session] = relationship('Session', back_populates='snapshots')