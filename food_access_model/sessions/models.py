from sqlalchemy.orm import Mapped, mapped_column, relationship

from food_access_model.database.database import Base
from food_access_model.database.mixin import CRUDMixin

class Session(Base, CRUDMixin):
    __tablename__ = 'sessions'
    id: Mapped[str] = mapped_column(primary_key=True)
    start_time: Mapped[str] = mapped_column(nullable=False)
    end_time: Mapped[str] = mapped_column(nullable=False)
    step: Mapped[int] = mapped_column(nullable=False)
