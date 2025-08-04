from sqlalchemy.orm import Mapped, mapped_column, relationship

from food_access_model.database.database import Base
from food_access_model.database.mixin import CRUDMixin
class FoodStore(Base, CRUDMixin):
    __tablename__ = 'food_stores'

    name: Mapped[str] = mapped_column(primary_key=True)
    shop: Mapped[str] = mapped_column(nullable=False)
    geometry: Mapped[str] = mapped_column(nullable=False)

class Household(Base, CRUDMixin):
    __tablename__ = 'households'

    id: Mapped[int] = mapped_column(primary_key=True)
    polygon: Mapped[str] = mapped_column(nullable=False)
    income: Mapped[int] = mapped_column(nullable=False)
    household_size: Mapped[int] = mapped_column(nullable=False)
    vehicles: Mapped[int] = mapped_column(nullable=False)
    number_of_workers: Mapped[int] = mapped_column(nullable=False)
    walking_time: Mapped[str] = mapped_column(nullable=False)
    biking_time: Mapped[str] = mapped_column(nullable=False)
    transit_time: Mapped[str] = mapped_column(nullable=False)
    driving_time: Mapped[str] = mapped_column(nullable=False)

class Roads(Base, CRUDMixin):
    __tablename__ = 'roads'

    name: Mapped[str] = mapped_column(primary_key=True)
    highway: Mapped[str] = mapped_column(nullable=False)
    length: Mapped[int] = mapped_column(nullable=False)
    geometry: Mapped[str] = mapped_column(nullable=False)
    service: Mapped[str] = mapped_column(nullable=False)



