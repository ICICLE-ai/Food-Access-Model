from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import UnaryExpression


class CRUDMixin:
    @classmethod
    async def create(
        cls, session: AsyncSession, **kwargs: dict[str, Any]
    ):
        print("Creating", cls.__name__, kwargs)
        try:
            instance = cls(**kwargs)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            return instance
        except IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"{cls.__name__} already exists",
            )

    @classmethod
    async def get(cls, session: AsyncSession, identifier: tuple[Any, ...]):
        instance = await session.get(cls, identifier)
        return instance

    @classmethod
    async def update(
        cls, session: AsyncSession, identifier: tuple[Any,...], **kwargs: dict[str, Any]
    ):
        instance = await session.get(cls, identifier)
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        await session.commit()
        await session.refresh(instance)
        return instance

    @classmethod
    async def delete(cls, session: AsyncSession, identifier: tuple[Any,...]):
        instance = await session.get(cls, identifier)
        await session.delete(instance)
        await session.commit()

    @classmethod
    async def get_all(cls, session: AsyncSession, identifier: dict[str, Any], order_by: UnaryExpression = None):
        conditions = [getattr(cls, key) == value for key,value in identifier.items()]
        result = await session.execute(
            select(cls).where(
                and_(
                    *conditions
                    )
            ).order_by(order_by)
        )
        return result.scalars().all()