from __future__ import annotations

from dataclasses import dataclass


class Units:
    """Unit conversion constants."""

    METERS_PER_MILE = 1609.34
    MILES_PER_METER = 1 / METERS_PER_MILE


@dataclass(frozen=True)
class Distance:
    """Immutable distance value object stored internally in meters."""

    _meters: float

    @property
    def meters(self) -> float:
        return self._meters

    @property
    def miles(self) -> float:
        return self._meters * Units.MILES_PER_METER

    @classmethod
    def from_meters(cls, meters: float) -> "Distance":
        return cls(float(meters))

    @classmethod
    def from_miles(cls, miles: float) -> "Distance":
        return cls(float(miles) * Units.METERS_PER_MILE)

    def scaled(self, factor: float) -> "Distance":
        return Distance.from_meters(self._meters * float(factor))

