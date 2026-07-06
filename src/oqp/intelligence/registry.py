"""Registry for modular advisory engines."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from oqp.intelligence.base import BaseEngine


EngineFactory = Callable[[], BaseEngine]


class EngineRegistry:
    """Small factory registry inspired by the alpha research lab structure."""

    def __init__(self) -> None:
        self._factories: dict[str, EngineFactory] = {}

    def register_factory(self, engine_id: str, factory: EngineFactory) -> None:
        key = self._normalize(engine_id)
        if key in self._factories:
            raise ValueError(f"Engine already registered: {engine_id!r}")
        self._factories[key] = factory

    def register_instance(self, engine: BaseEngine) -> None:
        self.register_factory(engine.engine_id, lambda engine=engine: engine)

    def create(self, engine_id: str) -> BaseEngine:
        key = self._normalize(engine_id)
        if key not in self._factories:
            supported = ", ".join(self.engine_ids())
            raise ValueError(f"Unknown engine: {engine_id!r}. Supported: {supported}")
        return self._factories[key]()

    def engine_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create_many(self, engine_ids: Iterable[str] | None = None) -> list[BaseEngine]:
        ids = self.engine_ids() if engine_ids is None else tuple(engine_ids)
        return [self.create(engine_id) for engine_id in ids]

    @staticmethod
    def _normalize(engine_id: str) -> str:
        return engine_id.strip().lower().replace("-", "_")
