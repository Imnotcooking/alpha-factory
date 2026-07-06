"""Central coordinator for advisory intelligence engines."""

from __future__ import annotations

from datetime import datetime, timezone
from traceback import format_exception_only

from oqp.intelligence.base import BaseEngine, EngineHealth, EngineResult, EngineStatus
from oqp.intelligence.context import EngineContext
from oqp.intelligence.registry import EngineRegistry
from oqp.intelligence.allocation_engine import AllocationAdvisoryEngine
from oqp.intelligence.portfolio_manager import PortfolioManagerEngine
from oqp.intelligence.regime_engine import RegimeSnapshotEngine
from oqp.intelligence.risk_engine import RiskControlRoomEngine


class IntelligenceCoordinator:
    """Run registered engines and return structured outputs by engine id."""

    def __init__(self, registry: EngineRegistry) -> None:
        self.registry = registry

    def run(
        self,
        context: EngineContext,
        *,
        engine_ids: tuple[str, ...] | None = None,
    ) -> dict[str, EngineResult]:
        results: dict[str, EngineResult] = {}
        for engine in self.registry.create_many(engine_ids):
            try:
                results[engine.engine_id] = engine.run(context)
            except Exception as exc:  # pragma: no cover - exercised by tests
                results[engine.engine_id] = self._failure_result(engine, exc)
        return results

    @staticmethod
    def _failure_result(engine: BaseEngine, exc: Exception) -> EngineResult:
        message = "".join(format_exception_only(type(exc), exc)).strip()
        health = EngineHealth(
            engine_id=engine.engine_id,
            status=EngineStatus.FAIL,
            message=message,
            metadata={"category": engine.category, "version": engine.version},
        )
        return EngineResult(
            engine_id=engine.engine_id,
            engine_name=engine.engine_name,
            status=EngineStatus.FAIL,
            generated_at=datetime.now(timezone.utc),
            summary=f"Engine failed: {message}",
            metadata={"category": engine.category, "version": engine.version},
            health=(health,),
        )


def default_intelligence_registry() -> EngineRegistry:
    registry = EngineRegistry()
    registry.register_factory(PortfolioManagerEngine.engine_id, PortfolioManagerEngine)
    registry.register_factory(AllocationAdvisoryEngine.engine_id, AllocationAdvisoryEngine)
    registry.register_factory(RegimeSnapshotEngine.engine_id, RegimeSnapshotEngine)
    registry.register_factory(RiskControlRoomEngine.engine_id, RiskControlRoomEngine)
    return registry


def default_intelligence_coordinator() -> IntelligenceCoordinator:
    return IntelligenceCoordinator(default_intelligence_registry())
