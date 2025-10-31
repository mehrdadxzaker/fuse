from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

Number = Union[int, float]
TemperatureSpec = Union[
    Number,
    Sequence[Tuple[int, Number]],
    Mapping[str, Any],
    "TemperatureSchedule",
    Callable[[int], Number],
]


class TemperatureSchedule:
    """Callable schedule that produces a temperature for a given iteration."""

    def value(self, iteration: int) -> float:
        raise NotImplementedError

    def manifest(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    def __call__(self, iteration: int) -> float:
        return self.value(iteration)


@dataclass
class ConstantSchedule(TemperatureSchedule):
    temperature: float

    def __post_init__(self) -> None:
        self.temperature = float(self.temperature)

    def value(self, iteration: int) -> float:
        return float(self.temperature)

    def manifest(self) -> Dict[str, Any]:
        return {"type": "constant", "temperature": float(self.temperature)}


@dataclass
class LinearRampSchedule(TemperatureSchedule):
    start: float
    end: float
    steps: int

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("LinearRampSchedule.steps must be positive")
        self.start = float(self.start)
        self.end = float(self.end)

    def value(self, iteration: int) -> float:
        if iteration <= 0:
            return self.start
        if iteration >= self.steps:
            return self.end
        ratio = iteration / float(self.steps)
        return self.start + (self.end - self.start) * ratio

    def manifest(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "start": float(self.start),
            "end": float(self.end),
            "steps": int(self.steps),
        }


@dataclass
class PiecewiseSchedule(TemperatureSchedule):
    points: Sequence[Tuple[int, Number]]
    _points: List[Tuple[int, float]] = field(init=False)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("PiecewiseSchedule requires at least one point")
        normalized: List[Tuple[int, float]] = []
        for step, value in self.points:
            step_int = int(step)
            normalized.append((step_int, float(value)))
        normalized.sort(key=lambda item: item[0])
        self._points = normalized

    def value(self, iteration: int) -> float:
        for step, value in reversed(self._points):
            if iteration >= step:
                return value
        return self._points[0][1]

    def manifest(self) -> Dict[str, Any]:
        return {
            "type": "piecewise",
            "points": [(int(step), float(value)) for step, value in self._points],
        }


@dataclass
class CallableSchedule(TemperatureSchedule):
    fn: Callable[[int], Number]
    description: Optional[str] = None

    def value(self, iteration: int) -> float:
        return float(self.fn(iteration))

    def manifest(self) -> Dict[str, Any]:
        desc = self.description or getattr(self.fn, "__name__", "callable")
        return {"type": "callable", "description": str(desc)}


def _coerce_callable(
    spec: TemperatureSchedule | Callable[[int], Number],
) -> TemperatureSchedule:
    if isinstance(spec, TemperatureSchedule):
        return spec
    desc = getattr(spec, "__name__", None)
    return CallableSchedule(spec, description=desc)


def _schedule_from_mapping(mapping: Mapping[str, Any]) -> TemperatureSchedule:
    schedule_type = mapping.get("type")
    if schedule_type is None:
        raise ValueError("Temperature schedule mapping requires a 'type' field")
    schedule_type = str(schedule_type).lower()
    if schedule_type == "constant":
        if "temperature" not in mapping and "value" not in mapping:
            raise ValueError("constant schedule requires 'temperature' or 'value'")
        temp = mapping.get("temperature", mapping.get("value"))
        return ConstantSchedule(temp)
    if schedule_type == "linear":
        for field in ("start", "end", "steps"):
            if field not in mapping:
                raise ValueError(f"linear schedule requires '{field}' field")
        return LinearRampSchedule(mapping["start"], mapping["end"], int(mapping["steps"]))
    if schedule_type == "piecewise":
        points = mapping.get("points")
        if not isinstance(points, Iterable):
            raise ValueError("piecewise schedule requires iterable 'points'")
        processed: List[Tuple[int, Number]] = []
        for item in points:
            if not isinstance(item, Iterable):
                raise ValueError("piecewise schedule points must be iterable pairs")
            pair = tuple(item)
            if len(pair) != 2:
                raise ValueError("piecewise schedule points must unpack into two values")
            step, value = pair
            processed.append((int(step), value))
        return PiecewiseSchedule(tuple(processed))
    raise ValueError(f"Unsupported temperature schedule type '{schedule_type}'")


def make_schedule(spec: TemperatureSpec) -> TemperatureSchedule:
    if isinstance(spec, TemperatureSchedule):
        return spec
    if isinstance(spec, Mapping):
        return _schedule_from_mapping(spec)
    if isinstance(spec, Sequence) and spec and isinstance(spec[0], (tuple, list)):
        processed: List[Tuple[int, Number]] = []
        for item in spec:
            if not isinstance(item, Iterable):
                raise TypeError("Piecewise schedule entries must be iterable")
            pair = tuple(item)
            if len(pair) != 2:
                raise TypeError("Piecewise schedule entries must be pairs")
            step, value = pair
            processed.append((int(step), value))
        return PiecewiseSchedule(tuple(processed))
    if callable(spec):
        return _coerce_callable(spec)
    if isinstance(spec, (int, float)):
        return ConstantSchedule(float(spec))
    raise TypeError(f"Unsupported temperature specification: {spec!r}")


def normalize_temperature_map(
    mapping: Optional[Mapping[str, TemperatureSpec]],
) -> Optional[Dict[str, TemperatureSchedule]]:
    if mapping is None:
        return None
    normalized: Dict[str, TemperatureSchedule] = {}
    for key, spec in mapping.items():
        if spec is None:
            continue
        normalized[str(key)] = make_schedule(spec)
    return normalized or None


def coerce_temperature_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception as exc:  # pragma: no cover - best effort
            raise TypeError("Temperature value must be convertible to float") from exc
    np_module: Any
    try:
        import numpy as np  # Lazy import to avoid hard dependency for callers

        np_module = np
    except Exception:  # pragma: no cover - optional dependency missing
        np_module = None
    if np_module is not None:
        try:
            arr = np_module.asarray(value)
            if arr.size != 1:
                raise ValueError("Temperature value must be scalar")
            return float(arr.reshape(()))
        except Exception as exc:  # pragma: no cover - fallback failures
            raise TypeError("Temperature value must be convertible to float") from exc
    raise TypeError(f"Temperature value '{value!r}' is not convertible to float")
