from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DirectorAction:
    """One time-bounded visual instruction for the renderer."""

    start: float
    end: float
    action: str
    reason: str = ""
    priority: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DirectorPlan:
    """Versioned contract between AI directors and the renderer."""

    version: int
    genre: str
    confidence: float
    actions: List[DirectorAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["actions"] = [a.to_dict() for a in self.actions]
        return data

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
