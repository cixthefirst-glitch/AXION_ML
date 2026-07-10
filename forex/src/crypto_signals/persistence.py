from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


@dataclass
class ModelMetadata:
    version: str
    created_at: str
    config: Dict[str, Any]


def save_model(model: Any, path: str, metadata: Optional[ModelMetadata] = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "metadata": asdict(metadata) if metadata else None}
    joblib.dump(payload, target)


def load_model(path: str) -> Optional[Dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return None
    payload = joblib.load(target)
    if not isinstance(payload, dict) or "model" not in payload:
        return None
    return payload
