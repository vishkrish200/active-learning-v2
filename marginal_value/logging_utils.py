from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def log_event(component: str, event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "component": component,
        "event": event,
    }
    payload.update(fields)
    print(json.dumps(payload, default=str, sort_keys=True), flush=True)


def log_progress(
    component: str,
    event: str,
    *,
    index: int,
    total: int,
    every: int,
    **fields: Any,
) -> bool:
    if total <= 0:
        return False
    should_log = index == total or (every > 0 and index % every == 0)
    if should_log:
        log_event(component, event, index=index, total=total, **fields)
    return should_log

