from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def leave_group_out_folds(rows: Iterable[dict[str, object]], group_key: str) -> list[dict[str, list[dict[str, object]]]]:
    by_group: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[row[group_key]].append(dict(row))

    folds = []
    all_groups = list(by_group)
    for group in all_groups:
        positives = by_group[group]
        support = [row for other_group in all_groups if other_group != group for row in by_group[other_group]]
        folds.append({"held_out_group": group, "positives": positives, "support": support})
    return folds

