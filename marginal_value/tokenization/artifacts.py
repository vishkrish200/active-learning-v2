from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TokenSequence:
    sample_id: str
    split: str
    base_token_ids: list[int]
    primitive_token_ids: list[str]
    primitive_durations_sec: list[float]
    quality_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.primitive_token_ids) != len(self.primitive_durations_sec):
            raise ValueError("primitive_token_ids and primitive_durations_sec must have the same length.")
        object.__setattr__(self, "base_token_ids", [int(token_id) for token_id in self.base_token_ids])
        object.__setattr__(self, "primitive_token_ids", [str(token_id) for token_id in self.primitive_token_ids])
        object.__setattr__(
            self,
            "primitive_durations_sec",
            [float(duration) for duration in self.primitive_durations_sec],
        )
        if self.quality_score is not None:
            object.__setattr__(self, "quality_score", float(self.quality_score))
        object.__setattr__(self, "metadata", dict(self.metadata))


def write_token_sequences_jsonl(path: str | Path, sequences: Iterable[TokenSequence]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sequence in sequences:
            handle.write(json.dumps(asdict(sequence), sort_keys=True, default=str))
            handle.write("\n")


def read_token_sequences_jsonl(path: str | Path) -> list[TokenSequence]:
    input_path = Path(path)
    sequences: list[TokenSequence] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            sequences.append(TokenSequence(**payload))
    return sequences


def write_bpe_merges_json(path: str | Path, primitives: Iterable[tuple[int, ...]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for primitive in primitives:
        codes = tuple(int(code) for code in primitive)
        rows.append(
            {
                "primitive_id": "_".join(str(code) for code in codes),
                "codes": list(codes),
                "length": len(codes),
            }
        )
    output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
