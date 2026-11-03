"""Typed access to the frozen recording index."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recording:
    participant_id: str
    pair_index: int
    role: str
    session: str
    run: int
    cache_filename: str
    cache_path: str

    @property
    def recording_id(self) -> str:
        return Path(self.cache_filename).stem


def load_recording_index(data_root: Path) -> list[Recording]:
    path = data_root / "recording_index.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"participant_id", "pair_index", "role", "session", "run", "cache_filename", "cache_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Index is missing required columns: {sorted(required)}")
    records = [
        Recording(
            participant_id=row["participant_id"],
            pair_index=int(row["pair_index"]),
            role=row["role"],
            session=row["session"],
            run=int(row["run"]),
            cache_filename=row["cache_filename"],
            cache_path=row["cache_path"],
        )
        for row in rows
    ]
    if len(records) != 120:
        raise ValueError(f"Expected 120 recordings, found {len(records)}")
    return records


def records_by_participant(records: list[Recording]) -> dict[str, list[Recording]]:
    result: dict[str, list[Recording]] = {}
    for record in records:
        result.setdefault(record.participant_id, []).append(record)
    return result
