"""Window generation with evaluation independence and provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_dl.data.index import Recording


@dataclass(frozen=True)
class Window:
    participant_id: str
    recording_id: str
    role: str
    session: str
    start: int
    stop: int
    split: str
    clipped_fraction: float = 0.0

    @property
    def provenance(self) -> tuple[str, str, str, int, int]:
        return (self.participant_id, self.recording_id, self.session, self.start, self.stop)


def window_starts(n_samples: int, window_len: int, hop: int) -> list[int]:
    if window_len <= 0 or hop <= 0 or n_samples < window_len:
        raise ValueError("Window length and hop must be positive and fit the signal")
    return list(range(0, n_samples - window_len + 1, hop))


def make_windows(
    record: Recording,
    split: str,
    window_len: int = 1280,
    hop: int = 640,
    n_samples: int = 30720,
    clipped_mask: np.ndarray | None = None,
    max_clipped_fraction: float = 0.01,
) -> list[Window]:
    result = []
    for start in window_starts(n_samples, window_len, hop):
        stop = start + window_len
        fraction = 0.0 if clipped_mask is None else float(np.asarray(clipped_mask[:, start:stop]).mean())
        if fraction <= max_clipped_fraction:
            result.append(Window(record.participant_id, record.recording_id, record.role, record.session, start, stop, split, fraction))
    return result


def assert_eval_windows_non_overlapping(windows: list[Window]) -> None:
    by_recording: dict[str, list[Window]] = {}
    for item in windows:
        by_recording.setdefault(item.recording_id, []).append(item)
    for recording_id, items in by_recording.items():
        ordered = sorted(items, key=lambda item: item.start)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start < previous.stop:
                raise AssertionError(f"Overlapping evaluation windows in {recording_id}")
