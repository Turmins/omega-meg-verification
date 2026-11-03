"""Participant-level six-fold rotation and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import random

from omega_dl.data.index import Recording, records_by_participant
from omega_dl.data.windows import Window


@dataclass(frozen=True)
class Fold:
    index: int
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]

    @property
    def gallery_subjects(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.val_subjects) | set(self.test_subjects)))


def make_blocks(records: list[Recording], seed: int = 1337, block_size: int = 10) -> list[tuple[str, ...]]:
    grouped = records_by_participant(records)
    if len(grouped) != 60 or block_size != 10:
        raise ValueError("The protocol requires 60 participants in six blocks of ten")
    rng = random.Random(seed)
    subjects = list(sorted(grouped))
    rng.shuffle(subjects)
    # Greedy balancing by probe-session counts, with deterministic tie-breaking.
    blocks: list[list[str]] = [[] for _ in range(6)]
    counts = [Counter() for _ in range(6)]
    shuffled_order = {subject: index for index, subject in enumerate(subjects)}
    subjects.sort(key=lambda s: (next(item.session for item in grouped[s] if item.role == "probe"), shuffled_order[s]))
    for subject in subjects:
        session = next(item.session for item in grouped[subject] if item.role == "probe")
        target = min(range(6), key=lambda i: (counts[i][session], len(blocks[i]), i))
        blocks[target].append(subject)
        counts[target][session] += 1
    if any(len(block) != block_size for block in blocks):
        raise AssertionError("Block construction did not produce six blocks of ten")
    return [tuple(sorted(block)) for block in blocks]


def make_folds(records: list[Recording], seed: int = 1337) -> list[Fold]:
    blocks = make_blocks(records, seed)
    folds = []
    for index in range(6):
        val = blocks[(index + 1) % 6]
        test = blocks[index]
        train = tuple(sorted(set().union(*[set(blocks[j]) for j in range(6) if j not in {index, (index + 1) % 6}])) )
        folds.append(Fold(index, train, val, test))
    return folds


def assert_no_leakage(records: list[Recording], folds: list[Fold]) -> None:
    all_subjects = {record.participant_id for record in records}
    seen_tests: list[str] = []
    for fold in folds:
        sets = [set(fold.train_subjects), set(fold.val_subjects), set(fold.test_subjects)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError(f"Subject leakage in fold {fold.index}")
        if set().union(*sets) != all_subjects:
            raise AssertionError(f"Fold {fold.index} does not cover all subjects")
        seen_tests.extend(fold.test_subjects)
        tags: dict[str, set[str]] = {}
        for record in records:
            if record.participant_id in fold.train_subjects:
                tags.setdefault(record.recording_id, set()).add("train")
            elif record.participant_id in fold.val_subjects:
                tags.setdefault(record.recording_id, set()).add("val")
            elif record.participant_id in fold.test_subjects:
                tags.setdefault(record.recording_id, set()).add("test")
        if any(len(values) != 1 for values in tags.values()):
            raise AssertionError(f"Recording leakage in fold {fold.index}")
    if sorted(seen_tests) != sorted(all_subjects):
        raise AssertionError("Each participant must be tested exactly once")


def split_for_subject(fold: Fold, subject: str) -> str:
    if subject in fold.train_subjects:
        return "train"
    if subject in fold.val_subjects:
        return "val"
    if subject in fold.test_subjects:
        return "test"
    raise KeyError(subject)


def assert_window_provenance(windows: list[Window], fold: Fold) -> None:
    for item in windows:
        if item.split != split_for_subject(fold, item.participant_id):
            raise AssertionError(f"Window has inconsistent split provenance: {item.provenance}")
