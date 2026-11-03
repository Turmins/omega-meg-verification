"""Frozen dataset and protocol assertions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omega_dl.data.filters import EXPECTED_RAW_SHAPE
from omega_dl.data.index import load_recording_index, records_by_participant


def validate_dataset(data_root: Path, check_arrays: bool = True) -> dict[str, object]:
    protocol = json.loads((data_root / "protocol_FROZEN.json").read_text(encoding="utf-8"))
    records = load_recording_index(data_root)
    grouped = records_by_participant(records)
    channels = (data_root / "channels.txt").read_text(encoding="utf-8").splitlines()
    if protocol.get("protocol_version") != "omega_student_broadband_T260_v1":
        raise ValueError("Unexpected protocol version")
    if len(channels) != 260 or len(set(channels)) != 260:
        raise ValueError("channels.txt must contain 260 unique frozen channels")
    if len(grouped) != 60 or any(len(items) != 2 for items in grouped.values()):
        raise ValueError("Expected exactly two recordings for each of 60 participants")
    if any({item.role for item in items} != {"enroll", "probe"} for items in grouped.values()):
        raise ValueError("Each participant must have one enrollment and one probe recording")
    if any(len({item.session for item in items}) != 2 for items in grouped.values()):
        raise ValueError("Enrollment and probe sessions must differ")
    results = []
    for record in records:
        path = data_root / "cache" / record.cache_filename
        if check_arrays:
            array = np.load(path, mmap_mode="r")
            if array.shape != EXPECTED_RAW_SHAPE or array.dtype != np.float32:
                raise ValueError(f"Invalid array for {record.recording_id}: {array.shape} {array.dtype}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values for {record.recording_id}")
        results.append({"participant_id": record.participant_id, "recording_id": record.recording_id, "valid": True})
    return {"protocol_version": protocol["protocol_version"], "participants": len(grouped), "recordings": len(records), "channels": len(channels), "valid_files": len(results), "records": results}
