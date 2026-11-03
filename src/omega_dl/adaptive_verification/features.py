"""Reuse verified v1 windows; store only compact, window-local graph features."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from omega_dl.paths import inside_extension, sha256_file
from omega_dl.verification.protocol import read_json, save_json, digest
from omega_dl.verification.preprocessing import EDGES


def graph_features(coherency, neighbors=12):
    values = np.asarray(coherency, dtype=np.complex64)
    if values.ndim != 3 or values.shape[1:] != (3, len(EDGES[0])):
        raise ValueError("Expected windows x 3 bands x upper-triangle coherency")
    if not np.isfinite(values).all() or not 1 <= neighbors < 260:
        raise ValueError("Invalid coherency or neighborhood size")
    matrix = np.zeros((len(values), 3, 260, 260), dtype=np.complex64)
    matrix[:, :, EDGES[0], EDGES[1]] = values
    matrix[:, :, EDGES[1], EDGES[0]] = values.conj()
    strength = np.abs(matrix)
    nodes = np.concatenate((strength.mean(-1), matrix.real.mean(-1),
                            matrix.imag.mean(-1), strength.std(-1)), axis=1).transpose(0, 2, 1)
    ranking = strength.mean(axis=1)
    ranking[:, np.arange(260), np.arange(260)] = -np.inf
    adjacent = np.argsort(-ranking, axis=-1, kind="stable")[:, :, :neighbors]
    selected = np.take_along_axis(matrix, adjacent[:, None], axis=-1).transpose(0, 2, 3, 1)
    magnitude = np.abs(selected)
    unit = np.divide(selected, magnitude, out=np.zeros_like(selected), where=magnitude > 1e-12)
    # Directed receiver-neighbor phase: reversing an edge negates its sine.
    edges = np.concatenate((unit.real, unit.imag, magnitude), axis=-1)
    return nodes.astype(np.float16), adjacent.astype(np.int16), edges.astype(np.float16)


class FeatureStore:
    def __init__(self, base_root, base_protocol, run_root, neighbors):
        self.base_root, self.base = base_root, base_protocol
        self.root = inside_extension(run_root / "graph_cache")
        self.neighbors, self.checked, self.loaded = neighbors, set(), {}
        self.transform_hash = sha256_file(Path(__file__))

    def source(self, record, kind):
        folder = self.base_root / "cache" / self.base["protocol_sha256"][:16] / record.recording_id
        manifest = read_json(folder / "manifest.json")
        signature = manifest["signature"]
        if (signature["protocol_sha256"] != self.base["protocol_sha256"]
                or signature["source"] != self.base["recordings"][record.recording_id]
                or signature["recording_id"] != record.recording_id):
            raise ValueError("Incompatible existing window cache")
        name = "cnn.npy" if kind == "eegnet" else "phase.npy"
        path = folder / name
        if str(path) not in self.checked:
            if sha256_file(path) != manifest["files_sha256"][name]:
                raise ValueError("Existing cache checksum differs")
            self.checked.add(str(path))
        return path, manifest["files_sha256"][name]

    def arrays(self, record, method):
        key = record.recording_id, method
        if key in self.loaded:
            return self.loaded[key]
        path, checksum = self.source(record, method)
        if method == "eegnet":
            result = (np.load(path, mmap_mode="r").reshape(24, 3, 260, 1280),)
        elif method == "phase":
            result = (np.load(path, mmap_mode="r"),)
        elif method == "graph":
            folder = self.root / record.recording_id
            signature = digest({"source": checksum, "neighbors": self.neighbors, "code": self.transform_hash})
            meta = folder / "manifest.json"
            names = ("nodes.npy", "neighbors.npy", "edges.npy")
            if not meta.exists():
                values = graph_features(np.load(path, mmap_mode="r"), self.neighbors)
                folder.mkdir(parents=True, exist_ok=True)
                for name, value in zip(names, values):
                    temporary = folder / (name + ".partial")
                    with temporary.open("wb") as stream:
                        np.save(stream, value, allow_pickle=False)
                    temporary.replace(folder / name)
                save_json(meta, {"signature": signature, "sha256": {n: sha256_file(folder / n) for n in names}})
            info = read_json(meta)
            if info["signature"] != signature or any(sha256_file(folder / n) != info["sha256"][n] for n in names):
                raise ValueError("Graph cache changed")
            result = tuple(np.load(folder / name, mmap_mode="r") for name in names)
        else:
            raise ValueError(method)
        self.loaded[key] = result
        return result

    def batch(self, records, indices, method, device):
        import torch
        items = [self.arrays(records[index // 24], method) for index in indices]
        result = []
        for component in range(len(items[0])):
            array = np.stack([item[component][index % 24] for item, index in zip(items, indices)])
            dtype = torch.long if method == "graph" and component == 1 else torch.float32
            result.append(torch.from_numpy(array).to(device=device, dtype=dtype))
        return result
