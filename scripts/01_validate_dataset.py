"""Validate the read-only dataset and write a compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from omega_dl.data.validate import validate_dataset
from omega_dl.paths import resolve_data_root, ensure_output_tree, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--skip-arrays", action="store_true")
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root, ROOT)
    report = validate_dataset(data_root, check_arrays=not args.skip_arrays)
    output = ensure_output_tree()["metrics"] / "dataset_validation.json"
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
