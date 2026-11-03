"""Explicit before/after checks for named input paths, without a legacy runner."""
from pathlib import Path
import argparse
import json
from _common import ROOT, child, sha256, write_new

def inventory(paths):
    result = {}
    for value in paths:
        path = Path(value).resolve(strict=True)
        if path.is_symlink():
            raise ValueError('Snapshot real input paths, not symlinks')
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in files:
            if file.is_symlink():
                raise ValueError('Snapshot contains a symlink')
            if file.is_file():
                result[str(file)] = {'bytes': file.stat().st_size, 'sha256': sha256(file)}
    if not result:
        raise ValueError('No input files found')
    return result

def snapshot(paths, output):
    output = child(ROOT, output)
    roots = [str(Path(p).resolve(strict=True)) for p in paths]
    for value in roots:
        p = Path(value)
        if output == p or p in output.parents:
            raise ValueError('Snapshot output must be outside the protected inputs')
    write_new(output, {'schema': 'publication_input_snapshot_v1', 'roots': roots, 'files': inventory(roots)})

def verify(path):
    before = json.loads(Path(path).read_text('utf-8'))
    if before.get('schema') != 'publication_input_snapshot_v1':
        raise ValueError('Unknown snapshot schema')
    if inventory(before['roots']) != before['files']:
        raise ValueError('Protected input files changed')
    return len(before['files'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['snapshot', 'verify'])
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--path', type=Path, nargs='+')
    args = parser.parse_args()
    if args.stage == 'snapshot':
        if not args.path:
            parser.error('snapshot requires explicit --path inputs')
        snapshot(args.path, args.snapshot)
        print('New input snapshot recorded')
    else:
        print(json.dumps({'unchanged_files': verify(args.snapshot)}))

if __name__ == '__main__':
    main()
