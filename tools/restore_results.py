"""Restore pinned saved runs and optionally three example caches; no inference."""
from pathlib import Path
import argparse
import json
import shutil
import uuid
from _common import ROOT, child, sha256, write_new

def plan(archive, include_example_cache=False):
    archive = Path(archive).resolve()
    if archive == ROOT or archive in ROOT.parents or ROOT in archive.parents:
        raise ValueError('Archive and publication checkout must be separate')
    manifest = json.loads((ROOT / 'artifacts/frozen_runs.json').read_text('utf-8'))
    files = dict(manifest['files'])
    if include_example_cache:
        protocol = json.loads(child(archive, 'outputs/verification_v1/protocol.json').read_text('utf-8'))
        first, second = protocol['folds'][0]['test_subjects'][:2]
        names = [name for name in protocol['recordings'] if name.startswith(first + '_') or name.startswith(second + '_probe')]
        if len(names) != 3:
            raise ValueError('Unexpected example recording grid')
        for name in names:
            for folder in [f"outputs/verification_v1/cache/{protocol['protocol_sha256'][:16]}/{name}",
                           f'outputs/adaptive_v2/graph_cache/{name}']:
                source = child(archive, folder)
                for source_file in source.iterdir():
                    if not source_file.is_file() or source_file.suffix not in {'.npy', '.json'}:
                        raise ValueError('Unexpected cache entry')
                    rel = source_file.relative_to(archive).as_posix()
                    files[rel] = {'sha256': sha256(source_file), 'bytes': source_file.stat().st_size}
    return archive, files

def restore(archive, files):
    # Validate every input and occupied destination before writing anything.
    for rel, entry in files.items():
        if not rel.startswith(('outputs/verification_v1/', 'outputs/adaptive_v2/')):
            raise ValueError('Manifest entry outside the declared runs')
        source, target = child(archive, rel), child(ROOT, rel)
        if source.stat().st_size != entry['bytes'] or sha256(source) != entry['sha256']:
            raise ValueError('Archive hash mismatch: ' + rel)
        if target.exists() and (not target.is_file() or sha256(target) != entry['sha256']):
            raise FileExistsError('Destination differs; no files overwritten: ' + rel)
    for rel, entry in files.items():
        source, target = child(archive, rel), child(ROOT, rel)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output, source.open('rb') as input_file:
                shutil.copyfileobj(input_file, output)
        if sha256(target) != entry['sha256']:
            raise ValueError('Copied file hash mismatch: ' + rel)
    receipt = ROOT / 'local_artifacts/imports' / (uuid.uuid4().hex + '.json')
    write_new(receipt, {'mode': 'byte-identical historical artifact import; not training provenance',
                        'archive': str(archive), 'files': files})
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=ROOT.parent / 'deep_learning_extension')
    parser.add_argument('--include-example-cache', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Show size without copying')
    args = parser.parse_args()
    archive, files = plan(args.archive_root, args.include_example_cache)
    result = {'files': len(files), 'bytes': sum(row['bytes'] for row in files.values()), 'training': False}
    if not args.plan:
        result['receipt'] = str(restore(archive, files).relative_to(ROOT))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
