"""Inspect pinned trained weights and prepare a Git-ignored release candidate.

Inspection is not redistribution permission or a test against model memorization.
No source data, individual score tables or optimizer states are copied.
"""
from pathlib import Path
import argparse
import collections
import json
import re
import shutil
import sys
import zipfile
from _common import ROOT, child, sha256, write_new

sys.path.insert(0, str(ROOT / 'src'))

def inspect(archive):
    import torch
    from omega_dl.models.siamese_cnn import SiameseCNN
    from omega_dl.adaptive_verification.models import make_model
    archive = Path(archive).resolve()
    registry = json.loads((ROOT / 'artifacts/frozen_runs.json').read_text('utf-8'))['files']
    protocols = {name: json.loads(child(archive, f'outputs/{name}/protocol.json').read_text('utf-8'))
                 for name in ('verification_v1', 'adaptive_v2')}
    rows = []
    for relative, entry in sorted(registry.items()):
        if not relative.endswith('/best.pt'):
            continue
        source = child(archive, relative)
        if source.stat().st_size != entry['bytes'] or sha256(source) != entry['sha256']:
            raise ValueError('Weight differs from pinned historical artifact: ' + relative)
        with zipfile.ZipFile(source) as checkpoint:
            pickles = [n for n in checkpoint.namelist() if n.endswith('/data.pkl')]
            if len(pickles) != 1:
                raise ValueError('Unexpected checkpoint structure')
            metadata = checkpoint.read(pickles[0])
        id_hits = len(re.findall(rb'sub-\d+', metadata))
        path_hits = len(re.findall(rb'[A-Za-z]:[\\/]|/home/|/Users/', metadata))
        if id_hits or path_hits:
            raise ValueError('Identifier/path pattern found; keep checkpoint private')
        state = torch.load(source, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError('Checkpoint must contain tensors only')
        if not all(bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError('Non-finite tensor')
        parts = relative.split('/')
        fold = parts[3]
        if parts[1] == 'verification_v1':
            method = 'v1'
            model = SiameseCNN(n_bands=3, **protocols['verification_v1']['config']['model'])
        else:
            method = parts[4]
            model = make_model(method, protocols['adaptive_v2']['config'])
        model.load_state_dict(state, strict=True)
        rows.append({'method': method, 'fold': fold, 'source_relative_path': relative,
                     'candidate_relative_path': f'{method}/{fold}.pt', 'bytes': entry['bytes'],
                     'sha256': entry['sha256'], 'tensor_count': len(state),
                     'tensor_elements': sum(v.numel() for v in state.values()),
                     'dtypes': dict(collections.Counter(str(v.dtype) for v in state.values())),
                     'strict_architecture_load': True, 'all_tensors_finite': True,
                     'text_identifier_patterns': id_hits, 'absolute_path_patterns': path_hits})
    if len(rows) != 18:
        raise ValueError('Expected six V1, six EEGNet and six Graph trained checkpoints')
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=ROOT.parent / 'deep_learning_extension')
    parser.add_argument('--out', default='local_artifacts/release_candidates/trained_weights')
    args = parser.parse_args()
    output = child(ROOT, args.out)
    output.relative_to((ROOT / 'local_artifacts').resolve())
    archive = args.archive_root.resolve()
    if output == archive or archive in output.parents:
        raise ValueError('Candidate must not write into the historical archive')
    if output.exists():
        raise FileExistsError('Use a fresh Git-ignored candidate directory')
    rows = inspect(archive)
    for row in rows:
        destination = child(output, row['candidate_relative_path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as target, child(archive, row['source_relative_path']).open('rb') as source:
            shutil.copyfileobj(source, target)
        if sha256(destination) != row['sha256']:
            raise ValueError('Candidate copy checksum mismatch')
    report = {'schema': 'trained_weights_release_candidate_v1',
              'distribution_status': 'pending_rights_confirmation', 'approved_for_publication': False,
              'privacy_limit': 'No text ID/path patterns found. This does not prove that trained biometric representations contain no participant information.',
              'files': rows, 'total_bytes': sum(row['bytes'] for row in rows),
              'training_or_inference_performed': False}
    write_new(output / 'manifest.json', report)
    print(json.dumps({'candidate_directory': str(output.relative_to(ROOT)), 'files': len(rows),
                      'total_bytes': report['total_bytes'], 'approved_for_publication': False}, indent=2))

if __name__ == '__main__':
    main()
