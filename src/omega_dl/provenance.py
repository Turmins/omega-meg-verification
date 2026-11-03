"""Append-only invocation records; confirmed training is recorded separately.

Historical environment.json is observed as legacy metadata, never rewritten
or interpreted as the environment of historical training.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import uuid

from omega_dl.paths import extension_root, inside_extension, sha256_file


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    """Publish a complete JSON atomically, refusing collisions and replacements."""
    path = inside_extension(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic create, unlike replace: never overwrites.
    finally:
        temporary.unlink(missing_ok=True)
    return path


def source_snapshot():
    root = extension_root()
    files = sorted(set(root.joinpath('src').rglob('*.py')) |
                   set(root.joinpath('scripts').glob('*.py')) |
                   set(root.joinpath('configs').rglob('*.yaml')))
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def current_environment():
    packages = {}
    for name in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'pytest', 'threadpoolctl'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {'python': platform.python_version(), 'executable': sys.executable,
              'platform': platform.platform(), 'packages': packages,
              'runtime_settings': {k: os.environ.get(k) for k in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')}}
    from threadpoolctl import threadpool_info
    result['threadpools'] = threadpool_info()
    torch = sys.modules.get('torch')
    if torch is not None:
        result['torch_runtime'] = {'version': str(torch.__version__),
                                  'cuda_version': torch.version.cuda,
                                  'cuda_available': torch.cuda.is_available(),
                                  'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                                  'cpu_threads': torch.get_num_threads(),
                                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                                  'cudnn_deterministic': torch.backends.cudnn.deterministic,
                                  'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                  'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}
    return result


class Invocation:
    def __init__(self, root, stage, arguments, invocation_id=None):
        self.root = inside_extension(root)
        self.id = invocation_id or uuid.uuid4().hex
        if not self.id or Path(self.id).name != self.id or self.id in ('.', '..'):
            raise ValueError('Invalid invocation ID')
        self.path = inside_extension(self.root / 'invocations' / self.id)
        self.stage, self.arguments = stage, arguments
        self.protocol = None
        self.sequence = 0

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        self.environment = current_environment()
        self.sources = source_snapshot()
        self.started = utc_now()
        legacy = self.root / 'environment.json'
        write_once(self.path / 'start.json', {
            'schema': 'omega_invocation_v1', 'invocation_id': self.id,
            'provenance_type': 'invocation', 'started_utc': self.started,
            'stage': self.stage, 'arguments': self.arguments, 'run_root': str(self.root),
            'environment': self.environment, 'source_sha256': self.sources,
            'legacy_environment': {'sha256': sha256_file(legacy),
                                   'interpretation': 'observed legacy metadata; training environment unknown'}
                                  if legacy.exists() else None})
        self.start_sha256 = sha256_file(self.path / 'start.json')
        return self

    def bind_protocol(self, protocol):
        self.protocol = protocol
        write_once(self.path / 'protocol.json', {'protocol_sha256': protocol['protocol_sha256'],
                   'effective_config': protocol['config'], 'frozen_code_sha256': protocol['code_sha256']})

    def event(self, kind, **values):
        self.sequence += 1
        write_once(self.path / f'event_{self.sequence:04d}.json',
                   {'kind': kind, 'observed_utc': utc_now(), **values})

    def __exit__(self, exc_type, exc, traceback):
        write_once(self.path / 'finish.json', {
            'invocation_id': self.id, 'finished_utc': utc_now(),
            'status': 'failed' if exc_type else 'completed',
            'error': {'type': exc_type.__name__, 'message': str(exc)} if exc_type else None,
            'protocol_sha256': self.protocol['protocol_sha256'] if self.protocol else None})
        return False


@contextmanager
def training_claim(invocation, summary_path):
    """Fail fast on a shared fold/model writer, before inspecting its summary.

The claim covers reuse, trainer execution, hashes and record publication.
All runners use this wrapper. A stale claim requires explicit inspection;
it is never guessed expired or stolen from another invocation.
    """
    summary_path = inside_extension(Path(summary_path).resolve())
    summary_path.relative_to(invocation.root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock = summary_path.with_name('.' + summary_path.name + '.p0_training.lock')
    owner = {'invocation_id': invocation.id, 'claim_id': uuid.uuid4().hex,
             'summary_path': str(summary_path), 'created_utc': utc_now()}
    payload = (json.dumps(owner, sort_keys=True) + '\n').encode('utf-8')
    try:
        handle = lock.open('xb')
    except FileExistsError as exc:
        invocation.event('training_busy', summary_path=str(summary_path))
        raise RuntimeError('Training operation already in progress; inspect the existing claim') from exc
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield owner
    finally:
        if not lock.exists() or lock.read_bytes() != payload:
            raise RuntimeError('Training ownership claim changed; refusing to remove it')
        lock.unlink()


def artifact_hashes(root, paths):
    result = {}
    for path in paths:
        path = inside_extension(Path(path).resolve())
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def run_training(invocation, trainer, summary_path, artifacts, *, fold, model, seed, args):
    """Confirm completion only while owning the shared operation; validate reuse."""
    summary_path = Path(summary_path).resolve()
    with training_claim(invocation, summary_path) as claim:
        if summary_path.exists():
            try:
                hashes = artifact_hashes(invocation.root, [summary_path, *artifacts])
                summary_key = summary_path.relative_to(invocation.root).as_posix()
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                protocol_sha = invocation.protocol['protocol_sha256'] if invocation.protocol else summary.get('protocol_sha256')
                context = {'protocol_sha256': protocol_sha, 'fold': fold, 'model': model, 'seed': seed}
                if any(key in summary and summary[key] != value for key, value in context.items()):
                    raise RuntimeError('Training summary context mismatch')
                matches = []
                for record_path in (invocation.root / 'training_provenance').glob('*.json'):
                    value = json.loads(record_path.read_text(encoding='utf-8'))
                    recorded = value.get('artifacts_sha256', {})
                    if recorded.get(summary_key) != hashes[summary_key]:
                        continue
                    if any(value.get(key) != expected for key, expected in context.items()):
                        raise RuntimeError('Training record context mismatch')
                    if not set(hashes).issubset(recorded):
                        raise RuntimeError('Training record required artifact set mismatch')
                    # Validate every artifact bound by the record, including extras.
                    current = artifact_hashes(invocation.root, [invocation.root / name for name in recorded])
                    if current != recorded:
                        raise RuntimeError('Training record artifact checksum mismatch')
                    matches.append(record_path.relative_to(invocation.root).as_posix())
                result = trainer(*args)
                if artifact_hashes(invocation.root, [summary_path, *artifacts]) != hashes:
                    raise RuntimeError('Training artifacts changed during reuse')
                if source_snapshot() != invocation.sources:
                    raise RuntimeError('Source changed during training reuse')
                invocation.event('training_reuse', fold=fold, model=model, summary_sha256=hashes[summary_key],
                                 artifacts_sha256=hashes, training_records=matches,
                                 training_environment='recorded' if matches else 'unknown')
                return result
            except BaseException as exc:
                invocation.event('training_reuse_failed', fold=fold, model=model,
                                 error_type=type(exc).__name__, error=str(exc))
                raise
        started = utc_now()
        invocation.event('training_attempt', fold=fold, model=model, seed=seed)
        try:
            result = trainer(*args)
            if not summary_path.exists():
                raise RuntimeError('Trainer did not produce a completed training summary')
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            epochs = summary.get('epochs', summary.get('epochs_completed', 0))
            context = {'protocol_sha256': invocation.protocol['protocol_sha256'],
                       'fold': fold, 'model': model, 'seed': seed}
            if epochs <= 0 or summary.get('protocol_sha256') != context['protocol_sha256'] or any(
                    key in summary and summary[key] != value for key, value in context.items()):
                raise RuntimeError('New training summary does not confirm this training')
            hashes = artifact_hashes(invocation.root, [summary_path, *artifacts])
            if source_snapshot() != invocation.sources:
                raise RuntimeError('Source changed during training; provenance cannot be confirmed')
            record_id = uuid.uuid4().hex
            record = write_once(invocation.root / 'training_provenance' / f'{record_id}.json', {
                'schema': 'omega_training_v2', 'provenance_type': 'confirmed_completed_training',
                'training_id': record_id, 'invocation_id': invocation.id, 'ownership_claim': claim,
                'operation_started_utc': started, 'recorded_utc': utc_now(),
                'completed_utc': summary.get('completed_utc'), **context,
                'effective_config': invocation.protocol['config'], 'environment': current_environment(),
                'environment_observation': 'at confirmed training completion; trainer runtime settings now effective',
                'invocation_start_environment': invocation.environment,
                'source_sha256': invocation.sources, 'artifacts_sha256': hashes})
            invocation.event('training_completed', fold=fold, model=model,
                             training_record=record.relative_to(invocation.root).as_posix())
            return result
        except BaseException as exc:
            invocation.event('training_failed', fold=fold, model=model,
                             error_type=type(exc).__name__, error=str(exc))
            raise
