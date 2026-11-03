"""Recompute saved V1/V2 metrics and sequential decisions, without neural inference."""
import argparse
import json
import sys
from _common import ROOT, child, sha256, write_new

sys.path.insert(0, str(ROOT / 'src'))

def compare(actual, filename):
    import pandas as pd
    expected = pd.read_csv(ROOT / 'results' / filename, float_precision='round_trip')
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected,
                                  check_dtype=False, check_exact=False, atol=1e-12, rtol=0)

def verify():
    from omega_dl.verification import reporting as v1
    from omega_dl.adaptive_verification import reporting as v2
    from omega_dl.adaptive_verification.protocol import load
    from omega_dl.adaptive_verification.evaluation import summarize_streams
    core = json.loads((ROOT / 'artifacts/FROZEN_CORE.json').read_text('utf-8'))
    for name, expected in {**core['core_sha256'], **core['reference_sha256']}.items():
        if sha256(child(ROOT, name)) != expected:
            raise ValueError('Publication core differs from the frozen source: ' + name)
    _, scores, metrics, _, _ = v1.load_validated(ROOT / 'outputs/verification_v1')
    compare(v1.summarize(scores, metrics), 'v1_summary.csv')
    root = ROOT / 'outputs/adaptive_v2'
    protocol, base = load(root)
    static_metrics, streams, _ = v2.validate(root, protocol, base)
    compare(v2.aggregate(static_metrics), 'v2_static_summary.csv')
    compare(v2.aggregate(summarize_streams(streams), stream=True), 'v2_stream_summary.csv')
    return {'status': 'passed', 'frozen_core_files': len(core['core_sha256']),
            'v1_score_rows': len(scores), 'v1_metric_rows': len(metrics),
            'v2_static_metric_rows': len(static_metrics), 'v2_stream_decisions_replayed': len(streams),
            'aggregate_tables_reproduced': 3, 'absolute_tolerance': 1e-12,
            'training': False, 'neural_inference': False,
            'phase_v2_dev_limit': 'Historical phase DEV rows are absent; locked phase thresholds and all saved phase TEST decisions/metrics are checked. Phase DEV reconstruction is retained only in the historical archive.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', help='Optional new JSON receipt inside this checkout')
    args = parser.parse_args()
    result = verify()
    if args.save:
        write_new(child(ROOT, args.save), result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
