"""Build a self-contained public demo from already tracked aggregate CSVs only."""
import csv
import json
from _common import ROOT

TABLES = {'v1': 'v1_summary.csv', 'v2_static': 'v2_static_summary.csv',
          'v2_stream': 'v2_stream_summary.csv'}
ALLOWED = {'method', 'duration_seconds', 'enrollment_seconds', 'policy', 'n_genuine', 'n_impostor',
           'false_accepts', 'false_rejects', 'FAR', 'FRR', 'HTER', 'TAR', 'mean_fold_AUC',
           'mean_fold_EER', 'min_fold_HTER', 'max_fold_HTER', 'updates', 'impostor_updates',
           'mean_final_drift'}
LABELS = {'method', 'policy'}
COUNTS = {'duration_seconds', 'enrollment_seconds', 'n_genuine', 'n_impostor', 'false_accepts',
          'false_rejects', 'updates', 'impostor_updates'}

def load_aggregates():
    tables = {}
    for name, filename in TABLES.items():
        with (ROOT / 'results' / filename).open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames) - ALLOWED:
                raise ValueError('Unexpected aggregate columns; never import individual score tables')
            tables[name] = [{key: value if key in LABELS else int(value) if key in COUNTS else float(value)
                             for key, value in row.items()} for row in reader]
    return {'schema': 'public_aggregate_demo_v1', 'contains_participant_records': False,
            'contains_neural_weights': False, 'tables': tables}

def render():
    data = json.dumps(load_aggregates(), ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    template = (ROOT / 'demo/template.html').read_text('utf-8')
    if template.count('__PUBLIC_AGGREGATES__') != 1:
        raise ValueError('Expected exactly one aggregate placeholder')
    return template.replace('__PUBLIC_AGGREGATES__', data.replace('<', '\\u003c'))

def main():
    output = ROOT / 'demo/index.html'
    output.write_text(render(), encoding='utf-8', newline='\n')
    print('Built demo/index.html from public aggregate tables; no participant rows or weights')

if __name__ == '__main__':
    main()
