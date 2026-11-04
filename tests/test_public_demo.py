from pathlib import Path
import json
import re
import numpy as np
import pytest
from tools import build_public_demo as demo

ROOT = Path(__file__).resolve().parents[1]

def test_public_demo_is_reproducible_and_has_only_aggregate_rows():
    html = (ROOT / 'demo/index.html').read_text('utf-8')
    assert html == demo.render()
    assert not re.search(r'sub-\d+|[A-Za-z]:[\\/]Users[\\/]', html)
    assert not re.search(r'<(?:script|img|link)[^>]+(?:src|href)=["\x27](?:https?:)?//', html)
    value = json.loads(re.search(r'<script id="public-aggregate-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert value['contains_participant_records'] is False
    assert value['contains_neural_weights'] is False
    assert set(value['tables']) == {'v1', 'v2_static', 'v2_stream'}
    for rows in value['tables'].values():
        for row in rows:
            assert set(row) <= demo.ALLOWED
            assert row['FAR'] == pytest.approx(row['false_accepts'] / row['n_impostor'])
            assert row['FRR'] == pytest.approx(row['false_rejects'] / row['n_genuine'])
            assert row['HTER'] == pytest.approx((row['FAR'] + row['FRR']) / 2)

def test_builder_rejects_individual_score_schema(tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/v1_summary.csv').write_text('participant_id,score\nSYNTHETIC,0.5\n')
    monkeypatch.setattr(demo, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='individual score tables'):
        demo.load_aggregates()
