"""Reporting must reject altered decisions and incomplete comparison grids."""
import numpy as np
import pandas as pd
import pytest
from omega_dl.verification.reporting import validate_group, summarize


def grid():
    rows = []
    for claimed in ["a", "b"]:
        for source in ["a", "b"]:
            for start in [0, 60]:
                genuine = int(claimed == source)
                score = 0.8 if genuine else 0.2
                rows.append(dict(claimed_id=claimed, source_id=source, duration_seconds=30, start_seconds=start, stage="test", is_genuine=genuine, score=score, threshold=0.5, accepted=genuine))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("defect", ["duplicate", "decision", "threshold", "label", "nonfinite"])
def test_saved_result_validation_catches_corruption(defect):
    rows = grid()
    validate_group(rows, ["a", "b"], 30, [0, 60], 0.5, "test")
    if defect == "duplicate":
        rows.iloc[0] = rows.iloc[1]
    elif defect == "decision":
        rows.loc[0, "accepted"] = 0
    elif defect == "threshold":
        rows.loc[0, "threshold"] = .4
    elif defect == "label":
        rows.loc[0, "is_genuine"] = 0
    else:
        rows.loc[0, "score"] = np.nan
    with pytest.raises(ValueError):
        validate_group(rows, ["a", "b"], 30, [0, 60], .5, "test")


def test_summary_uses_decisions_and_fold_auc_without_pooled_roc():
    rows = grid().assign(method="cnn")
    rows.loc[0, "accepted"] = 0
    rows.loc[2, "accepted"] = 1
    folds = pd.DataFrame([dict(method="cnn", duration_seconds=30, ROC_AUC=.6, EER_diagnostic=.3, HTER=.4), dict(method="cnn", duration_seconds=30, ROC_AUC=.8, EER_diagnostic=.2, HTER=.2)])
    result = summarize(rows, folds).iloc[0]
    assert result.false_accepts == result.false_rejects == 1
    assert result.FAR == result.FRR == result.HTER == .25
    assert result.mean_fold_AUC == pytest.approx(.7)
