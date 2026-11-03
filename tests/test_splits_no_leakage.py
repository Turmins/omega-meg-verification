from omega_dl.data.index import Recording
from omega_dl.data.splits import assert_no_leakage, make_folds, split_for_subject
from omega_dl.data.windows import Window


def _records():
    result = []
    for index in range(60):
        subject = f"sub-{index:04d}"
        result.extend([
            Recording(subject, index, "enroll", "ses-01", 1, f"{subject}_enroll.npy", ""),
            Recording(subject, index, "probe", "ses-02", 1, f"{subject}_probe.npy", ""),
        ])
    return result


def test_six_fold_subject_and_recording_disjointness():
    records = _records()
    folds = make_folds(records)
    assert_no_leakage(records, folds)
    assert all((len(f.train_subjects), len(f.val_subjects), len(f.test_subjects)) == (40, 10, 10) for f in folds)
    assert len({s for f in folds for s in f.test_subjects}) == 60


def test_window_provenance_resolves_to_one_split():
    records = _records()
    fold = make_folds(records)[0]
    for record in records:
        window = Window(record.participant_id, record.recording_id, record.role, record.session, 0, 1280, split_for_subject(fold, record.participant_id))
        assert split_for_subject(fold, window.participant_id) == window.split
