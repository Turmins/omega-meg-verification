from omega_dl.data.index import Recording
from omega_dl.data.windows import assert_eval_windows_non_overlapping, make_windows


def test_eval_hop_equals_window_length():
    record = Recording("sub-0001", 1, "enroll", "ses-01", 1, "sub-0001_enroll.npy", "")
    windows = make_windows(record, "val", hop=1280)
    assert [current.start - previous.start for previous, current in zip(windows, windows[1:])] == [1280] * 23
    assert_eval_windows_non_overlapping(windows)
