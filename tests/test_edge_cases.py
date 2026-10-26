import numpy as np
import pandas as pd
import pytest

from tawseel.data import make_parcels
from tawseel.model import add_history_features, build_pipeline, time_split


def test_history_ignores_input_order_and_timestamp_ties():
    original = pd.DataFrame(
        {
            "customer_id": ["a", "a", "a", "b"],
            "created_at": pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-02", "2026-01-01"]),
            "failed": [1, 1, 0, 0],
        },
        index=[10, 22, 4, 6],
    )
    unchanged = original.copy(deep=True)

    output = add_history_features(original)

    pd.testing.assert_frame_equal(original, unchanged)
    assert output.index.tolist() == original.index.tolist()
    assert output.loc[[10, 4], "customer_prior_parcels"].tolist() == [1, 1]
    assert output.loc[[10, 4], "customer_prior_fail_rate"].tolist() == [1.0, 1.0]


def test_time_split_keeps_boundary_timestamp_together_and_validates_fraction():
    frame = pd.DataFrame(
        {"created_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"])},
        index=[9, 8, 7, 6],
    )
    train, test = time_split(frame, test_fraction=0.5)

    assert train.index.tolist() == [9]
    assert test.index.tolist() == [8, 7, 6]
    with pytest.raises(ValueError):
        time_split(frame, test_fraction=0)
    with pytest.raises(ValueError):
        time_split(frame, test_fraction=1)


def test_pipeline_handles_unseen_missing_values_and_a_single_prediction():
    frame = add_history_features(make_parcels(300, seed=9))
    train, test = time_split(frame)
    model = build_pipeline(seed=2).fit(train, train["failed"])
    test = test.copy()
    test.loc[test.index[0], ["area", "merchant_id"]] = ["unseen-area", None]
    test.loc[test.index[0], ["cod_amount", "address_text"]] = [np.nan, ""]
    test["unexpected"] = "ignored"

    one = model.predict_proba(test.iloc[:1])
    batch = model.predict_proba(test)

    assert one.shape == (1, 2)
    assert one[0, 1] == pytest.approx(batch[0, 1])
