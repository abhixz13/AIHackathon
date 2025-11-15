import pandas as pd

from evalflow import (
    EvalDataFrame,
    col,
    load_from_pandas,
    Pipeline,
    accuracy,
)


def test_basic_dataframe_ops():
    df = pd.DataFrame(
        {
            "model": ["a", "b", "a"],
            "accuracy": [0.9, 0.8, 0.95],
            "output": [1, 0, 1],
            "target": [1, 0, 1],
        }
    )
    edf = load_from_pandas(df)
    assert edf.count() == 3
    filtered = edf.filter(col("accuracy") > 0.85)
    assert filtered.count() == 2


def test_compute_metrics_accuracy():
    df = pd.DataFrame(
        {
            "output": [1, 0, 1, 1],
            "target": [1, 0, 0, 1],
        }
    )
    edf = EvalDataFrame(df)
    metrics = edf.compute_metrics([accuracy])
    pdf = metrics.to_pandas()
    assert "accuracy" in pdf.columns
    assert 0.0 <= pdf.loc[0, "accuracy"] <= 1.0


def test_pipeline_metric_step():
    df = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "output": [1, 0, 1, 1],
            "target": [1, 0, 0, 1],
        }
    )
    edf = EvalDataFrame(df)
    pipe = Pipeline("p").metric_step("metrics", [accuracy])
    result = pipe.run(edf)
    assert result.count() == 1
