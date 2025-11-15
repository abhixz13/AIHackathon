"""Example workflows for the EvalFlow reference implementation."""

import evalflow as ef
from evalflow import col
import pandas as pd


def basic_usage():
    data = {
        "experiment_id": ["exp_001", "exp_001", "exp_002", "exp_002"],
        "model": ["gpt-4", "gpt-3.5", "gpt-4", "gpt-3.5"],
        "benchmark": ["mmlu", "mmlu", "hellaswag", "hellaswag"],
        "output": [1, 0, 1, 1],
        "target": [1, 0, 0, 1],
    }
    edf = ef.load_from_dict(data)
    metrics = edf.compute_metrics(["accuracy"], group_by=["model", "benchmark"])
    print("Grouped metrics:")
    print(metrics.to_pandas())


def model_comparison():
    data = {
        "model": ["gpt-4", "gpt-4", "gpt-3.5", "gpt-3.5"],
        "accuracy": [0.94, 0.95, 0.88, 0.89],
    }
    edf = ef.load_from_pandas(pd.DataFrame(data))
    comparison = edf.compare_models(model_col="model", metric_col="accuracy", baseline="gpt-3.5")
    print("Model comparison:")
    print(comparison.to_pandas())


def pipeline_example():
    df = pd.DataFrame(
        {
            "model": ["gpt-4", "gpt-3.5", "gpt-4", "gpt-3.5"],
            "benchmark": ["mmlu", "mmlu", "hellaswag", "hellaswag"],
            "output": [1, 0, 1, 1],
            "target": [1, 0, 0, 1],
            "accuracy": [0.95, 0.88, 0.93, 0.86],
        }
    )
    edf = ef.load_from_pandas(df)
    pipeline = ef.Pipeline("demo")
    pipeline.filter_step("only_mmlu", col("benchmark") == "mmlu")
    pipeline.metric_step("metrics", ["accuracy"], group_by=["model"])
    result = pipeline.run(edf)
    print("Pipeline result:")
    print(result.to_pandas())


if __name__ == "__main__":
    basic_usage()
    model_comparison()
    pipeline_example()
