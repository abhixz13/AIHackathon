# EvalFlow Quick Reference Guide

## Installation
```bash
pip install -e .
```

## Core Imports
```python
import evalflow as ef
from evalflow import col  # For column expressions
```

## Loading Data

### From Files
```python
# Single file (auto-detect format)
df = ef.load_results("data.parquet")
df = ef.load_results("data.csv")
df = ef.load_results("data.json")

# Multiple files with glob
df = ef.load_results("experiments/*.parquet")

# Specific loaders
df = ef.load_from_parquet("data.parquet")
df = ef.load_from_csv("data.csv", delimiter=",")
df = ef.load_from_json("data.jsonl")
```

### From Memory
```python
# From dictionary
df = ef.load_from_dict({
    "model": ["gpt-4", "gpt-3.5"],
    "accuracy": [0.95, 0.87]
})

# From Pandas
df = ef.load_from_pandas(pandas_df)
```

### Experiment/Benchmark Loaders
```python
# Load specific experiments
df = ef.load_experiment_results(["exp_001", "exp_002"])

# Load benchmark results
df = ef.load_benchmark_results("mmlu", models=["gpt-4"])
```

## Data Operations

### Filtering
```python
# Single condition
df = df.filter(col("model") == "gpt-4")
df = df.filter(col("accuracy") > 0.8)

# Multiple conditions
df = df.filter((col("model") == "gpt-4") & (col("benchmark") == "mmlu"))

# String operations
df = df.filter(col("model").str.contains("gpt"))

# List membership
df = df.filter(col("model").is_in(["gpt-4", "claude"]))
```

### Selection
```python
# Select columns
df = df.select("model", "accuracy", "benchmark")
```

### Adding/Modifying Columns
```python
# Add column
df = df.with_column("normalized", col("raw_score") / 100)

# Multiple operations
df = df.with_column("score_pct", col("score") * 100)
         .with_column("rank", col("score").rank())
```

### Grouping & Aggregation
```python
# Group by single column
grouped = df.group_by("model").agg({
    "accuracy": "mean",
    "latency": ["mean", "std"]
})

# Group by multiple columns
grouped = df.group_by("model", "benchmark").agg({
    "accuracy": ["mean", "min", "max"]
})
```

### Sorting
```python
# Ascending
df = df.sort("accuracy")

# Descending
df = df.sort("accuracy", desc=True)

# Multiple columns
df = df.sort("model", "accuracy", desc=True)
```

### Joining
```python
# Inner join
df = df1.join(df2, on="model")

# Left join
df = df1.join(df2, on="model", how="left")

# Multiple keys
df = df1.join(df2, on=["model", "benchmark"])
```

## Computing Metrics

### Standard Metrics
```python
# Single metric
result = df.compute_metrics("accuracy")

# Multiple metrics
result = df.compute_metrics(["accuracy", "f1_score", "precision"])

# With grouping
result = df.compute_metrics(
    ["accuracy", "f1_score"],
    group_by=["model", "benchmark"]
)
```

### Available Metrics
```python
# Standard metrics
"accuracy", "precision", "recall", "f1_score"
"bleu_score", "rouge_score", "perplexity"

# LLM-specific metrics
"gpt4_win_rate", "preference_score", "coherence_score"
"toxicity_score", "hallucination_rate", "instruction_following"
```

### Custom Metrics
```python
from evalflow.core.metrics import SimpleMetric, register_metric

def my_metric_fn(outputs, targets):
    return sum(o == t for o, t in zip(outputs, targets)) / len(outputs)

metric = SimpleMetric(
    name="my_metric",
    description="Custom metric",
    compute_fn=my_metric_fn
)

register_metric(metric)

# Use it
result = df.compute_metrics(["my_metric"])
```

## Analysis

### Model Comparison
```python
comparison = df.compare_models(
    model_col="model",
    metric_col="accuracy",
    baseline="gpt-3.5-turbo"  # Optional baseline for lift
)
```

### Filtering by Experiments/Benchmarks
```python
# Filter to experiments
df = df.filter_experiments(["exp_001", "exp_002"])
df = df.filter_experiments("exp_001")  # Single ID

# Filter to benchmarks
df = df.filter_benchmarks(["mmlu", "hellaswag"])
df = df.filter_benchmarks("mmlu")  # Single benchmark
```

## Pipelines

### Creating Pipelines
```python
from evalflow.core.pipeline import Pipeline

pipeline = Pipeline(name="my_pipeline")

# Add filter step
pipeline.filter_step("recent", col("date") > "2025-01-01")

# Add transformation step
pipeline.transform_step("normalize", "score_pct", col("score") * 100)

# Add metric step
pipeline.metric_step("compute", ["accuracy"], group_by=["model"])

# Run pipeline
result = pipeline.run(df)
```

### Pre-built Pipelines
```python
from evalflow.core.pipeline import (
    standard_comparison_pipeline,
    benchmark_analysis_pipeline
)

pipeline = standard_comparison_pipeline(baseline_model="gpt-3.5")
result = pipeline.run(df)
```

## Visualization

### Comparison Charts
```python
plot = ef.plot_comparison(
    df,
    x="model",
    y="accuracy",
    title="Model Performance",
    color="steelblue"
)
plot.save("comparison.png")
```

### Trend Lines
```python
plot = ef.plot_trends(
    df,
    x="date",
    y="accuracy",
    hue="model",  # Different line per model
    title="Performance Over Time"
)
plot.save("trends.png")
```

### Scatter Plots
```python
plot = ef.plot_scatter(
    df,
    x="latency_ms",
    y="accuracy",
    hue="model",
    title="Latency vs Accuracy"
)
plot.save("scatter.png")
```

### Heatmaps
```python
plot = ef.plot_heatmap(
    df,
    index="model",
    columns="benchmark",
    values="accuracy",
    title="Performance Heatmap",
    cmap="RdYlGn"
)
plot.save("heatmap.png")
```

### Distributions
```python
plot = ef.plot_distribution(
    df,
    column="accuracy",
    bins=30,
    title="Accuracy Distribution"
)
plot.save("distribution.png")
```

## Data Export

### To Pandas
```python
pandas_df = df.to_pandas()
```

### To Files
```python
df.write_parquet("output.parquet")
df.write_csv("output.csv")
```

### Preview Data
```python
# First N rows
df.head(10)

# Count rows
total = df.count()

# Get columns
columns = df.columns()

# Describe statistics
df.describe()
```

## Utility Functions

### Format Helpers
```python
from evalflow.utils import format_percentage, pretty_print_metrics

pct = format_percentage(0.856, decimals=2)  # "85.60%"

metrics = {"accuracy": 0.95, "f1": 0.89}
pretty_print_metrics(metrics)  # Formatted table
```

### Configuration
```python
from evalflow.utils import save_config, load_config

config = {"model": "gpt-4", "benchmark": "mmlu"}
save_config(config, "config.json")

loaded = load_config("config.json")
```

### Data Validation
```python
from evalflow.utils import validate_schema

data = {"model": "gpt-4", "accuracy": 0.95}
validate_schema(data, ["model", "accuracy"])  # Raises on missing fields
```

## Common Patterns

### Load → Filter → Compute → Visualize
```python
result = (
    ef.load_results("data.parquet")
    .filter(col("model").is_in(["gpt-4", "claude"]))
    .compute_metrics(["accuracy", "f1_score"], group_by=["model"])
)

ef.plot_comparison(result, x="model", y="accuracy").save("output.png")
```

### Grouped Analysis
```python
metrics_by_group = (
    df.group_by("model", "benchmark")
    .agg({"accuracy": ["mean", "std"], "latency": "median"})
)
```

### Time Series Analysis
```python
trends = (
    df.filter(col("date") >= "2025-01-01")
    .group_by("date", "model")
    .agg({"accuracy": "mean"})
    .sort("date")
)

ef.plot_trends(trends, x="date", y="accuracy_mean", hue="model").save("trends.png")
```

### A/B Testing
```python
ab_results = (
    df.filter(col("experiment").is_in(["control", "treatment"]))
    .compute_metrics(["accuracy", "preference_score"], group_by=["experiment"])
)
```

## Performance Tips

1. **Use Parquet** - Fastest format for large datasets
2. **Filter Early** - Reduce data size before expensive operations
3. **Lazy Evaluation** - Chain operations, compute once at the end
4. **Ray Cluster** - Enable for datasets > 10GB
5. **Appropriate Types** - Use correct data types for efficiency

## Troubleshooting

### Out of Memory
```python
# Use lazy operations
df = ef.load_results("huge.parquet")
df = df.filter(col("model") == "gpt-4")  # Not executed yet
result = df.collect()  # Execute here
```

### Slow Performance
```python
# Enable distributed computing
import ray
ray.init(num_cpus=8)

df = ef.load_results("large.parquet")
result = df.compute_metrics(["accuracy"])  # Automatically distributed
```

### Import Errors
```bash
# Reinstall dependencies
pip install -e ".[dev]"
```

---

**Version:** 1.0.0  
**Last Updated:** November 15, 2025  
**Support:** eval-team@openai.com | #evalflow-support
