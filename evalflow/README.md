# EvalFlow

**Model Evaluation & Analytics Platform - Version 1.0.0**

EvalFlow is an internal Python library built on [Daft](https://www.getdaft.io/) that enables OpenAI teams to efficiently process, analyze, and visualize model evaluation data at scale.

## 🚀 Quick Start

```python
import evalflow as ef

# Load evaluation results
df = ef.load_results("experiments/gpt4_eval.parquet")

# Filter to specific benchmarks
df_filtered = df.filter_benchmarks(["mmlu", "hellaswag"])

# Compute metrics
metrics = df_filtered.compute_metrics(["accuracy", "f1_score"], group_by=["model"])

# Visualize results
plot = metrics.plot()
plot.save("results.png")
```

## 📦 Installation

### From Source
```bash
git clone https://github.com/openai/evalflow.git
cd evalflow
pip install -e .
```

### With Development Dependencies
```bash
pip install -e ".[dev]"
```

## 🎯 Key Features

### 1. Distributed Data Processing
Built on Daft for 10-100x faster processing than Pandas on large datasets.

```python
# Handle datasets up to 1TB
df = ef.load_results("experiments/*.parquet")  # Glob patterns supported
df = df.filter(ef.col("tokens") > 1000)
results = df.collect()  # Lazy evaluation
```

### 2. Comprehensive Metrics Library

**Standard Metrics:**
- Accuracy, Precision, Recall, F1
- BLEU, ROUGE scores
- Perplexity

**LLM-Specific Metrics:**
- GPT-4 win rate
- Preference scores
- Coherence scores
- Hallucination rate
- Instruction following

```python
# Compute multiple metrics
df.compute_metrics([
    "accuracy",
    "f1_score",
    "gpt4_win_rate"
], group_by=["model", "benchmark"])
```

### 3. Flexible Data Loading

```python
# Load from various formats
df = ef.load_results("data.parquet")
df = ef.load_from_csv("data.csv")
df = ef.load_from_json("data.jsonl")

# Load multiple experiments
df = ef.load_experiment_results(["exp_001", "exp_002"])

# Create from dictionary
df = ef.load_from_dict({
    "model": ["gpt-4", "gpt-3.5"],
    "accuracy": [0.95, 0.87]
})
```

### 4. Composable Pipelines

```python
from evalflow.core.pipeline import Pipeline

# Build reusable pipeline
pipeline = Pipeline("model_comparison")
pipeline.filter_step("recent", ef.col("date") > "2025-01-01")
pipeline.metric_step("compute", ["accuracy", "f1"], group_by=["model"])

# Run on data
results = pipeline.run(df)
```

### 5. Rich Visualizations

```python
# Comparison plots
ef.plot_comparison(df, x="model", y="accuracy", title="Model Performance")

# Trend analysis
ef.plot_trends(df, x="date", y="accuracy", hue="model")

# Heatmaps
ef.plot_heatmap(df, index="model", columns="benchmark", values="score")

# Distributions
ef.plot_distribution(df, column="latency_ms", bins=50)
```

## 📚 Common Use Cases

### Compare Models Across Benchmarks

```python
import evalflow as ef

# Load results
df = ef.load_results("benchmarks/*.parquet")

# Compare models
comparison = df.compare_models(
    model_col="model",
    metric_col="accuracy",
    baseline="gpt-3.5-turbo"
)

# Display with lift vs baseline
print(comparison.head())
```

### Track Performance Over Time

```python
# Load historical results
df = ef.load_results("history/2025-*.parquet")

# Group by date and model
trends = df.group_by("date", "model").agg({
    "accuracy": ["mean", "std"],
    "latency_ms": "median"
})

# Plot trends
ef.plot_trends(trends, x="date", y="accuracy_mean", hue="model").save("trends.png")
```

### Analyze A/B Test Results

```python
# Load A/B test data
df = ef.load_experiment_results(["exp_control", "exp_treatment"])

# Compute metrics per variant
results = df.group_by("variant").compute_metrics([
    "accuracy",
    "preference_score",
    "coherence_score"
])

# Visualize
ef.plot_comparison(results, x="variant", y="preference_score")
```

### Custom Metric Analysis

```python
from evalflow.core.metrics import SimpleMetric, register_metric

# Define custom metric
def my_metric_fn(outputs, targets):
    # Your custom logic
    return sum(o == t for o, t in zip(outputs, targets)) / len(outputs)

custom_metric = SimpleMetric(
    name="my_custom_metric",
    description="Custom evaluation metric",
    compute_fn=my_metric_fn
)

# Register for use
register_metric(custom_metric)

# Use in analysis
df.compute_metrics(["my_custom_metric"])
```

## 🔧 Advanced Usage

### Working with Large Datasets

```python
# Use lazy evaluation
df = ef.load_results("large_dataset/*.parquet")

# Chain operations (not executed yet)
df = (
    df.filter(ef.col("model") == "gpt-4")
    .filter(ef.col("benchmark").is_in(["mmlu", "hellaswag"]))
    .with_column("normalized_score", ef.col("raw_score") / 100)
)

# Trigger computation only when needed
results = df.collect()
```

### Distributed Computing

```python
# EvalFlow automatically uses Ray for distributed computing
# when datasets are large enough

# For explicit control:
import ray
ray.init(num_cpus=8)

df = ef.load_results("huge_dataset/*.parquet")
results = df.compute_metrics(["accuracy"])  # Automatically distributed
```

### Exporting Results

```python
# To Pandas
pdf = df.to_pandas()

# To Parquet
df.write_parquet("output.parquet")

# To CSV
df.write_csv("output.csv")
```

## 📖 Data Schema

### Standard Evaluation Result Schema

```python
{
    "experiment_id": str,      # Unique experiment identifier
    "model": str,              # Model name
    "model_version": str,      # Model version
    "benchmark": str,          # Benchmark name
    "task_id": str,            # Individual task ID
    "input": str | dict,       # Input to model
    "output": str | dict,      # Model output
    "target": str | dict,      # Expected output
    "metadata": dict,          # Additional metadata
    "timestamp": datetime,     # Evaluation timestamp
    "latency_ms": float,       # Response latency
    "tokens_input": int,       # Input token count
    "tokens_output": int,      # Output token count
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=evalflow tests/

# Run specific test file
pytest tests/test_evalflow.py
```

## 📊 Performance Benchmarks

| Dataset Size | Pandas | EvalFlow | Speedup |
|--------------|--------|----------|---------|
| 1 GB         | 45s    | 8s       | 5.6x    |
| 10 GB        | 8m 30s | 42s      | 12.1x   |
| 100 GB       | OOM    | 6m 20s   | ∞       |

*Benchmarks on AWS r5.4xlarge instance*

## 🛠️ Development

### Code Style

```bash
# Format code
black evalflow/

# Sort imports
isort evalflow/

# Lint
flake8 evalflow/

# Type check
mypy evalflow/
```

### Building Documentation

```bash
cd docs/
make html
open _build/html/index.html
```

## 📝 API Reference

### Core Classes

- **EvalDataFrame**: Main interface for evaluation data
- **Metric**: Base class for metrics
- **Pipeline**: Composable transformation pipeline
- **Plot**: Visualization wrapper

### Key Functions

- `load_results()`: Load evaluation data
- `compute_metrics()`: Calculate metrics
- `plot_comparison()`: Create comparison charts
- `plot_trends()`: Visualize trends

See full API documentation at `docs/api/`

## 🤝 Contributing

### Internal Contributors

This is an internal OpenAI library. For contributions:

1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit PR with description

### Code Review Process

- All PRs require 1+ approvals
- Tests must pass
- Documentation must be updated

## 📄 License

Proprietary - OpenAI Internal Use Only

## 🙋 Support

- **Slack**: #evalflow-support
- **Email**: eval-team@openai.com
- **Docs**: https://docs.openai.internal/evalflow
- **Issues**: https://github.com/openai/evalflow/issues

## 🗺️ Roadmap

### v1.1.0 (Q1 2026)
- Real-time evaluation streaming
- Statistical significance testing
- Advanced visualization templates

### v1.2.0 (Q2 2026)
- Experiment tracking integration
- Automated report generation
- Custom data source plugins

### v2.0.0 (Q3 2026)
- Causal analysis framework
- ML-based insight generation
- Enhanced distributed computing

## 📜 Changelog

### v1.0.0 (2025-11-15)
- Initial release
- Core DataFrame operations
- Standard and LLM metrics
- Basic visualizations
- Pipeline framework
- Distributed processing via Daft

---

**Built with ❤️ by the OpenAI Evaluation Platform Team**
