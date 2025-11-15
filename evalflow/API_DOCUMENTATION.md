# EvalFlow API Documentation v1.0.0

## Table of Contents
- [Core Classes](#core-classes)
- [Data Loading](#data-loading)
- [Metrics](#metrics)
- [Pipelines](#pipelines)
- [Visualization](#visualization)
- [Utilities](#utilities)

---

## Core Classes

### EvalDataFrame

The primary interface for evaluation data manipulation.

```python
class EvalDataFrame:
    """
    High-level interface for evaluation data built on Daft DataFrame.
    """
```

#### Methods

**filter(predicate) → EvalDataFrame**
```python
df.filter(col("accuracy") > 0.8)
df.filter((col("model") == "gpt-4") & (col("benchmark") == "mmlu"))
```
Filter rows based on boolean predicate.

**select(*columns: str) → EvalDataFrame**
```python
df.select("model", "accuracy", "benchmark")
```
Select specific columns.

**group_by(*columns: str) → GroupedEvalDataFrame**
```python
df.group_by("model", "benchmark")
```
Group by one or more columns for aggregation.

**with_column(column_name: str, expression) → EvalDataFrame**
```python
df.with_column("normalized", col("raw_score") / 100)
```
Add or replace a column.

**join(other: EvalDataFrame, on: Union[str, List[str]], how: str = "inner") → EvalDataFrame**
```python
df.join(other_df, on="model")
df.join(other_df, on=["model", "benchmark"], how="left")
```
Join with another EvalDataFrame.

**sort(*columns: str, desc: bool = False) → EvalDataFrame**
```python
df.sort("accuracy", desc=True)
```
Sort by columns.

**compute_metrics(metrics: Union[str, List[str], Dict], group_by: Optional[List[str]] = None) → EvalDataFrame**
```python
df.compute_metrics("accuracy")
df.compute_metrics(["accuracy", "f1_score"], group_by=["model"])
```
Compute evaluation metrics.

**compare_models(model_col: str = "model", metric_col: str = "accuracy", baseline: Optional[str] = None) → EvalDataFrame**
```python
df.compare_models(model_col="model", metric_col="accuracy", baseline="gpt-3.5")
```
Compare models on a metric.

**filter_experiments(experiment_ids: Union[str, List[str]]) → EvalDataFrame**
```python
df.filter_experiments(["exp_001", "exp_002"])
```
Filter to specific experiments.

**filter_benchmarks(benchmarks: Union[str, List[str]]) → EvalDataFrame**
```python
df.filter_benchmarks(["mmlu", "hellaswag"])
```
Filter to specific benchmarks.

**to_pandas() → pd.DataFrame**
```python
pandas_df = df.to_pandas()
```
Convert to Pandas DataFrame.

**collect()**
```python
result = df.collect()
```
Trigger computation and collect results.

**write_parquet(path: str)**
```python
df.write_parquet("output.parquet")
```
Write to Parquet file.

**write_csv(path: str)**
```python
df.write_csv("output.csv")
```
Write to CSV file.

**head(n: int = 5)**
```python
df.head(10)
```
Return first n rows as Pandas DataFrame.

**count() → int**
```python
total = df.count()
```
Count total rows.

**columns() → List[str]**
```python
cols = df.columns()
```
Get list of column names.

**describe()**
```python
stats = df.describe()
```
Generate descriptive statistics.

**plot(**kwargs)**
```python
df.plot().save("chart.png")
```
Create visualization of data.

---

### GroupedEvalDataFrame

Grouped EvalDataFrame for aggregations.

#### Methods

**agg(aggregations: Dict[str, Union[str, List[str]]]) → EvalDataFrame**
```python
grouped.agg({
    "accuracy": ["mean", "std"],
    "latency": "median"
})
```
Perform aggregations on grouped data.

**compute_metrics(metrics: List[str]) → EvalDataFrame**
```python
grouped.compute_metrics(["accuracy", "f1_score"])
```
Compute metrics for each group.

---

### Metric

Base class for evaluation metrics.

```python
class Metric(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        higher_is_better: bool = True
    )
```

#### Methods

**compute(eval_df) → float**
```python
score = metric.compute(df)
```
Compute metric on an EvalDataFrame.

**get_aggregation()**
```python
agg_expr = metric.get_aggregation()
```
Get Daft aggregation expression.

---

### SimpleMetric

Simple metric computed from output and target columns.

```python
class SimpleMetric(Metric):
    def __init__(
        self,
        name: str,
        description: str,
        compute_fn: Callable,
        output_col: str = "output",
        target_col: str = "target",
        **kwargs
    )
```

**Example:**
```python
from evalflow.core.metrics import SimpleMetric, register_metric

def my_fn(outputs, targets):
    return sum(o == t for o, t in zip(outputs, targets)) / len(outputs)

metric = SimpleMetric(
    name="my_metric",
    description="Custom metric",
    compute_fn=my_fn
)

register_metric(metric)
```

---

### Pipeline

Composable pipeline for evaluation transformations.

```python
class Pipeline:
    """Composable pipeline for evaluation data transformations."""
    
    def __init__(self, name: Optional[str] = None)
```

#### Methods

**add_step(name: str, transform: Callable) → Pipeline**
```python
pipeline.add_step("custom", lambda df: df.filter(col("x") > 0))
```
Add transformation step.

**filter_step(name: str, predicate) → Pipeline**
```python
pipeline.filter_step("recent", col("date") > "2025-01-01")
```
Add filter step.

**transform_step(name: str, column: str, expression) → Pipeline**
```python
pipeline.transform_step("normalize", "score_pct", col("score") * 100)
```
Add column transformation step.

**metric_step(name: str, metrics: List[str], group_by: Optional[List[str]] = None) → Pipeline**
```python
pipeline.metric_step("compute", ["accuracy"], group_by=["model"])
```
Add metric computation step.

**run(df: EvalDataFrame) → EvalDataFrame**
```python
result = pipeline.run(df)
```
Execute pipeline on data.

**describe() → str**
```python
print(pipeline.describe())
```
Get description of pipeline steps.

---

## Data Loading

### load_results

```python
def load_results(
    path: Union[str, Path],
    format: Optional[str] = None,
    schema: Optional[Dict] = None,
    **kwargs
) → EvalDataFrame
```

Load evaluation results from file(s).

**Parameters:**
- `path`: File path or glob pattern
- `format`: File format (csv, parquet, json, jsonl). Auto-detected if None.
- `schema`: Optional schema override
- `**kwargs`: Additional arguments passed to loader

**Returns:** EvalDataFrame

**Examples:**
```python
df = ef.load_results("experiment_001.parquet")
df = ef.load_results("experiments/exp_*.parquet")
df = ef.load_results("data.txt", format="csv")
```

---

### load_from_parquet

```python
def load_from_parquet(
    path: str,
    schema: Optional[Dict] = None,
    columns: Optional[List[str]] = None,
    **kwargs
) → EvalDataFrame
```

Load from Parquet file(s).

**Examples:**
```python
df = ef.load_from_parquet("data.parquet")
df = ef.load_from_parquet("data.parquet", columns=["model", "accuracy"])
```

---

### load_from_csv

```python
def load_from_csv(
    path: str,
    schema: Optional[Dict] = None,
    delimiter: str = ",",
    has_header: bool = True,
    **kwargs
) → EvalDataFrame
```

Load from CSV file(s).

**Examples:**
```python
df = ef.load_from_csv("data.csv")
df = ef.load_from_csv("data.tsv", delimiter="\t")
```

---

### load_from_json

```python
def load_from_json(
    path: str,
    schema: Optional[Dict] = None,
    **kwargs
) → EvalDataFrame
```

Load from JSON or JSONL file(s).

**Examples:**
```python
df = ef.load_from_json("data.jsonl")
df = ef.load_from_json("data/*.json")
```

---

### load_from_pandas

```python
def load_from_pandas(df_pandas) → EvalDataFrame
```

Create EvalDataFrame from Pandas DataFrame.

**Examples:**
```python
df = ef.load_from_pandas(pandas_df)
```

---

### load_from_dict

```python
def load_from_dict(data: Dict) → EvalDataFrame
```

Create EvalDataFrame from dictionary.

**Examples:**
```python
df = ef.load_from_dict({
    "model": ["gpt-4", "gpt-3.5"],
    "accuracy": [0.95, 0.87]
})
```

---

### load_experiment_results

```python
def load_experiment_results(
    experiment_ids: Union[str, List[str]],
    base_path: str = "experiments",
    **kwargs
) → EvalDataFrame
```

Load results for specific experiments.

**Examples:**
```python
df = ef.load_experiment_results(["exp_001", "exp_002"])
df = ef.load_experiment_results("exp_001")
```

---

### load_benchmark_results

```python
def load_benchmark_results(
    benchmark: str,
    models: Optional[List[str]] = None,
    base_path: str = "benchmarks",
    **kwargs
) → EvalDataFrame
```

Load benchmark results, optionally filtered by models.

**Examples:**
```python
df = ef.load_benchmark_results("mmlu")
df = ef.load_benchmark_results("mmlu", models=["gpt-4", "claude"])
```

---

## Metrics

### Standard Metrics

**accuracy**
```python
from evalflow.metrics import accuracy
```
Proportion of correct predictions.

**precision**
```python
from evalflow.metrics import precision
```
True positives / (True positives + False positives).

**recall**
```python
from evalflow.metrics import recall
```
True positives / (True positives + False negatives).

**f1_score**
```python
from evalflow.metrics import f1_score
```
Harmonic mean of precision and recall.

**bleu_score**
```python
from evalflow.metrics import bleu_score
```
BLEU score for text generation.

**rouge_score**
```python
from evalflow.metrics import rouge_score
```
ROUGE-1 F1 score for text summarization.

**perplexity**
```python
from evalflow.metrics import perplexity
```
Perplexity from log probabilities.

---

### LLM Metrics

**gpt4_win_rate**
```python
from evalflow.metrics import gpt4_win_rate
```
Win rate against GPT-4 in pairwise comparisons.

**preference_score**
```python
from evalflow.metrics import preference_score
```
Average human preference score.

**coherence_score**
```python
from evalflow.metrics import coherence_score
```
Response coherence and structure quality.

**toxicity_score**
```python
from evalflow.metrics import toxicity_score
```
Toxicity level of generated content.

**hallucination_rate**
```python
from evalflow.metrics import hallucination_rate
```
Rate of factual hallucinations.

**instruction_following**
```python
from evalflow.metrics import instruction_following
```
How well model follows instructions.

---

### MetricRegistry

```python
class MetricRegistry:
    """Global registry for evaluation metrics."""
```

**get_instance() → MetricRegistry**
```python
registry = MetricRegistry.get_instance()
```
Get singleton instance.

**register(metric: Metric)**
```python
registry.register(custom_metric)
```
Register a metric.

**get(name: str) → Metric**
```python
metric = registry.get("accuracy")
```
Get metric by name.

**list_metrics() → Dict[str, Metric]**
```python
all_metrics = registry.list_metrics()
```
Get all registered metrics.

---

### register_metric

```python
def register_metric(metric: Metric)
```

Convenience function to register a metric.

**Example:**
```python
from evalflow.core.metrics import register_metric

register_metric(custom_metric)
```

---

## Visualization

### plot_comparison

```python
def plot_comparison(
    eval_df,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: str = "Model Comparison",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = "steelblue",
    **kwargs
) → Plot
```

Create bar chart for model comparison.

**Example:**
```python
plot = ef.plot_comparison(df, x="model", y="accuracy", title="Performance")
plot.save("comparison.png")
```

---

### plot_trends

```python
def plot_trends(
    eval_df,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    title: str = "Performance Trends",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs
) → Plot
```

Create line plot for performance trends.

**Example:**
```python
plot = ef.plot_trends(df, x="date", y="accuracy", hue="model")
plot.save("trends.png")
```

---

### plot_scatter

```python
def plot_scatter(
    eval_df,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: str = "Scatter Plot",
    **kwargs
) → Plot
```

Create scatter plot.

**Example:**
```python
plot = ef.plot_scatter(df, x="latency", y="accuracy", hue="model")
plot.save("scatter.png")
```

---

### plot_heatmap

```python
def plot_heatmap(
    eval_df,
    index: str,
    columns: str,
    values: str,
    title: str = "Performance Heatmap",
    cmap: str = "RdYlGn",
    **kwargs
) → Plot
```

Create heatmap from data.

**Example:**
```python
plot = ef.plot_heatmap(df, index="model", columns="benchmark", values="accuracy")
plot.save("heatmap.png")
```

---

### plot_distribution

```python
def plot_distribution(
    eval_df,
    column: str,
    bins: int = 30,
    title: Optional[str] = None,
    **kwargs
) → Plot
```

Create histogram/distribution plot.

**Example:**
```python
plot = ef.plot_distribution(df, column="accuracy", bins=20)
plot.save("distribution.png")
```

---

### Plot Class

```python
class Plot:
    """Wrapper for matplotlib plots with save capabilities."""
```

**save(path: str, dpi: int = 150, **kwargs)**
```python
plot.save("output.png", dpi=300)
```
Save plot to file.

**show()**
```python
plot.show()
```
Display plot (if in interactive environment).

**close()**
```python
plot.close()
```
Close the plot.

---

## Utilities

### format_percentage

```python
def format_percentage(value: float, decimals: int = 1) → str
```

Format value as percentage.

**Example:**
```python
pct = format_percentage(0.856, decimals=2)  # "85.60%"
```

---

### compute_confidence_interval

```python
def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) → Tuple[float, float]
```

Compute confidence interval using bootstrap.

**Example:**
```python
ci = compute_confidence_interval([0.85, 0.87, 0.89, 0.86], confidence=0.95)
```

---

### validate_schema

```python
def validate_schema(data: Dict, required_fields: List[str]) → bool
```

Validate that data contains required fields.

**Example:**
```python
validate_schema(data, ["model", "accuracy"])  # Raises on missing
```

---

### save_config / load_config

```python
def save_config(config: Dict, path: str)
def load_config(path: str) → Dict
```

Save/load configuration to/from JSON file.

**Example:**
```python
save_config({"model": "gpt-4"}, "config.json")
config = load_config("config.json")
```

---

### pretty_print_metrics

```python
def pretty_print_metrics(metrics: Dict[str, float])
```

Pretty print metrics dictionary.

**Example:**
```python
pretty_print_metrics({"accuracy": 0.95, "f1": 0.89})
```

---

## Column Expressions

### col

```python
def col(column_name: str)
```

Create a column reference for expressions.

**Examples:**
```python
df.filter(col("accuracy") > 0.8)
df.with_column("doubled", col("score") * 2)
df.filter((col("model") == "gpt-4") & (col("benchmark") == "mmlu"))
```

---

**Version:** 1.0.0  
**Last Updated:** November 15, 2025  
**Documentation:** https://docs.openai.internal/evalflow  
**Support:** eval-team@openai.com
