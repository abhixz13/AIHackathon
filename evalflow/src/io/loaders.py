from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..core.dataframe import EvalDataFrame


def _read_single(path: str, format: Optional[str] = None, **kwargs) -> pd.DataFrame:
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    if format is None:
        if ext in {".parquet"}:
            format = "parquet"
        elif ext in {".csv", ".tsv"}:
            format = "csv"
        elif ext in {".json", ".jsonl"}:
            format = "json"
        else:
            raise ValueError(f"Could not infer format from extension '{ext}' for path '{path}'")
    if format == "parquet":
        return pd.read_parquet(path_obj, **kwargs)
    elif format == "csv":
        return pd.read_csv(path_obj, **kwargs)
    elif format == "json":
        return pd.read_json(path_obj, lines=path_obj.suffix.lower() == ".jsonl", **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(
    path: Union[str, Path],
    format: Optional[str] = None,
    schema: Optional[Dict] = None,
    **kwargs,
) -> EvalDataFrame:
    """Load evaluation results from file(s).

    Parameters are intentionally aligned with the documented EvalFlow API,
    though this implementation relies on pandas under the hood.
    """

    path_str = str(path)
    paths: List[str]
    if any(ch in path_str for ch in ["*", "?", "["]):
        paths = sorted(glob.glob(path_str))
        if not paths:
            raise FileNotFoundError(f"No files matched glob pattern: {path_str!r}")
    else:
        paths = [path_str]

    frames = [_read_single(p, format=format, **kwargs) for p in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if schema is not None:
        # Apply simple schema projection if provided (keys become columns)
        cols = list(schema.keys())
        df = df[cols]

    return EvalDataFrame(df)


def load_from_parquet(
    path: str,
    schema: Optional[Dict] = None,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> EvalDataFrame:
    df = _read_single(path, format="parquet", columns=columns, **kwargs)
    if schema is not None:
        df = df[list(schema.keys())]
    return EvalDataFrame(df)


def load_from_csv(
    path: str,
    schema: Optional[Dict] = None,
    delimiter: str = ",",
    has_header: bool = True,
    **kwargs,
) -> EvalDataFrame:
    df = _read_single(
        path,
        format="csv",
        sep=delimiter,
        header=0 if has_header else None,
        **kwargs,
    )
    if schema is not None:
        df = df[list(schema.keys())]
    return EvalDataFrame(df)


def load_from_json(
    path: str,
    schema: Optional[Dict] = None,
    **kwargs,
) -> EvalDataFrame:
    df = _read_single(path, format="json", **kwargs)
    if schema is not None:
        df = df[list(schema.keys())]
    return EvalDataFrame(df)


def load_from_pandas(df_pandas: pd.DataFrame) -> EvalDataFrame:
    return EvalDataFrame(df_pandas)


def load_from_dict(data: Dict) -> EvalDataFrame:
    return EvalDataFrame(data)


def load_experiment_results(
    experiment_ids: Union[str, List[str]],
    base_path: str = "experiments",
    **kwargs,
) -> EvalDataFrame:
    if isinstance(experiment_ids, str):
        ids = [experiment_ids]
    else:
        ids = list(experiment_ids)
    paths = [str(Path(base_path) / f"{exp_id}.parquet") for exp_id in ids]
    frames = [_read_single(p, format="parquet", **kwargs) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return EvalDataFrame(df)


def load_benchmark_results(
    benchmark: str,
    models: Optional[List[str]] = None,
    base_path: str = "benchmarks",
    **kwargs,
) -> EvalDataFrame:
    pattern = str(Path(base_path) / f"{benchmark}_*.parquet")
    paths = sorted(glob.glob(pattern))
    frames = [_read_single(p, format="parquet", **kwargs) for p in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if models is not None and "model" in df.columns:
        df = df[df["model"].isin(models)]
    return EvalDataFrame(df)
