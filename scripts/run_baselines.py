#!/usr/bin/env python3
"""Week 1-2 baseline pipeline for lyrics-based multi-label auto-tagging.

Implements:
- data loading + cleaning
- fixed train/val/test split
- TF-IDF + OvR Logistic Regression
- TF-IDF + OvR Linear SVM
- metrics + per-tag reports + runtime tracking
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


@dataclass
class SplitData:
    x_train: pd.Series
    x_val: pd.Series
    x_test: pd.Series
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    labels: List[str]


@dataclass
class EvalResult:
    model: str
    split: str
    micro_f1: float
    macro_f1: float
    hamming_loss: float
    subset_accuracy: float
    train_seconds: float
    infer_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Week 1-2 baselines for lyrics-based auto-tagging."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV/Parquet/JSONL file.")
    parser.add_argument("--output-dir", default="outputs/week1_week2_baselines")
    parser.add_argument("--text-column", default="lyrics")
    parser.add_argument(
        "--tags-column",
        default="tags",
        help="Column containing delimiter-separated tags (used unless --tag-columns is set).",
    )
    parser.add_argument(
        "--tag-columns",
        default="",
        help="Comma-separated binary tag columns. If provided, overrides --tags-column.",
    )
    parser.add_argument("--tag-sep", default="|")
    parser.add_argument("--min-tag-frequency", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--class-weight", default="balanced", choices=["none", "balanced"])
    parser.add_argument("--logreg-c", type=float, default=4.0)
    parser.add_argument("--linearsvm-c", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for quick runs.")
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def split_tag_string(tag_string: str, sep: str) -> List[str]:
    if pd.isna(tag_string):
        return []
    return [t.strip() for t in str(tag_string).split(sep) if t.strip()]


def build_multilabel_targets(
    df: pd.DataFrame,
    tags_column: str,
    tag_columns_csv: str,
    tag_sep: str,
    min_tag_frequency: int,
) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    if tag_columns_csv.strip():
        tag_columns = [c.strip() for c in tag_columns_csv.split(",") if c.strip()]
        missing = [c for c in tag_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing tag columns: {missing}")

        y_df = df[tag_columns].copy()
        y_df = y_df.fillna(0)
        y_df = (y_df.astype(float) > 0).astype(int)
        support = y_df.sum(axis=0)
        keep_cols = support[support >= min_tag_frequency].index.tolist()
        if not keep_cols:
            raise ValueError("No tag columns left after min-tag-frequency filtering.")
        y_df = y_df[keep_cols]
        keep_rows = y_df.sum(axis=1) > 0
        return df.loc[keep_rows].copy(), y_df.loc[keep_rows].to_numpy(), keep_cols

    if tags_column not in df.columns:
        raise ValueError(
            f"Tag column '{tags_column}' not found. Use --tag-columns or set --tags-column."
        )

    tag_lists = df[tags_column].map(lambda x: split_tag_string(x, tag_sep))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(tag_lists)
    labels = list(mlb.classes_)
    if not labels:
        raise ValueError("No tags found after parsing tags column.")

    support = y.sum(axis=0)
    keep_mask = support >= min_tag_frequency
    if keep_mask.sum() == 0:
        raise ValueError("No labels left after min-tag-frequency filtering.")
    y = y[:, keep_mask]
    labels = [label for label, keep in zip(labels, keep_mask) if keep]
    keep_rows = y.sum(axis=1) > 0
    y = y[keep_rows]
    return df.loc[keep_rows].copy(), y, labels


def build_splits(args: argparse.Namespace, df: pd.DataFrame, y: np.ndarray, labels: List[str]) -> SplitData:
    x = df[args.text_column].astype(str)
    y_arr = y.astype(int)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y_arr, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )
    val_ratio_adjusted = args.val_size / (1.0 - args.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=args.random_state,
        shuffle=True,
    )
    return SplitData(
        x_train=x_train.reset_index(drop=True),
        x_val=x_val.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        labels=labels,
    )


def evaluate(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_seconds: float,
    infer_seconds: float,
) -> EvalResult:
    return EvalResult(
        model=model_name,
        split=split_name,
        micro_f1=f1_score(y_true, y_pred, average="micro", zero_division=0),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        hamming_loss=hamming_loss(y_true, y_pred),
        subset_accuracy=accuracy_score(y_true, y_pred),
        train_seconds=train_seconds,
        infer_seconds=infer_seconds,
    )


def per_tag_frame(labels: Iterable[str], y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return pd.DataFrame(
        {
            "tag": list(labels),
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
        }
    ).sort_values(["f1", "support"], ascending=[True, False])


def train_and_eval(
    args: argparse.Namespace,
    split_data: SplitData,
    classifier_name: str,
) -> tuple[List[EvalResult], pd.DataFrame]:
    class_weight = None if args.class_weight == "none" else args.class_weight
    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        strip_accents="unicode",
    )
    x_train_vec = vectorizer.fit_transform(split_data.x_train)
    x_val_vec = vectorizer.transform(split_data.x_val)
    x_test_vec = vectorizer.transform(split_data.x_test)

    if classifier_name == "logreg":
        base = LogisticRegression(
            C=args.logreg_c,
            max_iter=2000,
            solver="liblinear",
            class_weight=class_weight,
        )
        display_name = "TF-IDF + OvR LogisticRegression"
    elif classifier_name == "linearsvm":
        base = LinearSVC(C=args.linearsvm_c, class_weight=class_weight)
        display_name = "TF-IDF + OvR LinearSVC"
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    clf = OneVsRestClassifier(base)

    t0 = time.perf_counter()
    clf.fit(x_train_vec, split_data.y_train)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_val_pred = clf.predict(x_val_vec)
    infer_val_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    y_test_pred = clf.predict(x_test_vec)
    infer_test_seconds = time.perf_counter() - t2

    val_result = evaluate(
        display_name,
        "val",
        split_data.y_val,
        y_val_pred,
        train_seconds=train_seconds,
        infer_seconds=infer_val_seconds,
    )
    test_result = evaluate(
        display_name,
        "test",
        split_data.y_test,
        y_test_pred,
        train_seconds=train_seconds,
        infer_seconds=infer_test_seconds,
    )
    per_tag = per_tag_frame(split_data.labels, split_data.y_test, y_test_pred)
    return [val_result, test_result], per_tag


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(Path(args.input))
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in input data.")

    df = df.copy()
    df[args.text_column] = df[args.text_column].map(normalize_text)
    df = df[df[args.text_column] != ""].reset_index(drop=True)

    df, y, labels = build_multilabel_targets(
        df=df,
        tags_column=args.tags_column,
        tag_columns_csv=args.tag_columns,
        tag_sep=args.tag_sep,
        min_tag_frequency=args.min_tag_frequency,
    )
    split_data = build_splits(args, df, y, labels)

    all_results: List[EvalResult] = []
    for model_key in ("logreg", "linearsvm"):
        results, per_tag = train_and_eval(args, split_data, classifier_name=model_key)
        all_results.extend(results)
        per_tag.to_csv(out_dir / f"per_tag_metrics_{model_key}.csv", index=False)
        per_tag.head(20).to_csv(out_dir / f"lowest_f1_tags_{model_key}.csv", index=False)

    summary_df = pd.DataFrame([asdict(r) for r in all_results])
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    split_stats = {
        "num_samples_total": int(len(df)),
        "num_labels": int(len(labels)),
        "labels": labels,
        "label_support_total": dict(zip(labels, y.sum(axis=0).astype(int).tolist())),
        "split_sizes": {
            "train": int(split_data.y_train.shape[0]),
            "val": int(split_data.y_val.shape[0]),
            "test": int(split_data.y_test.shape[0]),
        },
        "args": vars(args),
    }
    save_json(out_dir / "run_metadata.json", split_stats)

    print("\nBaseline training complete.")
    print(f"Input: {args.input}")
    print(f"Output directory: {out_dir.resolve()}")
    print("\nMetrics summary (val/test):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
