#!/usr/bin/env python3
"""Generate report-ready plots from baseline experiment CSV outputs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# Use a writable matplotlib cache directory in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_local"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline metrics and per-tag diagnostics.")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing metrics_summary.csv and per_tag_metrics_*.csv files.",
    )
    parser.add_argument(
        "--figures-dir",
        default="",
        help="Output directory for figures (default: <results-dir>/figures).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "all"],
        help="Which split to plot from metrics_summary.csv.",
    )
    parser.add_argument(
        "--top-n-tags",
        type=int,
        default=10,
        help="How many lowest-F1 tags to show per model.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--title-prefix",
        default="Week 1-2 Baselines",
        help="Prefix for figure titles.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, needed: list[str], filename: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")


def save_fig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_main_metrics(metrics_df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    melt = metrics_df.melt(
        id_vars=["model", "split"],
        value_vars=["micro_f1", "macro_f1", "subset_accuracy"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melt, x="metric", y="value", hue="model", errorbar=None)
    plt.ylim(0, 1)
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title(f"{title_prefix}: Core Metrics")
    plt.legend(title="Model", loc="upper right")
    save_fig(out_dir / "core_metrics.png", dpi=dpi)


def plot_hamming_loss(metrics_df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=metrics_df, x="model", y="hamming_loss", hue="model", legend=False, errorbar=None)
    plt.ylim(0, max(1e-6, metrics_df["hamming_loss"].max() * 1.2))
    plt.xlabel("Model")
    plt.ylabel("Hamming Loss (lower is better)")
    plt.title(f"{title_prefix}: Hamming Loss")
    plt.xticks(rotation=10, ha="right")
    save_fig(out_dir / "hamming_loss.png", dpi=dpi)


def plot_runtime(metrics_df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    runtime = metrics_df[["model", "train_seconds", "infer_seconds"]].copy()
    runtime = runtime.groupby("model", as_index=False).mean(numeric_only=True)
    melt = runtime.melt(id_vars=["model"], var_name="stage", value_name="seconds")
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=melt, x="model", y="seconds", hue="stage", errorbar=None)
    plt.xlabel("Model")
    plt.ylabel("Seconds")
    plt.title(f"{title_prefix}: Runtime")
    plt.xticks(rotation=10, ha="right")
    plt.legend(title="Stage", labels=["train", "inference"])
    save_fig(out_dir / "runtime.png", dpi=dpi)


def plot_lowest_f1_tags(per_tag_file: Path, top_n: int, out_dir: Path, title_prefix: str, dpi: int) -> Path:
    df = pd.read_csv(per_tag_file)
    ensure_columns(df, ["tag", "f1", "support"], per_tag_file.name)
    df = df.sort_values(["f1", "support"], ascending=[True, False]).head(top_n)
    model_key = per_tag_file.stem.replace("per_tag_metrics_", "")
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=df, x="f1", y="tag", orient="h", color="#3b82f6", errorbar=None)
    plt.xlim(0, 1)
    plt.xlabel("F1")
    plt.ylabel("Tag")
    plt.title(f"{title_prefix}: Lowest F1 Tags ({model_key})")
    out_path = out_dir / f"lowest_f1_tags_{model_key}.png"
    save_fig(out_path, dpi=dpi)
    return out_path


def plot_support_vs_f1(per_tag_file: Path, out_dir: Path, title_prefix: str, dpi: int) -> Path:
    df = pd.read_csv(per_tag_file)
    ensure_columns(df, ["tag", "f1", "support"], per_tag_file.name)
    model_key = per_tag_file.stem.replace("per_tag_metrics_", "")
    plt.figure(figsize=(7, 4.8))
    sns.scatterplot(data=df, x="support", y="f1", s=40, alpha=0.85)
    plt.ylim(0, 1)
    plt.xlabel("Tag Support")
    plt.ylabel("F1")
    plt.title(f"{title_prefix}: Support vs F1 ({model_key})")
    out_path = out_dir / f"support_vs_f1_{model_key}.png"
    save_fig(out_path, dpi=dpi)
    return out_path


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics_summary.csv not found in: {results_dir}")

    metrics = pd.read_csv(metrics_path)
    ensure_columns(
        metrics,
        [
            "model",
            "split",
            "micro_f1",
            "macro_f1",
            "hamming_loss",
            "subset_accuracy",
            "train_seconds",
            "infer_seconds",
        ],
        metrics_path.name,
    )
    metrics["split"] = metrics["split"].astype(str)

    if args.split != "all":
        metrics = metrics[metrics["split"] == args.split].copy()
        if metrics.empty:
            raise ValueError(
                f"No rows found in metrics_summary.csv for split='{args.split}'. "
                "Use --split all or check the file."
            )

    prefix = f"{args.title_prefix} ({args.split})" if args.split != "all" else f"{args.title_prefix} (all)"
    plot_main_metrics(metrics, figures_dir, prefix, args.dpi)
    plot_hamming_loss(metrics, figures_dir, prefix, args.dpi)
    plot_runtime(metrics, figures_dir, prefix, args.dpi)

    per_tag_files = sorted(results_dir.glob("per_tag_metrics_*.csv"))
    if not per_tag_files:
        print("Warning: no per_tag_metrics_*.csv files found; skipping tag-level plots.")
    else:
        for per_tag_file in per_tag_files:
            plot_lowest_f1_tags(
                per_tag_file=per_tag_file,
                top_n=args.top_n_tags,
                out_dir=figures_dir,
                title_prefix=args.title_prefix,
                dpi=args.dpi,
            )
            plot_support_vs_f1(
                per_tag_file=per_tag_file,
                out_dir=figures_dir,
                title_prefix=args.title_prefix,
                dpi=args.dpi,
            )

    print("Plots generated:")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"- {p}")


if __name__ == "__main__":
    main()
