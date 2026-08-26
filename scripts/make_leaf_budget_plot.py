"""Create the camera-ready leaf-budget trade-off figure.

The manuscript averages metrics over datasets, giving each of the 21 paper
datasets equal weight. This script deliberately applies the same aggregation
instead of pooling all available datasets or outer-split rows.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_OPENML_IDS = {
    3, 15, 24, 27, 31, 37, 44, 55, 56, 151, 162, 1120, 1461, 1462,
    1464, 1489, 1494, 1504, 1590, 1597, 45557,
}


def extract_openml_id(value: str) -> int:
    match = re.search(r"(?:OpenML_|openml_)(\d+)", str(value))
    if match is None:
        raise ValueError(f"Cannot extract OpenML ID from {value!r}")
    return int(match.group(1))


def mean_for_paper_ids(frame: pd.DataFrame, value_col: str) -> float:
    subset = frame[frame["openml_id"].isin(PAPER_OPENML_IDS)]
    present = set(subset["openml_id"].unique())
    missing = PAPER_OPENML_IDS - present
    if missing:
        raise ValueError(f"Missing paper datasets for {value_col}: {sorted(missing)}")
    return float(subset.groupby("openml_id")[value_col].mean().mean())


def load_mbndt_points() -> pd.DataFrame:
    path = ROOT / "results" / "tables" / "ALL_leaf_budget_pareto_points.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["reported_model"] == "pp"].copy()
    frame["openml_id"] = frame["dataset_folder"].map(extract_openml_id)
    frame = frame[frame["openml_id"].isin(PAPER_OPENML_IDS)]

    label_order = {
        "K0004": "K=4",
        "K0008": "K=8",
        "K0016": "K=16",
        "K0032": "K=32",
        "K0064": "K=64",
        "HPO_Kstar": "HPO-selected",
        "NO_BUDGET": "No budget",
    }
    rows = []
    for setting, label in label_order.items():
        group = frame[frame["budget_setting"] == setting]
        if set(group["openml_id"]) != PAPER_OPENML_IDS:
            raise ValueError(f"Incomplete 21-dataset coverage for {setting}")
        rows.append(
            {
                "label": label,
                "accuracy": group.groupby("openml_id")["test_balanced_acc_mean"].mean().mean(),
                "leaves": group.groupby("openml_id")["routable_leaves_mean"].mean().mean(),
            }
        )
    return pd.DataFrame(rows)


def load_baselines() -> pd.DataFrame:
    specs = [
        (
            "CART",
            ROOT / "results" / "tables" / "CART_aggregated_report_metrics.csv",
            "test__bal_acc_mean",
            "cx__n_leaves_struct_mean",
        ),
        (
            "SPLIT",
            ROOT / "results" / "tables" / "SPLIT_aggregated_eport_metrics.csv",
            "test__bal_acc_mean",
            "cx__n_leaves_struct_mean",
        ),
        (
            "GradTree",
            ROOT / "results" / "tables" / "GradTree_aggregated_mbndt_metrics.csv",
            "gradtree__test__bal_acc_mean",
            "gradtree__cx__leaves_mean",
        ),
    ]
    rows = []
    for label, path, accuracy_col, leaves_col in specs:
        frame = pd.read_csv(path)
        frame["openml_id"] = frame["dataset"].map(extract_openml_id)
        rows.append(
            {
                "label": label,
                "accuracy": mean_for_paper_ids(frame, accuracy_col),
                "leaves": mean_for_paper_ids(frame, leaves_col),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    mbndt = load_mbndt_points()
    baselines = load_baselines()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.15), constrained_layout=True)

    k_sweep = mbndt[mbndt["label"].str.startswith("K=")]
    references = mbndt[~mbndt["label"].str.startswith("K=")]

    ax.plot(
        k_sweep["leaves"],
        k_sweep["accuracy"],
        color="#1f77b4",
        linewidth=0.8,
        alpha=0.55,
        zorder=1,
    )
    ax.scatter(
        k_sweep["leaves"],
        k_sweep["accuracy"],
        s=23,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.4,
        label="MBNDT K-sweep",
        zorder=3,
    )

    annotations = {
        "K=4": ((3, 5), "left"),
        "K=8": ((3, -12), "left"),
        "K=16": ((3, 6), "left"),
        "K=32": ((-10, -15), "right"),
        "K=64": ((-3, 8), "right"),
    }
    for row in k_sweep.itertuples(index=False):
        ax.annotate(
            row.label,
            (row.leaves, row.accuracy),
            xytext=annotations[row.label][0],
            textcoords="offset points",
            color="#155a8a",
            fontsize=6.5,
            horizontalalignment=annotations[row.label][1],
        )

    reference_styles = {
        "HPO-selected": ("*", "#7b3294"),
        "No budget": ("X", "#666666"),
    }
    for row in references.itertuples(index=False):
        marker, color = reference_styles[row.label]
        ax.scatter(
            row.leaves,
            row.accuracy,
            marker=marker,
            s=38,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=row.label,
            zorder=4,
        )

    styles = {
        "CART": ("s", "#d62728"),
        "SPLIT": ("D", "#2ca02c"),
        "GradTree": ("^", "#ff7f0e"),
    }
    for row in baselines.itertuples(index=False):
        marker, color = styles[row.label]
        ax.scatter(
            row.leaves,
            row.accuracy,
            marker=marker,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=row.label,
            zorder=4,
        )

    ax.set_xlabel("Mean realized leaves")
    ax.set_ylabel("Mean test balanced accuracy")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(loc="lower right", ncol=2, frameon=True, borderpad=0.4)
    ax.margins(x=0.10, y=0.10)

    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "leaf_budget_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "leaf_budget_tradeoff.png", dpi=300, bbox_inches="tight")
    print(mbndt.to_string(index=False))
    print(baselines.to_string(index=False))
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
