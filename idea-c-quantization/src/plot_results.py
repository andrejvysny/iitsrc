"""Generate paper-ready figures for Idea C quantization study."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = _root / "paper-c" / "figures"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"

# Consistent ordering
QUANT_ORDER = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q3_k_l", "q2_k"]
CROSS_QUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m"]
MODEL_ORDER = ["qwen2.5-3b", "llama-3.2-3b", "phi-3.5-mini", "mistral-7b"]
MODEL_DISPLAY = {
    "qwen2.5-3b": "Qwen2.5-3B",
    "llama-3.2-3b": "Llama-3.2-3B",
    "phi-3.5-mini": "Phi-3.5-mini",
    "mistral-7b": "Mistral-7B",
}

# IEEE-friendly styling
MODEL_COLORS = {
    "qwen2.5-3b": "#1f77b4",
    "llama-3.2-3b": "#ff7f0e",
    "phi-3.5-mini": "#2ca02c",
    "mistral-7b": "#d62728",
}
MODEL_MARKERS = {
    "qwen2.5-3b": "o",
    "llama-3.2-3b": "s",
    "phi-3.5-mini": "^",
    "mistral-7b": "D",
}

# Quant display labels (strip underscores for readability)
QUANT_DISPLAY = {
    "f16": "F16", "q8_0": "Q8_0", "q6_k": "Q6_K",
    "q5_k_m": "Q5_K_M", "q4_k_m": "Q4_K_M",
    "q3_k_m": "Q3_K_M", "q3_k_l": "Q3_K_L", "q2_k": "Q2_K",
}


def setup_ieee_style() -> None:
    """Configure matplotlib for IEEE paper figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def load_data() -> pd.DataFrame:
    df = pd.read_csv(EXPERIMENTS_CSV)
    df["quant"] = pd.Categorical(df["quant"], categories=QUANT_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


def get_common_pages(df: pd.DataFrame) -> set:
    from functools import reduce
    page_sets = [set(df[df["model"] == m]["page_id"]) for m in MODEL_ORDER]
    return reduce(lambda a, b: a & b, page_sets)


def get_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per (model, quant)."""
    return df.groupby(["model", "quant"], observed=True).agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        json_valid_pct=("json_valid", "mean"),
        mean_latency=("latency_s", "mean"),
        model_size_gb=("model_size_gb", "first"),
        mean_tokens_in=("tokens_in", "mean"),
        success_rate=("f1", lambda x: (x > 0).mean()),
        n=("f1", "size"),
    ).reset_index()


def fig1_f1_vs_quant(stats: pd.DataFrame) -> None:
    """Fig 1: F1 vs quantization level per model (line chart)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for model in MODEL_ORDER:
        ms = stats[stats["model"] == model].sort_values("quant")
        if ms.empty:
            continue
        x_labels = [QUANT_DISPLAY.get(str(q), str(q)) for q in ms["quant"]]
        ax.plot(
            x_labels, ms["mean_f1"],
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            label=MODEL_DISPLAY[model],
            linewidth=1.2,
            markersize=4,
        )
        # Error bands
        ax.fill_between(
            x_labels,
            ms["mean_f1"] - ms["std_f1"],
            ms["mean_f1"] + ms["std_f1"],
            alpha=0.1,
            color=MODEL_COLORS[model],
        )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Extraction Quality vs. Quantization Level")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "f1_vs_quant.pdf")
    fig.savefig(FIGURES_DIR / "f1_vs_quant.png")
    plt.close(fig)
    print("  Saved: f1_vs_quant.pdf")


def fig2_json_validity_heatmap(stats: pd.DataFrame) -> None:
    """Fig 2: JSON validity rate heatmap (model × quant)."""
    # Pivot: model rows, quant columns
    pivot = stats.pivot_table(
        index="model", columns="quant",
        values="json_valid_pct", observed=True,
    )
    pivot = pivot * 100  # to percentage
    # Reorder
    pivot = pivot.reindex(index=MODEL_ORDER, columns=[q for q in QUANT_ORDER if q in pivot.columns])
    pivot.index = [MODEL_DISPLAY.get(m, m) for m in pivot.index]
    pivot.columns = [QUANT_DISPLAY.get(str(q), str(q)) for q in pivot.columns]

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=0, vmax=100, ax=ax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "JSON Valid %", "shrink": 0.8},
        annot_kws={"size": 7},
    )
    ax.set_title("JSON Validity Rate by Model and Quantization")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "json_validity_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "json_validity_heatmap.png")
    plt.close(fig)
    print("  Saved: json_validity_heatmap.pdf")


def fig3_latency_vs_size(stats: pd.DataFrame) -> None:
    """Fig 3: Latency vs model file size, colored by quant level."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Color quants by a sequential colormap
    quant_list = [q for q in QUANT_ORDER if q in stats["quant"].unique()]
    cmap = plt.cm.viridis
    quant_colors = {q: cmap(i / max(len(quant_list) - 1, 1)) for i, q in enumerate(quant_list)}

    for model in MODEL_ORDER:
        ms = stats[stats["model"] == model]
        for _, row in ms.iterrows():
            q = str(row["quant"])
            ax.scatter(
                row["model_size_gb"], row["mean_latency"],
                color=quant_colors.get(q, "gray"),
                marker=MODEL_MARKERS[model],
                s=30, edgecolors="black", linewidth=0.3,
                zorder=3,
            )

    # Model legend
    for model in MODEL_ORDER:
        ax.scatter([], [], marker=MODEL_MARKERS[model], color="gray",
                   label=MODEL_DISPLAY[model], edgecolors="black", linewidth=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    # Quant colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, len(quant_list) - 1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_ticks(range(len(quant_list)))
    cbar.set_ticklabels([QUANT_DISPLAY.get(q, q) for q in quant_list])
    cbar.set_label("Quant Level", fontsize=7)

    ax.set_xlabel("Model Size (GB)")
    ax.set_ylabel("Mean Latency (s)")
    ax.set_title("Inference Latency vs. Model Size")
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "latency_vs_size.pdf")
    fig.savefig(FIGURES_DIR / "latency_vs_size.png")
    plt.close(fig)
    print("  Saved: latency_vs_size.pdf")


def fig4_success_vs_tokens(df: pd.DataFrame) -> None:
    """Fig 4: Success rate vs input token count (binned)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Bin tokens
    bins = [0, 500, 1000, 2000, 3000, 5000, 10000]
    labels = ["<500", "500-1k", "1k-2k", "2k-3k", "3k-5k", "5k+"]
    df = df.copy()
    df["token_bin"] = pd.cut(df["tokens_in"], bins=bins, labels=labels, right=False)
    df["success"] = (df["f1"] > 0).astype(int)

    for model in MODEL_ORDER:
        ms = df[df["model"] == model]
        if ms.empty:
            continue
        binned = ms.groupby("token_bin", observed=True)["success"].mean() * 100
        ax.plot(
            binned.index.astype(str), binned.values,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            label=MODEL_DISPLAY[model],
            linewidth=1.2,
            markersize=4,
        )

    ax.set_xlabel("Input Token Count")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Extraction Success vs. Input Length")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "success_vs_tokens.pdf")
    fig.savefig(FIGURES_DIR / "success_vs_tokens.png")
    plt.close(fig)
    print("  Saved: success_vs_tokens.pdf")


def main() -> None:
    setup_ieee_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    common_pages = get_common_pages(df)
    df_common = df[df["page_id"].isin(common_pages)].copy()

    # Exclude llama q3_k_l for cross-model comparison
    mask = ~((df_common["model"] == "llama-3.2-3b") & (df_common["quant"] == "q3_k_l"))
    df_filtered = df_common[mask]

    stats = get_stats(df_filtered)
    print(f"Generating figures from {len(df_common)} rows (100 common pages)...\n")

    fig1_f1_vs_quant(stats)
    fig2_json_validity_heatmap(stats)
    fig3_latency_vs_size(stats)
    fig4_success_vs_tokens(df_filtered)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
