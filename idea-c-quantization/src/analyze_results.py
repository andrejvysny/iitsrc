"""Analyze Idea C experiment results: filter, aggregate, export tables."""

import sys
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"

# Quant ordering: highest precision → lowest
QUANT_ORDER = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q3_k_l", "q2_k"]

# Cross-model comparison: only quants shared by all 4 models
CROSS_MODEL_QUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m"]

# Llama has q3_k_l instead of q3_k_m — exclude from cross-model q3 comparison
LLAMA_EXCLUDE_QUANTS = ["q3_k_l"]

MODEL_ORDER = ["qwen2.5-3b", "llama-3.2-3b", "phi-3.5-mini", "mistral-7b"]
MODEL_DISPLAY = {
    "qwen2.5-3b": "Qwen2.5-3B",
    "llama-3.2-3b": "Llama-3.2-3B",
    "phi-3.5-mini": "Phi-3.5-mini",
    "mistral-7b": "Mistral-7B",
}


def load_data() -> pd.DataFrame:
    """Load experiments CSV."""
    df = pd.read_csv(EXPERIMENTS_CSV)
    df["quant"] = pd.Categorical(df["quant"], categories=QUANT_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


def find_common_pages(df: pd.DataFrame) -> set:
    """Find page_ids present in all 4 models."""
    from functools import reduce
    page_sets = [set(df[df["model"] == m]["page_id"]) for m in MODEL_ORDER]
    return reduce(lambda a, b: a & b, page_sets)


def filter_common_pages(df: pd.DataFrame, common_pages: set) -> pd.DataFrame:
    """Filter to common pages only."""
    return df[df["page_id"].isin(common_pages)].copy()


def per_model_quant_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (model, quant): mean F1, JSON valid %, mean latency, success rate, N."""
    grouped = df.groupby(["model", "quant"], observed=True)
    stats = grouped.agg(
        n=("f1", "size"),
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        json_valid_pct=("json_valid", "mean"),
        schema_valid_pct=("schema_valid", "mean"),
        mean_hallucination=("hallucination_rate", "mean"),
        exact_match_pct=("exact_match", "mean"),
        mean_latency=("latency_s", "mean"),
        mean_tokens_in=("tokens_in", "mean"),
        model_size_gb=("model_size_gb", "first"),
        success_rate=("f1", lambda x: (x > 0).mean()),
    ).reset_index()

    # Convert rates to percentages
    for col in ["json_valid_pct", "schema_valid_pct", "exact_match_pct", "success_rate"]:
        stats[col] = (stats[col] * 100).round(1)
    for col in ["mean_f1", "std_f1", "mean_precision", "mean_recall", "mean_hallucination"]:
        stats[col] = stats[col].round(4)
    stats["mean_latency"] = stats["mean_latency"].round(2)
    stats["mean_tokens_in"] = stats["mean_tokens_in"].round(0).astype(int)

    return stats


def cross_model_stats(df: pd.DataFrame, common_pages: set) -> pd.DataFrame:
    """Stats for cross-model comparison: common pages × shared quants only."""
    df_common = filter_common_pages(df, common_pages)
    # Keep only shared quant levels
    df_cross = df_common[df_common["quant"].isin(CROSS_MODEL_QUANTS)].copy()
    return per_model_quant_stats(df_cross)


def per_model_all_quants(df: pd.DataFrame, common_pages: set) -> pd.DataFrame:
    """Stats per model using all their quants, but on common 100 pages."""
    df_common = filter_common_pages(df, common_pages)
    # Exclude llama's q3_k_l from this view
    mask = ~((df_common["model"] == "llama-3.2-3b") & (df_common["quant"] == "q3_k_l"))
    return per_model_quant_stats(df_common[mask])


def generate_latex_table(stats: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Quant & F1 & JSON\% & Latency(s) & Success\% \\")
    lines.append(r"\midrule")

    prev_model = None
    for _, row in stats.iterrows():
        model_name = MODEL_DISPLAY.get(str(row["model"]), str(row["model"]))
        quant = str(row["quant"])

        if prev_model is not None and str(row["model"]) != prev_model:
            lines.append(r"\midrule")
        prev_model = str(row["model"])

        lines.append(
            f"{model_name} & {quant} & "
            f"{row['mean_f1']:.3f} & "
            f"{row['json_valid_pct']:.1f} & "
            f"{row['mean_latency']:.1f} & "
            f"{row['success_rate']:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_model_summary_latex(stats: pd.DataFrame) -> str:
    """Model-level summary table (best quant per model)."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Model performance summary (100 common pages, best quantization)}")
    lines.append(r"\label{tab:model_summary}")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Model & Best F1 & JSON\% & Latency(s) & Size(GB) & Best Quant \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        ms = stats[stats["model"] == model]
        if ms.empty:
            continue
        best = ms.loc[ms["mean_f1"].idxmax()]
        model_name = MODEL_DISPLAY.get(model, model)
        lines.append(
            f"{model_name} & {best['mean_f1']:.3f} & "
            f"{best['json_valid_pct']:.1f} & "
            f"{best['mean_latency']:.1f} & "
            f"{best['model_size_gb']:.2f} & "
            f"{best['quant']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    print("Loading experiments data...")
    df = load_data()
    print(f"Total rows: {len(df)}")

    common_pages = find_common_pages(df)
    print(f"Common pages across all models: {len(common_pages)}")

    # 1. Cross-model comparison (shared quants, 100 pages)
    print("\n=== Cross-model comparison (shared quants) ===")
    cross_stats = cross_model_stats(df, common_pages)
    print(cross_stats.to_string(index=False))
    cross_stats.to_csv(RESULTS_DIR / "cross_model_comparison.csv", index=False)

    # 2. Per-model all quants (100 pages)
    print("\n=== Per-model all quants (100 pages) ===")
    all_stats = per_model_all_quants(df, common_pages)
    print(all_stats.to_string(index=False))
    all_stats.to_csv(RESULTS_DIR / "analysis_100pages.csv", index=False)

    # 3. Full dataset stats (all pages per model)
    print("\n=== Full dataset stats ===")
    full_stats = per_model_quant_stats(df)
    full_stats.to_csv(RESULTS_DIR / "analysis_full.csv", index=False)

    # 4. LaTeX tables
    latex_dir = RESULTS_DIR / "latex"
    latex_dir.mkdir(exist_ok=True)

    # Main results table
    latex_cross = generate_latex_table(
        cross_stats,
        caption="Cross-model comparison on 100 common pages (shared quantization levels)",
        label="tab:cross_model",
    )
    (latex_dir / "tab_cross_model.tex").write_text(latex_cross)

    # Per-model detailed table
    latex_all = generate_latex_table(
        all_stats,
        caption="Extraction performance across quantization levels (100 common pages)",
        label="tab:quant_results",
    )
    (latex_dir / "tab_quant_results.tex").write_text(latex_all)

    # Model summary
    latex_summary = generate_model_summary_latex(all_stats)
    (latex_dir / "tab_model_summary.tex").write_text(latex_summary)

    print(f"\nOutputs saved to {RESULTS_DIR}/")
    print(f"  - cross_model_comparison.csv")
    print(f"  - analysis_100pages.csv")
    print(f"  - analysis_full.csv")
    print(f"  - latex/tab_cross_model.tex")
    print(f"  - latex/tab_quant_results.tex")
    print(f"  - latex/tab_model_summary.tex")


if __name__ == "__main__":
    main()
