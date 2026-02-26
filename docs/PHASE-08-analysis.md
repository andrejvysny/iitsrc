# PHASE 8: Analysis & Visualization

**Duration**: ~2-3 hours
**Schedule**: Day 2, 18:00-22:00
**Dependencies**: Phase 5 (local results), Phase 6 (quant results), Phase 7 (cloud baselines)
**Blocks**: Phase 9 (paper needs figures and tables)

---

## 1. Objective

Create publication-quality analysis notebooks, charts, and statistical tables for both ideas. Export all figures as PDF vectors for LaTeX inclusion. Identify key findings for paper narrative.

---

## 2. Tasks

### 2.1 Idea B Analysis Notebook

**File**: `idea-b-schema-pruning/notebooks/analysis.ipynb`

#### Cell structure:

1. **Setup & Data Loading**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    'font.size': 9,
    'figure.figsize': (3.5, 2.5),  # IEEE single column width
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Load results
df_local = pd.read_csv("../results/experiments.csv")
df_cloud = pd.read_csv("../../shared/results/cloud_baselines_idea_b.csv")

print(f"Local: {len(df_local)} rows, {df_local.groupby(['pruning','format','model']).ngroups} configs")
print(f"Cloud: {len(df_cloud)} rows")
```

2. **Table 1: Main Results — Pruning × Format × Model → Avg F1**
```python
# Pivot table: rows = pruning×format, columns = model
pivot = df_local.groupby(["pruning", "format", "model"])["f1"].agg(["mean", "std"]).reset_index()
pivot_table = pivot.pivot_table(
    index=["pruning", "format"],
    columns="model",
    values="mean",
)
print(pivot_table.round(3).to_latex())
```

3. **Figure 1: F1 by Pruning Strategy (bar chart)**
```python
fig, ax = plt.subplots(figsize=(3.5, 2.5))
pruning_order = ["none", "generic", "heuristic", "semantic"]
avg_f1 = df_local.groupby("pruning")["f1"].mean().reindex(pruning_order)
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
bars = ax.bar(range(len(pruning_order)), avg_f1.values, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(pruning_order)))
ax.set_xticklabels(["No Pruning", "Generic", "Schema\nHeuristic", "Schema\nSemantic"], fontsize=8)
ax.set_ylabel("Macro F1")
ax.set_title("Extraction Quality by Pruning Strategy")
ax.set_ylim(0, 1)
for bar, val in zip(bars, avg_f1.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}",
            ha='center', va='bottom', fontsize=7)
plt.savefig("../figures/f1_by_pruning.pdf")
plt.show()
```

4. **Figure 2: Token Reduction vs F1 (scatter)**
```python
fig, ax = plt.subplots(figsize=(3.5, 2.5))
grouped = df_local.groupby(["pruning", "format", "model"]).agg(
    avg_f1=("f1", "mean"),
    avg_tokens=("tokens_in", "mean"),
).reset_index()

for model in grouped["model"].unique():
    mdf = grouped[grouped["model"] == model]
    ax.scatter(mdf["avg_tokens"], mdf["avg_f1"], label=model, alpha=0.7, s=30)

ax.set_xlabel("Average Input Tokens")
ax.set_ylabel("Macro F1")
ax.set_title("Token Reduction vs Extraction Quality")
ax.legend(fontsize=7, loc="lower right")
ax.set_xscale("log")
plt.savefig("../figures/tokens_vs_f1.pdf")
plt.show()
```

5. **Figure 3: Comparison with Cloud Baselines**
```python
fig, ax = plt.subplots(figsize=(3.5, 3.0))
# Best local config
best_local = df_local.groupby(["pruning", "format", "model"])["f1"].mean().idxmax()
best_local_f1 = df_local.groupby(["pruning", "format", "model"])["f1"].mean().max()

# Cloud baselines (best pruning)
cloud_f1 = df_cloud.groupby("model")["f1"].mean().sort_values(ascending=False)

# Combined bar chart
models = list(cloud_f1.index) + [f"Best Local\n({best_local[2]})"]
f1s = list(cloud_f1.values) + [best_local_f1]
colors = ["#1f77b4"] * len(cloud_f1) + ["#2ca02c"]

ax.barh(range(len(models)), f1s, color=colors, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=8)
ax.set_xlabel("Macro F1")
ax.set_title("Local vs Cloud Model Comparison")
ax.set_xlim(0, 1)
plt.savefig("../figures/local_vs_cloud.pdf")
plt.show()
```

6. **Statistical Summary**
```python
# Per-domain analysis
print("\nE-commerce vs Real Estate:")
for domain in ["ecommerce", "realestate"]:
    ddf = df_local[df_local["domain"] == domain]
    print(f"  {domain}: F1={ddf['f1'].mean():.3f} ± {ddf['f1'].std():.3f}")

# Per-site analysis
print("\nPer-site F1:")
df_local["site"] = df_local["page_id"].str.split("_").str[1]
print(df_local.groupby("site")["f1"].mean().round(3))

# JSON validity summary
print("\nJSON validity by config:")
print(df_local.groupby(["pruning", "model"])["json_valid"].mean().round(3))
```

7. **Ablation: Format Comparison**
```python
# Which format works best per pruning strategy?
format_ablation = df_local.groupby(["pruning", "format"])["f1"].mean().unstack()
print("\nFormat ablation (F1):")
print(format_ablation.round(3))
```

### 2.2 Idea C Analysis Notebook

**File**: `idea-c-quantization/notebooks/analysis.ipynb`

#### Cell structure:

1. **Setup & Data Loading** (same style as Idea B)

2. **Figure 1: Quality Degradation Curve**
```python
fig, ax = plt.subplots(figsize=(3.5, 2.5))
quant_order = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]
bits_map = {"f16": 16, "q8_0": 8, "q6_k": 6, "q5_k_m": 5, "q4_k_m": 4, "q3_k_m": 3, "q2_k": 2}

for model in df["model"].unique():
    mdf = df[df["model"] == model]
    by_quant = mdf.groupby("quant")["f1"].mean()
    bits = [bits_map[q] for q in quant_order if q in by_quant.index]
    f1s = [by_quant[q] for q in quant_order if q in by_quant.index]
    ax.plot(bits, f1s, marker="o", label=model, markersize=4, linewidth=1.5)

ax.set_xlabel("Bits per Weight")
ax.set_ylabel("Macro F1")
ax.set_title("Extraction Quality vs Quantization Level")
ax.legend(fontsize=7)
ax.invert_xaxis()  # F16 on left, Q2 on right
ax.set_xticks([2, 3, 4, 5, 6, 8, 16])
plt.savefig("../figures/degradation_curve.pdf")
plt.show()
```

3. **Figure 2: Pareto Frontier (Memory vs F1)**
```python
fig, ax = plt.subplots(figsize=(3.5, 2.5))
grouped = df.groupby(["model", "quant"]).agg(
    avg_f1=("f1", "mean"),
    model_size=("model_size_gb", "first"),
).reset_index()

markers = {"qwen2.5-3b": "o", "llama-3.2-3b": "s", "phi-3.5-mini": "^", "mistral-7b": "D"}
for model in grouped["model"].unique():
    mdf = grouped[grouped["model"] == model]
    ax.scatter(mdf["model_size"], mdf["avg_f1"], marker=markers.get(model, "o"),
               label=model, s=40, alpha=0.8)
    # Annotate each point with quant level
    for _, row in mdf.iterrows():
        ax.annotate(row["quant"], (row["model_size"], row["avg_f1"]),
                    fontsize=5, textcoords="offset points", xytext=(3, 3))

# Draw Pareto frontier
all_points = list(zip(grouped["model_size"], grouped["avg_f1"]))
pareto = []
for p in sorted(all_points):
    if not pareto or p[1] > pareto[-1][1]:
        pareto.append(p)
if pareto:
    pareto_x, pareto_y = zip(*pareto)
    ax.plot(pareto_x, pareto_y, "r--", alpha=0.5, linewidth=1, label="Pareto frontier")

ax.set_xlabel("Model Size (GB)")
ax.set_ylabel("Macro F1")
ax.set_title("Quality-Efficiency Pareto Frontier")
ax.legend(fontsize=6, loc="lower right")
plt.savefig("../figures/pareto_frontier.pdf")
plt.show()
```

4. **Figure 3: Per-Field-Type F1 Heatmap**
```python
# Requires per-field breakdown from experiments
# Categorize fields by type:
# - numeric: price, rating, bedrooms, bathrooms, area_sqm
# - string: name, description, address, brand
# - enum: currency, availability, type
# - array: specs

field_types = {
    "numeric": ["price", "rating", "bedrooms", "bathrooms", "area_sqm"],
    "string": ["name", "description", "address", "brand"],
    "enum": ["currency", "availability", "type"],
    "array": ["specs"],
}

# This requires per-field F1 data — computed from raw predictions stored during experiments
# If per-field data not available in CSV, re-compute from stored predictions

fig, ax = plt.subplots(figsize=(3.5, 3.0))
# Heatmap: rows = quant levels, columns = field types
# Values = average F1 for that field type at that quant level
# ... (populate from per-field analysis)
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
            xticklabels=["Numeric", "String", "Enum", "Array"],
            yticklabels=quant_labels, ax=ax)
ax.set_title("F1 by Field Type × Quantization")
plt.savefig("../figures/field_type_heatmap.pdf")
plt.show()
```

5. **Table: Main Results**
```python
# Model × Quant → F1, JSON%, Latency, Size
summary = df.groupby(["model", "quant"]).agg(
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
    json_valid=("json_valid", "mean"),
    schema_valid=("schema_valid", "mean"),
    latency_mean=("latency_s", "mean"),
    model_size=("model_size_gb", "first"),
).round(3)
print(summary.to_latex())
```

6. **Cliff Detection**
```python
print("\nCliff Detection (>5% F1 drop between adjacent quant levels):")
for model in df["model"].unique():
    mdf = df[df["model"] == model]
    by_quant = mdf.groupby("quant")["f1"].mean()
    prev_f1 = None
    prev_q = None
    for q in quant_order:
        if q not in by_quant.index:
            continue
        f1 = by_quant[q]
        if prev_f1 is not None:
            drop = prev_f1 - f1
            if drop > 0.05:
                print(f"  {model}: CLIFF at {prev_q}→{q}: {prev_f1:.3f}→{f1:.3f} (Δ={drop:.3f})")
        prev_f1 = f1
        prev_q = q
```

7. **Cloud Comparison**
```python
# At what quant level does local F1 >= cheapest cloud baseline?
cloud_f1 = df_cloud.groupby("model")["f1"].mean()
cheapest_cloud = cloud_f1.min()
cheapest_name = cloud_f1.idxmin()

print(f"\nCheapest cloud baseline: {cheapest_name} (F1={cheapest_cloud:.3f})")
for model in df["model"].unique():
    mdf = df[df["model"] == model]
    by_quant = mdf.groupby("quant")["f1"].mean()
    matching = by_quant[by_quant >= cheapest_cloud]
    if len(matching) > 0:
        worst_quant = matching.index[-1]  # most aggressive quant that still matches
        print(f"  {model}: matches at {worst_quant} (F1={by_quant[worst_quant]:.3f})")
    else:
        print(f"  {model}: never matches cloud baseline")
```

### 2.3 Export Figures

All figures saved as PDF vectors in respective `figures/` directories:

**Idea B figures** (`idea-b-schema-pruning/figures/`):
1. `f1_by_pruning.pdf` — Bar chart of F1 across 4 pruning strategies
2. `tokens_vs_f1.pdf` — Scatter of token reduction vs quality
3. `local_vs_cloud.pdf` — Horizontal bar comparing best local vs cloud models

**Idea C figures** (`idea-c-quantization/figures/`):
1. `degradation_curve.pdf` — Line chart of F1 vs bits/weight per model
2. `pareto_frontier.pdf` — Scatter with Pareto frontier annotated
3. `field_type_heatmap.pdf` — Heatmap of F1 by field type × quant level
4. `json_validity.pdf` — (optional) JSON validity rate across quant levels

---

## 3. Figure Specifications (IEEE Requirements)

| Property | Requirement |
|----------|-------------|
| Width | 3.5 inches (single column) or 7.16 inches (double column) |
| DPI | ≥300 for raster, vector PDF preferred |
| Font size | ≥8pt in figure, match body font |
| Colors | Distinguishable in grayscale (use markers too) |
| Format | PDF (vector) for LaTeX `\includegraphics` |
| Labels | Axis labels, title, legend with readable font |
| Border | No box around figure |

---

## 4. Key Analyses Checklist

### Idea B
- [ ] Main results table: pruning × format × model → F1
- [ ] Ablation: which format best per pruning strategy?
- [ ] Token reduction: % reduction vs raw HTML per strategy
- [ ] F1 improvement: schema-semantic vs no-pruning (absolute + relative)
- [ ] Cloud comparison: best local vs GPT-4o gap
- [ ] Per-domain: e-commerce vs real estate differences
- [ ] Per-site: variation across websites
- [ ] JSON validity and schema compliance rates
- [ ] Hallucination rate by strategy
- [ ] Statistical significance (if enough data points)

### Idea C
- [ ] Degradation curve: monotonic decrease confirmed?
- [ ] Pareto frontier: which model × quant is optimal?
- [ ] Cliff detection: where does quality drop sharply?
- [ ] Per-field-type: which field types degrade first?
- [ ] Cloud comparison: at what quant do we match cloud?
- [ ] Model family comparison: which model family most robust?
- [ ] JSON validity degradation: does it correlate with quant level?
- [ ] Latency improvement: how much faster at lower quant?
- [ ] Cost analysis: $/page for local vs cloud

---

## 5. Acceptance Criteria

- [ ] Idea B: ≥3 publication-quality figures exported as PDF
- [ ] Idea C: ≥3 publication-quality figures exported as PDF
- [ ] All figures readable at IEEE column width (3.5 inches)
- [ ] Key finding clearly visible in each chart
- [ ] LaTeX-ready table code generated for main results
- [ ] Statistical summary (mean ± std) for all configs
- [ ] Best configuration identified for each idea

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Results too noisy for clear trends | Medium | High | Average over more pages. Use error bars. Focus on robust trends |
| No clear cliff in degradation curve | Medium | Medium | Still interesting — report gradual decline. Focus on Pareto analysis |
| Per-field analysis needs raw predictions | Medium | Medium | Store predictions during experiments, not just CSV metrics |
| Charts too small at IEEE column width | Low | Low | Test at print size. Increase font if needed |

---

## 7. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Idea B notebook: load + tables | 20 min |
| Idea B: 3 figures | 30 min |
| Idea B: statistical analysis | 15 min |
| Idea C notebook: load + tables | 20 min |
| Idea C: 3-4 figures | 40 min |
| Idea C: cliff + Pareto analysis | 15 min |
| Export all PDFs + verify | 15 min |
| Fix formatting issues | 15 min |
| **Total** | **~2.5-3 hours** |
