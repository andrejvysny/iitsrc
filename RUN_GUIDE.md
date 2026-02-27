# Run Guide

## Prerequisites

```bash
source .venv/bin/activate
# Env vars needed: OPENAI_API_KEY, OPENROUTER_API_KEY
```

## Monitor Running Experiments

```bash
# Check processes
ps aux | grep "run_experiments\|run_baselines" | grep -v grep

# Idea C progress
wc -l idea-c-quantization/results/experiments.csv
tail -1 idea-c.log

# Baselines progress
wc -l shared/results/cloud_baselines.csv
tail -1 baselines.log

# Idea B progress (when running)
wc -l idea-b-schema-pruning/results/experiments.csv
tail -1 idea-b.log
```

## Start/Restart Experiments

All runners have **checkpoint/resume** — safe to kill and restart.

### Idea C (quantization, uses Metal GPU)

```bash
# Full run (500 pages × 25 model variants = 12,500 inferences)
nohup python idea-c-quantization/src/run_experiments.py --limit 500 > idea-c.log 2>&1 &

# Reduced scope (faster, still publishable)
nohup python idea-c-quantization/src/run_experiments.py --limit 100 > idea-c.log 2>&1 &

# Specific models/quants only
nohup python idea-c-quantization/src/run_experiments.py --limit 500 --models qwen2.5-3b llama-3.2-3b > idea-c.log 2>&1 &
nohup python idea-c-quantization/src/run_experiments.py --limit 500 --quants q4_k_m q8_0 f16 > idea-c.log 2>&1 &
```

### Idea B (schema pruning, uses Metal GPU)

**Cannot run simultaneously with Idea C** — both use Metal GPU.

```bash
# Kill Idea C first
kill <idea-c-pid>

# Full run (500 pages × 4 strategies × 3 formats × 3 models = ~18,000 inferences)
nohup python idea-b-schema-pruning/src/run_experiments.py --limit 500 > idea-b.log 2>&1 &

# Reduced scope
nohup python idea-b-schema-pruning/src/run_experiments.py --limit 100 > idea-b.log 2>&1 &
```

### Cloud Baselines (API calls, no GPU)

Can run in parallel with local experiments.

```bash
# Full run (4 models × 500 pages = 2,000 calls, ~$7)
nohup python shared/run_baselines.py --models gpt-4o qwen-72b llama-70b mistral-large --limit 500 > baselines.log 2>&1 &

# Single model test
python shared/run_baselines.py --models gpt-4o --limit 5
```

Available models: `gpt-4o`, `qwen-72b`, `llama-70b`, `mistral-large`

## Kill Experiments

```bash
# Kill specific process
kill <pid>

# Kill all experiments
pkill -f "run_experiments.py"
pkill -f "run_baselines.py"
```

## Scope Reduction Strategy (if tight on time)

Priority order for minimum viable paper:

1. **Idea C**: `--limit 100 --models qwen2.5-3b llama-3.2-3b phi-3.5-mini` (skip mistral-7b, 100 pages)
   → 100 × 20 = 2,000 inferences (~4h)
2. **Idea B**: `--limit 100` (100 pages only)
   → 100 × 36 = 3,600 inferences (~8h)
3. **Baselines**: `--models gpt-4o qwen-72b --limit 100` (2 models, 100 pages)
   → 200 calls (~10min)

## Results Locations

| Experiment | CSV Path                                        | Expected Rows  |
| ---------- | ----------------------------------------------- | -------------- |
| Idea C     | `idea-c-quantization/results/experiments.csv`   | ~12,500 (full) |
| Idea B     | `idea-b-schema-pruning/results/experiments.csv` | ~18,000 (full) |
| Baselines  | `shared/results/cloud_baselines.csv`            | ~2,000 (full)  |

## Next Steps After Experiments

```bash
# Analysis notebooks (Phase 8)
jupyter notebook idea-c-quantization/notebooks/analysis.ipynb
jupyter notebook idea-b-schema-pruning/notebooks/analysis.ipynb

# Compile papers (Phase 9)
cd paper-b && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd paper-c && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
