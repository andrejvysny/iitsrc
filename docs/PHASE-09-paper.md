# PHASE 9: Paper Writing

**Duration**: ~6-8 hours
**Schedule**: Day 3, full day (08:00-20:00)
**Dependencies**: Phase 8 (all analysis, figures, tables ready)
**Blocks**: Phase 10 (submission)

---

## 1. Objective

Write complete 6-page IEEE format paper(s) for IIT.SRC 2026. Generate both Paper B (schema pruning) and Paper C (quantization), supervisor selects one for submission. Full LaTeX source ready for Overleaf compilation.

---

## 2. Paper B: Schema-Conditioned DOM Pruning for Small LLM Web Extraction

### 2.1 Proposed Title

**"Schema-Conditioned DOM Pruning Enables Sub-3B Language Models for Structured Web Data Extraction"**

Alternative titles:
- "Schema-Aware HTML Preprocessing for Efficient Structured Data Extraction with Small Language Models"
- "Reducing Input Complexity for Small LLM Web Extraction via Schema-Conditioned DOM Pruning"

### 2.2 Section-by-Section Outline

#### Abstract (150-250 words)

Structure: Problem → Gap → Method → Key Result → Takeaway

```
Structured data extraction from web pages increasingly relies on large language models,
but frontier models (GPT-4o, Claude) incur significant cost and latency.
Small local models (<3B parameters) offer a cheaper alternative but struggle with
lengthy, noisy HTML input. We propose schema-conditioned DOM pruning, a preprocessing
strategy that selectively retains HTML elements semantically relevant to the target
extraction schema. Unlike prior generic pruning approaches, our method adapts the
pruning to what needs to be extracted. We evaluate four pruning strategies—none,
generic, schema-aware heuristic, and schema-aware semantic—across three output formats
and three sub-3B instruction-tuned models on [N] web pages spanning e-commerce and
real estate domains. Schema-semantic pruning achieves [X]% token reduction while
improving macro F1 from [Y] (no pruning) to [Z], reaching [W]% of GPT-4o quality
at [fraction]x the cost. Our results demonstrate that task-aware preprocessing
is critical for deploying small language models in structured web extraction pipelines.
```

#### 1. Introduction (0.75 page)

**Paragraph 1: Motivation** (4-5 sentences)
- Web contains vast structured data (products, listings, events)
- Traditional extraction: CSS selectors, XPath → brittle, site-specific
- LLM-based extraction: flexible, zero-shot, adapts to any schema
- BUT: GPT-4o costs ~$6/1K pages, requires internet, privacy concerns
- Small local models could solve this — but raw HTML overwhelms their limited context

**Paragraph 2: Gap** (3-4 sentences)
- Prior work: AXE (2025) shows generic DOM pruning helps small models
- But pruning is schema-agnostic — removes same elements regardless of extraction target
- No one has studied whether conditioning pruning on the target schema improves extraction
- Key insight: if extracting price+availability, keep product info nodes, drop reviews/related items

**Paragraph 3: Contributions** (bulleted list)
1. We propose schema-conditioned DOM pruning, first preprocessing approach that adapts HTML simplification to the target extraction schema
2. We implement and ablate four pruning strategies across three output formats and three sub-3B models
3. We show schema-semantic pruning achieves [X]% token reduction and [Y]% F1 improvement over no pruning, reaching [Z]% of GPT-4o quality
4. We release our evaluation dataset of [N] annotated pages across e-commerce and real estate domains

**Paragraph 4: Teaser result** (2 sentences)
- Best config: [model] with schema-semantic pruning + [format] achieves F1=[Z] using only [N] input tokens
- This matches [X]% of GPT-4o performance at [Y]× lower cost

#### 2. Related Work (0.75 page)

**2.1 LLM-based Web Extraction** (1 paragraph)
- AutoScraper (EMNLP 2024): GPT-4-Turbo zero-shot on SWDE, F1=88.69%
- AXE (2025): DOM pruning + 0.6B model, F1=88.1% on SWDE
- ScrapeGraphAI-100k (2025): fine-tuned Qwen3-1.7B competitive with 30B
- SLOT (EMNLP 2025): constrained decoding with Mistral-7B, 99.5% schema accuracy (fine-tuned)
- Position our work: zero-shot small models with preprocessing optimization

**2.2 HTML Preprocessing for LLMs** (1 paragraph)
- HtmlRAG (WWW 2025): HTML preserves more info than plain text for LLMs
- NEXT-EVAL (2025): flat JSON preprocessing yields F1=0.9567
- trafilatura: popular text extractor, but loses structural info
- markdownify: HTML→Markdown, preserves some structure
- Our contribution: preprocessing conditioned on extraction target

**2.3 Structured Output from LLMs** (1 paragraph)
- JSONSchemaBench (2025): benchmark for constrained decoding
- GoLLIE (ICLR 2024): code-based schema for IE
- llama.cpp grammar: GBNF grammars for constrained generation
- Our approach: prompt-based JSON with schema in prompt

#### 3. Method (1.5 pages)

**3.1 Pipeline Overview** (with Figure 1: architecture diagram)
```
HTML → Pruning(strategy, schema) → Format Conversion → LLM(prompt, schema) → JSON
```
- Input: raw HTML page + target JSON schema
- Output: structured JSON matching schema

**3.2 Pruning Strategies** (4 subsections with formulas)

*Strategy 1: No Pruning* — baseline, raw HTML
*Strategy 2: Generic Pruning* — remove `<script>`, `<style>`, `<nav>`, `<footer>`, ads, tracking attrs
*Strategy 3: Schema-Aware Heuristic*:
```
score(node) = |tokens(text(node)) ∩ K_schema| / |K_schema|
```
where K_schema is the set of keywords associated with the target schema fields.
Keep nodes where score > τ (tuned on validation set).

*Strategy 4: Schema-Aware Semantic*:
- Embed each DOM node text with all-MiniLM-L6-v2 (22M params)
- Embed each schema field name+description
- sim(node, field) = cosine(emb(node), emb(field))
- Keep nodes where max_field sim(node, field) > τ

**3.3 Output Formats** — Simplified HTML, Markdown (markdownify), Flat JSON (NEXT-EVAL style)

**3.4 Extraction Prompt** — System prompt + schema + content → JSON output

#### 4. Experiments (1.5 pages)

**4.1 Setup**
- Dataset: [N] pages (50 e-commerce from Amazon/eBay/Alza/Mall, 50 real estate from Zillow/Realtor/Nehnutelnosti/Reality)
- Annotation: 3-model consensus (Claude Haiku + GPT-4o-mini + Gemini Flash)
- Models: Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini (all Q4_K_M GGUF, Metal GPU)
- Baselines: GPT-4o, Claude Sonnet, Qwen-72B, Llama-70B, Mistral-Large
- Metrics: Macro F1 (primary), JSON validity, schema compliance, hallucination rate

**4.2 Main Results** (Table 1)
- Rows: pruning × format × model
- Columns: F1, JSON%, Tokens, Latency

**4.3 Ablation Analysis**
- Effect of pruning (Figure 2: bar chart)
- Effect of format (table or text)
- Token reduction vs F1 tradeoff (Figure 3: scatter)

**4.4 Cloud Comparison**
- Best local vs 5 cloud baselines (table or figure)
- Cost analysis: $/page for local vs cloud

**4.5 Per-Domain Analysis**
- E-commerce vs real estate performance differences
- English vs Slovak content

#### 5. Discussion & Conclusion (0.75 page)

**Key Findings** (3-4 sentences):
1. Schema-semantic pruning is the best strategy, achieving [X]% of cloud quality at [Y]% token reduction
2. Output format matters: [format] consistently outperforms alternatives
3. Among 3B models, [model] achieves highest F1
4. Even with optimal preprocessing, gap to GPT-4o remains [Z] F1 points

**Limitations** (2-3 bullets):
- Zero-shot only; fine-tuning may close the gap further
- Single-page extraction; multi-page/list-detail scenarios not studied
- Limited to 2 domains; generalization needs broader evaluation

**Future Work** (2-3 sentences):
- Learned neural pruner (replace heuristic with trained classifier)
- QLoRA fine-tuning on domain-specific extraction tasks
- Constrained decoding (GBNF grammar) for guaranteed JSON validity

#### References (0.5 page, 15-20 entries)

---

## 3. Paper C: Quantization Impact on Structured Data Extraction Quality

### 3.1 Proposed Title

**"How Low Can You Go? Quantization Effects on Small LLM Structured Web Data Extraction"**

Alternative titles:
- "Measuring Quantization Impact on Schema-Compliant Web Data Extraction with Small Language Models"
- "Quantization Trade-offs for Structured Data Extraction: A Systematic Study with Sub-7B Models"

### 3.2 Section-by-Section Outline

#### Abstract (150-250 words)

```
Deploying language models for structured data extraction on edge devices requires
aggressive model compression. While quantization (reducing weight precision from
16-bit to 2-bit) dramatically reduces memory and improves throughput, its impact
on structured output quality—JSON validity, schema compliance, and field-level
accuracy—remains unstudied for web extraction tasks. We present the first systematic
evaluation of quantization effects on structured data extraction, testing [N] model
variants (4 model families × up to 7 quantization levels from F16 to Q2_K) on
[M] web pages across e-commerce and real estate domains. We find that Q4_K_M
(4-bit) represents the optimal tradeoff, retaining [X]% of full-precision F1 at
[Y]% memory reduction. Below Q3_K_M, we observe a quality cliff where JSON validity
drops [Z]% and numeric field accuracy degrades [W]× faster than text fields.
Our Pareto analysis shows that [model] at Q4_K_M achieves the best quality-per-GB,
matching [cloud model] baseline quality at [fraction]× the memory footprint.
```

#### Sections follow similar structure to Paper B, adapted for quantization focus.

**Key differences**:
- Method section: focus on quantization levels, model variants, measurement protocol
- Experiments: degradation curves, Pareto frontier, cliff analysis, per-field-type breakdown
- Figures: degradation curve (primary), Pareto scatter, field-type heatmap

---

## 4. LaTeX Structure

### 4.1 Document Setup

```latex
\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{cleveref}

\begin{document}

\title{Schema-Conditioned DOM Pruning Enables Sub-3B Language Models for Structured Web Data Extraction}

% Double-blind: NO author names
% \author{} % left empty for blind review

\maketitle

\begin{abstract}
...
\end{abstract}

\begin{IEEEkeywords}
structured data extraction, web scraping, language models, DOM pruning, schema-aware preprocessing
\end{IEEEkeywords}

\section{Introduction}
...

\section{Related Work}
...

\section{Method}
...

\section{Experiments}
...

\section{Discussion and Conclusion}
...

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
```

### 4.2 Figure Inclusion

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/f1_by_pruning.pdf}
    \caption{Macro F1 scores across four pruning strategies, averaged over all models and output formats. Schema-semantic pruning achieves the highest quality with 95\% token reduction.}
    \label{fig:pruning_comparison}
\end{figure}
```

### 4.3 Table Format

```latex
\begin{table}[t]
    \centering
    \caption{Main Results: Macro F1 by Pruning Strategy, Output Format, and Model}
    \label{tab:main_results}
    \begin{tabular}{llccc}
        \toprule
        Pruning & Format & Qwen-3B & Llama-3B & Phi-3.5 \\
        \midrule
        None & HTML & 0.XX & 0.XX & 0.XX \\
        None & MD & 0.XX & 0.XX & 0.XX \\
        ...
        \bottomrule
    \end{tabular}
\end{table}
```

### 4.4 BibTeX References

```bibtex
@inproceedings{hu2022lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
    booktitle={ICLR},
    year={2022}
}

@article{axe2025,
    title={AXE: Automated Cross-domain Entity Extraction from HTML},
    author={...},
    journal={arXiv preprint arXiv:2602.01838},
    year={2025}
}

@inproceedings{autoscraper2024,
    title={AutoScraper: A Progressive Understanding Web Agent for Web Scraper Generation},
    author={...},
    booktitle={EMNLP},
    year={2024}
}

@inproceedings{slot2025,
    title={SLOT: Structured Language Output from LLMs for Web Information Extraction},
    author={...},
    booktitle={EMNLP Industry},
    year={2025},
    note={arXiv:2505.04016}
}

@article{scrapegraphai2025,
    title={ScrapeGraphAI: A Graph-Based Framework for Web Scraping Using Large Language Models},
    author={...},
    journal={arXiv preprint arXiv:2602.15189},
    year={2025}
}

@inproceedings{htmlrag2025,
    title={HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems},
    author={...},
    booktitle={WWW},
    year={2025},
    note={arXiv:2411.02959}
}

@article{nexteval2025,
    title={NEXT-EVAL: A Benchmark for Evaluating LLM-based Web Data Extraction},
    author={...},
    journal={arXiv preprint arXiv:2505.17125},
    year={2025}
}

@article{jsonschemabench2025,
    title={JSONSchemaBench: A Benchmark for Structured Output Generation},
    author={...},
    journal={arXiv preprint arXiv:2501.10868},
    year={2025}
}

@inproceedings{dettmers2023qlora,
    title={QLoRA: Efficient Finetuning of Quantized Language Models},
    author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
    booktitle={NeurIPS},
    year={2023}
}

@inproceedings{domlm2022,
    title={DOM-LM: Learning Generalizable Representations for HTML Documents},
    author={Deng, Xiang and others},
    booktitle={EMNLP Findings},
    year={2022},
    note={arXiv:2201.10608}
}

@inproceedings{gollie2024,
    title={GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction},
    author={...},
    booktitle={ICLR},
    year={2024}
}
```

**Note**: Fill in actual author names from arXiv pages before submission.

---

## 5. Writing Process

### 5.1 Order of Writing

1. **Figures & Tables first** — finalize all visual elements from Phase 8
2. **Method section** — describe what we did (most straightforward)
3. **Experiments section** — present results with reference to figures/tables
4. **Related Work** — position against prior art
5. **Introduction** — frame the problem, state contributions
6. **Abstract** — summarize the whole paper
7. **Conclusion** — discussion, limitations, future work
8. **References** — compile BibTeX, verify all cited

### 5.2 Writing Guidelines

- **Concise**: Every sentence earns its place. Cut filler.
- **Quantitative**: "improves F1 by 23%" not "significantly improves"
- **Honest**: Report failures and limitations
- **Double-blind**: No author/institution names anywhere in text, figures, or metadata
- **Active voice**: "We propose" not "It is proposed"
- **Present tense for claims**: "Schema pruning reduces tokens" not "reduced"
- **Past tense for experiments**: "We evaluated on 100 pages"

### 5.3 Page Budget

Total: 6 pages including references.

| Section | Target | Min | Max |
|---------|--------|-----|-----|
| Abstract | 0.25 | 0.20 | 0.30 |
| Introduction | 0.75 | 0.60 | 0.90 |
| Related Work | 0.75 | 0.60 | 0.85 |
| Method | 1.50 | 1.20 | 1.70 |
| Experiments | 1.50 | 1.30 | 1.80 |
| Discussion+Conclusion | 0.75 | 0.50 | 0.90 |
| References | 0.50 | 0.40 | 0.60 |

---

## 6. IIT.SRC Specific Requirements

| Requirement | Value |
|-------------|-------|
| Format | IEEE conference (2-column) |
| Pages | Exactly 6 |
| Language | English |
| Review | Double-blind |
| Template | IEEEtran `\documentclass[conference]{IEEEtran}` |
| Author info | Student name + study level (footnote), NOT in paper body |
| Supervisor | Listed in submission form only, NOT in paper |
| Keywords | 5-7 keywords |

### IIT.SRC Keywords

**Paper B**: structured data extraction, web scraping, DOM pruning, schema-aware preprocessing, small language models, HTML processing, zero-shot extraction

**Paper C**: model quantization, structured output, web data extraction, small language models, quality-efficiency tradeoff, GGUF, edge deployment

---

## 7. Acceptance Criteria

- [ ] Paper is exactly 6 pages in IEEE two-column format
- [ ] Double-blind: no author names, no identifiable references
- [ ] All figures render correctly in LaTeX (PDF included)
- [ ] All tables formatted with `booktabs` (no vertical lines)
- [ ] References complete and formatted (BibTeX + IEEEtran.bst)
- [ ] Abstract ≤ 250 words
- [ ] No grammatical errors
- [ ] All claims backed by experimental data
- [ ] Limitations section present
- [ ] No placeholder text (TODO, FIXME, XXX, [citation needed])
- [ ] Compiles without errors on Overleaf

---

## 8. Verification Checklist

```bash
# Before submission:
grep -i "todo\|fixme\|xxx\|citation needed" main.tex      # should return nothing
grep -i "author\|vysny\|bratislava\|fiit" main.tex          # anonymization check
wc -w main.tex                                              # rough word count
pdfinfo paper.pdf | grep Pages                              # should be 6
```

---

## 9. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Finalize figures + tables from Phase 8 | 30 min |
| Write Method section | 60 min |
| Write Experiments section | 60 min |
| Write Related Work | 45 min |
| Write Introduction | 45 min |
| Write Abstract | 20 min |
| Write Conclusion | 30 min |
| Compile references.bib | 30 min |
| LaTeX formatting + figure placement | 45 min |
| Proofread + polish | 30 min |
| Compile on Overleaf + page adjustment | 30 min |
| **Total** | **~7-8 hours** |

**Strategy**: Claude generates full draft, user reviews and edits. Multiple compile-check cycles on Overleaf.
