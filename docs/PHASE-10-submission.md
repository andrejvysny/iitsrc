# PHASE 10: Submission

**Duration**: ~1 hour
**Schedule**: Day 3, 18:00-20:00 (target submit by 22:00, hard deadline 23:59 CET)
**Dependencies**: Phase 9 (paper complete)
**Blocks**: Nothing (final phase)

---

## 1. Objective

Generate final PDF from LaTeX, verify compliance, submit to IIT.SRC 2026 via EasyChair before February 28, 23:59 CET.

---

## 2. Tasks

### 2.1 Final PDF Generation

1. Open paper LaTeX project on Overleaf (or compile locally)
2. Compile with `pdflatex` + `bibtex` cycle:
   ```
   pdflatex main
   bibtex main
   pdflatex main
   pdflatex main
   ```
3. Verify PDF output: 6 pages, no errors, all figures/tables rendered

### 2.2 Pre-Submission Checklist

#### Content Checks
- [ ] Title is descriptive and under 100 characters
- [ ] Abstract is 150-250 words
- [ ] All sections present: Abstract, Introduction, Related Work, Method, Experiments, Conclusion, References
- [ ] All figures referenced in text (`Figure~\ref{fig:...}`)
- [ ] All tables referenced in text (`Table~\ref{tab:...}`)
- [ ] All references cited in text (`\cite{...}`)
- [ ] No uncited references in bibliography
- [ ] No broken references (check for "??" in PDF)
- [ ] Page count = exactly 6

#### Anonymization Checks
- [ ] No author names anywhere in paper
- [ ] No institution names (FIIT, STU, Bratislava) in paper body
- [ ] No self-citations that reveal identity (e.g., "as we showed in [Our2024]")
- [ ] No identifying info in figure metadata
- [ ] No GitHub/repo links that reveal identity
- [ ] File properties don't contain author name

```bash
# Automated anonymization check
grep -i "andrej\|vysny\|fiit\|stuba\|bratislava\|slovak university" main.tex
# Should return nothing

# Check PDF metadata
pdfinfo paper.pdf | grep -i "author\|creator"
# Should not contain personal info
```

#### Format Checks
- [ ] IEEE conference format (2-column layout)
- [ ] Font size correct (10pt body text)
- [ ] Margins correct (IEEE standard)
- [ ] Figures at top of columns (`[t]` placement)
- [ ] Tables use `booktabs` style (no vertical lines)
- [ ] References formatted per IEEE style
- [ ] No widows/orphans (single lines at top/bottom of column)
- [ ] Figure captions below figures
- [ ] Table captions above tables

### 2.3 EasyChair Submission

1. Navigate to IIT.SRC 2026 EasyChair submission page
2. Create account or login
3. Start new submission

**Submission form fields**:

| Field | Value |
|-------|-------|
| Title | [Paper title] |
| Abstract | [Copy from paper] |
| Keywords | 5-7 keywords (comma-separated) |
| Topics | Select relevant tracks |
| Authors | Student name (first author) |
| Study level | Note in footnote (Bc./Ing./PhD.) |
| Supervisor | Add as last co-author in form (NOT in paper) |
| PDF | Upload final PDF |

**Keywords for Paper B**:
structured data extraction, web scraping, DOM pruning, schema-aware preprocessing, small language models, HTML processing

**Keywords for Paper C**:
model quantization, structured output, web data extraction, small language models, edge deployment, GGUF

4. Upload PDF
5. Verify upload (download and check PDF opens correctly)
6. Submit

### 2.4 Post-Submission Verification

- [ ] Received confirmation email from EasyChair
- [ ] Submission status shows "Submitted" on EasyChair
- [ ] Downloaded copy of submitted PDF matches local copy
- [ ] Screenshot of submission confirmation saved

---

## 3. IIT.SRC Submission Requirements

Based on IIT.SRC guidelines (verify at iitsrc.sk):

| Requirement | Details |
|-------------|---------|
| System | EasyChair |
| Deadline | February 28, 2026, 23:59 CET |
| Format | IEEE conference (6 pages) |
| Language | English |
| Review type | Double-blind |
| File format | PDF |
| Max file size | Usually 10 MB |
| Author footnote | Student study level (e.g., "1st year Ing.") |
| Supervisor | Listed in submission form, NOT in paper |

---

## 4. Emergency Procedures

### 4.1 If Paper Not Finished

- Submit incomplete paper (better than no submission)
- Focus on: Abstract + Introduction + Method + preliminary results
- Mark experiments as "preliminary" with available data

### 4.2 If EasyChair Down

- Try again in 30 min
- Email conference organizers with PDF attachment
- Check IIT.SRC website for alternative submission methods

### 4.3 If PDF Compilation Fails

- Fix LaTeX errors (check log file)
- Common issues: missing packages, broken figure paths, encoding
- Fallback: use Overleaf (handles dependencies automatically)
- Last resort: export from Overleaf even with warnings

### 4.4 If Figures Don't Render

- Replace with placeholder figures
- Add "(figure to be updated)" caption
- Still submit — content matters more than perfect figures

---

## 5. Acceptance Criteria

- [ ] Final PDF generated without compilation errors
- [ ] PDF is exactly 6 pages
- [ ] All anonymization checks pass
- [ ] PDF uploaded to EasyChair successfully
- [ ] Confirmation email received
- [ ] Submission status = "Submitted" on EasyChair
- [ ] Submitted before 22:00 CET (2 hours buffer before hard deadline)

---

## 6. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Final compilation + format check | 15 min |
| Anonymization verification | 10 min |
| EasyChair account setup / login | 10 min |
| Fill submission form | 10 min |
| Upload + verify PDF | 5 min |
| Post-submission verification | 5 min |
| **Total** | **~55 min** |

**Start by**: 20:00 CET at the latest (gives 4 hours before deadline)
**Target submit**: 22:00 CET (2-hour buffer)
**Hard deadline**: 23:59 CET, February 28, 2026
