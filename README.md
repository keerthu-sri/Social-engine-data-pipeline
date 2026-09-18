# Social Engine — Data Vortex A'26

## Round 1 — Phase 1 Submission
Data recovery pipeline for the corrupted Social Engine dataset.

### Contents
- `data_cleaning_and_eda.ipynb` — reproducible notebook: cleaning code, inline justification for every transformation, and exploratory data analysis. Run this top to bottom to regenerate everything from the raw files.
- `Social_Engine_Users.csv`, `Social_Engine_Posts_Corrupted.csv` — raw source files, recovered from the recovery-terminal challenge site.
- `Social_Engine_Users_Cleaned.csv`, `Social_Engine_Posts_Cleaned.csv` — cleaned output.
- `EDA_Report.md` — written EDA report with charts.

### How to reproduce
```
pip install pandas numpy matplotlib jupyter nbconvert
jupyter nbconvert --to notebook --execute --inplace data_cleaning_and_eda.ipynb
```

### Cleaning summary

| Issue | Rows affected | Action taken |
|---|---|---|
| Exact duplicate rows | 360 | Dropped (kept first occurrence) |
| Mixed timestamp formats (epoch / ISO / DD-MM-YYYY) | all rows | Normalized to ISO 8601 |
| Undecoded HTML entities in text | 341 | Decoded with `html.unescape` |
| Missing `platform` | 1,846 | Imputed via per-user mode where possible; else `"Unknown"` |
| Missing `text_content` | 1,746 | Flagged, not fabricated (`text_missing=True`) |
| Missing `likes` (literal `"NULL"`) | 1,858 | Kept as true missing, flagged (`likes_missing`) |
| Negative `likes` | 525 | Flagged as anomalous (`likes_anomalous`), excluded from `likes_valid` aggregate column, original value preserved |

All counts above are pre-deduplication; see the notebook for exact post-dedup figures.

### Key assumptions
- Users file required no structural cleaning — verified no nulls, no duplicate IDs, valid follower counts and dates.
- "Singapore" and "São Paulo, Brazil" location formats are legitimate, not corruption.
- Per-user platform mode imputation is a statistical assumption, explicitly flagged per-row (`platform_imputed`) for full transparency — not treated as ground truth.
- Free-text `text_content` is never fabricated. Missing text stays missing and flagged.
- Negative likes are data-quality anomalies (a "like" cannot be negative) but the true intended value is unknown, so rows are flagged rather than deleted or sign-corrected — a reversible decision documented for the evaluators.

---

## Round 2 — Rebuilding the Social Engine's Semantic Understanding Layer

Two independent supervised text-classification tasks trained on **Dataset 2** (9,000 labelled social-media posts), sharing one preprocessing and feature-engineering pipeline:
- **`sentiment_label`** — Positive / Negative / Neutral (perfectly balanced, 3,000 each)
- **`topic_category`** — Community_Discussion / Technical_Issues / Feature_Feedback / Account_Security (heavily imbalanced: 86.2% / 9.1% / 3.3% / 1.5%)

This report reflects the tuned pipeline (**v3**) versus the original submission.

### Contents
- `Round2_Technical_Report.pdf` — full pipeline writeup: preprocessing, feature engineering, model selection, tuning history, ablation study, error analysis, conclusion.
- `Round2_Evaluation_Metrics_Report.pdf` — evaluation-focused writeup: CV comparison, per-class metrics, confusion matrices, class distribution, sample misclassifications.

### Data preparation
- De-duplication: 1,100 of 9,000 rows were exact-duplicate texts with consistent labels; dropped before the train/test split to prevent data leakage, leaving 7,900 unique posts.
- 80/20 stratified split per task (`random_state=42`), performed **before** any vectorizer is fit: 6,320 train / 1,580 test posts.
- Unicode-escape repair, URL/@mention masking (`<URL>`/`<USER>`), hashtag unwrapping, elongation normalization (repeats capped at 2), light symbol stripping (keeps `! ? . , '`), lowercasing — intentionally lighter than a classic NLP pipeline since TF-IDF n-grams already down-weight uninformative words, and social text relies on exact phrasing/punctuation.

### Feature engineering
Three feature blocks concatenated into one sparse matrix per task:
1. Word n-grams (1–2), max 40,000 features
2. Character n-grams (3–5), max 30,000 features — robust to misspellings/slang
3. 10 engineered meta/lexicon features — VADER neg/neu/pos/compound, char length, word count, exclamation count, question-mark count, ALL-CAPS ratio, elongated-word count

Both TF-IDF vectorizers use `sublinear_tf=True` and `min_df=2`; meta features are Min-Max scaled to [0,1] so Complement Naive Bayes (needs non-negative input) can run on the combined matrix.

### Model selection
Four candidate classifiers compared via 5-fold stratified cross-validation on macro-F1 (chosen over accuracy since `topic_category` is imbalanced), plus a soft-voting ensemble of the top-3:
- Logistic Regression (L2)
- Linear SVM (`LinearSVC`, wrapped in `CalibratedClassifierCV`)
- SGD with modified-Huber loss (`alpha=5e-6`)
- Complement Naive Bayes

For `topic_category`, all classifiers use `class_weight="balanced"`.

**Final model for both tasks: LinearSVC** (the voting ensemble did not beat it on CV for either task).

| Task | CV Macro-F1 (LogReg) | CV Macro-F1 (LinearSVC) | CV Macro-F1 (SGD-SVM) | CV Macro-F1 (ComplementNB) | CV Macro-F1 (Voting Ensemble) |
|---|---|---|---|---|---|
| Sentiment | 0.637 ± 0.014 | 0.644 ± 0.011 | 0.610 ± 0.010 | 0.601 ± 0.012 | 0.627 ± 0.010 |
| Topic | 0.731 ± 0.027 | 0.799 ± 0.010 | 0.725 ± 0.044 | 0.257 ± 0.016 | 0.756 ± 0.019 |

### Tuning history (v1 → v3)
- **v1 → v2**: denser/more informative meta features → less regularization helped (LR `C: 5→8`, LinearSVC `C: 0.5→0.7`).
- **v2 → v3**: systematic grid search (LinearSVC `C: 0.2–1.0`, LogReg `C: 3–20`, 5-fold CV macro-F1) found LinearSVC `C=0.3` and LogReg `C=3` generalize slightly better for sentiment. Topic's v2 values (LinearSVC `C=0.7`, LR `C=8`) were kept.
- Sentiment test accuracy progression: 61.6% (v1) → 64.5% → **64.6% (v3)**; macro-F1: 0.617 → 0.646 → **0.647**.
- Topic classification stayed essentially flat (already near-ceiling for this feature family).

### Ablation study (sentiment task, v2 → v3)
Tried and rejected — none beat the tuned baseline (0.6441) by more than run-to-run noise (~±0.003):

| Technique | CV Macro-F1 | Verdict |
|---|---|---|
| Tuned baseline (TF-IDF + lexicon feats, C=0.3) | 0.6441 | kept — final pipeline |
| + NLTK scope-based negation tagging | 0.6407 | rejected — no gain |
| + TruncatedSVD (150 latent components) | 0.6420 | rejected — no gain |
| Stacking ensemble (LR meta-learner) vs. soft voting | 0.6409 | rejected — no gain |
| Wider n-grams (word 1-3 / char 2-6) | 0.6436 | rejected — no gain |
| Random oversampling of rare topic classes | inflated via CV leakage | rejected — hurt true test score |

Interpretation: remaining sentiment errors (sarcasm, terse/ambiguous text, reporting-register mismatches) need contextual language understanding beyond a classical TF-IDF pipeline.

### Held-out test results

| Task | Final Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Prec. | Macro-Recall |
|---|---|---|---|---|---|---|
| Sentiment (3-class) | LinearSVC | 64.6% | 0.647 | 0.646 | 0.647 | 0.649 |
| Topic (4-class) | LinearSVC | 96.8% | 0.831 | 0.965 | 0.956 | 0.754 |

Topic accuracy looks high mainly because 86% of posts are `Community_Discussion` — a majority-class baseline would already score ~86%. Macro-F1 (0.831) is the fairer headline number.

#### Per-class — Sentiment

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.66 | 0.70 | 0.68 | 486 |
| Neutral | 0.58 | 0.57 | 0.58 | 550 |
| Positive | 0.70 | 0.67 | 0.68 | 544 |

#### Per-class — Topic

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Account_Security | 1.00 | 0.62 | 0.76 | 26 |
| Community_Discussion | 0.97 | 1.00 | 0.98 | 1361 |
| Feature_Feedback | 0.86 | 0.50 | 0.63 | 50 |
| Technical_Issues | 0.99 | 0.90 | 0.95 | 143 |

### Error analysis
- **Sentiment** (559/1580 misclassified, 35.4%): news/factual headlines with negative real-world content but neutral reporting register still get predicted Negative; short/terse posts with little explicit sentiment language cause Neutral↔Negative/Positive confusion; sarcasm and mixed-sentiment posts are the hardest residual category — VADER/meta features catch overt polarity but not sarcasm.
- **Topic** (50/1580 misclassified, 3.2%): nearly all errors are minority classes (`Account_Security`, `Feature_Feedback`, `Technical_Issues`) pulled into the dominant `Community_Discussion` class — the classic imbalanced-classification failure mode. `Feature_Feedback` has the lowest recall (0.50): opinions about a feature often read lexically identical to ordinary discussion posts, and it has only 200 training examples. This is treated as a data-scarcity issue rather than a modelling weakness.

### Why not a Transformer?
No access to pretrained-weight hosts in this environment, so fine-tuning wasn't possible, and training a transformer from scratch on ~7,900 short posts would underfit versus a well-tuned classical pipeline. A carefully engineered TF-IDF + lexicon-feature + linear-model approach is the right fit for this dataset size and environment.

### Conclusion & future work
- Both models, vectorizers, and full evaluation metrics are saved to `models/` and `outputs/` for reproducibility and submission.
- Sentiment is near the practical ceiling for a classical TF-IDF pipeline at this dataset size (confirmed by the ablation study).
- Topic classification's honest next lever is more labelled data for the rare classes, not a fancier model.
- Given competition constraints (no GPU/internet for pretrained transformers), the natural next step is fine-tuning a pretrained encoder (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`) to better capture negation, context, and sarcasm.
