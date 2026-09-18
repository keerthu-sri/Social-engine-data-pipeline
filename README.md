# Social Engine — Pipeline & Model Submission (Round 1 & Round 2)

Comprehensive data recovery, cleaning, exploratory data analysis (EDA), and semantic understanding/classification pipeline for the Social Engine dataset.

---

## Repository Contents

* **`data_cleaning_and_eda.ipynb`**: Reproducible notebook containing cleaning code, inline justification for every transformation, and EDA. Run top-to-bottom to regenerate artifacts from raw files.
* **`Social_Engine_Users.csv` & `Social_Engine_Posts_Corrupted.csv**`: Raw source files recovered from the recovery-terminal challenge site.
* **`Social_Engine_Users_Cleaned.csv` & `Social_Engine_Posts_Cleaned.csv**`: Cleaned output tables.
* **`EDA_Report.md`**: Written EDA report complete with charts.
* **`models/` & `outputs/**`: Saved vectorizers, trained models, and evaluation artifacts for the semantic classification layer.

---

## How to Reproduce

```bash
pip install pandas numpy matplotlib jupyter nbconvert scikit-learn
jupyter nbconvert --to notebook --execute --inplace data_cleaning_and_eda.ipynb

```

---

## Cleaning Summary

| Issue | Rows Affected | Action Taken |
| --- | --- | --- |
| **Exact duplicate rows** | 360 | Dropped (kept first occurrence) |
| **Mixed timestamp formats** (epoch / ISO / DD-MM-YYYY) | All rows | Normalized to ISO 8601 |
| **Undecoded HTML entities** in text | 341 | Decoded with `html.unescape` |
| **Missing platform** | 1,846 | Imputed via per-user mode where possible; else `"Unknown"` |
| **Missing `text_content**` | 1,746 | Flagged, not fabricated (`text_missing=True`) |
| **Missing likes** (literal `"NULL"`) | 1,858 | Kept as true missing, flagged (`likes_missing`) |
| **Negative likes** | 525 | Flagged as anomalous (`likes_anomalous`), excluded from `likes_valid` aggregate column, original value preserved |

*(All counts above are pre-deduplication; see the notebook for exact post-dedup figures).*

---

## Key Assumptions

* Users file required no structural cleaning — verified no nulls, no duplicate IDs, valid follower counts and dates.
* "Singapore" and "São Paulo, Brazil" location formats are legitimate, not corruption.
* Per-user platform mode imputation is a statistical assumption, explicitly flagged per-row (`platform_imputed`) for full transparency — not treated as ground truth.
* Free-text `text_content` is never fabricated. Missing text stays missing and flagged.
* Negative likes are data-quality anomalies (a "like" cannot be negative) but the true intended value is unknown, so rows are flagged rather than deleted or sign-corrected — a reversible decision documented for the evaluators.

---

## Round 2: Semantic Understanding Layer (Classification Pipeline)

The Round 2 pipeline builds upon the cleaned dataset to solve two independent supervised text-classification tasks sharing a unified preprocessing and feature-engineering pipeline.

### 1. Dataset Split & Preprocessing

* **De-duplication:** 1,100 of 9,000 rows are exact duplicate texts with consistent labels; dropped before the train/test split to prevent data leakage, leaving **7,900 unique posts** (6,320 train / 1,580 test).


* **Pipeline Steps:** Unicode-escape repair, URL/mention masking (`<URL>` / `<USER>`), hashtag unwrapping, elongation normalization (repeats capped at 2), symbol stripping (retaining `!?.,'`), lowercasing, and whitespace normalization.



### 2. Feature Engineering

Three feature blocks are concatenated into a single sparse matrix per task:

* **Word n-grams (1-2):** Max 40,000 features (`sublinear_tf=True`, `min_df=2`).


* **Character n-grams (3-5):** Max 30,000 features — robust to misspellings, slang, and unusual tokenization.


* **10 Engineered Meta/Lexicon Features:** VADER sentiment lexicon scores (`neg`/`neu`/`pos`/`compound`), character length, word count, exclamation count, question-mark count, ALL-CAPS word ratio, and elongated-word count (Min-Max scaled to $[0, 1]$).



### 3. Model Evaluation & Performance Summary

| Task | Final Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Prec. | Macro-Recall |
| --- | --- | --- | --- | --- | --- | --- |
| **Sentiment (3-class)**<br> | LinearSVC

 | 64.6%

 | 0.647

 | 0.646

 | 0.647

 | 0.649

 |
| **Topic (4-class)**<br> | LinearSVC

 | 96.8%

 | 0.831

 | 0.965

 | 0.956

 | 0.754

 |

* **Sentiment Insights:** Perfectly balanced 3-way split, but lexically subtle due to sarcasm, mixed emotion, and terse text. LinearSVC ($C=0.3$) won via systematic grid search.


* **Topic Insights:** Heavily imbalanced (86% `Community_Discussion`). Accuracy is high due to majority class dominance; Macro-F1 ($0.831$) correctly indicates the model is learning minority classes (`Account_Security`, `Feature_Feedback`, `Technical_Issues`) using `class_weight="balanced"`.
