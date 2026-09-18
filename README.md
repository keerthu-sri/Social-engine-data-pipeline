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

## Round 1: Data Cleaning & EDA Summary

| Issue | Rows Affected | Action Taken |
| --- | --- | --- |
| **Exact duplicate rows** | 360 | Dropped (kept first occurrence) |
| **Mixed timestamp formats** (epoch / ISO / DD-MM-YYYY) | All rows | Normalized to ISO 8601 |
| **Undecoded HTML entities** in text | 341 | Decoded with `html.unescape` |
| **Missing platform** | 1,846 | Imputed via per-user mode where possible; else `"Unknown"` |
| **Missing `text_content**` | 1,746 | Flagged, not fabricated (`text_missing=True`) |
| **Missing likes** (literal `"NULL"`) | 1,858 | Kept as true missing, flagged (`likes_missing`) |
| **Negative likes** | 525 | Flagged as anomalous (`likes_anomalous`), excluded from `likes_valid` aggregate column, original value preserved |

*(Note: Counts above are pre-deduplication; see the notebook for exact post-dedup figures).*

### Key Data Assumptions

* **Users file integrity:** The users file required no structural cleaning—verified zero nulls, no duplicate IDs, and valid follower counts/dates.
* **Location formats:** `"Singapore"` and `"São Paulo, Brazil"` formats are legitimate, not corruption.
* **Platform imputation:** Per-user platform mode imputation is a statistical assumption, explicitly flagged per-row (`platform_imputed`) for full transparency.
* **Free-text preservation:** Free-text `text_content` is never fabricated. Missing text stays missing and flagged.
* **Negative likes:** Treated as data-quality anomalies (likes cannot be negative), but true values are unknown so rows are flagged rather than deleted or sign-corrected.

---

## Round 2: Semantic Understanding Layer (Classification Pipeline)

The Round 2 pipeline builds upon the cleaned dataset to solve two independent supervised text-classification tasks sharing a unified preprocessing and feature-engineering pipeline.

### 1. Dataset Split & Preprocessing

* **De-duplication:** 1,100 exact duplicate texts removed from the original 9,000 posts, resulting in **7,900 unique posts** split using an 80/20 stratified split (**6,320 train / 1,580 test**).


* **Pipeline Steps:** Unicode-escape repair, URL/mention masking (`<URL>`, `<USER>`), hashtag unwrapping, elongation normalization (capping repeats at 2), symbol stripping (retaining polarity-bearing punctuation `!?.,'`), lowercasing, and whitespace normalization.



### 2. Feature Engineering

Three feature blocks are concatenated into a sparse matrix per task:

* **Word n-grams (1-2):** Max 40,000 features (`sublinear_tf=True`, `min_df=2`).


* **Character n-grams (3-5):** Max 30,000 features for robustness against misspellings and slang.


* **Engineered Meta/Lexicon Features:** 10 features including VADER sentiment scores, character length, word count, exclamation/question marks, ALL-CAPS ratio, and elongated-word count, Min-Max scaled to $[0, 1]$.



### 3. Model Evaluation & Performance

#### Sentiment Classification (3-Class Balanced: Negative / Neutral / Positive)

* **Selected Model:** LinearSVC (tuned $C=0.3$).


* **Test Metrics:** Accuracy: **64.6%** | Macro-F1: **0.647** | Weighted-F1: **0.646** | Macro-Precision: **0.647** | Macro-Recall: **0.649**.


* **Key Challenges:** Dominated by sarcasm, terse/ambiguous text, and reporting-register mismatches where neutral news text contains negative real-world events.



#### Topic Classification (4-Class Imbalanced: Community_Discussion / Technical_Issues / Feature_Feedback / Account_Security)

* **Selected Model:** LinearSVC (tuned $C=0.7$, `class_weight="balanced"`).


* **Test Metrics:** Accuracy: **96.8%** | Macro-F1: **0.831** | Weighted-F1: **0.965** | Macro-Precision: **0.956** | Macro-Recall: **0.754**.


* **Context:** Accuracy is high because ~86% of posts belong to `Community_Discussion`; Macro-F1 highlights performance on minority classes (`Feature_Feedback` recall is lower at 0.50 due to severe data scarcity of rare classes).
