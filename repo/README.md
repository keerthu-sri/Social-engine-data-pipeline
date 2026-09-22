# Social Engine — Pipeline & Model Submission (Round 1 & Round 2)

Comprehensive data recovery, cleaning, exploratory data analysis (EDA), and semantic understanding/classification pipeline for the Social Engine dataset.

## Repository Contents

- `data_cleaning_and_eda.ipynb`: Reproducible notebook containing cleaning code, inline justification for every transformation, and EDA. Run top-to-bottom to regenerate artifacts from raw files.
- `Social_Engine_Users.csv` **&** `Social_Engine_Posts_Corrupted.csv`: Raw source files recovered from the recovery-terminal challenge site.
- `Social_Engine_Users_Cleaned.csv` **&** `Social_Engine_Posts_Cleaned.csv`: Cleaned output tables.
- `EDA_Report.md`: Written EDA report complete with charts.
- `models/` **&** `outputs/`: Saved vectorizers, trained models, and evaluation artifacts for the semantic classification layer.

## How to Reproduce

Bash

```
pip install pandas numpy matplotlib jupyter nbconvert scikit-learn
jupyter nbconvert --to notebook --execute --inplace data_cleaning_and_eda.ipynb
```

## Round 1: Data Cleaning & EDA Summary

<table class="notion-table" style="min-width: 75px;">
<colgroup><col style="min-width: 25px;"><col style="min-width: 25px;"><col style="min-width: 25px;"></colgroup><tbody><tr><td colspan="1" rowspan="1"><p><strong>Issue</strong></p></td><td colspan="1" rowspan="1"><p><strong>Rows Affected</strong></p></td><td colspan="1" rowspan="1"><p><strong>Action Taken</strong></p></td></tr><tr><td colspan="1" rowspan="1"><p><strong>Exact duplicate rows</strong></p></td><td colspan="1" rowspan="1"><p>360</p></td><td colspan="1" rowspan="1"><p>Dropped (kept first occurrence)</p></td></tr><tr><td colspan="1" rowspan="1"><p><strong>Mixed timestamp formats</strong> (epoch / ISO / DD-MM-YYYY)</p></td><td colspan="1" rowspan="1"><p>All rows</p></td><td colspan="1" rowspan="1"><p>Normalized to ISO 8601</p></td></tr><tr><td colspan="1" rowspan="1"><p><strong>Undecoded HTML entities</strong> in text</p></td><td colspan="1" rowspan="1"><p>341</p></td><td colspan="1" rowspan="1"><p>Decoded with <code>html.unescape</code></p></td></tr><tr><td colspan="1" rowspan="1"><p><strong>Missing platform</strong></p></td><td colspan="1" rowspan="1"><p>1,846</p></td><td colspan="1" rowspan="1"><p>Imputed via per-user mode where possible; else <code>"Unknown"</code></p></td></tr><tr><td colspan="1" rowspan="1"><p><strong>Missing </strong><code>text_content</code></p></td><td colspan="1" rowspan="1"><p>1,746</p></td><td colspan="1" rowspan="1"><p>Flagged, not fabricated (<code>text_missing=True</code>)</p></td></tr><tr><td colspan="1" rowspan="1"><p><strong>Missing likes</strong> (literal <code>"NULL"</code>)</p></td><td colspan="1" rowspan="1"><p>1,858</p></td><td colspan="1" rowspan="1"><p>Kept as true missing, flagged (<code>likes_missing</code>)</p></td></tr><tr><td colspan="1" rowspan="1"><p><strong>Negative likes</strong></p></td><td colspan="1" rowspan="1"><p>525</p></td><td colspan="1" rowspan="1"><p>Flagged as anomalous (<code>likes_anomalous</code>), excluded from <code>likes_valid</code> aggregate column, original value preserved</p></td></tr></tbody>
</table>

*(Note: Counts above are pre-deduplication; see the notebook for exact post-dedup figures).*

### Key Data Assumptions

- **Users file integrity:** The users file required no structural cleaning—verified zero nulls, no duplicate IDs, and valid follower counts/dates.
- **Location formats:** `"Singapore"` and `"São Paulo, Brazil"` formats are legitimate, not corruption.
- **Platform imputation:** Per-user platform mode imputation is a statistical assumption, explicitly flagged per-row (`platform_imputed`) for full transparency.
- **Free-text preservation:** Free-text `text_content` is never fabricated. Missing text stays missing and flagged.
- **Negative likes:** Treated as data-quality anomalies (likes cannot be negative), but true values are unknown so rows are flagged rather than deleted or sign-corrected.

## Round 2: Semantic Understanding Layer (Classification Pipeline)

The Round 2 pipeline builds upon the cleaned dataset to solve two independent supervised text-classification tasks sharing a unified preprocessing and feature-engineering pipeline.

### 1. Dataset Split & Preprocessing

- **De-duplication:** 1,100 exact duplicate texts removed from the original 9,000 posts, resulting in **7,900 unique posts** split using an 80/20 stratified split (**6,320 train / 1,580 test**).
- **Pipeline Steps:** Unicode-escape repair, URL/mention masking (`<URL>`, `<USER>`), hashtag unwrapping, elongation normalization (capping repeats at 2), symbol stripping (retaining polarity-bearing punctuation `!?.,'`), lowercasing, and whitespace normalization.

### 2. Feature Engineering

Three feature blocks are concatenated into a sparse matrix per task:

- **Word n-grams (1-2):** Max 40,000 features (`sublinear_tf=True`, `min_df=2`).
- **Character n-grams (3-5):** Max 30,000 features for robustness against misspellings and slang.
- **Engineered Meta/Lexicon Features:** 10 features including VADER sentiment scores, character length, word count, exclamation/question marks, ALL-CAPS ratio, and elongated-word count, Min-Max scaled to $\[0, 1\]$.

### 3. Model Evaluation & Performance

#### Sentiment Classification (3-Class Balanced: Negative / Neutral / Positive)

- **Selected Model:** LinearSVC (tuned $C=0.3$).
- **Test Metrics:** Accuracy: **64.6%** | Macro-F1: **0.647** | Weighted-F1: **0.646** | Macro-Precision: **0.647** | Macro-Recall: **0.649**.
- **Key Challenges:** Dominated by sarcasm, terse/ambiguous text, and reporting-register mismatches where neutral news text contains negative real-world events.

#### Topic Classification (4-Class Imbalanced: Community_Discussion / Technical_Issues / Feature_Feedback / Account_Security)

- **Selected Model:** LinearSVC (tuned $C=0.7$, `class_weight="balanced"`).
- **Test Metrics:** Accuracy: **96.8%** | Macro-F1: **0.831** | Weighted-F1: **0.965** | Macro-Precision: **0.956** | Macro-Recall: **0.754**.
- **Context:** Accuracy is high because \~86% of posts belong to `Community_Discussion`; Macro-F1 highlights performance on minority classes (`Feature_Feedback` recall is lower at 0.50 due to severe data scarcity of rare classes).

  # Round 3 — Real-Time Analysis Notebook

  **Competition:** Data Vortex A'26 **Topic 7:** Public Reaction to a Celebrity/Influencer Controversy **Subject:** Reported/alleged relationship between Tamil Nadu Chief Minister Vijay and actress Trisha Krishnan, and the surrounding political/fan commentary. **Note:** This notebook analyzes *public reaction only*. Neither party has confirmed the relationship; no claims about the underlying story are made or endorsed.

  ---

  ## 1. How this fits with Round 1 & Round 2

  | Round | Deliverable | Role in this notebook |
  | --- | --- | --- |
  | **Round 1** | `scraper.py` | Produces the raw input — `live_dataset.csv` (and optional `.json`), collected live from YouTube (and, when available, GDELT news) over a \~40-hour collection window. |
  | **Round 2** | `Round2_NLP_Notebook.ipynb`, `round2_nlp.py`, `models/` | Trains and pickles the sentiment and topic classifiers (`vectorizer_*` + `model_*` `.pkl` files) and provides the shared text-cleaning functions (`repair_text`, `clean_text`, `meta_features`, `UnionVectorizer`) that this notebook imports and re-applies to live data. |
  | **Round 3** (this notebook) | `Round3_Realtime_Analysis_Notebook.ipynb` | Loads the live dataset, cleans/filters it, applies the Round 2 models, and produces the Activity, Sentiment, Topic/Entity, and Trigger-Explanation analyses required for the Round 3 report. |

  **Run order:** `scraper.py` (Round 1) → `Round2_NLP_Notebook.ipynb` (Round 2, to produce `models/`) → this notebook (Round 3).

  ---

  ## 2. What this notebook does

  Run top to bottom. It is safe to re-run at any point during the live collection window — later rows appended to `live_dataset.csv` simply extend the timelines.
   1. **Load & clean** `live_dataset.csv` (dedupe on `source`+`post_id`, parse timestamps, drop empty text).
   2. **Restrict to the real controversy window** — `--yt-order relevance` scraping also pulls in old, unrelated high-comment videos (e.g. a 2013 song upload, years-old fan edits). A `WINDOW_START` cutoff removes this legacy noise before any analysis runs. **Check/update this date** to match your actual scrape start before trusting downstream charts.
   3. **Language flagging** — tags each post `en` / `ta` (Tamil script) / `mixed` / `tanglish` (Latin-script Tamil), since the Round 2 models and VADER are English-only.
   4. **Translation** — non-English/Tanglish text is machine-translated (`googletrans`) so it can be scored by the Round 2 models; a manual translation-quality audit sample is exported for spot-checking.
   5. **Apply Round 2 models** — predicts `sentiment` (Positive/Negative/Neutral) and `topic` on English-language text only.
   6. **Activity analysis** — hourly + daily post-volume buckets with rolling-baseline spike detection (≥1.75× baseline mean or ≥2σ above it).
   7. **Sentiment analysis** — daily sentiment mix + day-over-day shift detection (flags a shift when negative-share changes by more than `SHIFT_THRESHOLD`, default 0.15).
   8. **Topic/entity analysis** — Round 2 topic-category distribution (for reference) plus a direct daily keyword count of the story-specific entities: Vijay, Trisha, Udhayanidhi, Kangana, TVK, DMK, Sangeetha.
   9. **Trigger explanations** — pulls any `gdelt_news` rows within ±1 day of each flagged spike/shift as a starting point for explaining *why* the change happened. This surfaces candidate headlines only; the analyst still has to read them and write the explanation.
  10. **Export** — writes all key tables (`daily_volume_spikes.csv`, `hourly_volume_spikes.csv`, `sentiment_daily.csv`, `topic_daily.csv`, `entity_daily.csv`, `summary.json`) to `outputs/`, ready to drop into the Round 3 report.

  ---

  ## 3. Requirements
  - Python 3.12
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`
  - `googletrans==4.0.0-rc1` (installed in-notebook via `%pip install`)
  - `round2_nlp.py` present alongside this notebook (provides `repair_text`, `clean_text`, `meta_features`, `UnionVectorizer`)
  - A `models/` folder containing the four pickled files from Round 2:
    - `vectorizer_sentiment_classification.pkl`, `model_sentiment_classification.pkl`
    - `vectorizer_topic_classification.pkl`, `model_topic_classification.pkl`

  ## 4. Inputs / Outputs

  **Input:** `live_dataset.csv` (columns include `source`, `post_id`, `parent_id`, `created_utc`, `text`)

  **Outputs (written to** `outputs/`**):**

  | File | Contents |
  | --- | --- |
  | `daily_volume_spikes.csv` | Daily post volume + flagged spike days |
  | `hourly_volume_spikes.csv` | Hourly post volume + flagged spike hours |
  | `sentiment_daily.csv` | Daily Positive/Negative/Neutral share + day-over-day deltas |
  | `topic_daily.csv` | Daily Round 2 topic-category distribution |
  | `entity_daily.csv` | Daily mention counts for each tracked entity |
  | `summary.json` | Headline stats: record count, date range, source mix, language mix, spike/shift counts |
  | `translation_quality_audit.csv` | 100-row sample for manually rating translation quality |
  | `sentiment_evaluation.csv` | 100-row sample for manually rating model sentiment against human judgment |

  ---

  ## 5. Known Limitations (carry these into the Round 3 report)
  - **Single source:** all 1,994 in-window records are from YouTube comments only — no Twitter/X, Instagram, or forum data. Findings should not be generalized as "overall public opinion."
  - **Language coverage:** Round 2 models were trained on English only. Tanglish text is routed through the `en` pipeline but was never part of the training distribution — treat sentiment/topic predictions on Tanglish rows as lower-confidence.
  - **Topic-label mismatch:** the Round 2 topic categories (`Community_Discussion`, `Technical_Issues`, `Feature_Feedback`, `Account_Security`) were built for a product-feedback dataset, not celebrity/political discourse. Expect most posts to fall into `Community_Discussion` — this is a schema mismatch, not a real finding. The entity keyword counts are the more meaningful topic signal for this story.
  - **News cross-referencing gap:** the `gdelt_news` source returned no rows in the last run, so spike/shift triggers could not be confirmed against external headlines and were instead inferred from the comment text itself. Widen `--gdelt-timespan` or add a news source on the next scrape to close this gap.
  - **Translation quality:** machine-translated text (via `googletrans`) has not been independently verified at scale — only spot-checked via the manual audit sample.

  ---

  ## 6. Quick Start

  ```bash
  # 1. Make sure Round 1 + Round 2 deliverables are present
  #    live_dataset.csv, round2_nlp.py, models/*.pkl
  
  # 2. Install dependencies
  pip install pandas numpy matplotlib seaborn scikit-learn joblib googletrans==4.0.0-rc1
  
  # 3. Open and run top to bottom
  jupyter notebook Round3_Realtime_Analysis_Notebook.ipynb
  
  # 4. Check outputs/
  ls outputs/
  
  ```