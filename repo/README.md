# Social Engine — Round 1 Phase 1 Submission

Data recovery pipeline for the corrupted Social Engine dataset.

## Contents

- `data_cleaning_and_eda.ipynb` reproducible notebook: cleaning code, inline justification for every transformation, and exploratory data analysis. **Run this top to bottom to regenerate everything from the raw files.**
- `Social_Engine_Users.csv`, `Social_Engine_Posts_Corrupted.csv` : raw source files, recovered from the recovery-terminal challenge site.
- `Social_Engine_Users_Cleaned.csv`, `Social_Engine_Posts_Cleaned.csv` : cleaned output.
- `EDA_Report.md` : written EDA report with charts.

## How to reproduce

```bash
pip install pandas numpy matplotlib jupyter nbconvert
jupyter nbconvert --to notebook --execute --inplace data_cleaning_and_eda.ipynb
```

## Cleaning summary

| Issue | Rows affected | Action taken |
| --- | --- | --- |
| Exact duplicate rows | 360 | Dropped (kept first occurrence) |
| Mixed timestamp formats (epoch / ISO / DD-MM-YYYY) | all rows | Normalized to ISO 8601 |
| Undecoded HTML entities in text | 341 | Decoded with `html.unescape` |
| Missing `platform` | 1,846 | Imputed via per-user mode where possible; else `"Unknown"` |
| Missing `text_content` | 1,746 | **Flagged, not fabricated** (`text_missing=True`) |
| Missing `likes` (literal `"NULL"`) | 1,858 | Kept as true missing, flagged (`likes_missing`) |
| Negative `likes` | 525 | **Flagged as anomalous** (`likes_anomalous`), excluded from `likes_valid` aggregate column, original value preserved |

All counts above are pre-deduplication; see the notebook for exact post-dedup figures.

## Key assumptions

1. `users` file required no structural cleaning — verified no nulls, no duplicate IDs, valid follower counts and dates.
2. "Singapore" and "São Paulo, Brazil" location formats are legitimate, not corruption.
3. Per-user platform mode imputation is a statistical assumption, explicitly flagged per-row (`platform_imputed`) for full transparency — not treated as ground truth.
4. Free-text `text_content` is never fabricated. Missing text stays missing and flagged.
5. Negative likes are data-quality anomalies (a "like" cannot be negative) but the true intended value is unknown, so rows are flagged rather than deleted or sign-corrected — a reversible decision documented for the evaluators.duc