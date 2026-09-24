"""
Scores datasets/live_processing.* with the Round 2 sentiment + topic models
and saves it back to the same path, so dashboard_app_2.py's Live Monitoring
page (page_round4) picks up real Positive/Negative/topic values instead of 0.

Run this after every scraper refresh, before hitting "Refresh live" in the app.
"""
import os
import joblib
import pandas as pd

from round2_nlp import repair_text, clean_text, meta_features, UnionVectorizer  # noqa: F401

STEM = "datasets/live_processing"
MODEL_DIR = "models"

# 1. find whichever extension the dashboard is currently reading
for ext, reader, writer in [
    (".xlsx", pd.read_excel, lambda d, p: d.to_excel(p, index=False)),
    (".csv", pd.read_csv, lambda d, p: d.to_csv(p, index=False)),
    (".json", pd.read_json, lambda d, p: d.to_json(p, orient="records")),
]:
    path = STEM + ext
    if os.path.exists(path):
        df = reader(path)
        save = lambda d, p=path, w=writer: w(d, p)
        break
else:
    raise FileNotFoundError(f"No {STEM}.xlsx/.csv/.json found")

print(f"Loaded {len(df)} rows from {path}")

# 2. use translated_text if you have it (Tamil/Tanglish posts), else raw text
text_col = "translated_text" if "translated_text" in df.columns else "text"
df["model_text"] = df[text_col].astype(str)
df["repaired_text"] = df["model_text"].apply(repair_text)
df["clean_text"] = df["repaired_text"].apply(clean_text)

# 3. load the Round 2 models
sent_vec = joblib.load(f"{MODEL_DIR}/vectorizer_sentiment_classification.pkl")
sent_model = joblib.load(f"{MODEL_DIR}/model_sentiment_classification.pkl")
topic_vec = joblib.load(f"{MODEL_DIR}/vectorizer_topic_classification.pkl")
topic_model = joblib.load(f"{MODEL_DIR}/model_topic_classification.pkl")

meta = meta_features(df["repaired_text"].values)

X_sent = sent_vec.transform(df["clean_text"].values, meta)
X_topic = topic_vec.transform(df["clean_text"].values, meta)

df["sentiment"] = sent_model.predict(X_sent)
df["topic"] = topic_model.predict(X_topic)

print(df["sentiment"].value_counts(dropna=False))

# 4. save back to the exact same path the dashboard reads
save(df)
print(f"Saved scored data back to {path}")