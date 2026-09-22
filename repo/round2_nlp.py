"""
Round 2 NLP helpers, copied verbatim from Round2_NLP_Notebook.ipynb so the Round 3 notebook
can load the saved Round 2 models.

Usage in the Round 3 notebook (import the names into the notebook namespace - the pickles
were saved from the notebook's __main__, so they look for `UnionVectorizer` there):

    from round2_nlp import *
    sent_vec   = joblib.load("models/vectorizer_sentiment_classification.pkl")
    sent_model = joblib.load("models/model_sentiment_classification.pkl")
    topic_vec  = joblib.load("models/vectorizer_topic_classification.pkl")
    topic_model= joblib.load("models/model_topic_classification.pkl")

    df["repaired_text"] = df["text"].apply(repair_text)
    df["clean_text"]    = df["repaired_text"].apply(clean_text)
    M = meta_features(df["repaired_text"].values)
    X = sent_vec.transform(df["clean_text"].values, M)
    df["sentiment"] = sent_model.predict(X)
"""
import re, codecs
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTISPACE_RE = re.compile(r"\s+")
ELONG_RE = re.compile(r"(.)\1{2,}")


def fix_unicode_escapes(text):
    if "\\u" in text:
        try:
            return codecs.decode(text, "unicode_escape")
        except Exception:
            return text
    return text


def repair_text(text):
    """Unicode-fix only; keeps case & punctuation for meta-features / VADER."""
    return fix_unicode_escapes(str(text))


def clean_text(text):
    text = URL_RE.sub(" <URL> ", text)
    text = MENTION_RE.sub(" <USER> ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = ELONG_RE.sub(r"\1\1", text)
    text = re.sub(r"[^A-Za-z0-9<>!?.,\x27\s]", " ", text)
    text = MULTISPACE_RE.sub(" ", text).strip().lower()
    return text


def build_vectorizers():
    word_vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.9, ngram_range=(1, 2), max_features=40000)
    char_vec = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000)
    return word_vec, char_vec


sia = SentimentIntensityAnalyzer()


def meta_features(raw_texts):
    rows = []
    for t in raw_texts:
        vs = sia.polarity_scores(t)
        n_words = max(len(t.split()), 1)
        n_upper_words = sum(1 for w in t.split() if len(w) > 1 and w.isupper())
        rows.append([
            vs["neg"], vs["neu"], vs["pos"], vs["compound"],
            len(t), n_words, t.count("!"), t.count("?"),
            n_upper_words / n_words, len(ELONG_RE.findall(t)),
        ])
    return np.array(rows, dtype=float)


class UnionVectorizer:
    """TF-IDF (word + char) concatenated with scaled meta/lexicon features."""
    def __init__(self):
        self.word_vec, self.char_vec = build_vectorizers()
        self.scaler = MinMaxScaler()

    def fit(self, X_text, X_meta):
        self.word_vec.fit(X_text); self.char_vec.fit(X_text)
        self.scaler.fit(X_meta)
        return self

    def transform(self, X_text, X_meta):
        w = self.word_vec.transform(X_text)
        c = self.char_vec.transform(X_text)
        m = csr_matrix(self.scaler.transform(X_meta))
        return hstack([w, c, m]).tocsr()

    def fit_transform(self, X_text, X_meta):
        self.fit(X_text, X_meta)
        return self.transform(X_text, X_meta)
