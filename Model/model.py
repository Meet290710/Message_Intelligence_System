"""
Message Intelligence System
Train and compare KNN, SVM, and Naive Bayes classifiers.

Dataset target: spam_label (0 = Legitimate, 1 = Spam)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "Message_Intelligence_Dataset_5200.xlsx"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "spam_label"
TEXT_COLUMN = "message_text"
DROP_COLUMNS = ["message_id", "timestamp", TARGET]
NUMERIC_FEATURES = [
    "message_length",
    "word_count",
    "num_urls",
    "num_digits",
    "num_special_chars",
    "spam_keyword_score",
    "legit_keyword_score",
    "sender_activity_score",
    "sender_account_age_days",
    "messages_sent_last_24h",
    "hour_of_day",
    "day_of_week",
]
REQUIRED_COLUMNS = [TEXT_COLUMN, TARGET] + NUMERIC_FEATURES


def load_dataset(file):
    """Load an Excel or CSV dataset from a path or uploaded file-like object."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    data.columns = [str(c).strip() for c in data.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    data = data.copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(int)

    invalid_labels = sorted(set(data[TARGET].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"{TARGET} must contain only 0 and 1. Found: {invalid_labels}"
        )

    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna("").astype(str)
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[TARGET].nunique() < 2:
        raise ValueError("The dataset must contain both spam and legitimate labels.")
    return data


def build_preprocessor():
    # MinMaxScaler keeps numeric inputs non-negative, allowing the same
    # representation to be used with Multinomial Naive Bayes.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("message_text", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=12000,
                sublinear_tf=True,
            ), TEXT_COLUMN),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_models():
    """Return model pipelines using shared text/numeric preprocessing."""
    return {
        "KNN": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="minkowski", p=2
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", SVC(
                kernel="rbf", C=1.0, probability=True,
                class_weight="balanced", random_state=42
            )),
        ]),
        "Naive Bayes": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
    }


def train_and_evaluate(data, test_size=0.2, random_state=42):
    """Train all classifiers and return fitted models, metrics, and test data."""
    X = data[[TEXT_COLUMN] + NUMERIC_FEATURES].copy()
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = build_models()
    metrics = []
    predictions = {}
    reports = {}
    matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        reports[name] = classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["Legitimate", "Spam"],
            output_dict=True, zero_division=0
        )
        matrices[name] = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
        })

    results = pd.DataFrame(metrics).sort_values("F1 Score", ascending=False)
    details = {
        "y_test": y_test,
        "predictions": predictions,
        "reports": reports,
        "confusion_matrices": matrices,
    }
    return models, results.reset_index(drop=True), details


def save_model(model, path=None):
    """Save a fitted pipeline to a joblib file."""
    output_path = Path(path) if path else MODEL_DIR / "best_model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def make_message_features(
    message,
    sender_activity_score=50.0,
    sender_account_age_days=180.0,
    messages_sent_last_24h=5.0,
    timestamp=None,
):
    """Build a single-row feature DataFrame for a message typed into the app.

    Keyword scores are simple transparent keyword counts, not a trained NLP
    feature. For a production system, replace them with the same scoring logic
    used to create the original dataset.
    """
    import re
    from datetime import datetime

    text = str(message or "")
    urls = re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE)
    digits = re.findall(r"\d", text)
    special_chars = re.findall(r"[^A-Za-z0-9\s]", text)

    spam_keywords = [
        "free", "winner", "win", "prize", "claim", "urgent", "cash",
        "offer", "click", "bonus", "limited", "verify", "gift"
    ]
    legit_keywords = [
        "meeting", "report", "project", "invoice", "schedule", "please",
        "thanks", "attached", "team", "regarding", "confirm"
    ]
    lower = text.lower()
    spam_score = sum(lower.count(word) for word in spam_keywords)
    legit_score = sum(lower.count(word) for word in legit_keywords)

    dt = pd.to_datetime(timestamp, errors="coerce") if timestamp is not None else pd.NaT
    if pd.isna(dt):
        dt = pd.Timestamp(datetime.now())

    row = {
        TEXT_COLUMN: text,
        "message_length": len(text),
        "word_count": len(text.split()),
        "num_urls": len(urls),
        "num_digits": len(digits),
        "num_special_chars": len(special_chars),
        "spam_keyword_score": spam_score,
        "legit_keyword_score": legit_score,
        "sender_activity_score": float(sender_activity_score),
        "sender_account_age_days": float(sender_account_age_days),
        "messages_sent_last_24h": float(messages_sent_last_24h),
        "hour_of_day": int(dt.hour),
        "day_of_week": int(dt.dayofweek),
    }
    return pd.DataFrame([row])[[TEXT_COLUMN] + NUMERIC_FEATURES]


def predict_message(model, features):
    """Return label and optional spam probability for one message."""
    prediction = int(model.predict(features)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.named_steps["classifier"].classes_)
        probabilities = {
            int(label): float(prob)
            for label, prob in zip(classes, probabilities)
        }
    return prediction, probabilities
