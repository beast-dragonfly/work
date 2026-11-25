"""Обучение и оценка модели на отложенной выборке."""

import json
from collections import Counter
from math import ceil
from pathlib import Path

import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from .data import LABELS, load_dataset


DEFAULT_MODEL = Path("models/sentiment.joblib")
DEFAULT_METRICS = Path("reports/metrics.json")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_model():
    words = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_features=60_000,
        sublinear_tf=True,
    )
    chars = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=80_000,
        sublinear_tf=True,
    )
    features = FeatureUnion([("words", words), ("chars", chars)])
    return make_pipeline(features, LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=5000))


def evaluate(expected, predicted) -> dict:
    report = classification_report(
        expected, predicted, labels=LABELS, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(expected, predicted),
        "f1_macro": report["macro avg"]["f1-score"],
        "by_label": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
    }


def train(dataset_path: Path, model_path: Path, metrics_path: Path) -> dict:
    texts, labels, dataset_summary = load_dataset(dataset_path)
    if ceil(len(labels) * TEST_SIZE) < len(LABELS):
        raise ValueError("Размер тестовой выборки должен позволять оценить все три класса.")
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )
    if set(y_train) != set(LABELS) or set(y_test) != set(LABELS):
        raise ValueError("Добавьте отзывы редких классов, чтобы все классы вошли в обе выборки.")

    majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    model = build_model().fit(x_train, y_train)
    report = {
        "dataset": dataset_summary,
        "labels": LABELS,
        "split": {
            "random_state": RANDOM_STATE,
            "test_fraction": TEST_SIZE,
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "train_by_label": dict(Counter(y_train)),
            "test_by_label": dict(Counter(y_test)),
        },
        "scikit_learn_version": sklearn.__version__,
        "dummy_most_frequent": evaluate(y_test, majority.predict(x_test)),
        "tfidf_linear_svc": evaluate(y_test, model.predict(x_test)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
