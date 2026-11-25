"""Загрузка RuReviews и подготовка текстов к классификации."""

import csv
from collections import Counter
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from unicodedata import normalize
from urllib.request import urlopen


DATASET_REVISION = "2f6e61d30dc6c58ba47dce72e5803e1f0a3f38a7"
DATASET_NAME = "women-clothing-accessories.3-class.balanced.csv"
DATASET_URL = (
    f"https://raw.githubusercontent.com/sismetanin/rureviews/{DATASET_REVISION}/{DATASET_NAME}"
)
DATASET_SHA256 = "db226708d966a1506560136ef85b33ea326eaf6119465d4b1edb5bf9395330fd"
DEFAULT_DATA = Path("data") / DATASET_NAME

LABELS = ("positive", "neutral", "negative")
LABEL_NAMES = dict(zip(LABELS, ("позитивный", "нейтральный", "негативный")))


def normalize_text(text: str) -> str:
    return " ".join(normalize("NFKC", text).casefold().split())


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def download_dataset() -> Path:
    if DEFAULT_DATA.is_file():
        if sha256(DEFAULT_DATA) != DATASET_SHA256:
            raise ValueError(f"Неверная контрольная сумма {DEFAULT_DATA}. Укажите свой TSV через --data.")
        return DEFAULT_DATA

    DEFAULT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=DEFAULT_DATA.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            with urlopen(DATASET_URL, timeout=60) as response:
                copyfileobj(response, file)
            file.close()
            if sha256(temporary) != DATASET_SHA256:
                raise ValueError("Скачанный RuReviews не прошёл проверку SHA-256.")
            temporary.replace(DEFAULT_DATA)
        finally:
            temporary.unlink(missing_ok=True)
    return DEFAULT_DATA


def load_dataset(path: Path) -> tuple[list[str], list[str], dict]:
    label_counts_by_text: dict[str, Counter[str]] = {}
    raw_counts: Counter[str] = Counter()
    empty_rows = 0

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if not {"review", "sentiment"}.issubset(reader.fieldnames or []):
            raise ValueError("Для обучения нужен TSV с колонками review и sentiment.")
        for row in reader:
            if None in row or row["review"] is None or row["sentiment"] is None:
                raise ValueError(f"Некорректная строка TSV {reader.line_num}.")
            label = row["sentiment"].strip().casefold()
            if label == "neautral":
                label = "neutral"
            if label not in LABELS:
                raise ValueError(f"Неизвестная метка {label!r} в строке {reader.line_num}.")
            raw_counts[label] += 1
            text = normalize_text(row["review"])
            if not text:
                empty_rows += 1
                continue
            label_counts_by_text.setdefault(text, Counter())[label] += 1

    texts, labels = [], []
    duplicate_rows = conflicting_rows = conflicting_texts = 0
    for text, counts in label_counts_by_text.items():
        if len(counts) > 1:
            conflicting_texts += 1
            conflicting_rows += counts.total()
            continue
        label, count = next(iter(counts.items()))
        duplicate_rows += count - 1
        texts.append(text)
        labels.append(label)

    usable_counts = Counter(labels)
    if any(usable_counts[label] < 2 for label in LABELS):
        raise ValueError("После очистки нужно хотя бы два уникальных отзыва каждого класса.")
    summary = {
        "sha256": sha256(path),
        "raw_rows": raw_counts.total(),
        "raw_by_label": {label: raw_counts[label] for label in LABELS},
        "empty_rows_removed": empty_rows,
        "duplicate_rows_removed": duplicate_rows,
        "conflicting_rows_removed": conflicting_rows,
        "conflicting_texts_removed": conflicting_texts,
        "usable_rows": len(texts),
        "usable_by_label": {label: usable_counts[label] for label in LABELS},
    }
    return texts, labels, summary
