"""Запуск обучения и классификации из командной строки."""

import argparse
import csv
from pathlib import Path

import joblib

from .data import LABEL_NAMES, download_dataset, normalize_text
from .model import DEFAULT_METRICS, DEFAULT_MODEL, train


def predict_csv(model, source: Path, destination: Path) -> int:
    if source.resolve() == destination.resolve():
        raise ValueError("Для результата укажите отдельный файл.")

    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if "review" not in fields or "sentiment" in fields or len(set(fields)) != len(fields):
            raise ValueError("Ожидается CSV с колонкой review, уникальными заголовками и без sentiment.")
        rows, texts = [], []
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Некорректная строка CSV {reader.line_num}.")
            text = normalize_text(row["review"])
            if not text:
                raise ValueError(f"Пустой отзыв в строке CSV {reader.line_num}.")
            rows.append(row)
            texts.append(text)

    predictions = model.predict(texts) if texts else []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*fields, "sentiment"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow({**row, "sentiment": LABEL_NAMES[prediction]})
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Классификация тональности русскоязычных отзывов.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="обучить модель и сохранить метрики")
    train_parser.add_argument("--data", type=Path, help="TSV с отзывами (по умолчанию скачивается RuReviews)")
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="путь для сохранения модели")
    train_parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="путь для сохранения метрик")

    predict_parser = commands.add_parser("predict", help="определить тональность текста или отзывов из CSV")
    inputs = predict_parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="текст одного отзыва")
    inputs.add_argument("--input", type=Path, help="путь к CSV с колонкой review")
    predict_parser.add_argument("--output", type=Path, help="путь для сохранения CSV с прогнозами")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="путь к обученной модели")

    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.model.resolve() == args.metrics.resolve():
                raise ValueError("Для модели и метрик нужны разные пути.")
            dataset = args.data if args.data is not None else download_dataset()
            if dataset.resolve() in {args.model.resolve(), args.metrics.resolve()}:
                raise ValueError("Пути выходных файлов должны отличаться от пути датасета.")
            report = train(dataset, args.model, args.metrics)
            scores = report["tfidf_linear_svc"]
            print(f"Тестовая выборка — {report['split']['test_rows']} отзывов")
            print(f"Accuracy — {scores['accuracy']:.4f}, macro-F1 — {scores['f1_macro']:.4f}")
            print(f"Модель — {args.model}\nМетрики — {args.metrics}")
            return

        if (args.input is None) != (args.output is None):
            raise ValueError("Для пакетной классификации укажите одновременно --input и --output.")
        if args.output is not None and args.output.resolve() == args.model.resolve():
            raise ValueError("Путь выходного CSV должен отличаться от пути модели.")
        if args.text is not None:
            text = normalize_text(args.text)
            if not text:
                raise ValueError("Введите непустой отзыв.")
        if not args.model.is_file():
            raise ValueError(f"Модель {args.model} отсутствует. Запустите python -m sentiment train.")
        model = joblib.load(args.model)
        if args.text is not None:
            print(LABEL_NAMES[model.predict([text])[0]])
        else:
            count = predict_csv(model, args.input, args.output)
            print(f"Обработано отзывов — {count}. Результат сохранён в {args.output}.")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Ошибка. {error}\n")


if __name__ == "__main__":
    main()
