"""Веб-интерфейс классификатора тональности."""

from pathlib import Path

import joblib
from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from .data import LABEL_NAMES, normalize_text
from .model import DEFAULT_MODEL


MAX_REVIEW_LENGTH = 5_000
LENGTH_ERROR = f"Отзыв должен содержать не более {MAX_REVIEW_LENGTH} символов."


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 96 * 1024
    model_path = Path(app.root_path).parent / DEFAULT_MODEL
    model = joblib.load(model_path)

    def page(review="", label=None, error=None):
        return render_template(
            "index.html",
            review=review,
            label=label,
            label_name=LABEL_NAMES.get(label),
            error=error,
            max_length=MAX_REVIEW_LENGTH,
        )

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return page()

        review = request.form.get("review", "")
        if len(review) > MAX_REVIEW_LENGTH:
            return page(review=review, error=LENGTH_ERROR), 400

        text = normalize_text(review)
        if not text:
            return page(review=review, error="Введите текст отзыва."), 400

        label = model.predict([text])[0]
        return page(review=review, label=label)

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return page(error=LENGTH_ERROR), 413

    return app
