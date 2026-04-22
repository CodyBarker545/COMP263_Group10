from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path
from typing import Any

import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class AirlineSentimentLSTMTrainer:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.dataset_path = self._resolve_dataset_path()
        self.output_dir = self.repo_root / "model" / "lstm"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = 58
        self.text_col = "text"
        self.target_col = "airline_sentiment"
        self.max_words = 10000
        self.max_len = 50
        self.epochs = 10
        self.batch_size = 32

        self.label_encoder = LabelEncoder()
        self.tokenizer: Tokenizer | None = None
        self.model: Sequential | None = None

        self._set_seed()

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _resolve_dataset_path(self) -> Path:
        local_csv = Path(__file__).resolve().parent / "Tweets.csv"
        if local_csv.exists():
            return local_csv

        kaggle_dir = Path(kagglehub.dataset_download("crowdflower/twitter-airline-sentiment"))
        return kaggle_dir / "Tweets.csv"

    def clean_text(self, text: Any) -> str:
        cleaned = str(text).lower()
        cleaned = re.sub(r"http\S+", "", cleaned)
        cleaned = re.sub(r"@\w+", "", cleaned)
        cleaned = re.sub(r"#", "", cleaned)
        cleaned = re.sub(r"[^a-z\s]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)
        tweets = df[[self.text_col, self.target_col]].copy().dropna().drop_duplicates()
        tweets["clean_text"] = tweets[self.text_col].apply(self.clean_text)
        tweets["label"] = self.label_encoder.fit_transform(tweets[self.target_col])
        return tweets

    def split_dataset(self, tweets: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        X = tweets["clean_text"]
        y = tweets["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.seed,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test

    def prepare_sequences(self, X_train: pd.Series, X_test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_train)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding="post", truncating="post")
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding="post", truncating="post")
        return X_train_pad, X_test_pad

    def build_model(self) -> Sequential:
        model = Sequential(
            [
                Embedding(input_dim=self.max_words, output_dim=64),
                LSTM(64),
                Dropout(0.5),
                Dense(3, activation="softmax"),
            ]
        )
        model.build(input_shape=(None, self.max_len))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return model

    def compute_class_weights(self, y_train: pd.Series) -> dict[int, float]:
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        return {int(label): float(weight) for label, weight in zip(classes, class_weights)}

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray, test_loss: float, test_accuracy: float) -> dict[str, Any]:
        return {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=list(self.label_encoder.classes_),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def save_artifacts(self, history: dict[str, list[float]], metrics: dict[str, Any]) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must exist before saving artifacts.")

        self.model.save(self.output_dir / "model.keras")

        with open(self.output_dir / "tokenizer.pkl", "wb") as file:
            pickle.dump(self.tokenizer, file)

        with open(self.output_dir / "label_encoder.pkl", "wb") as file:
            pickle.dump(self.label_encoder, file)

        with open(self.output_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        with open(self.output_dir / "config.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "dataset_path": str(self.dataset_path),
                    "max_words": self.max_words,
                    "max_len": self.max_len,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "classes": list(self.label_encoder.classes_),
                },
                file,
                indent=2,
            )

    def run(self) -> None:
        tweets = self.load_dataset()
        X_train, X_test, y_train, y_test = self.split_dataset(tweets)
        X_train_pad, X_test_pad = self.prepare_sequences(X_train, X_test)
        class_weight_dict = self.compute_class_weights(y_train)

        model = self.build_model()
        history = model.fit(
            X_train_pad,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            class_weight=class_weight_dict,
            verbose=1,
        )

        test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
        y_pred = model.predict(X_test_pad, verbose=0)
        y_pred_classes = y_pred.argmax(axis=1)

        metrics = self.evaluate(y_test, y_pred_classes, test_loss, test_accuracy)
        self.save_artifacts(history.history, metrics)

        print(f"Saved LSTM artifacts to: {self.output_dir}")


if __name__ == "__main__":
    AirlineSentimentLSTMTrainer().run()
