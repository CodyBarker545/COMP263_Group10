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
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class AirlineSentimentTransformerTrainer:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.dataset_path = self._resolve_dataset_path()
        self.output_dir = self.repo_root / "model" / "transformer"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = 58
        self.text_col = "text"
        self.target_col = "airline_sentiment"
        self.model_name = "distilbert-base-uncased"
        self.max_len = 64
        self.epochs = 2
        self.batch_size = 16

        self.label_encoder = LabelEncoder()
        self.tokenizer: AutoTokenizer | None = None
        self.model: TFAutoModelForSequenceClassification | None = None

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
        cleaned = str(text).strip()
        cleaned = re.sub(r"http\S+|www\S+", " ", cleaned)
        cleaned = re.sub(r"@\w+", "@user", cleaned)
        cleaned = re.sub(r"#(\w+)", r"\1", cleaned)
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

    def encode_text(self, texts: pd.Series) -> dict[str, tf.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be initialized before encoding.")

        return self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="tf",
        )

    def build_model(self) -> TFAutoModelForSequenceClassification:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model
        return model

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

        self.model.save_pretrained(self.output_dir / "model")
        self.tokenizer.save_pretrained(self.output_dir / "tokenizer")

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
                    "model_name": self.model_name,
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_encodings = self.encode_text(X_train)
        test_encodings = self.encode_text(X_test)

        model = self.build_model()
        history = model.fit(
            {
                "input_ids": train_encodings["input_ids"],
                "attention_mask": train_encodings["attention_mask"],
            },
            y_train,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )

        test_loss, test_accuracy = model.evaluate(
            {
                "input_ids": test_encodings["input_ids"],
                "attention_mask": test_encodings["attention_mask"],
            },
            y_test,
            verbose=0,
        )

        predictions = model.predict(
            {
                "input_ids": test_encodings["input_ids"],
                "attention_mask": test_encodings["attention_mask"],
            },
            verbose=0,
        )
        y_pred_classes = np.argmax(predictions.logits, axis=1)

        metrics = self.evaluate(y_test, y_pred_classes, test_loss, test_accuracy)
        self.save_artifacts(history.history, metrics)

        print(f"Saved transformer artifacts to: {self.output_dir}")


if __name__ == "__main__":
    AirlineSentimentTransformerTrainer().run()
