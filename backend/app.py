from __future__ import annotations

import io
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class TextPredictionRequest(BaseModel):
    text: str


class BaseSentimentModel:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.label_encoder = self._load_pickle("label_encoder.pkl")

    def _load_pickle(self, filename: str) -> Any:
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

        with open(path, "rb") as file:
            return pickle.load(file)

    def predict(self, text: str) -> dict[str, Any]:
        raise NotImplementedError


class LSTMSentimentModel(BaseSentimentModel):
    def __init__(self, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.tokenizer = self._load_pickle("tokenizer.pkl")
        self.model = load_model(model_dir / "model.keras")
        self.max_len = int(self.config["max_len"])

    def clean_text(self, text: Any) -> str:
        cleaned = str(text).lower()
        cleaned = re.sub(r"http\S+", "", cleaned)
        cleaned = re.sub(r"@\w+", "", cleaned)
        cleaned = re.sub(r"#", "", cleaned)
        cleaned = re.sub(r"[^a-z\s]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = self.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding="post", truncating="post")
        probabilities = self.model.predict(padded, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = str(self.label_encoder.inverse_transform([predicted_index])[0])

        return {
            "label": predicted_label,
            "confidence": float(np.max(probabilities)),
            "probabilities": {
                str(label): float(probabilities[index])
                for index, label in enumerate(self.label_encoder.classes_)
            },
        }


class TransformerSentimentModel(BaseSentimentModel):
    def __init__(self, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_dir / "model")
        self.max_len = int(self.config["max_len"])

    def clean_text(self, text: Any) -> str:
        cleaned = str(text).strip()
        cleaned = re.sub(r"http\S+|www\S+", " ", cleaned)
        cleaned = re.sub(r"@\w+", "@user", cleaned)
        cleaned = re.sub(r"#(\w+)", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = self.clean_text(text)
        encoded = self.tokenizer(
            [cleaned],
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="tf",
        )
        logits = self.model(encoded, training=False).logits.numpy()[0]
        probabilities = tf.nn.softmax(logits).numpy()
        predicted_index = int(np.argmax(probabilities))
        predicted_label = str(self.label_encoder.inverse_transform([predicted_index])[0])

        return {
            "label": predicted_label,
            "confidence": float(np.max(probabilities)),
            "probabilities": {
                str(label): float(probabilities[index])
                for index, label in enumerate(self.label_encoder.classes_)
            },
        }


class SentimentInferenceService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.model_root = repo_root / "model"
        self.lstm_model = self._load_optional_model(LSTMSentimentModel, self.model_root / "lstm")
        self.transformer_model = self._load_optional_model(TransformerSentimentModel, self.model_root / "transformer")

    def _load_optional_model(self, model_class: type[BaseSentimentModel], model_dir: Path) -> BaseSentimentModel | None:
        try:
            if not model_dir.exists() or not any(model_dir.iterdir()):
                return None
            return model_class(model_dir)
        except Exception:
            return None

    def health(self) -> dict[str, Any]:
        return {
            "lstm_ready": self.lstm_model is not None,
            "transformer_ready": self.transformer_model is not None,
            "model_root": str(self.model_root),
        }

    def predict_text(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty.")

        if self.lstm_model is None or self.transformer_model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "One or more saved model artifacts are missing. "
                    "Make sure model/lstm and model/transformer contain trained artifacts."
                ),
            )

        return {
            "tweet": text,
            "lstm": self.lstm_model.predict(text),
            "transformer": self.transformer_model.predict(text),
        }

    def parse_uploaded_tweets(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and "tweets" in payload:
            payload = payload["tweets"]

        if not isinstance(payload, list):
            raise HTTPException(status_code=400, detail="Uploaded JSON must be a list or an object with a 'tweets' list.")

        parsed_rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, str):
                parsed_rows.append({"tweet": item, "_original": {"tweet": item}})
                continue

            if isinstance(item, dict):
                tweet_text = None
                for key in ("text", "tweet", "content", "message"):
                    if key in item and str(item[key]).strip():
                        tweet_text = str(item[key])
                        break

                if tweet_text is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Each tweet object must include one of: text, tweet, content, message.",
                    )

                parsed_rows.append({"tweet": tweet_text, "_original": item})
                continue

            raise HTTPException(
                status_code=400,
                detail="Uploaded JSON items must be either strings or tweet objects.",
            )

        return parsed_rows

    def classify_uploaded_tweets(self, payload: Any) -> list[dict[str, Any]]:
        if self.lstm_model is None or self.transformer_model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "One or more saved model artifacts are missing. "
                    "Make sure model/lstm and model/transformer contain trained artifacts."
                ),
            )

        rows = self.parse_uploaded_tweets(payload)
        classified_rows: list[dict[str, Any]] = []

        for row in rows:
            tweet = row["tweet"]
            output = dict(row["_original"])
            output["tweet"] = tweet
            output["lstm_output"] = self.lstm_model.predict(tweet)
            output["transformer_output"] = self.transformer_model.predict(tweet)
            classified_rows.append(output)

        return classified_rows


repo_root = Path(__file__).resolve().parents[1]
service = SentimentInferenceService(repo_root)

app = FastAPI(title="Airline Sentiment API", version="1.0.0")
api_router = APIRouter(prefix="/api")

_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_allowed_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@api_router.get("/health")
def health_check() -> dict[str, Any]:
    return service.health()


@app.post("/predict/text")
@api_router.post("/predict/text")
def predict_text(request: TextPredictionRequest) -> JSONResponse:
    return JSONResponse(service.predict_text(request.text))


@app.post("/predict/file")
@api_router.post("/predict/file")
async def predict_file(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported.")

    content = await file.read()
    try:
        payload = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {error.msg}") from error

    classified_rows = service.classify_uploaded_tweets(payload)
    output_bytes = json.dumps(classified_rows, indent=2).encode("utf-8")

    output_name = f"{Path(file.filename).stem}_classified.json"
    headers = {"Content-Disposition": f'attachment; filename="{output_name}"'}
    return StreamingResponse(io.BytesIO(output_bytes), media_type="application/json", headers=headers)


app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
