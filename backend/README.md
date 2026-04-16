# Backend

This folder contains the FastAPI backend for Airline Sentiment Detection.

Main file:

- [app.py]

## Purpose

The backend is responsible for:

- loading the saved LSTM and transformer models from the `model` folder
- exposing HTTP endpoints for single-tweet prediction
- accepting uploaded JSON files of tweets
- returning or downloading prediction results

## Files

- [app.py]
  Main API application and model-loading logic
- [requirements.txt]
  Backend-only dependencies

## Main Classes

### `TextPredictionRequest`

Pydantic request model for `POST /predict/text`.

Field:

- `text`: the user-entered tweet or sentence

### `BaseSentimentModel`

Shared base class for sentiment model wrappers.

Responsibilities:

- store the model folder path
- load shared artifacts such as `label_encoder.pkl`
- define the common `predict()` interface

Key method:

- `_load_pickle(filename)`: loads a saved Python object from the model folder

### `LSTMSentimentModel`

Wrapper around the saved Keras LSTM model.

Responsibilities:

- load `model.keras`
- load `tokenizer.pkl`
- clean incoming text using the same logic as the LSTM trainer
- tokenize and pad text
- return the predicted label, confidence, and class probabilities

Key methods:

- `clean_text(text)`: reproduces the LSTM training-time text cleaning
- `predict(text)`: runs inference on one text input

### `TransformerSentimentModel`

Wrapper around the saved transformer model.

Responsibilities:

- load the transformer model from `model/transformer/model`
- load the tokenizer from `model/transformer/tokenizer`
- clean text using the transformer-specific preprocessing
- tokenize text and run transformer inference

Key methods:

- `clean_text(text)`: lighter preprocessing that preserves more context
- `predict(text)`: runs transformer inference for one text input

### `SentimentInferenceService`

Main coordination layer used by the API routes.

Responsibilities:

- find and load the saved models from the root `model` folder
- report backend health and model readiness
- predict one tweet with both models
- parse uploaded JSON payloads
- classify whole uploaded tweet lists

Key methods:

- `_load_optional_model(model_class, model_dir)`: loads a model if artifacts exist
- `health()`: returns whether each model is ready
- `predict_text(text)`: runs both models on one input
- `parse_uploaded_tweets(payload)`: validates and normalizes uploaded JSON
- `classify_uploaded_tweets(payload)`: runs both models on every uploaded tweet

## API Routes

### Health

- `GET /health`
- `GET /api/health`

Returns whether the LSTM and transformer artifacts are loaded successfully.

### Single Text Prediction

- `POST /predict/text`
- `POST /api/predict/text`

Input:

```json
{
  "text": "My flight was delayed again."
}
```

Output:

- original tweet
- LSTM output
- transformer output

### File Prediction

- `POST /predict/file`
- `POST /api/predict/file`

Upload:

- one `.json` file

Behavior:

- validates the JSON
- extracts tweet text from each item
- runs both models
- returns a downloadable JSON file

## How To Run

From the project root:

```powershell
.\.venv\Scripts\activate
cd backend
python app.py
```

Because `app.py` contains a `__main__` block, running it directly starts the Uvicorn server on port `8000`.

## Dependencies

Install from:

```powershell
pip install -r backend\requirements.txt
```

Main backend packages:

- `fastapi`
- `uvicorn`
- `python-multipart`

The backend also depends on the ML stack already installed in the main environment:

- `tensorflow`
- `transformers`
- `numpy`

## Notes

- If one model is missing artifacts, the backend health endpoint will show that.
- Prediction endpoints expect both models to be available.
- The frontend uses `/api/...` routes through the Vite proxy, but the original non-prefixed routes are still available too.
