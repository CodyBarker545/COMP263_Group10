# Airline Sentiment Detection

This project classifies airline-related tweets as `positive`, `neutral`, or `negative` using two trained models:

- an LSTM model built with TensorFlow/Keras
- a DistilBERT transformer model served through TensorFlow Transformers

It also includes:

- training code in the `ml` folder
- a FastAPI backend in the `backend` folder
- a React frontend in the `frontend` folder

## What The App Does

The app supports two user workflows:

1. Type one airline tweet into the textbox and get predictions from both models.
2. Upload a JSON file of tweets and download a new JSON file that contains:
   - the original tweet
   - the LSTM model output
   - the transformer model output

## Installation

### Python environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r backend\requirements.txt
```

Notes:

- `requirements.txt` contains the ML stack such as TensorFlow, Transformers, scikit-learn, and notebook tools.
- `backend/requirements.txt` contains the API-specific packages such as FastAPI and Uvicorn.

### Frontend environment

From the project root:

```powershell
cd frontend
npm install
```

## Training The Models

The two OOP training scripts are in the `ml` folder:

- [lstm_trainer.py]
- [transformer_trainer.py]

Run them from the project root or the `ml` folder:

```powershell
cd ml
python lstm_trainer.py
python transformer_trainer.py
```

They save artifacts into:

- [model/lstm](
- [model/transformer]

## Running The App

### Start the backend

From the project root:

```powershell
.\.venv\Scripts\activate
cd backend
python app.py
```

The API will run on:

```text
http://127.0.0.1:8000
```

Useful endpoints:

- `GET /health`
- `GET /api/health`
- `POST /predict/text`
- `POST /api/predict/text`
- `POST /predict/file`
- `POST /api/predict/file`

### Start the frontend

Open another terminal:

```powershell
cd frontend
npm run dev
```

The frontend runs through Vite and proxies `/api` requests to the backend.

## JSON Upload Format

The backend accepts either:

```json
["tweet one", "tweet two"]
```

or:

```json
{
  "tweets": [
    { "text": "My flight was delayed again." },
    { "tweet": "The crew was very helpful today." }
  ]
}
```

Each object should include one of:

- `text`
- `tweet`
- `content`
- `message`

## End-To-End Flow

1. Train and save the models with the scripts in `ml`.
2. Start the backend from `backend/app.py`.
3. Start the frontend from the `frontend` folder.
4. Enter a tweet or upload a JSON file.
5. The frontend calls the backend.
6. The backend loads the saved models from `model` and returns predictions.

## Important Current Behavior

The backend only reports a model as ready if its saved artifacts are present and load correctly. If one model folder is missing required files, the health check will show that and prediction requests that require both models will fail until the missing artifacts are restored.
