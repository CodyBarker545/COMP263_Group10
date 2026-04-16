# ML

This folder contains the notebooks, dataset copy, and training scripts for the two sentiment models.

Files:

- [preprocessing.ipynb]
- [transformer_model.ipynb]
- [lstm_trainer.py]
- [transformer_trainer.py]
- [Tweets.csv]

## Purpose

This folder is where the models are trained.

It contains:

- notebook versions used for experimentation and exploration
- OOP script versions used for repeatable training and artifact saving
- the local dataset file used during training

## Notebooks

### `preprocessing.ipynb`

Notebook version of the LSTM pipeline.

Main flow:

- load dataset
- explore sentiment distribution
- clean tweet text
- encode labels
- tokenize and pad sequences
- train an LSTM classifier
- evaluate performance with metrics and plots

### `transformer_model.ipynb`

Notebook version of the transformer pipeline.

Main flow:

- load dataset
- clean tweets with lighter preprocessing
- tokenize with DistilBERT tokenizer
- fine-tune a transformer classifier
- evaluate with metrics and plots

## Script: `lstm_trainer.py`

Main class:

### `AirlineSentimentLSTMTrainer`

This class turns the LSTM notebook into a runnable script.

Responsibilities:

- locate the dataset
- clean tweet text
- split train/test data
- tokenize and pad sequences
- compute class weights
- build and train the Keras LSTM model
- evaluate the model
- save artifacts into `model/lstm`

Key methods:

- `_set_seed()`
  Sets Python, NumPy, and TensorFlow seeds
- `_resolve_dataset_path()`
  Uses `ml/Tweets.csv` if present, otherwise falls back to Kaggle download
- `clean_text(text)`
  Applies the LSTM-specific text cleaning
- `load_dataset()`
  Loads CSV data and creates cleaned text and labels
- `split_dataset(tweets)`
  Creates training and test splits
- `prepare_sequences(X_train, X_test)`
  Builds the tokenizer and padded sequences
- `build_model()`
  Creates the Keras LSTM model
- `compute_class_weights(y_train)`
  Balances the classes during training
- `evaluate(...)`
  Builds a metrics dictionary
- `save_artifacts(...)`
  Saves model, tokenizer, label encoder, config, history, and metrics
- `run()`
  Executes the full training pipeline

Run it with:

```powershell
cd ml
python lstm_trainer.py
```

## Script: `transformer_trainer.py`

Main class:

### `AirlineSentimentTransformerTrainer`

This class turns the transformer notebook into a runnable script.

Responsibilities:

- locate the dataset
- clean text for transformer input
- split train/test data
- tokenize with DistilBERT tokenizer
- fine-tune the transformer model
- evaluate results
- save artifacts into `model/transformer`

Key methods:

- `_set_seed()`
  Sets deterministic seeds
- `_resolve_dataset_path()`
  Finds the local CSV or downloads through Kaggle
- `clean_text(text)`
  Applies lighter transformer preprocessing
- `load_dataset()`
  Loads CSV data and creates labels
- `split_dataset(tweets)`
  Creates training and test splits
- `encode_text(texts)`
  Converts text into transformer tensors
- `build_model()`
  Builds and compiles the TensorFlow DistilBERT classifier
- `evaluate(...)`
  Creates a metrics summary
- `save_artifacts(...)`
  Saves model, tokenizer, label encoder, config, history, and metrics
- `run()`
  Executes the full transformer training pipeline

Run it with:

```powershell
cd ml
python transformer_trainer.py
```

## Dataset

The local dataset file is:

- [Tweets.csv]

This is the Twitter Airline Sentiment dataset used for all training and evaluation.

## Training Output

The training scripts save into:

- [model/lstm]
- [model/transformer]

Those saved artifacts are later loaded by the backend API.
