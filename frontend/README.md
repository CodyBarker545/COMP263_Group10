# Frontend

This folder contains the React single-page application for Airline Sentiment Detection.

Main source files:

- [src/App.jsx]
- [src/styles.css]
- [src/main.jsx]

## Purpose

The frontend provides a polished one-page interface where the user can:

- enter a single tweet in a textbox
- submit it for prediction
- choose a JSON file of tweets
- submit the JSON file separately
- download the classified JSON result
- see backend and model readiness badges

## Files

- [index.html]
  HTML entry point used by Vite
- [package.json]
  Frontend scripts and dependencies
- [vite.config.js]
  Vite config with a proxy from `/api` to the backend
- [src/main.jsx]
  React app bootstrap
- [src/App.jsx]
  Main application logic and UI
- [src/styles.css]
  Page styling, layout, and responsive behavior

## Main Component

### `App`

This is the entire frontend application.

Responsibilities:

- manage the textarea value
- manage API results and error states
- show whether the backend models are ready
- let the user select a JSON file
- submit the selected JSON file only when the submit button is pressed
- trigger download of the backend-generated classified JSON

## Important Functions Inside `App.jsx`

### `formatPercent(value)`

Formats a probability/confidence number into a user-friendly percentage string.

### `handleAnalyze(event)`

Triggered when the user submits the single-tweet form.

Responsibilities:

- validate that text exists
- call `POST /api/predict/text`
- store the returned predictions
- show an error if the backend request fails

### `handleFileChange(event)`

Triggered when the user chooses a file.

Responsibilities:

- store the selected file in state
- show the chosen filename
- does not upload automatically

### `handleUploadSubmit()`

Triggered only when the user presses the JSON submit button.

Responsibilities:

- validate that a file has been selected
- send the selected file to `POST /api/predict/file`
- receive the generated JSON result
- trigger browser download of the returned file

### `handleClear()`

Resets:

- tweet text
- prediction result
- error messages
- upload status
- selected filename

## Styling

The page styling is defined in [styles.css]
It controls:

- layout
- button styling
- upload controls
- status messages
- model result cards
- health badges
- responsive behavior for smaller screens

## API Integration

The frontend uses:

```js
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";
```

That means:

- by default, development requests go through the Vite proxy
- if needed, a custom backend URL can be provided with `VITE_API_BASE_URL`

The proxy is configured in [vite.config.js]

## How To Install

From the project root:

```powershell
cd frontend
npm install
```

## How To Run

Start the backend first.

Then run:

```powershell
cd frontend
npm run dev
```

For a production build:

```powershell
npm run build
```

## How To Use

### Single tweet

1. Type a tweet into the textbox.
2. Press `Analyze Sentiment`.
3. The app displays both the LSTM and transformer predictions.

### JSON upload

1. Press `Choose Tweet JSON`.
2. Select a valid `.json` file.
3. Press `Submit JSON`.
4. The backend processes the file.
5. The browser downloads a new classified JSON file.

## Accepted JSON Shapes

Example 1:

```json
["tweet one", "tweet two"]
```

Example 2:

```json
{
  "tweets": [
    { "text": "My flight was delayed again." },
    { "tweet": "The staff was very helpful." }
  ]
}
```
