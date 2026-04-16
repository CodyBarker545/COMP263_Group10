import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }

  return `${Math.round(value * 100)}%`;
}

export default function App() {
  const [tweet, setTweet] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [health, setHealth] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    async function loadHealth() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        if (isMounted) {
          setHealth(data);
        }
      } catch {
        if (isMounted) {
          setHealth({
            lstm_ready: false,
            transformer_ready: false,
            unreachable: true,
          });
        }
      }
    }

    loadHealth();

    return () => {
      isMounted = false;
    };
  }, []);

  const handleAnalyze = async (event) => {
    event.preventDefault();

    if (!tweet.trim()) {
      setError("Enter a tweet before running sentiment analysis.");
      setResult(null);
      return;
    }

    setIsAnalyzing(true);
    setError("");
    setUploadMessage("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict/text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: tweet }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Unable to analyze the tweet right now.");
      }

      setResult(data);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setSelectedFileName(file.name);
    setError("");
    setUploadMessage("");
    setSelectedFile(file);
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) {
      setError("Choose a JSON file before submitting it.");
      return;
    }

    setIsUploading(true);
    setError("");
    setUploadMessage("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict/file`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = "Unable to process the uploaded JSON file.";
        try {
          const errorData = await response.json();
          detail = errorData.detail || detail;
        } catch {
          // Keep the fallback detail if the response cannot be parsed.
        }
        throw new Error(detail);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const contentDisposition = response.headers.get("Content-Disposition");
      const matchedName = contentDisposition?.match(/filename="(.+)"/);
      const downloadName = matchedName?.[1] || "classified_tweets.json";

      anchor.href = url;
      anchor.download = downloadName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(url);

      setUploadMessage(`Processed ${selectedFile.name} and downloaded ${downloadName}.`);
      setSelectedFile(null);
      setSelectedFileName("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleClear = () => {
    setTweet("");
    setResult(null);
    setError("");
    setUploadMessage("");
    setSelectedFileName("");
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <main className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <section className="hero-card">
        <div className="eyebrow">Airline Sentiment Detection</div>
        <h1>Understand how a passenger tweet feels in one click.</h1>
        <p className="hero-copy">
          Enter a sentence about an airline or upload a JSON file of tweets and
          get sentiment predictions from both trained models.
        </p>

        <div className="health-row">
          <span className={`health-chip ${health?.lstm_ready ? "healthy" : "unhealthy"}`}>
            LSTM {health?.lstm_ready ? "ready" : "not ready"}
          </span>
          <span className={`health-chip ${health?.transformer_ready ? "healthy" : "unhealthy"}`}>
            Transformer {health?.transformer_ready ? "ready" : "not ready"}
          </span>
          {health?.unreachable ? (
            <span className="health-chip unhealthy">Backend offline</span>
          ) : null}
        </div>

        <form className="sentiment-form" onSubmit={handleAnalyze}>
          <label className="input-label" htmlFor="tweet-input">
            Passenger tweet
          </label>
          <textarea
            id="tweet-input"
            className="tweet-input"
            rows="5"
            placeholder="Example: My flight was delayed again and the support team still hasn't helped."
            value={tweet}
            onChange={(event) => setTweet(event.target.value)}
          />

          <div className="form-actions">
            <button className="analyze-button" type="submit" disabled={isAnalyzing}>
              {isAnalyzing ? "Analyzing..." : "Analyze Sentiment"}
            </button>

            <button
              className="upload-button"
              type="button"
              disabled={isUploading}
              onClick={() => fileInputRef.current?.click()}
            >
              Choose Tweet JSON
            </button>

            <button
              className="submit-upload-button"
              type="button"
              disabled={isUploading || !selectedFile}
              onClick={handleUploadSubmit}
            >
              {isUploading ? "Submitting..." : "Submit JSON"}
            </button>

            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept=".json,application/json"
              onChange={handleFileChange}
            />

            <button className="ghost-button" type="button" onClick={handleClear}>
              Clear
            </button>
          </div>

          <div className="helper-row">
            <span className="helper-text">
              Upload a JSON array of tweet strings or tweet objects with a
              `text` or `tweet` field.
            </span>
            {selectedFileName ? (
              <span className="file-chip">{selectedFileName}</span>
            ) : null}
          </div>
        </form>

        {error ? <p className="status-message error-message">{error}</p> : null}
        {uploadMessage ? (
          <p className="status-message success-message">{uploadMessage}</p>
        ) : null}

        <section className="result-panel">
          {result ? (
            <>
              <div className="result-badge">Prediction Ready</div>
              <p className="result-rationale">
                The API classified this tweet using both saved airline
                sentiment models.
              </p>

              <div className="model-grid">
                <article className="model-card">
                  <span className="result-label">LSTM Model</span>
                  <strong className="model-value">{result.lstm.label}</strong>
                  <span className="confidence-pill">
                    Confidence {formatPercent(result.lstm.confidence)}
                  </span>
                </article>

                <article className="model-card">
                  <span className="result-label">Transformer Model</span>
                  <strong className="model-value">
                    {result.transformer.label}
                  </strong>
                  <span className="confidence-pill">
                    Confidence {formatPercent(result.transformer.confidence)}
                  </span>
                </article>
              </div>
            </>
          ) : (
            <>
              <div className="result-badge pending">Awaiting input</div>
              <p className="placeholder-copy">
                Paste a sentence above for live model scoring, or upload a JSON
                file to download tweet-by-tweet classifications from both
                models.
              </p>
            </>
          )}
        </section>
      </section>
    </main>
  );
}
