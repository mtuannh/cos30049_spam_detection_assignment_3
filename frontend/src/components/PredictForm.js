import React, { useState } from "react";
import { api } from "../services/api";

export default function PredictForm() {
const [text, setText] = useState("");
const [loading, setLoading] = useState(false);
const [result, setResult] = useState(null);
const [error, setError] = useState("");

const onSubmit = async (e) => {
e.preventDefault();
setError("");
setResult(null);

const clean = text.trim();
if (clean.length < 2) {
    setError("Please enter at least 2 characters.");
    return;
}
setLoading(true);
try {
    const r = await api.predict(clean);
    setResult(r);
} catch (err) {
    setError(err.message || "Prediction failed");
} finally {
    setLoading(false);
}
};

const labelBadge = (label) =>
label ? <span className="badge spam">SPAM</span> : <span className="badge ham">HAM</span>;

return (
<section className="card">
    <h2>Single Prediction</h2>
    <form onSubmit={onSubmit} className="form">
    <textarea
        rows="5"
        placeholder="Type a message to classify…"
        value={text}
        onChange={(e) => setText(e.target.value)}
    />
    <div className="actions">
        <button type="submit" disabled={loading}>{loading ? "Predicting…" : "Predict"}</button>
        <button type="button" className="secondary" onClick={() => { setText(""); setResult(null); setError(""); }}>
        Clear
        </button>
    </div>
    </form>

    {error && <div className="error">{error}</div>}

    {result && (
    <div className="result">
        <div>{labelBadge(result.label)}</div>
        <p>Probability (spam): <b>{(result.probability * 100).toFixed(2)}%</b></p>
        {result.top_tokens?.length > 0 && (
        <p>Top tokens: {result.top_tokens.join(", ")}</p>
        )}
    </div>
    )}
</section>
);
}
