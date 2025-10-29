import React, { useState } from "react";
import { api } from "../services/api";

export default function BatchPredict() {
const [raw, setRaw] = useState("");
const [rows, setRows] = useState([]);
const [loading, setLoading] = useState(false);
const [error, setError] = useState("");

const toList = (txt) =>
txt.split("\n").map((s) => s.trim()).filter((s) => s.length > 0).slice(0, 200);

const runBatch = async () => {
setError("");
const items = toList(raw);
if (items.length === 0) { setError("Please enter at least one line."); return; }
setLoading(true);
try {
    const r = await api.batchPredict(items);
    const results = r.results || [];
    setRows(items.map((t, i) => ({ text: t, ...(results[i] || {}) })));
} catch (err) {
    setError(err.message || "Batch failed");
} finally {
    setLoading(false);
}
};

return (
<section className="card">
    <h2>Batch Prediction</h2>
    <p>Enter one message per line (max 200 lines).</p>
    <textarea rows="8" value={raw} onChange={(e) => setRaw(e.target.value)} placeholder="Line 1&#10;Line 2&#10;…" />
    <div className="actions">
    <button onClick={runBatch} disabled={loading}>{loading ? "Predicting…" : "Run batch"}</button>
    <button className="secondary" onClick={() => { setRaw(""); setRows([]); setError(""); }}>Clear</button>
    </div>
    {error && <div className="error">{error}</div>}

    {rows.length > 0 && (
    <div className="table-wrap">
        <table>
        <thead>
            <tr><th>#</th><th>Text</th><th>Label</th><th>Spam Prob.</th></tr>
        </thead>
        <tbody>
            {rows.map((r, i) => (
            <tr key={i}>
                <td>{i + 1}</td>
                <td className="mono">{r.text}</td>
                <td>{r.label ? "SPAM" : "HAM"}</td>
                <td>{(r.probability * 100).toFixed(2)}%</td>
            </tr>
            ))}
        </tbody>
        </table>
    </div>
    )}
</section>
);
}
