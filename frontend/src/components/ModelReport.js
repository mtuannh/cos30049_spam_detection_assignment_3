import React, { useEffect, useState } from "react";
import { api } from "../services/api";

export default function ModelReport() {
const [metrics, setMetrics] = useState(null);
const [scores, setScores] = useState(null);
const [health, setHealth] = useState(null);
const [error, setError] = useState("");

useEffect(() => {
(async () => {
    try {
    const [m, s, h] = await Promise.all([api.metrics(), api.kmeansScores(), api.health()]);
    setMetrics(m); setScores(s); setHealth(h);
    } catch (err) { setError(err.message || "Failed to load report"); }
})();
}, []);

if (error) return <div className="error">{error}</div>;
if (!metrics || !scores || !health) return <div className="card">Loading…</div>;

const fmt = (x) => (typeof x === "number" ? x.toFixed(3) : x);

return (
<section className="card">
    <h2>Model Report</h2>
    <div className="grid2">
    <div>
        <h4>Classifier (TF-IDF + MultinomialNB)</h4>
        <ul>
        <li>Accuracy: <b>{fmt(metrics.accuracy)}</b></li>
        <li>Precision: <b>{fmt(metrics.precision)}</b></li>
        <li>Recall: <b>{fmt(metrics.recall)}</b></li>
        <li>F1: <b>{fmt(metrics.f1)}</b></li>
        </ul>
    </div>
    <div>
        <h4>KMeans (k=2) – Unsupervised</h4>
        <ul>
        <li>Silhouette: <b>{fmt(scores.silhouette)}</b></li>
        <li>V-measure: <b>{fmt(scores.v_measure)}</b></li>
        </ul>
    </div>
    </div>
    <details>
    <summary>System</summary>
    <pre>{JSON.stringify(health, null, 2)}</pre>
    </details>
</section>
);
}
