import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { Bar, Pie, Line } from "react-chartjs-2";
import {
    Chart as ChartJS, CategoryScale, LinearScale, BarElement, ArcElement, PointElement, LineElement, Tooltip, Legend,
} from "chart.js";
ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, PointElement, LineElement, Tooltip, Legend);

export default function ChartsDashboard() {
const [core, setCore] = useState(null);
const [pr, setPr] = useState(null);
const [cal, setCal] = useState(null);
const [elbow, setElbow] = useState(null);
const [scores, setScores] = useState(null);
const [error, setError] = useState("");

useEffect(() => {
(async () => {
    try {
    const [c, p, k, e, s] = await Promise.all([
        api.charts(), api.prCurve(), api.calibration(), api.elbow(), api.kmeansScores()
    ]);
    setCore(c); setPr(p); setCal(k); setElbow(e); setScores(s);
    } catch (err) { setError(err.message || "Failed to load charts"); }
})();
}, []);

if (error) return <div className="error">{error}</div>;
if (!core) return <div className="card">Loading charts…</div>;

const labelDist = core.label_distribution || { ham: 0, spam: 0 };
const topWords = core.top_spam_words || [];
const histLengths = core.message_length_hist || [];

// build histogram (50 bins)
const bins = 50;
const maxLen = Math.max(1, ...histLengths);
const step = Math.ceil(maxLen / bins);
const counts = new Array(bins).fill(0);
histLengths.forEach((L) => {
const idx = Math.min(bins - 1, Math.floor(L / step));
counts[idx] += 1;
});
const histLabels = counts.map((_, i) => `${i * step}–${(i + 1) * step}`);

return (
<div className="grid">
    <section className="card">
    <h3>Spam vs Ham</h3>
    <Pie data={{
        labels: ["Ham", "Spam"],
        datasets: [{ data: [labelDist.ham, labelDist.spam] }]
    }} />
    </section>

    <section className="card">
    <h3>Top 20 Spam Words</h3>
    <Bar data={{
        labels: topWords.map(t => t[0]),
        datasets: [{ label: "Frequency", data: topWords.map(t => t[1]) }]
    }}
    options={{ responsive: true, plugins: { legend: { display: false } }}} />
    </section>

    <section className="card">
    <h3>Message Length Histogram</h3>
    <Bar data={{
        labels: histLabels,
        datasets: [{ label: "Count", data: counts }]
    }}
    options={{ responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 0 }}} }} />
    </section>

    {pr && (
    <section className="card">
        <h3>Precision–Recall Curve</h3>
        <Line data={{
        labels: pr.recall,
        datasets: [{ label: "Precision", data: pr.precision }]
        }}
        options={{ parsing: false, scales: { x: { title: { display: true, text: "Recall" }}, y: { title: { display: true, text: "Precision" }}}}} />
    </section>
    )}

    {scores && (
    <section className="card">
        <h3>KMeans Scores (k=2)</h3>
        <Bar
        data={{
            labels: ["Silhouette", "V-measure"],
            datasets: [
            {
                label: "Score",
                data: [
                scores.silhouette ?? 0,
                scores.v_measure ?? 0
                ],
            },
            ],
        }}
        options={{
            responsive: true,
            scales: {
            y: { beginAtZero: true, max: 1, title: { display: true, text: "Score (0–1)" } },
            x: { title: { display: true, text: "Metric" } },
            },
            plugins: {
            legend: { display: false },
            tooltip: { enabled: true },
            },
        }}
        />
        {/* hiển thị thêm số raw nhỏ gọn */}
        <p style={{ marginTop: 8 }}>
        Silhouette: <b>{(scores.silhouette ?? 0).toFixed(3)}</b> ·
        V-measure: <b>{(scores.v_measure ?? 0).toFixed(3)}</b>
        </p>
    </section>
)}
</div>
);
}
