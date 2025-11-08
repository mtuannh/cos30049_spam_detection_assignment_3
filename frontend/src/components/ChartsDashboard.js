import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { Bar, Pie, Line } from "react-chartjs-2";
import {
    Chart as ChartJS, CategoryScale, LinearScale, BarElement, ArcElement, 
    PointElement, LineElement, Title, Tooltip, Legend, Filler
} from "chart.js";
ChartJS.register(
    CategoryScale, LinearScale, BarElement, ArcElement, 
    PointElement, LineElement, Title, Tooltip, Legend, Filler
);

export default function ChartsDashboard() {
const [core, setCore] = useState(null);
const [pr, setPr] = useState(null);
const [cal, setCal] = useState(null);
// elbow chart removed
const [scores, setScores] = useState(null);
const [predStats, setPredStats] = useState(null);
const [error, setError] = useState("");
const [resetting, setResetting] = useState(false);

useEffect(() => {
(async () => {
    try {
    const [c, p, k, s, ps] = await Promise.all([
        api.charts(), api.prCurve(), api.calibration(), api.kmeansScores(), api.predictionStats()
    ]);
    setCore(c); setPr(p); setCal(k); setScores(s); setPredStats(ps);
    } catch (err) { setError(err.message || "Failed to load charts"); }
})();
}, []);

//prediction stats every 3 seconds to keep the dynamic chart updated
useEffect(() => {
const interval = setInterval(async () => {
    try {
    const ps = await api.predictionStats();
    setPredStats(ps);
    } catch (err) {
    console.error("Failed to update prediction stats:", err);
    }
}, 3000);
return () => clearInterval(interval);
}, []);

// Scroll to pie chart if hash is present in URL
useEffect(() => {
    if (window.location.hash === '#realtime-pie-chart') {
        // Wait for data to load and then scroll
        const scrollToPieChart = () => {
            const element = document.getElementById("realtime-pie-chart");
            if (element) {
                element.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        };
        // Try immediately and also after a delay to handle async data loading
        scrollToPieChart();
        const timeout = setTimeout(scrollToPieChart, 500);
        return () => clearTimeout(timeout);
    }
}, [predStats, core]);

// Reset prediction history
const handleReset = async () => {
    if (!window.confirm("Are you sure you want to reset all prediction history? This action cannot be undone.")) {
        return;
    }
    
    setResetting(true);
    try {
        await api.resetPredictions();
        // Refresh prediction stats immediately
        const ps = await api.predictionStats();
        setPredStats(ps);
    } catch (err) {
        setError(err.message || "Failed to reset predictions");
    } finally {
        setResetting(false);
    }
};

if (error) return <div className="error">{error}</div>;
if (!core) return <div className="card">Loading charts…</div>;

const labelDist = core.label_distribution || { ham: 0, spam: 0 };
const topWords = core.top_spam_words || [];
const histLengths = core.message_length_hist || [];

//build histogram(50 bins)
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
<>
    <h2 style={{ textAlign: 'center', marginBottom: '20px', marginTop: '10px' }}>
        Visualizations of Real-Time Update
    </h2>
    <div className="grid" style={{ display: 'flex', justifyContent: 'center' }} id="realtime-pie-chart">
    {predStats && (
        <section className="card" style={{ maxWidth: '500px', width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h3 style={{ margin: 0 }}>Live Prediction Distribution</h3>
                {predStats.total > 0 && (
                    <button 
                        onClick={handleReset}
                        disabled={resetting}
                        style={{
                            backgroundColor: resetting ? '#999' : '#ff6b6b',
                            color: '#fff',
                            border: 'none',
                            padding: '6px 12px',
                            borderRadius: '4px',
                            cursor: resetting ? 'not-allowed' : 'pointer',
                            fontSize: '12px',
                            fontWeight: 'bold',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            transition: 'background-color 0.3s'
                        }}
                        onMouseOver={(e) => {
                            if (!resetting) e.target.style.backgroundColor = '#ff5252';
                        }}
                        onMouseOut={(e) => {
                            if (!resetting) e.target.style.backgroundColor = '#ff6b6b';
                        }}
                    >
                        <span>🔄</span>
                        {resetting ? 'Resetting...' : 'Reset'}
                    </button>
                )}
            </div>
            {predStats.total > 0 ? (
                <>
                    <div style={{ maxWidth: '350px', margin: '0 auto' }}>
                    <Pie data={{
                        labels: ["Ham", "Spam"],
                        datasets: [{
                            data: [predStats.ham, predStats.spam],
                            backgroundColor: ['rgba(75, 192, 192, 0.8)', 'rgba(255, 99, 132, 0.8)'],
                            borderColor: '#FFFFFF',
                            borderWidth: 2
                        }]
                    }} 
                    options={{
                        responsive: true,
                        plugins: {
                            legend: { 
                                display: true,
                                position: 'bottom'
                            },
                            tooltip: {
                                callbacks: {
                                    label: (context) => {
                                        const label = context.label || '';
                                        const value = context.parsed || 0;
                                        const total = predStats.total;
                                        const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                        return `${label}: ${value} (${percentage}%)`;
                                    }
                                }
                            }
                        }
                    }}
                    />
                    </div>
                    <p style={{ marginTop: 12, textAlign: 'center' }}>
                        <b>Total Predictions:</b> {predStats.total} ·
                        <b> Ham:</b> {predStats.ham} ({predStats.total > 0 ? ((predStats.ham / predStats.total) * 100).toFixed(1) : 0}%) ·
                        <b> Spam:</b> {predStats.spam} ({predStats.total > 0 ? ((predStats.spam / predStats.total) * 100).toFixed(1) : 0}%)
                    </p>
                </>
            ) : (
                <p style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
                    No predictions yet. Make some predictions on the <b>Predict</b> or <b>Batch</b> pages to see the distribution here.
                </p>
            )}
        </section>
    )}
    </div>

    <h2 style={{ textAlign: 'center', marginBottom: '20px', marginTop: '30px' }}>
        Visualizations of the 2025 Dataset
    </h2>
    <div className="grid">
    <section className="card">
    <h3>Spam vs Ham</h3>
    <Pie data={{
        labels: ["Ham", "Spam"],
        datasets: [{
            data: [labelDist.ham, labelDist.spam],
            backgroundColor: ['rgba(255, 255, 255, 0.8)', 'rgba(255, 255, 255, 0.4)'],
            borderColor: '#FFFFFF',
            borderWidth: 1
        }]
    }} />
    </section>

    <section className="card">
    <h3>Top 20 Spam Words</h3>
    <Bar data={{
        labels: topWords.map(t => t[0]),
        datasets: [{
            label: "Frequency",
            data: topWords.map(t => t[1]),
            backgroundColor: 'rgba(255, 255, 255, 0.7)',
            borderColor: '#FFFFFF',
            borderWidth: 1
        }]
    }}
    options={{ responsive: true, plugins: { legend: { display: false } }}} />
    </section>

    <section className="card">
    <h3>Message Length Histogram</h3>
    <Bar data={{
        labels: histLabels,
        datasets: [{
            label: "Count",
            data: counts,
            backgroundColor: 'rgba(255, 255, 255, 0.7)',
            borderColor: '#FFFFFF',
            borderWidth: 1
        }]
    }}
    options={{ responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 0 }}} }} />
    </section>

    {pr && (
<section className="card">
    <h3>Precision–Recall Curve</h3>
    <Line 
    data={{
    datasets: [{
        label: "PR Curve",
        data: pr.recall.map((r, i) => ({ 
        x: parseFloat(r), 
        y: parseFloat(pr.precision[i]) 
        })),
        borderColor: '#FFFFFF',
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
    }]
    }}
    options={{
    responsive: true,
    maintainAspectRatio: true,
    scales: {
        x: { 
        type: 'linear',
        title: { 
            display: true, 
            text: "Recall",
            font: { size: 14 }
        },
        min: 0,
        max: 1,
        ticks: { stepSize: 0.1 }
        },
        y: { 
        type: 'linear',
        title: { 
            display: true, 
            text: "Precision",
            font: { size: 14 }
        },
        min: 0,
        max: 1,
        ticks: { stepSize: 0.1 }
        }
    },
    plugins: {
        legend: { 
        display: true,
        position: 'top'
        },
        tooltip: {
        callbacks: {
            label: (context) => {
            return `Recall: ${context.parsed.x.toFixed(3)}, Precision: ${context.parsed.y.toFixed(3)}`;
            }
        }
        }
    }
    }} 
    />
</section>
)}
</div>
</>
);
}
