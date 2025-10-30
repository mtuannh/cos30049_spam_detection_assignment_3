const BASE = "http://localhost:8000";


async function getJSON(path) {
    const res = await fetch(`${BASE}${path}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

async function postJSON(path, body) {
    const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export const api = {
    health: () => getJSON("/health"),
    predict: (text) => postJSON("/predict", { text }),
    batchPredict: (texts) => postJSON("/batch_predict", { texts }),
    metrics: () => getJSON("/metrics"),
    charts: () => getJSON("/charts-data"),
    prCurve: () => getJSON("/pr-curve"),
    calibration: () => getJSON("/calibration"),
    elbow: () => getJSON("/kmeans/elbow"),
    kmeansScores: () => getJSON("/kmeans/scores"),
};
