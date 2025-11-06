const API_BASE = "http://127.0.0.1:8000"; 

function showTab(id) {
  document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

async function predictOne() {
  const text = document.getElementById("singleText").value.trim();
  const resultDiv = document.getElementById("singleResult");
  resultDiv.innerHTML = "<em>Detecting...</em>";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const data = await res.json();

    const pct = (data.confidence * 100).toFixed(2);
    const color = data.label === "spam" ? "red" : "green";
    resultDiv.innerHTML = `
      <strong style="color:${color}">${data.label.toUpperCase()}</strong>
      <br>Confidence: ${pct}% 
      <br>P(spam): ${(data.prob_spam*100).toFixed(1)}%, P(ham): ${(data.prob_ham*100).toFixed(1)}%
    `;
  } catch (err) {
    resultDiv.innerHTML = `<span style="color:red">Error: ${err}</span>`;
  }
}

async function predictBatch() {
  const raw = document.getElementById("batchText").value.trim();
  const lines = raw.split("\n").filter(x => x.trim() !== "");
  const resultDiv = document.getElementById("batchResult");

  if (lines.length === 0) {
    resultDiv.innerHTML = "<em>Please enter at least one email.</em>";
    return;
  }

  resultDiv.innerHTML = "<em>Processing batch...</em>";

  const payload = {
    items: lines.map((txt, idx) => ({ id: `email_${idx+1}`, text: txt }))
  };

  try {
    const res = await fetch(`${API_BASE}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    let html = `<table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">
      <tr><th>ID</th><th>Label</th><th>Confidence</th><th>P(spam)</th><th>P(ham)</th></tr>`;

    data.results.forEach(r => {
      const color = r.label === "spam" ? "red" : "green";
      html += `<tr>
        <td>${r.id}</td>
        <td style="color:${color};font-weight:bold">${r.label}</td>
        <td>${(r.confidence*100).toFixed(1)}%</td>
        <td>${(r.prob_spam*100).toFixed(1)}%</td>
        <td>${(r.prob_ham*100).toFixed(1)}%</td>
      </tr>`;
    });

    html += `</table><br>
    <strong>Summary:</strong>
    <pre>${JSON.stringify(data.summary, null, 2)}</pre>`;

    resultDiv.innerHTML = html;
  } catch (err) {
    resultDiv.innerHTML = `<span style="color:red">Error: ${err}</span>`;
  }
}

let pieChart, histChart, lineChart;

async function loadAnalytics() {
  const res = await fetch(`${API_BASE}/analytics`);
  const data = await res.json();

  // PIE CHART
  const pieCtx = document.getElementById('chartPie').getContext('2d');
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(pieCtx, {
    type: 'pie',
    data: {
      labels: ['Spam', 'Ham'],
      datasets: [{
        data: [data.totals.spam, data.totals.ham],
        backgroundColor: ['#e74c3c', '#27ae60']
      }]
    },
    options: {
      plugins: { title: { display: true, text: 'Spam vs Ham Ratio' } }
    }
  });

  // HISTOGRAM
  const histCtx = document.getElementById('chartHist').getContext('2d');
  if (histChart) histChart.destroy();
  histChart = new Chart(histCtx, {
    type: 'bar',
    data: {
      labels: data.confidence_hist.map(x => x.bucket),
      datasets: [{
        label: 'Confidence Count',
        data: data.confidence_hist.map(x => x.count),
        backgroundColor: '#3498db'
      }]
    },
    options: {
      plugins: { title: { display: true, text: 'Confidence Histogram' } }
    }
  });

  // LINE CHART
  const lineCtx = document.getElementById('chartLine').getContext('2d');
  if (lineChart) lineChart.destroy();
  lineChart = new Chart(lineCtx, {
    type: 'line',
    data: {
      labels: data.last_24h.map(x => x.hour),
      datasets: [
        { label: 'Spam', data: data.last_24h.map(x => x.spam), borderColor: '#e74c3c', fill: false },
        { label: 'Ham', data: data.last_24h.map(x => x.ham), borderColor: '#27ae60', fill: false }
      ]
    },
    options: {
      plugins: { title: { display: true, text: 'Predictions per Hour (24h)' } }
    }
  });
}


async function loadReport() {
  const res = await fetch(`${API_BASE}/model/report`);
  const data = await res.json();

  let html = `<h3>Model Info</h3>
  <ul>
    <li><strong>Name:</strong> ${data.model.name}</li>
    <li><strong>Vectorizer:</strong> ${data.model.vectorizer}</li>
    <li><strong>Trained At:</strong> ${data.model.trained_at}</li>
    <li><strong>Threshold:</strong> ${data.threshold}</li>
    <li><strong>Vocab Size:</strong> ${data.feature_info.vocab_size}</li>
  </ul>`;

  html += `<h3>Metrics</h3><table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">`;
  for (const [k, v] of Object.entries(data.metrics)) {
    html += `<tr><td>${k}</td><td>${v}</td></tr>`;
  }
  html += `</table>`;

  html += `<h3>Confusion Matrix</h3>
  <pre>${JSON.stringify(data.confusion_matrix, null, 2)}</pre>`;

  document.getElementById("reportContent").innerHTML = html;
}

showTab('predict');
