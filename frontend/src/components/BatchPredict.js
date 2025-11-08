import React, { useState, useEffect } from "react";
import { api } from "../services/api";

export default function BatchPredict() {
const [raw, setRaw] = useState("");
const [rows, setRows] = useState([]);
const [loading, setLoading] = useState(false);
const [error, setError] = useState("");
const [lineCount, setLineCount] = useState(0);
const [validLines, setValidLines] = useState(0);
const [warnings, setWarnings] = useState([]);
const [previewLines, setPreviewLines] = useState([]);

// Real-time validation and feedback
useEffect(() => {
    const lines = raw.split("\n");
    const trimmedLines = lines.map(s => s.trim()).filter(s => s.length > 0);
    
    setLineCount(lines.length);
    setValidLines(trimmedLines.length);
    
    // Preview first 3 non-empty lines
    setPreviewLines(trimmedLines.slice(0, 3));
    
    // Generate warnings
    const newWarnings = [];
    
    if (trimmedLines.length > 200) {
        newWarnings.push(`Too many lines (${trimmedLines.length}). Only first 200 will be processed.`);
    }
    
    const emptyLines = lines.length - trimmedLines.length;
    if (emptyLines > 0) {
        newWarnings.push(`${emptyLines} empty line(s) will be skipped.`);
    }
    
    const tooShort = trimmedLines.filter(l => l.length < 2).length;
    if (tooShort > 0) {
        newWarnings.push(`${tooShort} line(s) are too short (< 2 characters) and may not predict accurately.`);
    }
    
    const tooLong = trimmedLines.filter(l => l.length > 500).length;
    if (tooLong > 0) {
        newWarnings.push(`${tooLong} line(s) exceed 500 characters - consider splitting longer messages.`);
    }
    
    setWarnings(newWarnings);
}, [raw]);

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

// Calculate statistics from results
const getStats = () => {
    if (rows.length === 0) return null;
    const spamCount = rows.filter(r => r.label === 1).length;
    const hamCount = rows.filter(r => r.label === 0).length;
    const avgSpamProb = rows.reduce((sum, r) => sum + (r.probability || 0), 0) / rows.length;
    return { spamCount, hamCount, avgSpamProb };
};

const stats = getStats();

return (
<section className="card">
    <h2>Batch Prediction</h2>
    <p>Enter one message per line (max 200 lines).</p>
    
    <div style={{ position: 'relative' }}>
        <textarea 
            rows="8" 
            value={raw} 
            onChange={(e) => setRaw(e.target.value)} 
            placeholder="Line 1&#10;Line 2&#10;…"
            style={{
                borderColor: validLines === 0 && raw.length > 0 ? '#ff6b6b' : '#ddd',
                borderWidth: validLines === 0 && raw.length > 0 ? '2px' : '1px'
            }}
        />
        
        {/* Real-time Stats */}
        <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            marginTop: '5px', 
            fontSize: '12px',
            color: '#666'
        }}>
            <span>
                <strong>Total lines:</strong> {lineCount}
            </span>
            <span style={{ color: validLines > 200 ? '#ff6b6b' : validLines > 0 ? '#4caf50' : '#999' }}>
                <strong>Valid lines:</strong> {validLines > 200 ? '200 (max)' : validLines}
            </span>
        </div>
    </div>
    
    {/* Warnings */}
    {warnings.length > 0 && (
        <div style={{ 
            backgroundColor: '#fff3cd', 
            border: '1px solid #ffc107', 
            borderRadius: '4px', 
            padding: '10px', 
            marginTop: '10px' 
        }}>
            <strong style={{ color: '#856404' }}>Warnings:</strong>
            <ul style={{ margin: '5px 0 0 20px', padding: 0 }}>
                {warnings.map((warn, i) => (
                    <li key={i} style={{ color: '#856404', fontSize: '13px' }}>{warn}</li>
                ))}
            </ul>
        </div>
    )}
    
    <div className="actions">
        <button 
            onClick={runBatch} 
            disabled={loading || validLines === 0}
            style={{ 
                opacity: (loading || validLines === 0) ? 0.6 : 1,
                cursor: (loading || validLines === 0) ? 'not-allowed' : 'pointer'
            }}
        >
            {loading ? "Predicting…" : `Run batch ${validLines > 0 ? `(${Math.min(validLines, 200)} messages)` : ''}`}
        </button>
        <button className="secondary" onClick={() => { setRaw(""); setRows([]); setError(""); }}>Clear</button>
    </div>
    
    {error && <div className="error">{error}</div>}
    
    {stats && (
        <div style={{ 
            backgroundColor: 'transparent', 
            border: '2px solid rgba(255, 255, 255, 0.5)', 
            borderRadius: '4px', 
            padding: '15px', 
            marginTop: '15px',
            marginBottom: '15px'
        }}>
            <h3 style={{ marginTop: 0, marginBottom: '10px', color: '#fff' }}>Batch Results Summary</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px', color: '#fff' }}>
                <div>
                    <strong>Total Processed:</strong> {rows.length}
                </div>
                <div style={{ color: '#ff6b6b' }}>
                    <strong>Spam:</strong> {stats.spamCount} ({((stats.spamCount / rows.length) * 100).toFixed(1)}%)
                </div>
                <div style={{ color: '#4caf50' }}>
                    <strong>Ham:</strong> {stats.hamCount} ({((stats.hamCount / rows.length) * 100).toFixed(1)}%)
                </div>
                <div>
                    <strong>Avg Spam Probability:</strong> {(stats.avgSpamProb * 100).toFixed(2)}%
                </div>
            </div>
        </div>
    )}

    {rows.length > 0 && (
    <div className="table-wrap">
        <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Text</th>
                <th>Label</th>
                <th>Spam Prob.</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {rows.map((r, i) => {
                const confidence = r.probability > 0.9 || r.probability < 0.1 ? 'Very High' : 
                                r.probability > 0.7 || r.probability < 0.3 ? 'High' : 
                                r.probability > 0.6 || r.probability < 0.4 ? 'Medium' : 'Low';
                return (
                <tr key={i} style={{ 
                    backgroundColor: r.label ? 'rgba(255, 99, 99, 0.1)' : 'rgba(99, 255, 99, 0.1)' 
                }}>
                    <td>{i + 1}</td>
                    <td className="mono" style={{ maxWidth: '400px', wordBreak: 'break-word' }}>
                        {r.text}
                    </td>
                    <td>
                        <span style={{ 
                            padding: '3px 8px', 
                            borderRadius: '3px', 
                            backgroundColor: r.label ? '#fee' : '#efe',
                            color: r.label ? '#c33' : '#3c3',
                            fontWeight: 'bold',
                            fontSize: '12px'
                        }}>
                            {r.label ? "SPAM" : "HAM"}
                        </span>
                    </td>
                    <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                            <div style={{ 
                                flex: 1, 
                                height: '8px', 
                                backgroundColor: '#eee', 
                                borderRadius: '4px',
                                overflow: 'hidden'
                            }}>
                                <div style={{ 
                                    width: `${r.probability * 100}%`, 
                                    height: '100%', 
                                    backgroundColor: r.probability > 0.5 ? '#ff6b6b' : '#4caf50',
                                    transition: 'width 0.3s'
                                }}></div>
                            </div>
                            <span style={{ fontSize: '12px', minWidth: '50px' }}>
                                {(r.probability * 100).toFixed(2)}%
                            </span>
                        </div>
                    </td>
                    <td style={{ fontSize: '12px', color: '#666' }}>{confidence}</td>
                </tr>
                );
            })}
        </tbody>
        </table>
    </div>
    )}
</section>
);
}
