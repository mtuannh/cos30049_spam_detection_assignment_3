import React, { useState, useEffect } from "react";
import { api } from "../services/api";

export default function PredictForm({ text, setText }) {
const [loading, setLoading] = useState(false);
const [result, setResult] = useState(null);
const [error, setError] = useState("");
const [validationErrors, setValidationErrors] = useState([]);
const [charCount, setCharCount] = useState(0);
const [wordCount, setWordCount] = useState(0);
const [suggestions, setSuggestions] = useState([]);

// Real-time validation and feedback
useEffect(() => {
    const errors = [];
    const clean = text.trim();
    
    setCharCount(text.length);
    setWordCount(clean ? clean.split(/\s+/).length : 0);
    
    // Comprehensive validation
    if (clean.length === 0) {
        // No error when empty, just waiting for input
    } else if (clean.length < 2) {
        errors.push("Message too short (minimum 2 characters)");
    } else if (clean.length > 5000) {
        errors.push("Message too long (maximum 5000 characters)");
    }
    
    // Check for suspicious patterns
    const suggestions = [];
    if (clean.match(/(.)\1{4,}/)) {
        suggestions.push("Contains repeated characters - may affect accuracy");
    }
    if (clean.match(/[^\x00-\x7F]{10,}/)) {
        suggestions.push("Contains many non-ASCII characters - model trained on English");
    }
    if (clean.split(/\s+/).length < 3 && clean.length > 20) {
        suggestions.push("Very few words detected - consider adding more context");
    }
    if (clean.match(/https?:\/\//gi)?.length > 3) {
        suggestions.push("Multiple URLs detected - common in spam");
    }
    if (clean.match(/[A-Z]{10,}/)) {
        suggestions.push("Excessive capitalization detected - common in spam");
    }
    if (clean.match(/(!{2,}|\?{2,})/)) {
        suggestions.push("Multiple exclamation/question marks - common in spam");
    }
    
    setValidationErrors(errors);
    setSuggestions(suggestions);
}, [text]);

const onSubmit = async (e) => {
e.preventDefault();
setError("");
setResult(null);

const clean = text.trim();
if (validationErrors.length > 0) {
    setError("Please fix validation errors before submitting.");
    return;
}
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

// Helper to get color based on character count
const getCharCountColor = () => {
    if (charCount === 0) return '#999';
    if (charCount < 2) return '#ff6b6b';
    if (charCount > 5000) return '#ff6b6b';
    if (charCount > 4500) return '#ffa500';
    return '#4caf50';
};

return (
<section className="card">
    <h2>Single Prediction</h2>
    <form onSubmit={onSubmit} className="form">
    <div style={{ position: 'relative' }}>
        <textarea
            rows="5"
            placeholder="Type a message to classify…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            style={{
                borderColor: validationErrors.length > 0 ? '#ff6b6b' : '#ddd',
                borderWidth: validationErrors.length > 0 ? '2px' : '1px'
            }}
        />
        <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            marginTop: '5px', 
            fontSize: '12px',
            color: '#666'
        }}>
            <span>
                <strong>Words:</strong> {wordCount}
            </span>
            <span style={{ color: getCharCountColor() }}>
                <strong>Characters:</strong> {charCount}/5000
            </span>
        </div>
    </div>
    
    {/* Validation Errors */}
    {validationErrors.length > 0 && (
        <div style={{ 
            backgroundColor: '#fee', 
            border: '1px solid #fcc', 
            borderRadius: '4px', 
            padding: '10px', 
            marginTop: '10px' 
        }}>
            <strong style={{ color: '#c33' }}>Validation Issues:</strong>
            <ul style={{ margin: '5px 0 0 20px', padding: 0 }}>
                {validationErrors.map((err, i) => (
                    <li key={i} style={{ color: '#c33' }}>{err}</li>
                ))}
            </ul>
        </div>
    )}
    
    {/* Dynamic Suggestions */}
    {suggestions.length > 0 && validationErrors.length === 0 && (
        <div style={{ 
            backgroundColor: 'transparent', 
            border: '2px solid rgba(255, 193, 7, 0.6)', 
            borderRadius: '4px', 
            padding: '10px', 
            marginTop: '10px' 
        }}>
            <strong style={{ color: '#ffc107' }}>💡 Insights:</strong>
            <ul style={{ margin: '5px 0 0 20px', padding: 0 }}>
                {suggestions.map((sug, i) => (
                    <li key={i} style={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '13px' }}>{sug}</li>
                ))}
            </ul>
        </div>
    )}
    
    <div className="actions">
        <button 
            type="submit" 
            disabled={loading || validationErrors.length > 0}
            style={{ 
                opacity: (loading || validationErrors.length > 0) ? 0.6 : 1,
                cursor: (loading || validationErrors.length > 0) ? 'not-allowed' : 'pointer'
            }}
        >
            {loading ? "Predicting…" : "Predict"}
        </button>
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
        <div style={{ 
            marginTop: '10px', 
            padding: '10px', 
            backgroundColor: 'transparent',
            borderRadius: '4px',
            border: `2px solid ${result.label ? 'rgba(255, 107, 107, 0.6)' : 'rgba(76, 175, 80, 0.6)'}`
        }}>
            <strong style={{ color: '#fff' }}>Confidence Level:</strong> <span style={{ color: '#fff' }}>{
                result.probability > 0.9 || result.probability < 0.1 ? 
                'Very High' : 
                result.probability > 0.7 || result.probability < 0.3 ? 
                'High' : 
                result.probability > 0.6 || result.probability < 0.4 ? 
                'Medium' : 
                'Low'
            }</span>
        </div>
        {result.top_tokens?.length > 0 && (
        <p style={{ marginTop: '10px' }}>
            <strong style={{ color: '#fff' }}>Key indicators:</strong> {result.top_tokens.map((token, i) => (
                <span key={i} style={{ 
                    display: 'inline-block',
                    backgroundColor: 'rgba(255, 255, 255, 0.2)',
                    color: '#fff',
                    padding: '2px 8px',
                    margin: '2px',
                    borderRadius: '3px',
                    fontSize: '12px',
                    border: '1px solid rgba(255, 255, 255, 0.3)'
                }}>{token}</span>
            ))}
        </p>
        )}
    </div>
    )}
</section>
);
}
