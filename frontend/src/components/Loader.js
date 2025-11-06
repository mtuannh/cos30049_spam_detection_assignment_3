import React from "react";

export default function Loader({ text = "Loading..." }) {
return (
<div style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    padding: "40px 0",
    color: "#94a3b8"
}}>
    <div className="spinner" style={{
    border: "4px solid #1e293b",
    borderTop: "4px solid #38bdf8",
    borderRadius: "50%",
    width: "36px",
    height: "36px",
    animation: "spin 1s linear infinite"
    }}></div>
    <p style={{ marginTop: 10 }}>{text}</p>
    <style>
    {`
        @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
        }
    `}
    </style>
</div>
);
}
