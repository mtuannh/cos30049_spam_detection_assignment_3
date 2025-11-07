import React, { useState } from "react";

import { 
    NavLink, Routes, Route 
} 
from "react-router-dom";
import PredictForm from "./components/PredictForm";
import BatchPredict from "./components/BatchPredict";
import ChartsDashboard from "./components/ChartsDashboard";
import ModelReport from "./components/ModelReport";

export default function App() {
    const [text, setText] = useState("");
    return (
    <div className="container">
        <header>
        <h1>Spam Detection — Assignment 3</h1>
        <nav>
            <NavLink to="/" end>Predict</NavLink>
            <NavLink to="/batch">Batch</NavLink>
            <NavLink to="/analytics">Analytics</NavLink>
            <NavLink to="/report">Model Report</NavLink>
        </nav>
        </header>

        <main>
        <Routes>
            <Route path="/" element={<PredictForm text={text} setText={setText} />} />
            <Route path="/batch" element={<BatchPredict />} />
            <Route path="/analytics" element={<ChartsDashboard />} />
            <Route path="/report" element={<ModelReport />} />
        </Routes>
        </main>

        <footer>
        <small>FastAPI at http://localhost:8000 • v1.1.0</small>
        </footer>
    </div>
    );
}
