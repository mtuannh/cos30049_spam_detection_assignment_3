Spam Email Detection — Assignment 3
1. Backend Setup (FastAPI + ML Model)
   **Install dependencies:**
     cd backend
     pip install -r requirements.txt
   ** Required Libraries**
   (fastapi, uvicorn, scikit-learn, numpy/pandas, joblib, pydantic, collections)
   **Run backend**
   uvicorn app:app --reload --port 8000
2. Frontend Setup
   **Install dependencies:**
     cd frontend
     npm install
3. Configuration for AI Model Integration
   **Model Overview**
   The backend uses a classic Multinomial Naive Bayes classifier trained on text features extracted by TF-IDF Vectorizer.
   **Model Integration Steps**
   The SpamModel class in model_utils.py manages:
      - Training and saving the model to /Models/model.joblib
      - Loading the model at startup (for faster response)
      - Predicting single or batch emails
      - Returning confidence probability and top tokens
   During FastAPI startup, the model loads automatically:
        MODEL = SpamModel()
        MODEL.load()
   API Endpoint:
      - /healt: Check API status and model version
      - /predict: Predict single email
      - /batch_predict: Predict multiple emails
      - /charts-data: Return analytics (spam vs ham, histograms, etc.)
      - /model-report: Return model performance metrics
    The React frontend connects to these routes through /src/services/api.js:
      const API_BASE = "http://localhost:8000";
export const api = {
    predict: async (text) => {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        return res.json();
    },
    chartsData: async () => (await fetch(`${API_BASE}/charts-data`)).json(),
};
4. Analytics Dashboard
   After every prediction:
      Backend appends result into PRED_HISTORY.
      The /charts-data endpoint dynamically aggregates:
      Spam vs Ham counts
      Probability histogram
      Prediction volume over time
      React dashboard (ChartsDashboard.js) auto refreshes upon new predictions.
5. Common commands
   -Run backend: uvicorn app:app --reload --port 8000
   - Run frontend: npm start
   - Install python libs: pip install -r requirements.txt
   - Install React: npm install
   - Access API Docs: http://127.0.0.1:8000/docs
6. Credit
   Frameworks: React 18, FastAPI 0.110+
   Libraries: scikit-learn, numpy, joblib, recharts
   Dataset: Email Spam Collection
   Developed by: Group 9 — COS30049 Computing Technology Innovation Project 
