Spam Email Detection — Assignment 3

1. Virtual Environment Setup
   **Prerequisites**
     - Python 3.8+ installed on the system
     - Node.js 14+ and npm installed

   **Creating and Activating Virtual Environment**

   ## Windows
   **Navigate to the project root directory**
     - cd cos30049_spam_detection_assignment_3-1

   **Create virtual environment**
     - python -m venv venv

   **Fix PowerShell execution policy (if needed)**
     - Set-ExecutionPolicy Bypass -Scope Process

   **Activate virtual environment (PowerShell)**
     - .\venv\Scripts\Activate.ps1

   **Activate virtual environment (Command Prompt)**
     - .\venv\Scripts\activate

   ## macOS/Linux
   **Navigate to the project root directory**
     - cd cos30049_spam_detection_assignment_3-1

   **Create virtual environment**
     - python -m venv venv

   **Activate virtual environment**
     - source venv/bin/activate

2. Backend Setup (FastAPI + ML Model)
   **Install dependencies:**
     - cd backend
     - pip install -r requirements.txt

   **Required Libraries**
   (fastapi, uvicorn, scikit-learn, numpy/pandas, joblib, pydantic, collections)

   **Run backend**
   - uvicorn app:app --reload --port 8000

3. Frontend Setup (Open a new terminal)

   **Install dependencies:**
     - cd cos30049_spam_detection_assignment_3-1
     - cd frontend
     - npm install
     - npm start

   **Notice**
     - If you see any notification like this "9 vulnerabilities (3 moderate, 6 high)" in the terminal after running "npm install", just ignore it then straightforward to "npm start". 

4. Configuration for AI Model Integration

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

5. Analytics Dashboard
   After every prediction:
      Backend appends result into PRED_HISTORY.
      The /charts-data endpoint dynamically aggregates:
      Spam vs Ham counts
      Probability histogram
      Prediction volume over time
      React dashboard (ChartsDashboard.js) auto refreshes upon new predictions.

6. Common commands
   - Run backend: uvicorn app:app --reload --port 8000
   - Run frontend: npm start
   - Install python libs: pip install -r requirements.txt
   - Install React: npm install
   - Access API Docs: http://127.0.0.1:8000/docs

7. Credit
   Frameworks: React 18, FastAPI 0.110+
   Libraries: scikit-learn, numpy, joblib, recharts
   Dataset: Email Spam Collection
   Developed by: Group 9 — COS30049 Computing Technology Innovation Project 
