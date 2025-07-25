# 📊 ML Churn Prediction Pipeline

This is a complete end-to-end **Churn Prediction API** project built with:

* 🧠 Machine Learning (Random Forest)
* 🔍 SHAP for interpretability
* ⚙️ FastAPI for serving the model
* 🚀 Render for deployment

Live API: [https://ml-churn-pipeline.onrender.com](https://ml-churn-pipeline.onrender.com)

---

## 📌 Project Structure

```
.
├── app.py                  # FastAPI app
├── model.pkl              # Trained model
├── requirements.txt       # Python dependencies
├── Dockerfile             # (Optional) for Docker deployment
├── README.md              # Project documentation
└── notebooks/
    └── training.ipynb     # Model training and SHAP analysis
```

---

## 🧠 Step 1: Train the Model

1. Open `notebooks/training.ipynb`
2. Run all cells to:

   * Clean & preprocess data
   * Train `RandomForestClassifier`
   * Save model as `model.pkl`
   * Generate SHAP explanations

> <img width="1000" height="800" alt="Screenshot 2025-07-25 at 5 52 15 PM" src="https://github.com/user-attachments/assets/9bb02d81-869d-4684-a24f-5f4c8f6e4950" />
<img width="1000" height="800" alt="Screenshot 2025-07-25 at 6 06 13 PM" src="https://github.com/user-attachments/assets/5e163c6c-68e2-43a1-b7fe-415de52da4df" />



---

## ⚙️ Step 2: Build the API (FastAPI)

**File:** `app.py`

* Loads model from `model.pkl`
* Exposes two routes:

  * `GET /` → returns health status
  * `POST /predict` → accepts features and returns prediction

Visit after local run: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Step 3: Test Locally

```bash
uvicorn app:app --reload
```

Then test with Swagger UI or cURL:

**Test Body:**

```json
{
  "features": [2.5, 0, 1, 3.8]
}
```

**cURL Command:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"features": [2.5, 0, 1, 3.8]}'
```

---

## 🚀 Step 4: Deploy on Render

1. Go to [Render.com](https://render.com/)
2. Create a new Web Service
3. Connect to your GitHub repo
4. Set:

   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn app:app --host=0.0.0.0 --port=10000`
5. Use Free Web Service tier and deploy.

> After deployment, your live app will be accessible at something like:
> `https://ml-churn-pipeline.onrender.com`

Use `/docs` at the end to access Swagger UI.

---

## 📦 Sample Live Test (Render)

```bash
curl -X 'POST' \
  'https://ml-churn-pipeline.onrender.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"features": [2.5, 0, 1, 3.8]}'
```

Response:

```json
{
  "prediction": 1
}
```

---

## 📈 SHAP Explainability

SHAP values are generated in the training notebook for feature importance.

> Add a screenshot of your SHAP summary or waterfall plot if available.

---

## ✅ Project Highlights

* 🧠 ML Model: Random Forest
* 🔍 Explainability with SHAP
* ⚙️ FastAPI for serving predictions
* 🌐 Deployed on Render (free tier)

---

## 🙌 Contributing

Feel free to fork and improve! PRs welcome.

---

## 📄 License

MIT License
