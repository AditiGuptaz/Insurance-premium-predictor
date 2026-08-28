# Insurance Premium Category Predictor

An end-to-end machine learning application that predicts a person's insurance premium category from their profile. A **RandomForest** model is trained on insurance data, served through a **FastAPI** backend, and used via an interactive **Streamlit** frontend.

This project covers the full ML lifecycle: feature engineering, model training, serialization, API serving, and a user-facing frontend.

---

## What it does

Enter a person's details (age, weight, height, income, city, occupation, smoker status) and the app predicts their insurance premium category (e.g. Low / Medium / High).

```
User fills the form (Streamlit)
      ↓
Frontend sends the data to the FastAPI backend
      ↓
Backend engineers features (BMI, age group, lifestyle risk, city tier)
      ↓
Trained RandomForest model predicts the premium category
      ↓
Prediction shown back in the browser
```

---

## Architecture

Three parts working together:

```
┌────────────────┐      HTTP POST      ┌────────────────┐
│   Streamlit    │  ───────────────▶   │    FastAPI     │
│   frontend     │                     │    backend     │
│  (frontend.py) │  ◀───────────────   │    (app.py)    │
└────────────────┘   JSON prediction   └───────┬────────┘
                                               │
                                        ┌──────▼──────┐
                                        │  model.pkl  │
                                        │ (RandomForest)
                                        └─────────────┘
```

---

## Key features

- **End-to-end ML pipeline** — from raw data to a served, usable prediction
- **Feature engineering** — derives BMI, age group, lifestyle risk, and city tier from raw inputs
- **Consistent train/serve features** — the same feature engineering used in training is reproduced at prediction time via Pydantic **computed fields**, so the model always receives the format it expects
- **Input validation** — Pydantic enforces valid ranges (age, height, weight) and restricted values (occupation) before prediction
- **REST API** — FastAPI backend with an interactive `/docs` page
- **Interactive frontend** — Streamlit form for easy, non-technical use

---

## Tech stack

`scikit-learn` (RandomForest) · `FastAPI` · `Pydantic` · `Streamlit` · `pandas` · `Uvicorn` · `Python`

---

## Project structure

| File | Role |
|---|---|
| `fastapi_ml_model.ipynb` | Trains the model — feature engineering, pipeline, RandomForest, saves `model.pkl` |
| `app.py` | FastAPI backend — loads the model, validates input, engineers features, serves predictions |
| `frontend.py` | Streamlit frontend — input form that calls the API and displays the prediction |
| `model.pkl` | The trained model (RandomForest pipeline with preprocessing) |
| `insurance.csv` | The training dataset |

---

## The features the model uses

Raw inputs are transformed into engineered features before prediction:

| Engineered feature | Derived from |
|---|---|
| `bmi` | weight / height² |
| `age_group` | age (young / adult / middle_aged / senior) |
| `lifestyle_risk` | smoker status + BMI (low / medium / high) |
| `city_tier` | city (tier 1 / 2 / 3 Indian cities) |
| `income_lpa` | used directly |
| `occupation` | used directly |

This transformation happens identically in training (`fastapi_ml_model.ipynb`) and serving (`app.py`) — a key requirement for a model to work correctly in production.

---

## Run it

This app has two parts that run at the same time, in **two separate terminals**.

**Prerequisites:**
```bash
pip install fastapi uvicorn streamlit scikit-learn pandas
```

**Terminal 1 — start the API backend:**
```bash
uvicorn app:app --reload
```
The API runs at `http://localhost:8000` (interactive docs at `/docs`).

**Terminal 2 — start the frontend:**
```bash
streamlit run frontend.py
```
The app opens at `http://localhost:8501`. Fill in the form and click **Predict Premium Category**.

---

## How to use the API directly

You can also call the API without the frontend:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "weight": 75, "height": 1.7, "income_lpa": 12, "smoker": false, "city": "Mumbai", "occupation": "private_job"}'
```

Response:
```json
{ "predicted_category": "Medium" }
```

---

## What this project demonstrates

- Training an ML model with a proper **scikit-learn pipeline** (preprocessing + RandomForest)
- **Feature engineering** and reproducing it consistently at serving time
- **Model serialization** and loading (pickle)
- Building a **REST API** to serve a model (FastAPI, Pydantic validation, computed fields)
- Creating an **interactive frontend** (Streamlit) that consumes the API
- Understanding the full path from a trained model to a usable application

---

## Note on model versions

The model is saved with pickle, which ties it to the scikit-learn version used to train it. If you retrain or run into a version-mismatch error when loading `model.pkl`, ensure the serving environment uses the same scikit-learn version the model was trained with. (This is why production ML pins library versions or uses containers like Docker.)
