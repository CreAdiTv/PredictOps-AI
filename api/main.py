from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Load trained model
model = joblib.load(ARTIFACT_DIR / "best_model.pkl")

feature_columns = joblib.load(
    ARTIFACT_DIR / "feature_columns.pkl"
)

app = FastAPI(
    title="PredictOps AI",
    description="Predictive Maintenance API",
    version="1.0"
)


class PredictionInput(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "PredictOps AI API is running"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    try:
        input_data = data.features

        missing = set(feature_columns) - set(input_data)

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {list(missing)}"
            )

        df = pd.DataFrame(
            [[input_data[col] for col in feature_columns]],
            columns=feature_columns
        )

        prediction = float(model.predict(df)[0])

        return {
            "predicted_rul": round(prediction, 2),
            "unit": "operating cycles"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )