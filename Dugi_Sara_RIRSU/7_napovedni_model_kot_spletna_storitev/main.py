import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Optional
import uvicorn
from keras._tf_keras.keras.models import load_model
from sklearn import set_config

app = FastAPI()

#6
model_dir = "Dugi_Sara_RIRSU/6_rekurentne_nevronske_mreze_in_casovne_vrste/models"

scaler = joblib.load(f"{model_dir}/scaler.pkl")
rnn_model = load_model(f"{model_dir}/rnn_model.h5", compile=False)
gru_model = load_model(f"{model_dir}/gru_model.h5", compile=False)
lstm_model = load_model(f"{model_dir}/lstm_model.h5", compile=False)

rnn_model.compile(optimizer="adam", loss="mse", metrics=["mse"])
gru_model.compile(optimizer="adam", loss="mse", metrics=["mse"])
lstm_model.compile(optimizer="adam", loss="mse", metrics=["mse"])

class MBajkModel(BaseModel):
    date: str
    available_bike_stands: float

def create_windows(data, window_size=186):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

#4
MODEL_PATH = 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/scalers/AdaBoost_Regresija.joblib'
SCALER_PATH = 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/scalers/scaler_standard_new.pkl'
SELECTED_FEATURES_PATH = "Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/selected_features.pkl"
ALL_FEATURES_PATH = "Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/all_features.pkl"
model_nal4= joblib.load(MODEL_PATH)
scaler_nal4 = joblib.load(SCALER_PATH)
all_features = joblib.load(ALL_FEATURES_PATH)
selected_features = joblib.load(SELECTED_FEATURES_PATH)

def categorize_hour(hour):
    if 5 <= hour <= 11:
        return 0.7
    elif 12 <= hour <= 16:
        return 0.8
    elif 17 <= hour <= 20:
        return 0.6
    else:
        return 0.1
    
try:
    print("Značilnice, ki jih model pričakuje:")
    expected_features = model_nal4.feature_names_in_
    print(expected_features)
except Exception as e:
    raise RuntimeError(f"Težava pri nalaganju modela ali scalerja: {e}")

class PredictRequest(BaseModel):
    hour: float
    temperature: float
    humidity: float
    wind_speed: float
    visibility: float
    dew_point_temperature: float
    solar_radiation: float
    rainfall: float
    snowfall: float
    seasons: int
    work_hours: int
    day: int
    month: int
    year: int
    temp_humidity_interaction: float
    temp_squared: float
    humidity_inverse: float
    temperature_log: float

required_features = all_features

@app.get("/")
async def root():
    return {"message": "Časovna vrsta napovedi za izposojo koles."}

@app.post("/predict/task4")
async def predict_line(data: List[PredictRequest]):
    try:
        df_bike_data = pd.DataFrame([item.dict() for item in data])

        df_bike_data[['day', 'month', 'year']] = df_bike_data['date'].str.split('/', expand=True)
        df_bike_data['day'] = pd.to_numeric(df_bike_data['day'])
        df_bike_data['month'] = pd.to_numeric(df_bike_data['month'])
        df_bike_data['year'] = pd.to_numeric(df_bike_data['year'])
        df_bike_data = df_bike_data.drop('date', axis=1)

        df_bike_data['temp_humidity_interaction'] = df_bike_data['temperature'] * df_bike_data['humidity']
        df_bike_data['temp_squared'] = df_bike_data['temperature'] ** 2
        df_bike_data['humidity_inverse'] = 1 / (df_bike_data['humidity'] + 1e-3)
        df_bike_data['temperature_log'] = np.log1p(df_bike_data['temperature'])

        df_bike_data['solar_radiation'] = np.log1p(df_bike_data['solar_radiation'])
        df_bike_data['rainfall'] = np.log1p(df_bike_data['rainfall'])
        df_bike_data['snowfall'] = np.log1p(df_bike_data['snowfall'])

        columns_to_scale = ['temperature', 'humidity', 'wind_speed', 'dew_point_temperature']
        df_bike_data[columns_to_scale] = scaler_nal4.transform(df_bike_data[columns_to_scale])
        df_selected = df_bike_data.reindex(columns=selected_features, fill_value=0)
        predictions = model_nal4.predict(df_selected)
        predictions_original_scale = np.expm1(predictions)

        return {"predictions": predictions_original_scale.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/task6")
async def predict_task6(data: List[MBajkModel], model_type: str = "RNN"):
    try:
        data = sorted([d.dict() for d in data], key=lambda x: x['date'])
        bike_stands = np.array([d['available_bike_stands'] for d in data]).reshape(-1, 1)

        if len(bike_stands) < 186:
            raise HTTPException(status_code=400, detail="Vhodni podatki morajo vsebovati vsaj 186 vrednosti.")

        bike_stands_normalized = scaler.transform(bike_stands)

        try:
            X_data = create_windows(bike_stands_normalized, window_size=186)
            if X_data.shape[0] == 0:
                raise ValueError("Not enough data points to create the required window size of 186.")
            X_data = X_data.reshape(X_data.shape[0], 1, X_data.shape[1])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error in data preparation: {str(e)}")

        if model_type.upper() == "RNN":
            predictions = rnn_model.predict(X_data)
        elif model_type.upper() == "GRU":
            predictions = gru_model.predict(X_data)
        elif model_type.upper() == "LSTM":
            predictions = lstm_model.predict(X_data)
        else:
            raise HTTPException(status_code=400, detail="Neveljaven model_type. Uporabite 'RNN', 'GRU', ali 'LSTM'.")

        predictions_original = scaler.inverse_transform(predictions)
        prediction_value = float(predictions_original[-1][0])

        return {"model_type": model_type, "prediction": prediction_value}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)