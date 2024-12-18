import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import SimpleRNN, GRU, LSTM, Dense

output_dir = "Dugi_Sara_RIRSU/6_rekurentne_nevronske_mreze_in_casovne_vrste/dodatni_del"
os.makedirs(output_dir, exist_ok=True)

file_path = "Dugi_Sara_RIRSU/6_rekurentne_nevronske_mreze_in_casovne_vrste/mbajk.csv"
try:
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    print("Podatki uspešno naloženi!")
except Exception as e:
    print(f"Napaka pri nalaganju datoteke: {e}")

# Dodajanje dodatnih značilnic
features = ["available_bike_stands", "apparent_temperature", "dew_point", 
            "precipitation_probability", "surface_pressure"]

# Imputacija manjkajočih vrednosti
imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Random Forest za izboljšano imputacijo manjkajočih vrednosti
for column in features:
    if data[column].isnull().sum() > 0:
        rf = RandomForestRegressor()
        non_null_data = data.dropna()
        rf.fit(non_null_data.drop(column, axis=1), non_null_data[column])
        data[column] = data[column].fillna(rf.predict(data.drop(column, axis=1)))

# Normalizacija podatkov
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[features])

train_size = len(data_scaled) - 1302
train, test = data_scaled[:train_size], data_scaled[train_size:]

# Priprava časovnih oken za multivariatno učenje
def create_multivariate_time_windows(data, window_size, target_index):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_index])
    return np.array(X), np.array(y)

window_size = 60
target_index = 0  # Indeks za 'available_bike_stands'

X_train, y_train = create_multivariate_time_windows(train, window_size, target_index)
X_test, y_test = create_multivariate_time_windows(test, window_size, target_index)

# Preoblikovanje za modele RNN
X_train = X_train.transpose(0, 2, 1)
X_test = X_test.transpose(0, 2, 1)

print(f"X_train oblika: {X_train.shape}, y_train oblika: {y_train.shape}")
print(f"X_test oblika: {X_test.shape}, y_test oblika: {y_test.shape}")

# Gradnja in učenje
def build_and_train_model_multivariate(model_type, X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
    model = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])  # (features, timesteps)

    if model_type == 'RNN':
        model.add(SimpleRNN(16, return_sequences=True, input_shape=input_shape))
        model.add(SimpleRNN(16))
    elif model_type == 'GRU':
        model.add(GRU(16, return_sequences=True, input_shape=input_shape))
        model.add(GRU(16))
    elif model_type == 'LSTM':
        model.add(LSTM(16, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(16))
    else:
        raise ValueError("Unsupported model type. Use 'RNN', 'GRU', or 'LSTM'.")

    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1)
    return model, history

# ovrednotenje modela
def evaluate_model_multivariate(model, X_test, y_test, scaler, model_name):
    test_pred = model.predict(X_test)
    test_pred_inv = scaler.inverse_transform(np.c_[test_pred, np.zeros((len(test_pred), 4))])[:, 0]
    y_test_inv = scaler.inverse_transform(np.c_[y_test, np.zeros((len(y_test), 4))])[:, 0]

    num_steps_to_show = 200
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv[-num_steps_to_show:], label='Resnična vrednost', color='orange', linewidth=2)
    plt.plot(test_pred_inv[-num_steps_to_show:], label='Napoved', linestyle='--', color='green', linewidth=2)
    plt.title(f'{model_name} Model - Zadnjih {num_steps_to_show} korakov\nMAE: {mean_absolute_error(y_test_inv, test_pred_inv):.2f}, MSE: {mean_squared_error(y_test_inv, test_pred_inv):.2f}')
    plt.xlabel('Časovni koraki')
    plt.ylabel('Število prostih stojal za kolesa')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'napoved_multivariatno_{model_name}.png'))
    plt.close()

# Treniranje modelov
print("\nGradnja in učenje RNN modela...")
rnn_model, rnn_history = build_and_train_model_multivariate('RNN', X_train, y_train, X_test, y_test)

print("\nGradnja in učenje GRU modela...")
gru_model, gru_history = build_and_train_model_multivariate('GRU', X_train, y_train, X_test, y_test)

print("\nGradnja in učenje LSTM modela...")
lstm_model, lstm_history = build_and_train_model_multivariate('LSTM', X_train, y_train, X_test, y_test)

# Ovrednotenje
evaluate_model_multivariate(rnn_model, X_test, y_test, scaler, 'RNN')
evaluate_model_multivariate(gru_model, X_test, y_test, scaler, 'GRU')
evaluate_model_multivariate(lstm_model, X_test, y_test, scaler, 'LSTM')

print("Vsi modeli so ovrednoteni in rezultati shranjeni.")