import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import SimpleRNN, GRU, LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

output_dir = "Dugi_Sara_RIRSU/6_rekurentne_nevronske_mreze_in_casovne_vrste/obvezni_del"
os.makedirs(output_dir, exist_ok=True)

# 1. Naložimo podatke
file_path = "Dugi_Sara_RIRSU/6_rekurentne_nevronske_mreze_in_casovne_vrste/mbajk.csv"
try:
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    print("Podatki uspešno naloženi!")
except Exception as e:
    print(f"Napaka pri nalaganju datoteke: {e}")

# 2. Sortiranje zapisov glede na čas
data = data.sort_index()
print(data.head())

# 3. Izris grafa vrednosti izposoje koles glede na čas
# Resample data to daily average
data_resampled = data['available_bike_stands'].resample('D').mean()

# Plot daily average
plt.figure(figsize=(12, 6))
plt.plot(data_resampled, label='Daily Average', color='blue')
plt.title('Časovna vrsta - Povprečno število prostih stojal za kolesa (Dnevno)')
plt.xlabel('Čas')
plt.ylabel('Available Bike Stands')
plt.legend()
plt.savefig(os.path.join(output_dir, 'casovna_vrsta_povp_stevilo.png'))
plt.close()

# Moving average for smoothing (7-day window)
rolling_avg = data_resampled.rolling(window=7).mean()

# Plot rolling average for clarity
plt.figure(figsize=(12, 6))
plt.plot(data_resampled, alpha=0.5, label='Daily Average')
plt.plot(rolling_avg, color='red', label='7-Day Moving Average')
plt.title('Časovna vrsta - 7-dnevno drseče povprečje')
plt.xlabel('Čas')
plt.ylabel('Available Bike Stands')
plt.legend()
plt.savefig(os.path.join(output_dir, 'casovna_vrsta_drseco_povprecje.png'))
plt.close()

# 4. Filtriranje ciljnega stolpca
target_column = 'available_bike_stands'
time_series = data[[target_column]]
print("Oblikovana univariatna časovna vrsta:")
print(time_series.head())

# 5. Pretvorba podatkov v numpy array za lažje delo
time_series_values = time_series.values

# 6. Razdelitev podatkov na učno in testno množico
train_size = len(time_series_values) - 1302
train, test = time_series_values[:train_size], time_series_values[train_size:]

print(f"Velikost učne množice: {len(train)}")
print(f"Velikost testne množice: {len(test)}")

# 7. Normalizacija podatkov (Min-Max Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

print("Podatki po normalizaciji:")
print(train_scaled[:5])
print(test_scaled[:5])

# 8. Ustvarjanje časovnih oken za RNN
def create_time_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 10  # Dolžina časovnega okna
X_train, y_train = create_time_windows(train_scaled, window_size)
X_test, y_test = create_time_windows(test_scaled, window_size)

# Preoblikovanje podatkov za RNN [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Oblika učnih podatkov za RNN:")
print(X_train.shape, y_train.shape)

# 9. Gradnja in učenje RNN modela
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 10. Napoved in vrednotenje modela
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverzna transformacija napovedi
train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Izračun napake (RMSE)
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
print(f"RMSE na učnem naboru: {train_rmse:.2f}")
print(f"RMSE na testnem naboru: {test_rmse:.2f}")

# 11. Prikaz napovedi
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test_inv):], y_test_inv, label='Dejanske vrednosti')
plt.plot(data.index[-len(test_pred_inv):], test_pred_inv, label='Napovedi', linestyle='--')
plt.title('Primerjava napovedi in dejanskih vrednosti')
plt.xlabel('Čas')
plt.ylabel('Available Bike Stands')
plt.legend()
plt.savefig(os.path.join(output_dir, 'primerjava_napovedi.png'))
plt.close()

# 8. Priprava podatkov na učenje z okni velikosti 186
def create_time_windows_fixed(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0])  # Okno velikosti 186
        y.append(data[i + window_size, 0])   # Naslednja vrednost v vrsti
    return np.array(X), np.array(y)

# Definicija velikosti okna
window_size = 186  

# Ustvarjanje učnih in testnih podatkov
X_train, y_train = create_time_windows_fixed(train_scaled, window_size)
X_test, y_test = create_time_windows_fixed(test_scaled, window_size)

print(f"Oblika X_train: {X_train.shape}, Oblika y_train: {y_train.shape}")
print(f"Oblika X_test: {X_test.shape}, Oblika y_test: {y_test.shape}")

# Preoblikovanje vhodnih podatkov za RNN v (samples, 1, window_size)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"Preoblikovani X_train: {X_train.shape}")
print(f"Preoblikovani X_test: {X_test.shape}")


# 9. Gradnja treh arhitektur napovednih modelov
# Funkcija za izgradnjo in učenje modela
def build_and_train_model(model_type, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    model = Sequential()
    if model_type == 'RNN':
        model.add(SimpleRNN(32, return_sequences=True, input_shape=(1, X_train.shape[2])))
        model.add(SimpleRNN(32))
    elif model_type == 'GRU':
        model.add(GRU(32, return_sequences=True, input_shape=(1, X_train.shape[2])))
        model.add(GRU(32))
    elif model_type == 'LSTM':
        model.add(LSTM(32, return_sequences=True, input_shape=(1, X_train.shape[2])))
        model.add(LSTM(32))
    else:
        raise ValueError("Unsupported model type. Use 'RNN', 'GRU', or 'LSTM'.")

    # Dense layers
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Izhodni sloj z 1 nevronom

    # Kompilacija modela
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Učenje modela
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test, y_test), verbose=1)
    return model, history

# Funkcija za izris rezultatov učenja
def plot_learning_history(history, model_name):
    plt.figure(figsize=(12, 6))

    # Graf izgube
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()

# Nastavitev parametrov
EPOCHS = 30
BATCH_SIZE = 32

# Gradnja in učenje RNN modela
print("\nGradnja in učenje RNN modela...")
rnn_model, rnn_history = build_and_train_model('RNN', X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)
plot_learning_history(rnn_history, 'RNN Model')

# Gradnja in učenje GRU modela
print("\nGradnja in učenje GRU modela...")
gru_model, gru_history = build_and_train_model('GRU', X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)
plot_learning_history(gru_history, 'GRU Model')

# Gradnja in učenje LSTM modela
print("\nGradnja in učenje LSTM modela...")
lstm_model, lstm_history = build_and_train_model('LSTM', X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)
plot_learning_history(lstm_history, 'LSTM Model')

# Funkcija za napoved in ovrednotenje modela
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, model_name):
    # Napovedi za učne in testne podatke
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Ensure predictions have correct shape before inverse transformation
    train_pred = train_pred.reshape(-1, 1)
    test_pred = test_pred.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Inverzna transformacija napovedi
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)
    train_pred_inv = scaler.inverse_transform(train_pred)
    test_pred_inv = scaler.inverse_transform(test_pred)

    # Izračun metrik
    test_mae = mean_absolute_error(y_test_inv, test_pred_inv)
    test_mse = mean_squared_error(y_test_inv, test_pred_inv)

    # Vizualizacija rezultatov
    plt.figure(figsize=(14, 6))

    # Učna množica in napovedi
    plt.plot(range(len(y_train_inv)), y_train_inv, label='Učna množica', color='blue')
    plt.plot(range(len(y_train_inv)), train_pred_inv, label='Napoved (Učna)', color='cyan', linestyle='--')

    # Resnična vrednost in napovedi testnega seta
    test_start = len(y_train_inv)  # Align test data after training data
    plt.plot(range(test_start, test_start + len(y_test_inv)), y_test_inv, label='Resnična vrednost (Testna)', color='orange')
    plt.plot(range(test_start, test_start + len(test_pred_inv)), test_pred_inv, label='Napoved (Testna)', color='green', linestyle='--')

    # Naslov in oznake
    plt.title(f'Napoved {model_name} modela\nMAE: {test_mae:.2f}, MSE: {test_mse:.2f}')
    plt.xlabel('Časovni koraki')
    plt.ylabel('Število izposojenih koles')
    plt.legend()

    # Shranjevanje grafa
    plot_path = os.path.join(output_dir, f'napoved_{model_name}.png')
    plt.savefig(plot_path)
    print(f"Graf shranjen na: {plot_path}")
    plt.close()

    # Graf residualov (napake)
    residuals = y_test_inv.flatten() - test_pred_inv.flatten()
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label='Reziduali (Napake)', color='red')
    plt.title(f'{model_name} - Reziduali (Napake)')
    plt.xlabel('Časovni koraki')
    plt.ylabel('Razlika (Dejanske - Napovedane)')
    plt.legend()

    # Shranjevanje grafa residualov
    residuals_path = os.path.join(output_dir, f'reziduali_{model_name}.png')
    plt.savefig(residuals_path)
    print(f"Graf rezidualov shranjen na: {residuals_path}")
    plt.close()

    return test_mae, test_mse

# Ovrednotenje vseh modelov
rnn_metrics = evaluate_model(rnn_model, X_train, y_train, X_test, y_test, scaler, 'RNN')
gru_metrics = evaluate_model(gru_model, X_train, y_train, X_test, y_test, scaler, 'GRU')
lstm_metrics = evaluate_model(lstm_model, X_train, y_train, X_test, y_test, scaler, 'LSTM')