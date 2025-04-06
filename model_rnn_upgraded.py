import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1. Veri Yükleme
df = pd.read_csv("temizlenmis_veri.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# 2. Girdi & Hedef
X = df.drop(columns=['date', 'T'])
y = df[['T']]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 3. Sliding Window - horizon destekli
def create_sliding_window(X_scaled, y_scaled, window_size=5, horizon=3):
    X_win, y_win = [], []
    for i in range(len(X_scaled) - window_size - horizon + 1):
        X_win.append(X_scaled[i:i+window_size])
        y_win.append(y_scaled[i+window_size + horizon - 1])  # horizon gün sonrası
    return np.array(X_win), np.array(y_win)

# 4. Model Oluşturucu
def build_rnn_model(input_shape, units=64):
    model = Sequential()
    model.add(SimpleRNN(units=units, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 5. Walk-Forward Validation Fonksiyonu
def walk_forward_validation(X, y, window_size=5, horizon=3, n_test=100, retrain_epochs=3, batch_size=64):
    actuals, predictions = [], []
    train_X = X[:-n_test]
    train_y = y[:-n_test]
    test_X = X[-n_test:]
    test_y = y[-n_test:]
    
    for i in range(len(test_X)):
        model = build_rnn_model(input_shape=(window_size, train_X.shape[2]))
        model.fit(train_X, train_y, epochs=retrain_epochs, batch_size=batch_size, verbose=0)
        
        yhat = model.predict(test_X[i:i+1])[0][0]
        predictions.append(yhat)
        actuals.append(test_y[i][0])
        
        # Update train set with the current test sample
        train_X = np.concatenate([train_X, test_X[i:i+1]], axis=0)
        train_y = np.concatenate([train_y, test_y[i:i+1]], axis=0)
    
    return np.array(actuals), np.array(predictions)

# 6. Parametreler
window_size = 5
horizon = 3
n_test = 100
epochs = 3
batch_size = 64

# 7. Sliding window verisi
X_window, y_window = create_sliding_window(X_scaled, y_scaled, window_size, horizon)

# 8. Walk-Forward Uygulama
actuals_scaled, preds_scaled = walk_forward_validation(
    X_window, y_window,
    window_size=window_size,
    horizon=horizon,
    n_test=n_test,
    retrain_epochs=epochs,
    batch_size=batch_size
)

# 9. Geri ölçekleme ve metrikler
y_true = scaler_y.inverse_transform(actuals_scaled.reshape(-1, 1))
y_pred = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"[Walk-Forward] Horizon={horizon} -> MSE: {mse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

# 10. Grafik
plt.figure(figsize=(12, 6))
plt.plot(y_true, label="Gerçek Sıcaklık", marker='o', linestyle='-')
plt.plot(y_pred, label="Tahmin Edilen Sıcaklık", marker='x', linestyle='--')
plt.title(f"Walk-Forward: Gerçek vs Tahmin (Horizon={horizon})")
plt.xlabel("Adım")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.grid()
plt.show()
