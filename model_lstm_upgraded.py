import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==== Ayarlanabilir Parametreler ====
window_size = 5
# Temel eğitim için epochs, batch_size, units parametreleri walk-forward içinde kullanılmayacak,
# walk-forward kısmında retrain_epochs kullanılacak.
retrain_epochs = 3  
batch_size = 64
units = 64
target_column = 'T'
horizon = 3
n_test = 100  # Walk-forward için test adım sayısı

# ==== 1. Veri Yükleme ====
df = pd.read_csv("temizlenmis_veri.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# ==== 2. Girdi ve Hedef Ayır ====
X = df.drop(columns=['date', target_column])
y = df[[target_column]]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ==== 3. Sliding Window Oluştur ====
def create_sliding_window(X_data, y_data, window_size, horizon):
    X_win, y_win = [], []
    for i in range(len(X_data) - window_size - horizon + 1):
        X_win.append(X_data[i:i+window_size])
        y_win.append(y_data[i+window_size + horizon - 1])
    return np.array(X_win), np.array(y_win)

X_window, y_window = create_sliding_window(X_scaled, y_scaled, window_size, horizon)

# ==== 4. Eğitim/Test Ayır ====
# Walk-forward için tüm veri üzerinden bir sliding window oluşturulduktan sonra test seti en son n_test pencere olarak belirleniyor.
X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.2, shuffle=False)

# ==== 5. Upgraded LSTM Modeli için Walk-Forward Validation ====
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=units, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def walk_forward(model_builder, X, y, n_test, retrain_epochs=3, batch_size=64):
    preds, actuals = [], []
    train_X, train_y = X[:-n_test], y[:-n_test]
    test_X, test_y = X[-n_test:], y[-n_test:]
    
    for i in range(len(test_X)):
        # Her adımda yeni model sıfırdan oluşturuluyor
        model = model_builder((train_X.shape[1], train_X.shape[2]))
        model.fit(train_X, train_y, epochs=retrain_epochs, batch_size=batch_size, verbose=0)
        
        pred = model.predict(test_X[i:i+1])[0][0]
        preds.append(pred)
        actuals.append(test_y[i])
        
        # Yeni test örneği eğitim setine ekleniyor
        train_X = np.concatenate([train_X, test_X[i:i+1]], axis=0)
        train_y = np.concatenate([train_y, test_y[i:i+1]], axis=0)
        
    return np.array(actuals), np.array(preds)

actuals_scaled, preds_scaled = walk_forward(build_lstm, X_window, y_window, n_test, retrain_epochs, batch_size)

# ==== 6. Geri Ölçekleme ve Performans Metriği Hesaplama ====
actuals = scaler_y.inverse_transform(actuals_scaled.reshape(-1, 1))
preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))

mse_wf = mean_squared_error(actuals, preds)
mae_wf = mean_absolute_error(actuals, preds)
r2_wf = r2_score(actuals, preds)
print(f"[Walk-Forward - Horizon={horizon}] MSE: {mse_wf:.5f}, MAE: {mae_wf:.5f}, R²: {r2_wf:.5f}")

# ==== 7. Grafik (Walk-Forward Sonuçları) ====
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Gerçek Sıcaklık", marker='o')
plt.plot(preds, label="Tahmin Edilen Sıcaklık", marker='x', linestyle='--')
plt.title(f"Walk-Forward Sonuçları (Horizon={horizon})")
plt.xlabel("Adım")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.grid(True)
plt.show()
