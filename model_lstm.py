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
epochs = 10
batch_size = 64
units = 64
target_column = 'T'

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
def create_sliding_window(X_data, y_data, window_size):
    X_win, y_win = [], []
    for i in range(len(X_data) - window_size):
        X_win.append(X_data[i:i+window_size])
        y_win.append(y_data[i+window_size])
    return np.array(X_win), np.array(y_win)

X_window, y_window = create_sliding_window(X_scaled, y_scaled, window_size)

# ==== 4. Eğitim/Test Ayır ====
X_train, X_test, y_train, y_test = train_test_split(
    X_window, y_window, test_size=0.2, shuffle=False
)

# ==== 5. LSTM Model Tanımı ====
model = Sequential()
model.add(LSTM(units=units, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# ==== 6. Eğit ====
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

# ==== 7. Test Tahminleri ====
y_pred_scaled = model.predict(X_test)
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)
y_test_inv = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
print(f"[Test Sonucu] MSE: {mse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

# ==== 8. Grafikler ====
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label="Eğitim Kaybı")
plt.plot(history.history['val_loss'], label="Doğrulama Kaybı")
plt.title("LSTM Eğitim Süreci")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Gerçek Sıcaklık", alpha=0.8)
plt.plot(y_pred_inv, label="Tahmin Edilen Sıcaklık", alpha=0.7)
plt.title("Gerçek vs Tahmin (LSTM)")
plt.xlabel("Test Örneği")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.show()
