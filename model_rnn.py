import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# --- PARAMETRELER ---
window_size = 5     # geçmiş kaç gün?
epochs = 10             # eğitim döngüsü
batch_size = 64         # batch boyutu

# --- VERİ YÜKLEME ---
df = pd.read_csv("temizlenmis_veri.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

X = df.drop(columns=['date', 'T'])
y = df[['T']]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# --- SLIDING WINDOW ---
def create_sliding_window(X, y, window_size=5):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

X_win, y_win = create_sliding_window(X_scaled, y_scaled, window_size)

# --- EĞİTİM/TEST AYIRIMI ---
X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, test_size=0.2, shuffle=False)

# --- RNN MODELİ ---
model = Sequential()
model.add(SimpleRNN(units=64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# --- EĞİTİM ---
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

# --- TAHMİN ---
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)

# --- METRİKLER ---
mse = mean_squared_error(y_test_real, y_pred)
mae = mean_absolute_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred)
print(f"\n[Test Sonucu] MSE: {mse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

# --- GRAFİKLER ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title("Model Eğitim Süreci")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_test_real, label='Gerçek Sıcaklık', alpha=0.8)
plt.plot(y_pred, label='Tahmin Edilen Sıcaklık', alpha=0.7)
plt.title("Gerçek vs Tahmin")
plt.xlabel("Test Örneği")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.show()
