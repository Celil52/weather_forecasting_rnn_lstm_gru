import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# ==== Ayarlanabilir Parametreler ====
window_size = 5
horizon = 5
epochs = 10
batch_size = 64
units = 64

# ==== 1. Veri Yükleme ====
df = pd.read_csv("temizlenmis_veri.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

X = df.drop(columns=['date', 'T'])
y = df[['T']].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ==== 2. Sliding Window ====
def create_sliding_window(X, y, window_size, horizon):
    X_win, y_win = [], []
    for i in range(len(X) - window_size - horizon + 1):
        X_win.append(X[i:i+window_size])
        y_win.append(y[i+window_size + horizon - 1])
    return np.array(X_win), np.array(y_win)

X_window, y_window = create_sliding_window(X_scaled, y_scaled, window_size, horizon)

# ==== 3. Eğitim/Test Ayır ====
X_train, X_test, y_train, y_test = train_test_split(
    X_window, y_window, test_size=0.2, shuffle=False
)

# ==== 4. GRU Model (Temel Eğitim) ====
model = Sequential()
model.add(GRU(units=units, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_real, y_pred)
mae = mean_absolute_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred)
print(f"[Temel Test - Horizon={horizon}] MSE: {mse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

# ==== 5. Walk-Forward ====
def build_gru(input_shape):
    model = Sequential()
    model.add(GRU(units=units, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def walk_forward(model_builder, X, y, n_test, retrain_epochs=3, batch_size=64):
    preds, actuals = [], []
    train_X, train_y = X[:-n_test], y[:-n_test]
    test_X, test_y = X[-n_test:], y[-n_test:]

    for i in range(len(test_X)):
        model = model_builder((train_X.shape[1], train_X.shape[2]))
        model.fit(train_X, train_y, epochs=retrain_epochs, batch_size=batch_size, verbose=0)

        pred = model.predict(test_X[i:i+1])[0][0]
        preds.append(pred)
        actuals.append(test_y[i])

        train_X = np.concatenate([train_X, test_X[i:i+1]], axis=0)
        train_y = np.concatenate([train_y, test_y[i:i+1]], axis=0)

    return np.array(actuals), np.array(preds)

n_test = 100
actuals_scaled, preds_scaled = walk_forward(build_gru, X_window, y_window, n_test=n_test, retrain_epochs=3)

actuals = scaler_y.inverse_transform(actuals_scaled.reshape(-1, 1))
preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))

mse_wf = mean_squared_error(actuals, preds)
mae_wf = mean_absolute_error(actuals, preds)
r2_wf = r2_score(actuals, preds)
print(f"[Walk-Forward - Horizon={horizon}] MSE: {mse_wf:.5f}, MAE: {mae_wf:.5f}, R²: {r2_wf:.5f}")

# ==== 6. Grafik ====
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Gerçek", marker='o')
plt.plot(preds, label="Tahmin", marker='x', linestyle='--')
plt.title(f"Walk-Forward Sonuçları (Horizon={horizon})")
plt.legend()
plt.show()
