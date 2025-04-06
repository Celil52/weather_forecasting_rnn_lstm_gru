import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Veri Yükleme ve Tarih İşlemleri
df = pd.read_csv('weather_forecast.csv')  # Dosya adınızı gerektiği gibi ayarlayın
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# 2. Eksik Değerleri Doldurma ve Aykırı Değer Güncelleme
# Eksik değerlerin lineer interpolasyon yöntemiyle doldurulması
df_interpolated = df.interpolate(method='linear')

# Aykırı değer güncelleme: Sıcaklık (T) sütunu için IQR yöntemi kullanılarak clipping
Q1 = df_interpolated['T'].quantile(0.25)
Q3 = df_interpolated['T'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_interpolated['T'] = df_interpolated['T'].clip(lower=lower_bound, upper=upper_bound)

# 3. Ölçeklendirme
# Ölçeklenecek sütunlar (sayısal sütunların tamamı)
numeric_columns = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef',
                   'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd', 'rain', 'raining',
                   'SWDR', 'PAR', 'max. PAR', 'Tlog']

scaler = StandardScaler()
df_interpolated[numeric_columns] = scaler.fit_transform(df_interpolated[numeric_columns])

# 4. Sliding Window Oluşturma: Geçmiş 'n' gün verisini alarak bir sonraki günün sıcaklığını tahmin etme
def create_sliding_window(data, target_column, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        # penceredeki tüm sütunları kullanabiliriz, burada tüm sütunlardan feature set oluşturuluyor
        X_window = data.iloc[i:i+window_size].drop(columns=['date']).values
        X.append(X_window)
        # hedef: pencere sonrasındaki günün sıcaklık değeri (T sütunu, ölçeklendirilmiş hali)
        y.append(data.iloc[i+window_size][target_column])
    return np.array(X), np.array(y)

# Örneğin, pencere boyutunu 5 gün olarak belirleyelim
window_size = 5
X, y = create_sliding_window(df_interpolated, target_column='T', window_size=window_size)

print("Girdi (X) veri şekli:", X.shape)
print("Hedef (y) veri şekli:", y.shape)

# İsteğe bağlı: Model eğitimi için veriyi eğitim ve test setlerine bölebilirsiniz
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("Eğitim seti şekli:", X_train.shape, y_train.shape)
print("Test seti şekli:", X_test.shape, y_test.shape)
df_interpolated.to_csv("temizlenmis_veri.csv", index=False)