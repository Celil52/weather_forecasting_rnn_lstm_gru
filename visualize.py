import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Veriyi oku ve sıcaklığı ayır
df = pd.read_csv("temizlenmis_veri.csv")
X = df.drop(columns=['date', 'T'])  # Özellikler
y = df['T']  # Sıcaklık

# 2. Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. PCA grafiği (sıcaklığa göre renklendirme)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
plt.title("PCA ile 2B Görselleştirme (Renk: Sıcaklık)")
plt.xlabel("Bileşen 1")
plt.ylabel("Bileşen 2")
plt.colorbar(scatter, label="Sıcaklık (°C)")
plt.show()
