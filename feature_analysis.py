import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
df = pd.read_csv("temizlenmis_veri.csv")
cov_matrix = df.cov(numeric_only=True)
import seaborn as sns
sns.histplot(df['T'], kde=True)
plt.title("T (Sıcaklık) Değişkeni Dağılımı")
plt.show()