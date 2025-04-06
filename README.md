
# 🌦️ Time Series Weather Forecasting using RNN, LSTM, and GRU

Bu proje, geçmiş meteorolojik verilerden yola çıkarak sıcaklık tahmini yapmak amacıyla geliştirilmiştir. RNN tabanlı farklı derin öğrenme modelleri (Simple RNN, LSTM, GRU) kullanılarak hem kısa vadeli (1 gün sonrası) hem de uzun vadeli (5 gün sonrası) tahmin senaryoları test edilmiştir.

## Veri Hakkında

Bu proje için kullanılan ham hava durumu verileri [Kaggle'dan]([https://www.kaggle.com/...](https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting 

  )) alınmıştır.


Veri büyüklüğü nedeniyle bu repoda paylaşılmamıştır. Aşağıdaki adımları izleyerek veriyi indirip `temizlenmis_veri.csv` dosyasını oluşturabilirsiniz:

1. Kaggle hesabı oluşturun.
2. Veriyi yukardıdaki linkten indirin:
3. Dosyayı `data/raw/` klasörüne yerleştirin.
4. `prepare_data.py` dosyasını çalıştırarak `temizlenmis_veri.csv`'yi oluşturun.
   
## 📁 Proje Yapısı

```
.
├── prepare_data.py              # Veriyi ön işleme (temizleme, normalize etme)
├── feature_analysis.py          # Korelasyon, öznitelik dağılımı analizleri
├── model_rnn.py                 # Basit RNN modeli
├── model_lstm.py                # LSTM modeli
├── model_gru.py                 # GRU modeli
├── model_rnn_upgraded.py       # Walk-forward kullanılan RNN
├── model_lstm_upgraded.py      # Walk-forward kullanılan LSTM
├── model_gru_upgraded.py       # Walk-forward kullanılan GRU
├── visualize.py                # Tahmin - gerçek karşılaştırma grafikleri
├── temizlenmis_veri.csv        # Temizlenmiş ve işlenmiş veri
├── requirements.txt            # Gereken kütüphaneler
└── README.md                   # Bu dosya
```
## 📂 Dosyalar Ne İşe Yarar?

- `prepare_data.py`: Veriyi temizler ve normalize eder.
- `model_rnn.py`: Basit RNN modeliyle tahmin yapar.
- `model_lstm.py`: LSTM ile tahmin yapar.
- `model_gru.py`: GRU ile tahmin yapar.
- `model_*_upgraded.py`: Walk-forward validation içeren gelişmiş sürümler.
- `visualize.py`: Tahmin edilen ve gerçek sıcaklıkları karşılaştırır.

## 📦 Gereksinimler

Bu proje için **Python 3.10.x** kullanılması önerilir. Python 3.11 ve sonrası TensorFlow ile uyumlu olmayabilir.  
Sanal ortam oluşturup bağımlılıkları yüklemek için aşağıdaki adımları izleyin:

```bash
# Sanal ortam oluştur (Python 3.10 ile)
py -3.10 -m venv venv

# Ortamı aktif et (Windows için)
venv\Scripts\activate

# Pip'i güncelle ve gerekli paketleri yükle
pip install --upgrade pip
pip install -r requirements.txt
```
💡 Not: Python 3.13 kullanıyorsanız TensorFlow yüklenemez, bu yüzden Python 3.10 yükleyip sanal ortamı onunla oluşturmalısınız.

Python 3.10+ önerilir.

## 🔧 Kurulum ve Çalıştırma

> 🛠️ Kurulumdan önce sanal ortamı oluşturduğunuzdan ve aktif ettiğinizden emin olun (`venv\Scripts\activate`).

1. **Veriyi Hazırla:**
   ```bash
   python prepare_data.py
   ```

2. **Modeli Eğit:**
   - Örneğin Basit GRU:
     ```bash
     python model_gru.py
     ```
   - Walk-forward LSTM:
     ```bash
     python model_lstm_upgraded.py
     ```

3. **Görselleştirme:**
   - Tahmin/Gerçek sıcaklık grafiği çizmek için:
     ```bash
     python visualize.py
     ```

## 🧠 Kullanılan Modeller

- `SimpleRNN`: Temel geri besleme mekanizmalı model  
- `LSTM`: Uzun süreli bağımlılıkları öğrenmede başarılı  
- `GRU`: LSTM’e benzer performans gösteren daha sade yapı  

Modeller `walk-forward validation` yöntemiyle test edilmiştir. Kısa ve uzun vadeli (1 gün – 5 gün sonrası) tahminler yapılmıştır.

## 📊 Performans Değerlendirmesi

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Determinasyon Katsayısı)

## 📌 Notlar

- `temizlenmis_veri.csv` dosyası hazır olarak klasöre eklenmiştir.
- Projede `"shuffle=False"` ile zaman serisi yapısı korunmuştur.
- `venv/` klasörü `.gitignore` içinde olup repoya dahil edilmemelidir.
- en iyi model GRU ve LSTM olarak bulunmuştur, isterseniz sadece onları deneyebilirsiniz. 

## 👤 Yazar

**Celil Kaan Güngör**  
TOBB ETÜ - Computer Engineering  
201101014
