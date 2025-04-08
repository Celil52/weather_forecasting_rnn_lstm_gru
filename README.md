
# 🌦️ Time Series Weather Forecasting using RNN, LSTM, and GRU

Bu proje, geçmiş meteorolojik verilerden yola çıkarak sıcaklık tahmini yapmak amacıyla geliştirilmiştir. RNN tabanlı farklı derin öğrenme modelleri (Simple RNN, LSTM, GRU) kullanılarak hem kısa vadeli (1 gün sonrası) hem de uzun vadeli (5 gün sonrası) tahmin senaryoları test edilmiştir.

## Veri Hakkında

Bu proje için kullanılan ham hava durumu verileri [Kaggle'dan]([https://www.kaggle.com/...](https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting 

  )) alınmıştır.
   
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
├── visualize.py                # Veriyi görselleştirmede kullanılan dosya
├── weather_forecast.csv        # Ham Veri
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
- `visualize.py`: Veriyi görselleştirir.

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

## 🔄 Veri Ön İşleme

Projede kullanılan verinin orijinal hali `weather_forecast.csv` dosyasında yer almaktadır.  
Veriyi modele uygun hale getirmek için aşağıdaki komutu çalıştırabilirsiniz:

```bash
python prepare_data.py
```
Bu işlem sonucunda temizlenmis_veri.csv adlı dosya oluşur ve modeller bu veriyi kullanır.
-ya da doğrudan temizlenmis_veri.csv dosyasını kullanabilirsiniz.-

2. **Modeli Eğit:**
   - Örneğin Basit GRU:
     ```bash
     python model_gru.py
     ```
   - Walk-forward LSTM:
     ```bash
     python model_lstm_upgraded.py
     ```
     dosyalarda sonuçları görselleştirelecek yapı hazır bulunmaktadır.

## 🧠 Kullanılan Modeller

- `SimpleRNN`: Temel geri besleme mekanizmalı model  
- `LSTM`: Uzun süreli bağımlılıkları öğrenmede başarılı  
- `GRU`: LSTM’e benzer performans gösteren daha sade yapı  

Modeller `walk-forward validation` yöntemiyle test edilmiştir.Bu test edilen dosyalara model_*_upgraded ile ulaşabilirsiniz. Kısa ve uzun vadeli (1 gün – 5 gün sonrası) tahminler yapılmıştır.

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
