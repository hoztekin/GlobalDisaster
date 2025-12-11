# 🌍 Global Disaster AI: Risk & Trend Analysis System

**Miuul Data Scientist Bootcamp - Final Projesi** **Takım:** Resque Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://resqdata.haliloztekin.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Proje Hakkında

Bu proje, **Resque Data** takımı tarafından **Miuul Data Scientist Bootcamp** mezuniyet projesi olarak geliştirilmiştir.

**Amaç:** 2018-2024 yılları arasındaki 50.000+ küresel afet verisini analiz ederek; uluslararası yardım kuruluşları, devletler ve sigorta şirketleri için yapay zeka destekli bir **Karar Destek Sistemi (DSS)** oluşturmaktır.

Proje, statik veri analizinin ötesine geçerek; **World Bank API** ile canlı veri zenginleştirme, **K-Means** ile risk kümeleme ve **XGBoost** ile anlık risk tahminlemesi yapan uçtan uca bir üründür.

🔗 **Canlı Demo:** [globaldisaster.haliloztekin.com](https://globaldisaster.haliloztekin.com)

---

## 🏗️ Proje Mimarisi ve Klasör Yapısı

Proje, veri zenginleştirme, stratejik analiz, makine öğrenmesi modellemesi ve dashboard sunumu olmak üzere 4 ana katmandan oluşur.

```text
GlobalDisaster/
├── README.md                                    # Proje Dokümantasyonu
├── requirements.txt                             # Gerekli Kütüphaneler (aiohttp, streamlit vb.)
├── app.py                                       # 🚀 Streamlit Dashboard (Ana Uygulama)
├── docker-compose.yml                           # Docker Deployment
├── Dockerfile                                   # Docker Image Tanımı
│
├── data/
│   ├── raw/                                     # Ham Kaggle Verisi (global_disaster.csv)
│   └── processed/
│       └── disaster_data_final.csv              # World Bank API ile zenginleştirilmiş Final Veri
│
├── models/                                      # Eğitilmiş Modeller ve Meta Veriler
│   ├── severity_prediction_model.pkl            # Model 01 (XGBoost Classifier)
│   ├── severity_prediction_scaler.pkl           # Model 01 Scaler
│   ├── model_01_features.pkl                    # Model 01 Feature Listesi
│   ├── risk_kmeans_model.pkl                    # Model 04 (K-Means Clustering)
│   ├── country_risk_clusters.csv                # Risk Kümeleme Çıktısı
│   └── disaster_momentum_analysis.csv           # Trend Analiz Çıktısı
│
├── notebooks/                                   # Analiz ve Modelleme Scriptleri
│   ├── 01_Ultimate_EDA.py                       # Veri Zenginleştirme (API) + EDA Pipeline
│   ├── 02_Strategic_Analysis.py                 # Ekonomik Analiz ve HTML Rapor Üretimi
│   ├── 05_Model_01_SeverityPrediction.py        # Şiddet Tahmin Modeli (Classification)
│   ├── 06_Model_04_RiskClustering.py            # Coğrafi Risk Kümeleme (Clustering)
│   └── 07_Model_05_TrendForecasting.py          # Trend ve Momentum Analizi (Time Series)
│
└── reports/                                     # Dashboard İçin Üretilen Dinamik Raporlar
    ├── strategic_analysis/                      # Waterfall, ROI, Radar Grafikleri
    ├── model_01/                                # Confusion Matrix vb.
    ├── model_04_clustering/                     # Risk Haritası (Map)
    └── model_05_trend/                          # Gelecek Tahmin Grafikleri


💡 Temel Özellikler

1. 🤖 AI Risk Simülatörü (Async API)

* Kullanıcı bir afet senaryosu girdiğinde (Örn: "Türkiye'de Kışın 7.2 Deprem"), sistem arka planda Asenkron (Async) olarak World Bank API'ye bağlanır.

* Ülkenin güncel nüfusu, gelir grubu ve yüzölçümü verilerini canlı çeker.

* XGBoost modelini kullanarak olayın risk seviyesini (Düşük, Orta, Yüksek, Kritik) tahmin eder.

* Fail-Safe Mekanizması: Yüksek can kayıplarında modeli ezerek "Güvenlik Protokolü" gereği riski otomatik yükseltir.

2. 🗺️ Coğrafi Risk Kümeleme (Clustering)

* Ülkeleri sadece konumlarına göre değil; afet sıklığı, ekonomik direnç ve kayıp oranlarına göre K-Means algoritmasıyla 3 sınıfa ayırır:

   🟢 Low Risk: Afet yönetimi güçlü veya riski düşük ülkeler.

   🟡 Moderate Risk: Orta seviye risk grubu.

   🔴 High Risk: Kırılgan ve yüksek etkilenme oranına sahip ülkeler.

3. 📈 Trend ve Momentum Analizi

* Zaman Serisi: Polynomial Regression ile gelecek 12 ayın küresel afet yoğunluğunu tahmin eder.

* Momentum: Hangi afet tipinin (örn. Tornado) son 2 yılda artış trendinde olduğunu (RISING/FALLING) analiz eder.

4. 📊 Stratejik Raporlama

* Ekonomik Kayıp Şelalesi: Ülkelerin kümülatif kaybını görselleştirir.

* ROI Analizi: Afet öncesi 1$ yatırımın, afet sonrası kaç $ tasarruf sağladığını simüle eder.

* Yardım Etkinliği: Hangi afet tiplerinin yeterli yardım alamadığını (Aid Gap) gösterir.

🛠️ Kurulum ve Çalıştırma

Yöntem 1: Yerel Kurulum (Local)

1. Repoyu Klonlayın:

git clone [https://github.com/haliloztekin/GlobalDisaster.git](https://github.com/haliloztekin/GlobalDisaster.git)
cd GlobalDisaster

2. Sanal Ortam Oluşturun ve Kütüphaneleri Yükleyin:

pip install -r requirements.txt

3. Veri Pipeline'ını Çalıştırın (Sırasıyla): (Not: Hazır modeller models/ klasöründe mevcuttur, bu adımı atlayıp direkt uygulamayı başlatabilirsiniz.)


     # 1. Veriyi indir, API ile zenginleştir ve temizle
       python notebooks/01_Ultimate_EDA.py

    # 2. Stratejik raporları ve grafikleri üret
      python notebooks/02_Strategic_Analysis.py

    # 3. Modelleri eğit ve kaydet
      python notebooks/05_Model_01_SeverityPrediction.py
      python notebooks/06_Model_04_RiskClustering.py
      python notebooks/07_Model_05_TrendForecasting.py
      
    # 4. Uygulamayı Başlatın:

      streamlit run app.py
      
Yöntem 2: Docker Deployment 🐳
Proje Docker ile konteynerize edilmeye hazırdır.

docker-compose up --build
Uygulama http://localhost:8505 adresinde çalışacaktır.

📊 Kullanılan Teknolojiler
    * Dil: Python 3.10
    * Arayüz: Streamlit, Streamlit-Option-Menu
    * Veri İşleme: Pandas, NumPy, Aiohttp (Async API Entegrasyonu)
    * Makine Öğrenmesi: Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn (SMOTE)
    * Görselleştirme: Plotly (İnteraktif), Matplotlib, Seaborn

👨‍💻 Takım: Resque Data

Bu proje Miuul Data Scientist Bootcamp kapsamında aşağıdaki ekip tarafından geliştirilmiştir:

Ali Özdemir, Nadide Yücel, Aslı Güldağ Bekaroğlu, İbrahim Alnıaçık, Halil Öztekin

Lisans: MIT License