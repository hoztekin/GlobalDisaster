# =============================================================================
# MODEL 05: DISASTER TREND FORECASTING (Time Series)
# =============================================================================

"""
AMAÇ:
1. Gelecek 12 ay için küresel afet sıklığını tahmin etmek.
2. Hangi afet tiplerinin yükseliş trendinde olduğunu (Momentum) bulmak.

YÖNTEM:
- Seasonal Decompose (Mevsimsellik Analizi)
- Polynomial Regression (Trend Tahmini)
- Growth Rate Calculation (Afet Tipi Momentumu)

ÇIKTI:
- Gelecek 1 yılın tahmin grafiği (HTML)
- Yükselen yıldızlar (Hangi afet tipi artıyor?)
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import timedelta

# Windows Koruması
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
warnings.filterwarnings('ignore')


class Config:
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

    INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'disaster_data_final.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'models'
    REPORT_DIR = PROJECT_ROOT / 'reports' / 'model_03_trend'

    FORECAST_MONTHS = 12  # Gelecek 12 ayı tahmin et


def run_trend_pipeline():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📈 MODEL 05: TREND FORECASTING & MOMENTUM ANALYSIS BAŞLATILIYOR")
    print("=" * 80)

    # 1. VERİ YÜKLEME
    if not Config.INPUT_FILE.exists():
        raise FileNotFoundError(f"❌ Veri bulunamadı: {Config.INPUT_FILE}")

    df = pd.read_csv(Config.INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])

    print(f"✅ Veri yüklendi: {len(df):,} satır")
    print(f"📅 Veri Aralığı: {df['date'].min().date()} - {df['date'].max().date()}")

    # =========================================================================
    # ANALİZ 1: GLOBAL AFET TRENDİ (GELECEK TAHMİNİ)
    # =========================================================================
    print("\n🔮 Global Trend Tahmini Yapılıyor...")

    # Aylık afet sayısını hesapla (Resampling)
    monthly_counts = df.set_index('date').resample('M').size().reset_index(name='count')

    # Zamanı sayısal değere çevir (Regresyon için)
    monthly_counts['time_idx'] = np.arange(len(monthly_counts))

    # Model: Polynomial Regression (Hem doğrusal hem eğrisel artışları yakalar)
    # Degree=2 veya 3 genelde iyidir, aşırı uyumdan (overfit) kaçınır.
    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

    X = monthly_counts[['time_idx']]
    y = monthly_counts['count']

    model.fit(X, y)

    # Gelecek 12 ayı oluştur
    last_idx = monthly_counts['time_idx'].max()
    future_indices = np.arange(last_idx + 1, last_idx + 1 + Config.FORECAST_MONTHS).reshape(-1, 1)
    future_dates = [monthly_counts['date'].max() + pd.DateOffset(months=i) for i in
                    range(1, Config.FORECAST_MONTHS + 1)]

    # Tahmin yap
    future_preds = model.predict(future_indices)

    # Geçmiş veriler üzerindeki trend çizgisi (Modelin geçmişi nasıl öğrendiği)
    historical_trend = model.predict(X)

    # GÖRSELLEŞTİRME
    fig = go.Figure()

    # Gerçek Veriler
    fig.add_trace(go.Scatter(
        x=monthly_counts['date'], y=monthly_counts['count'],
        mode='lines+markers', name='Actual Data',
        line=dict(color='#2c3e50', width=2), opacity=0.6
    ))

    # Trend Çizgisi (Geçmiş)
    fig.add_trace(go.Scatter(
        x=monthly_counts['date'], y=historical_trend,
        mode='lines', name='Historical Trend',
        line=dict(color='#e67e22', width=3, dash='dash')
    ))

    # Gelecek Tahmini
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        mode='lines+markers', name='Forecast (Next 12 Months)',
        line=dict(color='#e74c3c', width=3)
    ))

    fig.update_layout(
        title=f"🔮 Global Disaster Frequency Forecast (Next {Config.FORECAST_MONTHS} Months)",
        xaxis_title="Date", yaxis_title="Number of Disasters",
        template="plotly_white", hovermode="x unified"
    )

    fig.write_html(Config.REPORT_DIR / "global_trend_forecast.html")
    print(f"✅ Trend grafiği kaydedildi: {Config.REPORT_DIR}/global_trend_forecast.html")

    # =========================================================================
    # ANALİZ 2: AFET TİPİ MOMENTUMU (HANGİSİ YÜKSELİŞTE?)
    # =========================================================================
    print("\n🚀 Afet Tipi Momentumu Hesaplanıyor...")

    # Son 2 yıl vs Önceki 2 yıl karşılaştırması (Growth Rate)
    current_year = df['date'].dt.year.max()
    last_2_years = df[df['date'].dt.year >= (current_year - 1)]
    prev_2_years = df[(df['date'].dt.year < (current_year - 1)) & (df['date'].dt.year >= (current_year - 3))]

    recent_counts = last_2_years['disaster_type'].value_counts()
    old_counts = prev_2_years['disaster_type'].value_counts()

    momentum_data = []

    for dtype in recent_counts.index:
        if dtype in old_counts:
            recent = recent_counts[dtype]
            old = old_counts[dtype]
            # Basit büyüme oranı formülü: (Yeni - Eski) / Eski
            growth_rate = ((recent - old) / old) * 100
            momentum_data.append({
                'Disaster Type': dtype,
                'Recent Count (2y)': recent,
                'Growth Rate (%)': round(growth_rate, 2),
                'Trend': '⬆️ RISING' if growth_rate > 5 else ('⬇️ FALLING' if growth_rate < -5 else '➡️ STABLE')
            })

    momentum_df = pd.DataFrame(momentum_data).sort_values('Growth Rate (%)', ascending=False)

    print("\n🔥 TOP 5 RISING DISASTERS (Momentum):")
    print(momentum_df.head(5).to_string(index=False))

    # Momentum Bar Chart
    fig2 = px.bar(
        momentum_df.head(10),
        x='Growth Rate (%)', y='Disaster Type',
        orientation='h', color='Growth Rate (%)',
        color_continuous_scale=['red', 'gray', 'green'],
        # Kırmızı=Düşüş, Yeşil=Artış (Tersi mantık: Artış kötü aslında ama görsel olarak büyüme yeşil algılanır, biz kırmızıyı artış yapalım)
        color_continuous_midpoint=0,
        title="📈 Disaster Type Momentum: Which types are increasing fast?"
    )
    # Renk skalasını düzelt (Artış = Kırmızı/Kötü, Azalış = Yeşil/İyi)
    fig2.update_traces(marker_coloraxis=None,
                       marker=dict(color=np.where(momentum_df.head(10)['Growth Rate (%)'] > 0, '#e74c3c', '#2ecc71')))

    fig2.write_html(Config.REPORT_DIR / "disaster_momentum.html")

    # =========================================================================
    # 3. KAYIT
    # =========================================================================
    momentum_df.to_csv(Config.OUTPUT_DIR / 'disaster_momentum_analysis.csv', index=False)

    # Trend Modelini Kaydet (Gerekirse Streamlit'te canlı tahmin için)
    with open(Config.OUTPUT_DIR / 'trend_forecasting_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f"\n💾 Analiz verileri kaydedildi: {Config.OUTPUT_DIR}")
    print("✅ MODEL 05 SÜRECİ TAMAMLANDI!")


if __name__ == '__main__':
    run_trend_pipeline()