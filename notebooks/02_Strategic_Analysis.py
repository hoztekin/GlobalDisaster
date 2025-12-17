# =============================================================================
# 02_Strategic_Analysis.py
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Path Ayarları
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "disaster_data_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "strategic_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Renkler
COLORS = {'primary': '#FF6B35', 'secondary': '#004E89', 'danger': '#FC5130', 'success': '#2BAE66', 'navy': '#2C3E50'}

print("=" * 100)
print("📊 GÖRSELLEŞTİRME BAŞLATILIYOR (OKUNABİLİRLİK MODU)")
print("=" * 100)

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])

# =============================================================================
# 1. WATERFALL (Ülke Kayıp)
# =============================================================================
country_loss = df.groupby('country')['economic_loss_usd'].sum().sort_values(ascending=False).head(15)
fig = go.Figure(go.Waterfall(
    name="Loss", orientation="v", measure=['relative']*len(country_loss),
    x=country_loss.index, y=country_loss.values,
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": COLORS['danger']}}
))
fig.update_layout(title="1. En Çok Kayıp Yaşayan 15 Ülke", height=600)
fig.write_html(OUTPUT_DIR / "1_waterfall_country_economic_loss.html")

# =============================================================================
# 2. AFET TİPİ (BAR + PIE)
# =============================================================================
disaster_cost = df.groupby('disaster_type')['economic_loss_usd'].sum().sort_values(ascending=False)
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]], column_widths=[0.6, 0.4])

fig.add_trace(go.Bar(x=disaster_cost.values, y=disaster_cost.index, orientation='h', marker_color=COLORS['primary'], name='Kayıp'), row=1, col=1)
fig.add_trace(go.Pie(labels=disaster_cost.index, values=disaster_cost.values, hole=0.3), row=1, col=2)

fig.update_layout(
    title="2. Afet Türlerine Göre Maliyet",
    height=600,
    legend=dict(orientation="h", y=-0.2),
    margin=dict(l=20, r=20, t=50, b=50)
)
fig.write_html(OUTPUT_DIR / "2_disaster_type_cost_comparison.html")

# =============================================================================
# 3. TREND (Line)
# =============================================================================
yearly = df.groupby('year')[['economic_loss_usd', 'aid_amount_usd']].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['economic_loss_usd'], name='Kayıp', line=dict(color=COLORS['danger'], width=3)))
fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['aid_amount_usd'], name='Yardım', line=dict(color=COLORS['success'], width=3)))
fig.update_layout(title="3. Yıllık Kayıp Trendi", height=500, legend=dict(orientation="h", y=1.1))
fig.write_html(OUTPUT_DIR / "3_temporal_economic_trend.html")

# =============================================================================
# 4. RADAR (Yardım)
# =============================================================================
aid = df[df['economic_loss_usd']>0].groupby('disaster_type')[['economic_loss_usd', 'aid_amount_usd']].sum()
aid['eff'] = (aid['aid_amount_usd'] / aid['economic_loss_usd']) * 100
fig = go.Figure(go.Scatterpolar(r=aid['eff'], theta=aid.index, fill='toself', name='Kapsama %'))
fig.update_layout(title="4. Yardım Karşılama Oranı", height=600, margin=dict(l=80, r=80)) # Yazılar kesilmesin diye boşluk
fig.write_html(OUTPUT_DIR / "4_aid_efficiency_radar.html")

# =============================================================================
# 7. SEVERITY SCATTER : Log Scale
# =============================================================================
sample = df.sample(n=min(3000, len(df)), random_state=42)
fig = px.scatter(sample, x='severity_index', y='economic_loss_usd', color='disaster_type', log_y=True, title="7. Şiddet vs Ekonomik Kayıp")
fig.write_html(OUTPUT_DIR / "7_severity_economic_scatter.html")

# =============================================================================
# 8. TOP 20 (BAR) : Yüksek Grafik
# =============================================================================
top20 = df.nlargest(20, 'economic_loss_usd')
top20['lbl'] = top20['disaster_type'] + ' - ' + top20['country'] + ' (' + top20['year'].astype(str) + ')'
fig = go.Figure(go.Bar(x=top20['economic_loss_usd'], y=top20['lbl'], orientation='h', marker=dict(color=top20['severity_index'], colorscale='Reds')))
fig.update_layout(title="8. En Maliyetli 20 Afet", height=800, yaxis={'categoryorder':'total ascending'}) # Okunabilir yükseklik
fig.write_html(OUTPUT_DIR / "8_top_20_expensive_disasters.html")

# Diğerleri (Standart)
# 5. Map
grouped = df.groupby('country').agg({'economic_loss_usd':'sum', 'casualties':'sum'}).reset_index()
fig = px.scatter_geo(grouped, locations="country", locationmode='country names', size=np.log1p(grouped['economic_loss_usd']), color="casualties", title="5. Global Harita")
fig.update_layout(height=700, margin={"r":0,"t":50,"l":0,"b":0})
fig.write_html(OUTPUT_DIR / "5_geographic_impact_map.html")

# 6. ROI
fig = px.bar(x=['Erken Uyarı', 'Altyapı', 'Yardım'], y=[4.5, 3.2, 2.1], title="6. ROI Analizi")
fig.write_html(OUTPUT_DIR / "6_roi_investment_analysis.html")

# 9. Gap
gap = df.groupby('disaster_type')[['economic_loss_usd', 'aid_amount_usd']].sum()
gap['diff'] = gap['economic_loss_usd'] - gap['aid_amount_usd']
gap = gap.sort_values('diff', ascending=False)
fig = go.Figure(go.Waterfall(name="Gap", orientation="v", measure=['relative']*len(gap), x=gap.index, y=gap['diff'], connector={"line": {"color": "rgb(63, 63, 63)"}}))
fig.update_layout(title="9. Yardım Açığı", height=600)
fig.write_html(OUTPUT_DIR / "9_aid_gap_waterfall.html")

# 10. Recovery
monthly = df.groupby(df['date'].dt.to_period('M').astype(str))['economic_loss_usd'].sum().reset_index()
fig = px.line(monthly, x='date', y='economic_loss_usd', title="10. İyileşme Maliyeti")
fig.write_html(OUTPUT_DIR / "10_recovery_cost_timeline.html")

# 11 & 12 Stratejik
quad = df.groupby('country').agg({'response_time_hours':'median', 'economic_loss_usd':'mean', 'severity_index':'mean'}).dropna()
fig = px.scatter(quad, x="response_time_hours", y="economic_loss_usd", log_y=True, size="severity_index", title="11. Hız vs Başarı")
fig.write_html(OUTPUT_DIR / "11_speed_vs_success_quadrant.html")

fig = px.scatter(quad, x="severity_index", y="response_time_hours", title="12. Kapasite vs Hız")
fig.write_html(OUTPUT_DIR / "12_capacity_vs_speed.html")

print("✅ BİTTİ. Tüm HTML dosyaları güncellendi.")