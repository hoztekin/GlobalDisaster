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
from datetime import datetime

# Ayarlar
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Dizin Ayarları
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "disaster_data_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "strategic_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Renk Paleti
COLORS = {
    'primary': '#FF6B35', 'secondary': '#004E89', 'accent': '#F7B801',
    'success': '#2BAE66', 'danger': '#FC5130', 'purple': '#9B59B6',
    'teal': '#1ABC9C', 'navy': '#2C3E50'
}

print("=" * 100)
print("📊 STRATEJİK ANALİZ VE FULL GÖRSELLEŞTİRME BAŞLATILIYOR")
print("=" * 100)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"❌ Veri bulunamadı: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])

print(f"✅ Veri Yüklendi: {len(df):,} kayıt")

# =============================================================================
# 1. ÜLKE BAZLI KAYIP (WATERFALL)
# =============================================================================
print("\n[1/10] Ekonomik Kayıp Şelalesi (Waterfall)...")
country_loss = df.groupby('country')['economic_loss_usd'].sum().sort_values(ascending=False).head(15)
measures = ['relative'] * len(country_loss)

fig = go.Figure(go.Waterfall(
    name="Economic Loss", orientation="v", measure=measures,
    x=country_loss.index, y=country_loss.values,
    text=[f'${val/1e9:.1f}B' for val in country_loss.values],
    textposition="outside",
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": COLORS['danger']}},
    decreasing={"marker": {"color": COLORS['success']}},
))
fig.update_layout(title="💰 1. Top 15 Countries by Economic Loss", height=600)
fig.write_html(OUTPUT_DIR / "1_waterfall_country_economic_loss.html")

# =============================================================================
# 2. AFET TİPİ MALİYET KARŞILAŞTIRMASI (BAR + PIE)
# =============================================================================
print("\n[2/10] Afet Tipi Maliyet Analizi...")
disaster_cost = df.groupby('disaster_type')['economic_loss_usd'].sum().sort_values(ascending=False)

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]],
                    subplot_titles=('Total Loss by Type', 'Distribution'))

fig.add_trace(go.Bar(x=disaster_cost.values, y=disaster_cost.index, orientation='h', marker_color=COLORS['primary'], name='Loss'), row=1, col=1)
fig.add_trace(go.Pie(labels=disaster_cost.index, values=disaster_cost.values, hole=0.3), row=1, col=2)

fig.update_layout(title="🌪️ 2. Disaster Type Cost Analysis", height=500)
fig.write_html(OUTPUT_DIR / "2_disaster_type_cost_comparison.html")

# =============================================================================
# 3. ZAMANSAL TREND (TREND LINE)
# =============================================================================
print("\n[3/10] Zamansal Trend Analizi...")
yearly_data = df.groupby('year')[['economic_loss_usd', 'aid_amount_usd']].sum().reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=yearly_data['year'], y=yearly_data['economic_loss_usd'], mode='lines+markers', name='Economic Loss', line=dict(color=COLORS['danger'], width=3)))
fig.add_trace(go.Scatter(x=yearly_data['year'], y=yearly_data['aid_amount_usd'], mode='lines+markers', name='Aid Provided', line=dict(color=COLORS['success'], width=3)))
fig.add_trace(go.Scatter(x=yearly_data['year'], y=yearly_data['economic_loss_usd'], fill='tonexty', name='Aid Gap', line=dict(width=0), showlegend=False, opacity=0.1))

fig.update_layout(title="📈 3. Economic Loss vs Aid Trends (2018-2024)", height=500)
fig.write_html(OUTPUT_DIR / "3_temporal_economic_trend.html")

# =============================================================================
# 4. YARDIM ETKİNLİĞİ (RADAR)
# =============================================================================
print("\n[4/10] Yardım Etkinliği (Radar)...")
aid_stats = df[df['economic_loss_usd'] > 0].groupby('disaster_type')[['economic_loss_usd', 'aid_amount_usd']].sum()
aid_stats['efficiency'] = (aid_stats['aid_amount_usd'] / aid_stats['economic_loss_usd']) * 100
aid_stats = aid_stats.sort_values('efficiency', ascending=False)

fig = go.Figure(go.Scatterpolar(r=aid_stats['efficiency'], theta=aid_stats.index, fill='toself', name='Aid Coverage %', line=dict(color=COLORS['secondary'])))
fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="💼 4. Aid Efficiency by Disaster Type (%)")
fig.write_html(OUTPUT_DIR / "4_aid_efficiency_radar.html")

# =============================================================================
# 5. COĞRAFİ ETKİ HARİTASI (BUBBLE MAP - YENİLENMİŞ)
# =============================================================================
print("\n[5/10] Coğrafi Etki Haritası...")
grouped = df.groupby('country')
country_stats = pd.DataFrame({
    'economic_loss_usd': grouped['economic_loss_usd'].sum(),
    'casualties': grouped['casualties'].sum(),
    'population_density': grouped['population_density'].mean() # Yeni Veri
}).reset_index()
country_stats['bubble_size'] = np.log1p(country_stats['economic_loss_usd'])

fig = px.scatter_geo(
    country_stats, locations="country", locationmode='country names',
    size="bubble_size", color="casualties",
    hover_name="country",
    hover_data={"economic_loss_usd": ":,.0f", "casualties": ":,.0f", "population_density": ":.1f", "bubble_size": False},
    title="🌍 5. Global Disaster Impact Map", projection="natural earth"
)
fig.update_layout(height=700, width=1400, showlegend=True)
fig.write_html(OUTPUT_DIR / "5_geographic_impact_map.html")

# =============================================================================
# 6. ROI ANALİZİ (BAR)
# =============================================================================
print("\n[6/10] ROI Yatırım Analizi...")
total_loss = df['economic_loss_usd'].sum()
scenarios = {'Early Warning': {'c': 0.02, 's': 0.08}, 'Resilience': {'c': 0.05, 's': 0.35}, 'Aid Boost': {'c': 0.03, 's': 0.10}}
roi_data = [{'Scenario': k, 'Investment': total_loss*v['c'], 'Savings': total_loss*v['s'], 'ROI': v['s']/v['c']} for k,v in scenarios.items()]
roi_df = pd.DataFrame(roi_data)

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'bar'}]], subplot_titles=("Investment vs Savings", "ROI Multiplier"))
fig.add_trace(go.Bar(x=roi_df['Scenario'], y=roi_df['Investment'], name='Invest', marker_color=COLORS['danger']), row=1, col=1)
fig.add_trace(go.Bar(x=roi_df['Scenario'], y=roi_df['Savings'], name='Save', marker_color=COLORS['success']), row=1, col=1)
fig.add_trace(go.Bar(x=roi_df['Scenario'], y=roi_df['ROI'], name='ROI', marker_color=COLORS['accent'], text=roi_df['ROI'].round(1), textposition='auto'), row=1, col=2)
fig.update_layout(title="💰 6. ROI Analysis", height=500)
fig.write_html(OUTPUT_DIR / "6_roi_investment_analysis.html")

# =============================================================================
# 7. ŞİDDET vs KAYIP (SCATTER)
# =============================================================================
print("\n[7/10] Şiddet vs Kayıp İlişkisi...")
# Performans için örneklem alıyoruz (5000 kayıt)
sample_df = df.sample(n=min(5000, len(df)), random_state=42)
fig = px.scatter(
    sample_df, x='severity_index', y='economic_loss_usd', color='disaster_type',
    size='casualties', hover_data=['country', 'date'], log_y=True,
    title="⚠️ 7. Severity vs Economic Loss Relationship"
)
fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Severity: %{x}<br>Loss: $%{y:,.0f}<extra></extra>')
fig.update_layout(height=700, width=1600, showlegend=True)
fig.write_html(OUTPUT_DIR / "7_severity_economic_scatter.html")

# =============================================================================
# 8. EN PAHALI 20 AFET (BAR)
# =============================================================================
print("\n[8/10] En Pahalı 20 Afet...")
top20 = df.nlargest(20, 'economic_loss_usd').copy()
top20['label'] = top20['disaster_type'] + ' - ' + top20['country'] + ' (' + top20['date'].dt.strftime('%Y') + ')'

fig = go.Figure(go.Bar(
    x=top20['economic_loss_usd'], y=top20['label'], orientation='h',
    marker=dict(color=top20['severity_index'], colorscale='Reds', colorbar=dict(title="Severity"))
))
fig.update_layout(title="🔥 8. Top 20 Most Expensive Disasters", height=800, yaxis={'categoryorder':'total ascending'})
fig.write_html(OUTPUT_DIR / "8_top_20_expensive_disasters.html")

# =============================================================================
# 9. YARDIM AÇIĞI (GAP WATERFALL)
# =============================================================================
print("\n[9/10] Yardım Açığı Analizi...")
gap_data = df.groupby('disaster_type')[['economic_loss_usd', 'aid_amount_usd']].sum()
gap_data['aid_gap'] = gap_data['economic_loss_usd'] - gap_data['aid_amount_usd']
gap_data = gap_data.sort_values('aid_gap', ascending=False)

fig = go.Figure(go.Waterfall(
    name="Aid Gap", orientation="v", measure=['relative']*len(gap_data),
    x=gap_data.index, y=gap_data['aid_gap'],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": COLORS['navy']}}
))
fig.update_layout(title="📉 9. Aid Gap Analysis (Unmet Needs)", height=600)
fig.write_html(OUTPUT_DIR / "9_aid_gap_waterfall.html")

# =============================================================================
# 10. İYİLEŞME MALİYETİ ZAMAN ÇİZELGESİ (LINE)
# =============================================================================
print("\n[10/10] İyileşme Maliyeti Zaman Çizelgesi...")
df['daily_recovery_cost'] = df['economic_loss_usd'] / (df['recovery_days'] + 1)
df['year_month'] = df['date'].dt.to_period('M').astype(str)
monthly_recovery = df.groupby('year_month')['daily_recovery_cost'].sum().reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly_recovery['year_month'], y=monthly_recovery['daily_recovery_cost'],
    mode='lines', fill='tozeroy', line=dict(color=COLORS['purple']), name='Recovery Cost'
))
fig.update_layout(title="⏱️ 10. Recovery Cost Timeline (Monthly)", height=500)
fig.write_html(OUTPUT_DIR / "10_recovery_cost_timeline.html")

print("\n" + "=" * 100)
print("✅ TÜM GÖRSELLEŞTİRMELER TAMAMLANDI!")
print(f"📂 Raporlar: {OUTPUT_DIR}")
print("=" * 100)