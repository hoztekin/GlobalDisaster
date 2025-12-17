# =============================================================================
# 03_Model_05_GridAnalysis.py
# (ULTIMATE: Smart Filter + Coordinates + DETAILED CONSOLE REPORT)
# =============================================================================

import os
# Windows Joblib Çakışmasını Önle
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ayarlar
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Path
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "disaster_data_final.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_05"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("🌍 MODEL 05: SPATIAL GRID ANALYSIS (FULL REPORT MODE)")
print("=" * 100)

# 1. Veri Yükle
if not DATA_FILE.exists():
    raise FileNotFoundError(f"❌ Dosya bulunamadı: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
print(f"✅ Ham Veri Yüklendi: {len(df)} satır")

# 2. Grid Oluşturma (2 Derece)
GRID_SIZE = 2.0
df['grid_lat'] = np.floor(df['latitude'] / GRID_SIZE) * GRID_SIZE + (GRID_SIZE / 2)
df['grid_lon'] = np.floor(df['longitude'] / GRID_SIZE) * GRID_SIZE + (GRID_SIZE / 2)
df['grid_id'] = df['grid_lat'].astype(str) + "_" + df['grid_lon'].astype(str)

# 3. Aggregation (Özetleme)
print("\n[1/4] Gridler Oluşturuluyor ve Özetleniyor...")
grid_df = df.groupby(['grid_id', 'grid_lat', 'grid_lon']).agg({
    'severity_index': 'mean',
    'economic_loss_usd': 'sum',
    'casualties': 'sum',
    'date': 'count'
}).reset_index()

grid_df.columns = ['grid_id', 'lat', 'lon', 'avg_severity', 'total_loss', 'total_casualties', 'event_count']
print(f"   -> Toplam Ham Grid Sayısı: {len(grid_df)}")

# 4. Risk Skoru ve Filtreleme
print("\n[2/4] Risk Skoru Hesaplanıyor ve Çöp Veriler Atılıyor...")
grid_df['risk_score'] = (
    (grid_df['avg_severity'] * 0.3) +
    (np.log1p(grid_df['total_loss']) * 0.3) +
    (np.log1p(grid_df['total_casualties']) * 0.3) +
    (np.log1p(grid_df['event_count']) * 0.1)
)

# Filtre: Top %15
threshold = grid_df['risk_score'].quantile(0.85)
filtered_df = grid_df[grid_df['risk_score'] > threshold].copy()

print(f"   -> Risk Eşik Değeri (Threshold): {threshold:.2f}")
print(f"   -> Filtre Öncesi: {len(grid_df)} Grid")
print(f"   -> Filtre Sonrası: {len(filtered_df)} Grid (Sadece Kritik Bölgeler)")

# 5. Clustering
print("\n[3/4] K-Means Kümeleme ve Model Performansı...")
X = filtered_df[['risk_score', 'avg_severity']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
filtered_df['cluster'] = kmeans.fit_predict(X_scaled)

# Küme İsimlendirme
rank = filtered_df.groupby('cluster')['risk_score'].mean().sort_values().index
labels = {rank[0]: 'High Risk', rank[1]: 'Very High Risk', rank[2]: 'Extreme Danger Zone'}
filtered_df['risk_label'] = filtered_df['cluster'].map(labels)

# --- DETAYLI KONSOL RAPORU (GERİ GELDİ) ---
print("-" * 50)
print("📈 MODEL İSTATİSTİKLERİ")
print("-" * 50)

# Silhouette Score
score = silhouette_score(X_scaled, filtered_df['cluster'])
print(f"🔹 Silhouette Score: {score:.4f} (Ayrışma Kalitesi)")

# Küme Profilleri (Okunaklı Tablo)
summary = filtered_df.groupby('risk_label').agg({
    'risk_score': 'mean',
    'avg_severity': 'mean',
    'total_loss': 'mean',
    'total_casualties': 'mean',
    'event_count': 'count'
}).sort_values('risk_score', ascending=False)

# Para birimini okunur yapalım (Milyon $)
summary['total_loss'] = (summary['total_loss'] / 1e6).map('${:,.1f}M'.format)
summary['total_casualties'] = summary['total_casualties'].map('{:,.0f}'.format)
summary = summary.rename(columns={'event_count': 'grid_count'})

print("\n🔹 Küme Profilleri (Ortalamalar):")
print(summary)

print("\n🔹 Yorum:")
print(f"   👉 En tehlikeli '{labels[rank[2]]}' grubunda {summary.loc['Extreme Danger Zone', 'grid_count']} adet bölge var.")
print(f"   👉 Bu bölgelerdeki ortalama maddi kayıp: {summary.loc['Extreme Danger Zone', 'total_loss']}")

# 6. Kayıt
joblib.dump(kmeans, MODEL_DIR / "kmeans_grid_model.pkl")
filtered_df.to_csv(REPORT_DIR / "grid_risk_map_data.csv", index=False)

# 7. Görselleştirme
print("\n[4/4] HTML Harita Raporu Hazırlanıyor (Koordinatlı)...")

fig = px.scatter_geo(
    filtered_df,
    lat="lat",
    lon="lon",
    color="risk_label",
    size="risk_score",
    hover_name="grid_id",
    # Koordinatları ve detayları gösteren ayar
    hover_data={
        "lat": ":.2f",
        "lon": ":.2f",
        "total_loss": ":,.0f",
        "total_casualties": ":,.0f",
        "risk_score": ":.2f"
    },
    title=f"🌍 Global Disaster Hotspots (Top 15% Risk) - {len(filtered_df)} Zones Identified",
    projection="natural earth",
    color_discrete_map={
        "High Risk": "#FFA500",         # Turuncu
        "Very High Risk": "#FF4500",    # Koyu Turuncu
        "Extreme Danger Zone": "#8B0000" # Kan Kırmızısı
    },
    opacity=0.8
)

fig.update_geos(
    showcoastlines=True, coastlinecolor="#333333",
    showland=True, landcolor="#f4f4f4",
    showocean=True, oceancolor="#eef"
)
fig.update_layout(height=700, margin={"r":0,"t":50,"l":0,"b":0})

out_path = REPORT_DIR / "model_05_grid_map.html"
fig.write_html(out_path)
print(f"✅ Rapor Kaydedildi: {out_path}")
print("=" * 100)