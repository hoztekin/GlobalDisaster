# =============================================================================
# 03_Model_04_CountryClustering.py
# (Country Risk & Capacity Clustering)
# =============================================================================

import os
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

# Path Ayarı
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "disaster_data_final.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_04"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print(f"📂 ÇALIŞMA DİZİNİ: {PROJECT_ROOT}")
print("🤖 MODEL 04: COUNTRY RISK CLUSTERING (K-MEANS) - ANALİZ MODU")
print("=" * 100)

# 1. Veri Yükle
if not DATA_FILE.exists():
    raise FileNotFoundError(f"❌ Dosya bulunamadı: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

# 2. Aggregation
agg_dict = {
    'severity_index': 'mean',
    'economic_loss_usd': 'mean',
    'casualties': 'mean',
    'military_power_index': 'mean',
    'human_capital_index': 'mean',
    'response_time_hours': 'median',
    'experience_index': 'max'
}
available_cols = [c for c in agg_dict.keys() if c in df.columns]
final_agg = {c: agg_dict[c] for c in available_cols}

country_profile = df.groupby('country').agg(final_agg).reset_index()
country_profile = country_profile.fillna(country_profile.median(numeric_only=True))

print(f"✅ Analiz Edilen Ülke Sayısı: {len(country_profile)}")

# 3. Modelleme Hazırlığı
features = [col for col in ['severity_index', 'military_power_index', 'experience_index', 'response_time_hours'] if col in country_profile.columns]
print(f"📊 Modele Giren Özellikler:\n   {features}")

X = country_profile[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means (4 Cluster)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
country_profile['cluster'] = kmeans.fit_predict(X_scaled)
country_profile['cluster_str'] = "Cluster " + country_profile['cluster'].astype(str)

# --- DETAYLI KONSOL ÇIKTILARI (ARTIK SESSİZ DEĞİL) ---
print("\n" + "-" * 50)
print("📈 MODEL PERFORMANS VE İSTATİSTİKLERİ")
print("-" * 50)

# Silhouette Score (Kümeleme Kalitesi)
score = silhouette_score(X_scaled, country_profile['cluster'])
print(f"🔹 Silhouette Score: {score:.4f} (1'e ne kadar yakınsa ayrışma o kadar iyi)")

# Küme Profilleri
print("\n🔹 Küme Ortalamaları (Cluster Profiles):")
# Okunabilirlik için sayıları yuvarla
summary = country_profile.groupby('cluster')[features].mean().reset_index()
print(summary.to_string(index=False))

print("\n🔹 Hangi Kümede Kaç Ülke Var?")
print(country_profile['cluster'].value_counts().sort_index())

# Küme Yorumlama (Otomatik Analiz)
print("\n🔹 KÜME YORUMLARI (OTOMATİK ANALİZ):")
for i, row in summary.iterrows():
    risk_level = "YÜKSEK" if row['severity_index'] > 0 else "DÜŞÜK" # Scale edilmiş veride 0 ortalamadır ama burada ham veri var
    cap_level = "YÜKSEK" if row['military_power_index'] > summary['military_power_index'].mean() else "DÜŞÜK"
    print(f"   👉 Cluster {i}: Ortalama Risk {row['severity_index']:.2f} ({risk_level}) | Askeri Kapasite: {cap_level}")

# 5. Kayıt (Model ve CSV)
joblib.dump(kmeans, MODEL_DIR / "kmeans_country_model.pkl")
joblib.dump(scaler, MODEL_DIR / "kmeans_country_scaler.pkl")
country_profile.to_csv(REPORT_DIR / "country_risk_clusters.csv", index=False)

# 6. Görselleştirme
print("\n🎨 HTML Raporlar Hazırlanıyor...")

if 'military_power_index' in country_profile.columns:
    fig_scatter = px.scatter(
        country_profile,
        x="military_power_index",
        y="severity_index",
        color="cluster_str",
        size="economic_loss_usd",
        hover_name="country",
        title="🤖 Model 04: Capacity vs Risk Matrix",
        labels={"military_power_index": "Military Capacity (Std)", "severity_index": "Avg Severity (Risk)"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    scatter_path = REPORT_DIR / "model_04_cluster_scatter.html"
    fig_scatter.write_html(scatter_path)
    print(f"   📄 Scatter Raporu: {scatter_path}")

fig_map = px.choropleth(
    country_profile,
    locations="country",
    locationmode="country names",
    color="cluster_str",
    hover_name="country",
    hover_data=["military_power_index", "severity_index"],
    title="🌍 Model 04: Global Risk & Capacity Clusters Map",
    projection="natural earth",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_map.update_geos(showframe=False, showcoastlines=True)
fig_map.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})

map_path = REPORT_DIR / "model_04_cluster_map.html"
fig_map.write_html(map_path)
print(f"   📄 Harita Raporu: {map_path}")

print("\n" + "=" * 100)
print("✅ İŞLEM BAŞARIYLA TAMAMLANDI")
print("=" * 100)