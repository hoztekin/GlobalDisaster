# =============================================================================
# MODEL 02: GEOGRAPHIC RISK CLUSTERING (K-Means) - FIXED VERSION
# =============================================================================

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Windows Çökme Koruması
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')


class Config:
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

    INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'disaster_data_final.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'models'
    REPORT_DIR = PROJECT_ROOT / 'reports' / 'model_02_clustering'

    N_CLUSTERS = 3
    RANDOM_STATE = 42


def run_clustering_pipeline():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🌍 MODEL 02: RISK CLUSTERING (FIXED VERSION) BAŞLATILIYOR")
    print("=" * 80)

    if not Config.INPUT_FILE.exists():
        raise FileNotFoundError(f"❌ Veri bulunamadı: {Config.INPUT_FILE}")

    df = pd.read_csv(Config.INPUT_FILE)
    print(f"✅ Veri yüklendi: {len(df):,} satır")

    # Feature Engineering (Country Profile)
    country_profile = df.groupby('country').agg({
        'disaster_type': 'count',
        'severity_index': 'mean',
        'economic_loss_usd': 'sum',
        'casualties': 'sum',
        'population_density': 'mean'
    }).reset_index()

    country_profile.columns = ['country', 'frequency', 'avg_severity', 'total_loss', 'total_casualties', 'density']

    # Log Dönüşümleri
    country_profile['log_loss'] = np.log1p(country_profile['total_loss'])
    country_profile['log_casualties'] = np.log1p(country_profile['total_casualties'])
    country_profile = country_profile.fillna(country_profile.mean(numeric_only=True))

    # Clustering
    features = ['frequency', 'avg_severity', 'log_loss', 'log_casualties', 'density']
    X = country_profile[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🧩 K-Means Modeli Çalıştırılıyor...")
    kmeans = KMeans(n_clusters=Config.N_CLUSTERS, random_state=Config.RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    country_profile['cluster_id'] = clusters

    print("\n📊 Composite Risk Score Hesaplanıyor...")
    country_profile['risk_score'] = (
            (country_profile['frequency'] / country_profile['frequency'].max()) * 0.35 +  # 35%: frequency
            (country_profile['avg_severity'] / country_profile['avg_severity'].max()) * 0.35 +  # 35%: severity
            (country_profile['log_loss'] / country_profile['log_loss'].max()) * 0.20 +  # 20%: economic loss
            (country_profile['log_casualties'] / country_profile['log_casualties'].max()) * 0.10  # 10%: casualties
    )

    # Cluster'ı composite risk_score'a göre sırala
    cluster_risk = country_profile.groupby('cluster_id')['risk_score'].mean().sort_values().index
    remap = {old: new for new, old in enumerate(cluster_risk)}
    country_profile['risk_level'] = country_profile['cluster_id'].map(remap)

    risk_labels = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
    country_profile['risk_label'] = country_profile['risk_level'].map(risk_labels)

    print("\n📊 Risk Dağılımı:")
    print(country_profile['risk_label'].value_counts())

   # Görselleştirme - Interactive Tooltip ile
    fig = px.choropleth(
        country_profile, locations="country", locationmode='country names',
        color="risk_label", title="🌍 Global Disaster Risk Clusters",
        color_discrete_map={'Low Risk': '#2ecc71', 'Moderate Risk': '#f1c40f', 'High Risk': '#e74c3c',
                            'Critical Risk': '#d70015'},
        hover_data={
            'country': True,
            'risk_label': True,
            'frequency': ':.0f',
            'avg_severity': ':.2f',
            'log_loss': ':.2f',
            'risk_score': ':.3f'
        },
        hover_name='country'
    )
    fig.update_layout(height=700, width=1400, showlegend=True)
    fig.write_html(Config.REPORT_DIR / "risk_cluster_map.html")

    # Kayıt
    country_profile.to_csv(Config.OUTPUT_DIR / 'country_risk_clusters.csv', index=False)
    with open(Config.OUTPUT_DIR / 'risk_kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"\n💾 Harita ve Veri Kaydedildi: {Config.REPORT_DIR}")
    print("✅ MODEL 02 SÜRECİ TAMAMLANDI (FIXED)!")

    # Debug: Top 10 country risk scores
    print("\n📈 Top 10 Ülke Risk Scores:")
    top10 = country_profile.nlargest(10, 'risk_score')[
        ['country', 'risk_score', 'frequency', 'avg_severity', 'risk_label']]
    print(top10.to_string())


if __name__ == '__main__':
    run_clustering_pipeline()