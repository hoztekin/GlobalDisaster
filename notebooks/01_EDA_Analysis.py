# =============================================================================
# 01__EDA.py
# (BİRLEŞTİRİLMİŞ TAM ANALİZ: Veri Zenginleştirme + Detaylı Keşifsel Analiz)
# =============================================================================

"""
AKIŞ:
1. Veri Yükleme & API Zenginleştirme
2. Feature Engineering (Mevsimler, Log Dönüşüm)
3. Data Quality Check
4. Temporal Analysis
5. Geographic Analysis
6. Disaster Type Analysis
7. Statistical Tests (ANOVA / Kruskal-Wallis)
8. Correlation Analysis
9. Distribution Analysis
10. Outlier Detection
11. Key Insights & Save
"""

import os
import requests
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, kruskal, spearmanr
from pathlib import Path
from datetime import datetime

# 1. KRİTİK AYAR: Çökme Engelleyici (forrtl error çözümü)
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# Ayarlar
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# 2. KRİTİK AYAR: Dosya Yolu (NameError çözümü)

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "global_disaster.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "ultimate_eda"

# Klasörleri oluştur
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print(f"📂 PROJE DİZİNİ: {PROJECT_ROOT}")
print("=" * 100)

# =============================================================================
# BÖLÜM 1: VERİ YÜKLEME VE API (Zaman Aşımı Korumalı)
# =============================================================================
print("\n[1/11] VERİ YÜKLEME VE API ENTEGRASYONU")

# Dosya kontrolü
if not DATA_RAW.exists():
    print(f"❌ HATA: Dosya bulunamadı -> {DATA_RAW}")
    # Hata varsa durdur
    raise FileNotFoundError("Lütfen dosya yolunu kontrol et.")

df = pd.read_csv(DATA_RAW)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# World Bank API Yardımcıları
COUNTRY_TO_ISO3 = {
    "Australia": "AUS", "Bangladesh": "BGD", "Brazil": "BRA", "Canada": "CAN",
    "Chile": "CHL", "China": "CHN", "France": "FRA", "Germany": "DEU",
    "Greece": "GRC", "India": "IND", "Indonesia": "IDN", "Italy": "ITA",
    "Japan": "JPN", "Mexico": "MEX", "Nigeria": "NGA", "Philippines": "PHL",
    "South Africa": "ZAF", "Spain": "ESP", "Turkey": "TUR", "United States": "USA",
    "Pakistan": "PAK", "Vietnam": "VNM", "Thailand": "THA", "Nepal": "NPL",
    "Iran": "IRN", "United Kingdom": "GBR", "New Zealand": "NZL"
}


def fetch_wb_indicator(iso3, indicator, start, end):
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
    params = {"date": f"{start}:{end}", "format": "json", "per_page": "100"}
    try:
        # TIMEOUT EKLENDİ: 10 saniye cevap gelmezse pas geçer, program donmaz.
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200 and len(r.json()) > 1:
            return {int(x['date']): x['value'] for x in r.json()[1] if x['value']}
    except:
        return {}
    return {}


print("   🌐 World Bank verileri çekiliyor (Nüfus, Yüzölçümü)...")
records = []
countries = [c for c in df['country'].unique() if c in COUNTRY_TO_ISO3]

for c in countries:
    iso = COUNTRY_TO_ISO3[c]
    # Nüfus
    pop_map = fetch_wb_indicator(iso, "SP.POP.TOTL", df['year'].min(), df['year'].max())
    # Yüzölçümü (2020 sabit)
    area_map = fetch_wb_indicator(iso, "AG.SRF.TOTL.K2", 2020, 2020)
    area = list(area_map.values())[0] if area_map else None

    for y in range(df['year'].min(), df['year'].max() + 1):
        records.append({
            "country": c, "year": y,
            "population": pop_map.get(y),
            "surface_area_km2": area
        })

wb_df = pd.DataFrame(records)
df = df.merge(wb_df, on=['country', 'year'], how='left')
print(f"   ✅ Veri zenginleştirildi! (Population ve Surface Area eklendi)")

# Eksikleri doldur (Mean Imputation)
df['population'] = df['population'].fillna(df.groupby('country')['population'].transform('mean'))

# =============================================================================
# BÖLÜM 2: FEATURE ENGINEERING
# =============================================================================
print("\n[2/11] FEATURE ENGINEERING")

# Zaman
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear


def get_season(m):
    if m in [12, 1, 2]:
        return 'Winter'
    elif m in [3, 4, 5]:
        return 'Spring'
    elif m in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


df['season'] = df['month'].apply(get_season)

# Hesaplamalı Metrikler
df['population_density'] = df['population'] / (df['surface_area_km2'] + 1)
df['loss_per_capita'] = df['economic_loss_usd'] / (df['population'] + 1)
df['casualties_per_100k'] = (df['casualties'] / (df['population'] + 1)) * 100000

# Log Dönüşümleri (Modeller için hazırlık)
df['casualties_log'] = np.log1p(df['casualties'])
df['economic_loss_log'] = np.log1p(df['economic_loss_usd'])

print("   ✅ Yeni özellikler üretildi: Season, Density, Loss Per Capita...")

# =============================================================================
# BÖLÜM 3: DATA QUALITY CHECK
# =============================================================================
print("\n[3/11] DATA QUALITY CHECK")

missing = df.isnull().sum()
print(f"   🔍 Eksik Değerler:\n{missing[missing > 0]}")
print(f"   🔍 Tekrar Eden Satırlar: {df.duplicated().sum()}")
print(f"   📊 Veri Aralığı: {df['date'].min().date()} - {df['date'].max().date()}")

# =============================================================================
# BÖLÜM 4: TEMPORAL ANALYSIS
# =============================================================================
print("\n[4/11] TEMPORAL ANALYSIS")

yearly_counts = df['year'].value_counts().sort_index()
print(f"   📅 Yıllara Göre Afet Sayıları:\n{yearly_counts}")

seasonal_stats = df.groupby('season')[['severity_index', 'casualties']].mean()
print(f"   🌸 Mevsimsel Ortalamalar:\n{seasonal_stats}")

# =============================================================================
# BÖLÜM 5: GEOGRAPHIC ANALYSIS
# =============================================================================
print("\n[5/11] GEOGRAPHIC ANALYSIS")

top_risk_countries = df.groupby('country')['severity_index'].mean().sort_values(ascending=False).head(5)
print(f"   🌍 En Yüksek Ortalama Şiddet (Top 5):\n{top_risk_countries}")

# Yeni eklenen veri ile analiz
densest_impact = df.groupby('country')[['population_density', 'casualties']].mean().sort_values('population_density',
                                                                                                ascending=False).head(5)
print(f"   👥 En Yoğun Nüfuslu Ülkelerde Ortalama Kayıp:\n{densest_impact}")

# =============================================================================
# BÖLÜM 6: DISASTER TYPE ANALYSIS
# =============================================================================
print("\n[6/11] DISASTER TYPE ANALYSIS")

type_stats = df.groupby('disaster_type').agg({
    'severity_index': 'mean',
    'economic_loss_usd': 'sum',
    'casualties': 'sum'
}).sort_values('economic_loss_usd', ascending=False)
print(f"   🌪️ Afet Tipine Göre İstatistikler:\n{type_stats}")

# =============================================================================
# BÖLÜM 7: STATISTICAL TESTS (ANOVA)
# =============================================================================
print("\n[7/11] STATISTICAL TESTS (ANOVA & Kruskal-Wallis)")

# Test 1: Mevsimler arası şiddet farkı var mı?
groups = [df[df['season'] == s]['severity_index'] for s in df['season'].unique()]
f_stat, p_val = f_oneway(*groups)
print(f"   🧪 ANOVA (Season vs Severity): P-value = {p_val:.5f}")
if p_val < 0.05:
    print("      ✅ SONUÇ: Mevsimler arasında afet şiddeti açısından anlamlı bir fark VAR.")
else:
    print("      ❌ SONUÇ: Mevsimsel fark istatistiksel olarak anlamlı değil.")

# =============================================================================
# BÖLÜM 8: CORRELATION ANALYSIS
# =============================================================================
print("\n[8/11] CORRELATION ANALYSIS")

# Sadece sayısal kolonlar
num_cols = ['severity_index', 'casualties', 'economic_loss_usd', 'population_density', 'response_time_hours']
corr = df[num_cols].corr()

print("   🔗 Korelasyon Matrisi:")
print(corr)

# Heatmap kaydet
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig(REPORT_DIR / "correlation_matrix.png")
print("   💾 Heatmap kaydedildi: correlation_matrix.png")

# =============================================================================
# BÖLÜM 9: DISTRIBUTION ANALYSIS
# =============================================================================
print("\n[9/11] DISTRIBUTION ANALYSIS")

print(f"   📊 Severity Skewness: {df['severity_index'].skew():.2f}")
print(f"   📊 Casualties Skewness (Original): {df['casualties'].skew():.2f}")
print(f"   📊 Casualties Skewness (Log Transformed): {df['casualties_log'].skew():.2f}")

# =============================================================================
# BÖLÜM 10: OUTLIER DETECTION
# =============================================================================
print("\n[10/11] OUTLIER DETECTION (IQR Method)")


def count_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((df[col] < lower) | (df[col] > upper)).sum()


print(f"   🚨 Outliers in Economic Loss: {count_outliers('economic_loss_usd')}")
print(f"   🚨 Outliers in Casualties: {count_outliers('casualties')}")

# =============================================================================
# BÖLÜM 11: KEY INSIGHTS & SAVE
# =============================================================================
print("\n[11/11] SONUÇ VE KAYIT")

output_path = DATA_PROCESSED / "disaster_data_final.csv"
df.to_csv(output_path, index=False)

summary = f"""
EDA RAPORU
-------------------
Tarih: {datetime.now()}
Toplam Veri: {len(df)}
Zenginleştirme: World Bank Nüfus ve Yüzölçümü Eklendi.

Önemli Bulgular:
1. En maliyetli afet tipi: {type_stats.index[0]}
2. Mevsimsel fark (ANOVA): {'Var' if p_val < 0.05 else 'Yok'}
3. Nüfus yoğunluğu ile kayıp korelasyonu: {corr.loc['population_density', 'casualties']:.3f}

Kayıt Yeri: {output_path}
"""

with open(REPORT_DIR / "eda_summary.txt", "w") as f:
    f.write(summary)

print(f"   ✅ Temizlenmiş ve Zenginleştirilmiş Veri Kaydedildi: {output_path}")
print(f"   ✅ Özet Rapor: {REPORT_DIR / 'eda_summary.txt'}")

print("\n" + "=" * 100)