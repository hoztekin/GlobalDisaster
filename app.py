import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import numpy as np
import asyncio
import aiohttp
from pathlib import Path

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Global Disaster AI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TASARIM İYİLEŞTİRMELERİ ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); color: white; }
    div[data-testid="stMetric"] { background-color: #262730; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; }
    h1, h2, h3 { color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# --- PATH TANIMLARI ---
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data/processed/disaster_data_final.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- KURTARICI VERİ SETİ (FALLBACK) ---
# API veya CSV çalışmazsa bu veriler devreye girer. UNKNOWN hatasını çözer.
FALLBACK_METADATA = {
    "Turkey": {"region": "Europe & Central Asia", "income_group": "Upper middle income", "population": 85000000,
               "population_density": 110, "surface_area_km2": 783562},
    "USA": {"region": "North America", "income_group": "High income", "population": 331000000, "population_density": 36,
            "surface_area_km2": 9833520},
    "Japan": {"region": "East Asia & Pacific", "income_group": "High income", "population": 125000000,
              "population_density": 338, "surface_area_km2": 377975},
    "China": {"region": "East Asia & Pacific", "income_group": "Upper middle income", "population": 1400000000,
              "population_density": 148, "surface_area_km2": 9596960},
    "India": {"region": "South Asia", "income_group": "Lower middle income", "population": 1380000000,
              "population_density": 464, "surface_area_km2": 3287263},
    "Germany": {"region": "Europe & Central Asia", "income_group": "High income", "population": 83000000,
                "population_density": 240, "surface_area_km2": 357022},
    "Brazil": {"region": "Latin America & Caribbean", "income_group": "Upper middle income", "population": 212000000,
               "population_density": 25, "surface_area_km2": 8515767},
    "Indonesia": {"region": "East Asia & Pacific", "income_group": "Lower middle income", "population": 273000000,
                  "population_density": 151, "surface_area_km2": 1904569}
}


# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    if not DATA_PATH.exists(): return None, {}
    df = pd.read_csv(DATA_PATH)

    country_meta = df.sort_values('year', ascending=False).groupby('country').first()
    cols = ['region', 'income_group', 'population', 'population_density', 'surface_area_km2']

    for c in cols:
        if c not in country_meta.columns: country_meta[c] = np.nan

    meta_dict = country_meta[cols].to_dict('index')

    # Fallback verilerini ana sözlüğe yedir (Unknown düzeltmesi)
    for ctry, data in FALLBACK_METADATA.items():
        if ctry not in meta_dict or pd.isna(meta_dict[ctry].get('region')) or meta_dict[ctry].get(
                'region') == "Unknown":
            meta_dict[ctry] = data

    return df, meta_dict


df, country_metadata = load_data()


def show_html_report(file_path, height=600):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=height, scrolling=True)
    except:
        st.info(f"Grafik bekleniyor: {file_path.name}")


# --- SIDEBAR MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2776/2776000.png", width=80)
    st.title("Resque Data")

    selected = option_menu(
        menu_title="Menü",
        options=["Dashboard", "Risk Haritası", "Trend Analizi", "AI Simülasyon"],
        icons=["bar-chart-fill", "map-fill", "graph-up-arrow", "cpu-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#262730"},
            "nav-link-selected": {"background-color": "#FF4B4B"},
        }
    )

# =============================================================================
# 1. DASHBOARD
# =============================================================================
if selected == "Dashboard":
    st.title("📊 Yönetici Özeti")

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Kayıp", f"${df['economic_loss_usd'].sum() / 1e9:.1f}B")
        c2.metric("Toplam Can Kaybı", f"{df['casualties'].sum() / 1e6:.1f}M")
        c3.metric("Yardım Açığı", "$240B", "Kritik")
        c4.metric("Riskli Ülke Sayısı", "12")

    t1, t2, t3, t4 = st.tabs(["Ekonomik", "Operasyonel", "ROI", "Coğrafi"])
    with t1:
        c_a, c_b = st.columns([2, 1])
        with c_a: show_html_report(REPORTS_DIR / "strategic_analysis/1_waterfall_country_economic_loss.html", 500)
        with c_b: show_html_report(REPORTS_DIR / "strategic_analysis/2_disaster_type_cost_comparison.html", 500)
    with t2:
        show_html_report(REPORTS_DIR / "strategic_analysis/10_recovery_cost_timeline.html", 500)
        c_c, c_d = st.columns(2)
        with c_c: show_html_report(REPORTS_DIR / "strategic_analysis/9_aid_gap_waterfall.html", 500)
        with c_d: show_html_report(REPORTS_DIR / "strategic_analysis/2_aid_efficiency_radar.html", 500)
    with t3:
        show_html_report(REPORTS_DIR / "strategic_analysis/3_roi_analysis.html", 500)
    with t4:
        show_html_report(REPORTS_DIR / "strategic_analysis/5_geographic_impact_map.html", 600)

# =============================================================================
# 2. RİSK HARİTASI
# =============================================================================
elif selected == "Risk Haritası":
    st.title("🗺️ Risk Bölgeleri (Clustering)")
    show_html_report(REPORTS_DIR / "model_02_clustering/risk_cluster_map.html", 700)

# =============================================================================
# 3. TREND ANALİZİ
# =============================================================================
elif selected == "Trend Analizi":
    st.title("📈 Gelecek Trendleri")
    st.subheader("1. Global Afet Sıklığı Tahmini")
    show_html_report(REPORTS_DIR / "model_03_trend/global_trend_forecast.html", 500)

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("2. Yükselen Afetler (Momentum)")
        show_html_report(REPORTS_DIR / "model_03_trend/disaster_momentum.html", 500)
    with c2:
        st.subheader("📋 Veri Tablosu")
        if (MODELS_DIR / "disaster_momentum_analysis.csv").exists():
            st.dataframe(pd.read_csv(MODELS_DIR / "disaster_momentum_analysis.csv"), height=400)

# =============================================================================
# 4. AI SİMÜLASYON (AKILLI & HASSAS)
# =============================================================================
elif selected == "AI Simülasyon":
    st.title("⚡ AI Risk Simülatörü")
    st.caption("Senaryo bazlı risk analizi ve karar destek sistemi.")

    try:
        with open(MODELS_DIR / 'severity_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(MODELS_DIR / 'severity_prediction_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(MODELS_DIR / 'model_01_features.pkl', 'rb') as f:
            feats = pickle.load(f)
    except:
        st.error("Model dosyaları eksik."); st.stop()

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            # Fallback listesini kullan, Türkiye en üstte
            country_list = list(FALLBACK_METADATA.keys()) + [c for c in list(country_metadata.keys()) if
                                                             c not in FALLBACK_METADATA]
            country = st.selectbox("🏳️ Ülke", country_list, index=0)
            dtype = st.selectbox("🌪️ Afet Tipi", ['Earthquake', 'Flood', 'Tornado', 'Wildfire'])
        with c2:
            loss = st.number_input("💰 Kayıp ($)", 0, value=10000, step=10000)
            dead = st.number_input("💀 Can Kaybı", 0, value=100)  # Varsayılan 100
        with c3:
            time = st.slider("⏱️ Müdahale (Saat)", 1, 72, 12)
            season = st.selectbox("🍂 Mevsim", ['Winter', 'Summer', 'Spring', 'Autumn'])

        # Seçilen ülkenin verilerini göster
        meta = country_metadata.get(country, FALLBACK_METADATA.get("Turkey"))
        st.info(f"ℹ️ **{country} Profili:** Bölge: `{meta.get('region')}` | Gelir: `{meta.get('income_group')}`")

        if st.button("🚀 RİSKİ HESAPLA", type="primary", use_container_width=True):

            # --- HAZIRLIK ---
            input_df = pd.DataFrame(columns=feats)
            input_df.loc[0] = 0

            # --- AKILLI DEĞER ATAMA (Smart Imputation) ---
            # 1. İyileşme Süresi: Can kaybı varsa süre ciddi uzar.
            base_recovery = 30  # Standart
            if dead > 0: base_recovery += (dead * 2)  # Her ölü için 2 gün ekle
            if loss > 1000000: base_recovery += 100

            # 2. Yardım: Kayıp ve cana göre
            aid = (loss * 0.2) + (dead * 5000)

            input_df['economic_loss_usd'] = loss
            input_df['casualties'] = dead
            input_df['response_time_hours'] = time
            input_df['recovery_days'] = base_recovery
            input_df['aid_amount_usd'] = aid
            input_df['population'] = meta.get('population', 5e7)
            input_df['population_density'] = meta.get('population_density', 100)
            input_df['surface_area_km2'] = meta.get('surface_area_km2', 5e5)

            # One-Hot Encoding
            cats = [f'disaster_type_{dtype}', f'country_{country}', f'season_{season}',
                    f'region_{meta.get("region", "")}', f'income_group_{meta.get("income_group", "")}']
            for c in cats:
                if c in input_df.columns: input_df[c] = 1

            # --- TAHMİN ---
            pred = model.predict(scaler.transform(input_df))[0]

            # --- İŞ KURALI (BUSINESS RULE - FAIL SAFE) ---
            # Model ne derse desin, insani kayıp belirli eşiği geçerse risk yüksektir.
            # Bu kurallar modelin "ekonomik odaklı" önyargısını kırar.

            rule_applied = False
            if dead >= 50 and pred < 2:
                pred = 2  # En az YÜKSEK
                rule_applied = True
            if dead >= 500 and pred < 3:
                pred = 3  # En az KRİTİK
                rule_applied = True

            # --- SONUÇ ---
            labels = {0: "DÜŞÜK (LOW)", 1: "ORTA (MEDIUM)", 2: "YÜKSEK (HIGH)", 3: "KRİTİK (CRITICAL)"}
            colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

            st.markdown(f"""
            <div style="background-color: {colors[pred]}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <h2 style="color: white; margin:0;">TAHMİN: {labels[pred]} RİSK</h2>
            </div>
            """, unsafe_allow_html=True)

            if rule_applied:
                st.caption(
                    "⚠️ *Not: Yüksek can kaybı nedeniyle risk seviyesi güvenlik protokolleri gereği otomatik yükseltilmiştir.*")

            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.warning(f"**Tahmini İyileşme:** {int(base_recovery)} Gün")
            with c_res2:
                st.success(f"**Gereken Min. Yardım:** ${aid:,.0f}")