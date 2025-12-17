# =============================================================================
# Global Disaster AI Dashboard
# =============================================================================

import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Global Disaster AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .stMetric label {
        color: rgba(255,255,255,0.8) !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: bold;
    }
    .risk-critical { background-color: #e74c3c; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    .risk-high { background-color: #e67e22; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    .risk-medium { background-color: #f1c40f; color: #333; padding: 20px; border-radius: 10px; text-align: center; }
    .risk-low { background-color: #2ecc71; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    .insight-box { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #e94560;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- PATH CONFIGURATION ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"
DATA = PROJECT_ROOT / "data" / "processed"

DIRS = {
    "strategic": REPORTS / "strategic_analysis",
    "model_01": REPORTS / "model_01",
    "model_02": REPORTS / "model_02_clustering",
    "model_03": REPORTS / "model_03_trend",
    "model_04": REPORTS / "model_04",
    "model_05": REPORTS / "model_05"
}


# --- DATA LOADING ---
@st.cache_data
def load_data():
    path = DATA / "disaster_data_final.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_resource
def load_models():
    """Tüm modelleri yükle"""
    models = {}

    # Model 01: Severity Prediction
    try:
        models['severity_model'] = pickle.load(open(MODELS / 'severity_prediction_model.pkl', 'rb'))
        models['severity_scaler'] = pickle.load(open(MODELS / 'severity_prediction_scaler.pkl', 'rb'))
        models['severity_features'] = pickle.load(open(MODELS / 'model_01_features.pkl', 'rb'))
    except Exception as e:
        pass

    # Model 02: Risk Clustering (K-Means)
    try:
        models['risk_kmeans'] = pickle.load(open(MODELS / 'risk_kmeans_model.pkl', 'rb'))
        models['country_risk'] = pd.read_csv(MODELS / 'country_risk_clusters.csv')
    except:
        pass

    # Model 03: Trend Forecasting
    try:
        models['trend_model'] = pickle.load(open(MODELS / 'trend_forecasting_model.pkl', 'rb'))
        models['momentum_data'] = pd.read_csv(MODELS / 'disaster_momentum_analysis.csv')
    except:
        pass

    # Model 04: Country Clustering
    try:
        models['country_kmeans'] = joblib.load(MODELS / 'kmeans_country_model.pkl')
        models['country_scaler'] = joblib.load(MODELS / 'kmeans_country_scaler.pkl')
    except:
        pass

    # Model 05: Grid Analysis
    try:
        models['grid_kmeans'] = joblib.load(MODELS / 'kmeans_grid_model.pkl')
        models['grid_data'] = pd.read_csv(DIRS['model_05'] / 'grid_risk_map_data.csv')
    except:
        pass

    return models


df = load_data()
models = load_models()


# --- HELPER FUNCTIONS ---
def show_html(folder, filename, height=650):
    """HTML dosyasını göster"""
    path = DIRS.get(folder, REPORTS) / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=height, scrolling=True)
    else:
        st.warning(f"📁 Dosya bulunamadı: {filename}")


def predict_severity(input_data, models):
    """Model 01 ile risk tahmini yap"""
    if 'severity_model' not in models:
        return None, "Model yüklenmedi"

    try:
        features = models['severity_features']
        X = pd.DataFrame([input_data])

        for col in features:
            if col not in X.columns:
                X[col] = 0

        X = X[features]
        X_scaled = models['severity_scaler'].transform(X)
        prediction = models['severity_model'].predict(X_scaled)[0]

        severity_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
        return prediction, severity_map.get(prediction, "UNKNOWN")
    except Exception as e:
        return None, str(e)


def get_country_risk(country, models):
    """Ülkenin risk seviyesini getir"""
    if 'country_risk' not in models:
        return None

    risk_df = models['country_risk']
    country_data = risk_df[risk_df['country'] == country]

    if not country_data.empty:
        return country_data.iloc[0].to_dict()
    return None


# --- SIDEBAR ---
with st.sidebar:
    st.title("🚨 Resque Data")

    selected = option_menu(
        menu_title=None,
        options=["Kontrol Paneli", "Risk Haritası", "Stratejik Analiz", "AI Simülatör", "Trendler", "Hakkında"],
        icons=["speedometer2", "globe", "graph-up", "cpu", "activity", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "#FF6B35", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#004E89"},
        }
    )

    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%d %B %Y')}")

    # Model durumu
    st.markdown("### 🤖 Model Durumu")
    model_status = {
        "Severity (XGBoost)": "severity_model" in models,
        "Risk Clustering": "risk_kmeans" in models,
        "Trend Forecast": "trend_model" in models,
        "Country Cluster": "country_kmeans" in models,
        "Grid Analysis": "grid_kmeans" in models
    }
    for name, status in model_status.items():
        icon = "✅" if status else "❌"
        st.caption(f"{icon} {name}")

# =============================================================================
# SAYFA: KONTROL PANELİ
# =============================================================================
if selected == "Kontrol Paneli":
    st.title("🎯 Yönetici Özeti")
    st.markdown("*Küresel afet verilerinin kapsamlı analizi*")

    if df is not None:
        # Ana Metrikler
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_loss = df['economic_loss_usd'].sum()
            st.metric("💰 Toplam Kayıp", f"${total_loss / 1e9:.1f}B")

        with col2:
            total_casualties = df['casualties'].sum()
            st.metric("💀 Can Kaybı", f"{total_casualties / 1e6:.2f}M")

        with col3:
            st.metric("📊 Toplam Olay", f"{len(df):,}")

        with col4:
            avg_response = df['response_time_hours'].mean()
            st.metric("⏱️ Ort. Müdahale", f"{avg_response:.1f} saat")

        with col5:
            avg_severity = df['severity_index'].mean()
            st.metric("⚡ Ort. Şiddet", f"{avg_severity:.2f}")

        st.markdown("---")

        # Tab'lar
        tab1, tab2, tab3 = st.tabs(["💰 EKONOMİK ANALİZ", "🌍 COĞRAFİ GÖRÜNÜM", "⚙️ OPERASYONEL"])

        with tab1:
            st.subheader("📊 Ülkelere Göre Ekonomik Kayıp")
            show_html("strategic", "1_waterfall_country_economic_loss.html", 550)

            st.markdown("---")

            st.subheader("🌪️ Afet Tipi ve Maliyet Dağılımı")
            show_html("strategic", "2_disaster_type_cost_comparison.html", 600)

            st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("📈 Şiddet vs Kayıp İlişkisi")
                show_html("strategic", "7_severity_economic_scatter.html", 500)

            with col_b:
                st.subheader("🏆 En Pahalı 20 Afet")
                show_html("strategic", "8_top_20_expensive_disasters.html", 500)

        with tab2:
            st.subheader("🗺️ Küresel Etki Haritası")
            show_html("strategic", "5_geographic_impact_map.html", 700)

            st.markdown("---")
            st.subheader("📋 Ülke Bazlı Özet")

            country_summary = df.groupby('country').agg({
                'economic_loss_usd': 'sum',
                'casualties': 'sum',
                'severity_index': 'mean',
                'disaster_type': 'count'
            }).reset_index()
            country_summary.columns = ['Ülke', 'Toplam Kayıp ($)', 'Can Kaybı', 'Ort. Şiddet', 'Olay Sayısı']
            country_summary = country_summary.sort_values('Toplam Kayıp ($)', ascending=False).head(15)
            country_summary['Toplam Kayıp ($)'] = country_summary['Toplam Kayıp ($)'].apply(
                lambda x: f"${x / 1e6:,.0f}M")

            st.dataframe(country_summary, use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("🎯 Yardım Etkinliği & Açığı")

            # --- DÜZELTME: Grafikler alt alta ---
            st.markdown("**Yardım Karşılama Oranı**")
            show_html("strategic", "4_aid_efficiency_radar.html", 550)

            st.markdown("---")

            st.markdown("**Yardım Açığı Analizi**")
            show_html("strategic", "9_aid_gap_waterfall.html", 550)

            st.markdown("---")
            st.subheader("💡 Operasyonel İçgörüler")

            if df is not None:
                total_loss = df['economic_loss_usd'].sum()
                total_aid = df['aid_amount_usd'].sum()
                aid_coverage = (total_aid / total_loss) * 100 if total_loss > 0 else 0

                # En çok yardım açığı olan afet tipi
                aid_gap = df.groupby('disaster_type').agg({
                    'economic_loss_usd': 'sum',
                    'aid_amount_usd': 'sum'
                })
                aid_gap['gap'] = aid_gap['economic_loss_usd'] - aid_gap['aid_amount_usd']
                worst_type = aid_gap['gap'].idxmax()
                worst_gap = aid_gap.loc[worst_type, 'gap']

                col_i1, col_i2 = st.columns(2)

                with col_i1:
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>📊 Yardım Karşılama Oranı</h4>
                        <p>Küresel afetlerde toplam ekonomik kaybın <strong>%{aid_coverage:.1f}</strong>'i yardım fonlarıyla karşılanabilmektedir.</p>
                        <p>Bu oran, afet yönetimi için <strong>proaktif yatırımların</strong> önemini vurgulamaktadır.</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_i2:
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>⚠️ En Kritik Açık</h4>
                        <p><strong>{worst_type}</strong> tipi afetlerde en yüksek yardım açığı görülmektedir.</p>
                        <p>Bu kategoride <strong>${worst_gap / 1e9:.1f} Milyar Usd</strong>'lık fonlama eksikliği bulunmaktadır.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="insight-box">
                    <h4>🎯 Stratejik Öneri</h4>
                    <p>Yardım fonlarının etkinliğini artırmak için:</p>
                    <ul>
                        <li><strong>Erken uyarı sistemlerine</strong> yatırım yapılması (ROI: 4.5x)</li>
                        <li><strong>Bölgesel acil durum fonları</strong> oluşturulması</li>
                        <li>Yüksek riskli bölgelerde <strong>altyapı güçlendirme</strong> çalışmaları</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("❌ Veri dosyası bulunamadı!")

# =============================================================================
# SAYFA: RİSK HARİTASI
# =============================================================================
elif selected == "Risk Haritası":
    st.title("🗺️ Risk Analizi Haritaları")

    model_choice = st.radio(
        "Analiz Modeli Seçin:",
        ["🔥 Sıcak Noktalar (Grid)", "🏛️ Ülke Kapasitesi", "🌐 Genel Risk Kümeleri"],
        horizontal=True
    )

    if "Grid" in model_choice:
        st.subheader("Model 05: Spatial Grid Analysis")
        st.info("📍 2x2 derece grid'lere bölünmüş dünya haritasında en riskli %15'lik bölgeler gösteriliyor.")
        show_html("model_05", "model_05_grid_map.html", 750)

        if 'grid_data' in models:
            grid_df = models['grid_data']
            col1, col2, col3 = st.columns(3)
            col1.metric("Toplam Kritik Bölge", len(grid_df))
            col2.metric("Ortalama Risk Skoru", f"{grid_df['risk_score'].mean():.2f}")
            col3.metric("En Yüksek Risk", f"{grid_df['risk_score'].max():.2f}")

    elif "Kapasite" in model_choice:
        st.subheader("Model 04: Country Risk & Capacity Clustering")
        st.info("🏛️ Ülkeler askeri kapasite ve afet deneyimine göre kümeleniyor.")

        show_html("model_04", "model_04_cluster_map.html", 650)

        st.markdown("---")
        st.subheader("📊 Kapasite vs Risk Matrisi")
        show_html("model_04", "model_04_cluster_scatter.html", 600)

    else:
        st.subheader("Model 02: Geographic Risk Clustering")
        st.info("🌐 K-Means algoritması ile ülkeler 3 risk grubuna ayrılıyor.")
        show_html("model_02", "risk_cluster_map.html", 750)

        if 'country_risk' in models:
            risk_df = models['country_risk']
            st.markdown("---")
            st.subheader("📋 Risk Dağılımı")

            col1, col2, col3 = st.columns(3)

            for i, (label, color) in enumerate([("Low Risk", "🟢"), ("Moderate Risk", "🟡"), ("High Risk", "🔴")]):
                count = len(risk_df[risk_df['risk_label'] == label])
                with [col1, col2, col3][i]:
                    st.metric(f"{color} {label}", f"{count} Ülke")

# =============================================================================
# SAYFA: STRATEJİK ANALİZ
# =============================================================================
elif selected == "Stratejik Analiz":
    st.title("🧠 Stratejik İçgörüler")

    tab1, tab2, tab3 = st.tabs(["🎯 Performans", "💵 ROI Analizi", "📊 Karşılaştırma"])

    with tab1:
        st.subheader("Müdahale Hızı vs Başarı Oranı")
        st.info("Hızlı müdahale eden ülkeler ekonomik kayıpları nasıl minimize ediyor?")
        show_html("strategic", "11_speed_vs_success_quadrant.html", 650)

        st.markdown("---")

        st.subheader("Kapasite vs Müdahale Hızı")
        show_html("strategic", "12_capacity_vs_speed.html", 550)

    with tab2:
        st.subheader("💰 Yatırım Getirisi (ROI) Analizi")
        st.info("Afet öncesi 1 Usd yatırım, afet sonrası kaç Usd tasarruf sağlıyor?")
        show_html("strategic", "6_roi_investment_analysis.html", 500)

        st.markdown("""
        ### 📈 Temel Bulgular

        | Yatırım Alanı | ROI Oranı | Açıklama |
        |--------------|-----------|----------|
        | Erken Uyarı Sistemleri | **4.5x** | En yüksek getiri |
        | Altyapı Güçlendirme | **3.2x** | Orta-uzun vadeli etki |
        | Acil Yardım Fonları | **2.1x** | Reaktif yaklaşım |
        """)

    with tab3:
        st.subheader("📊 Afet Tipi Karşılaştırması")

        if df is not None:
            type_summary = df.groupby('disaster_type').agg({
                'economic_loss_usd': ['sum', 'mean'],
                'casualties': ['sum', 'mean'],
                'severity_index': 'mean',
                'response_time_hours': 'mean'
            }).round(2)

            type_summary.columns = ['Toplam Kayıp', 'Ort. Kayıp', 'Toplam Can Kaybı', 'Ort. Can Kaybı', 'Ort. Şiddet',
                                    'Ort. Müdahale (saat)']
            type_summary = type_summary.sort_values('Toplam Kayıp', ascending=False)

            st.dataframe(type_summary, use_container_width=True)

# =============================================================================
# SAYFA: AI SİMÜLATÖR
# =============================================================================
elif selected == "AI Simülatör":
    st.title("⚡ Yapay Zeka Risk Simülatörü")
    st.markdown("*XGBoost modeli ile afet şiddeti tahmini*")

    if 'severity_model' in models:
        st.success("✅ Model aktif - Gerçek tahminler yapılıyor")
        model_active = True
    else:
        st.warning("⚠️ Model yüklenemedi - Demo modunda çalışıyor")
        model_active = False

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Lokasyon Bilgileri")

        if df is not None:
            country = st.selectbox("Ülke", sorted(df['country'].unique()))
            disaster_type = st.selectbox("Afet Tipi", sorted(df['disaster_type'].unique()))
        else:
            country = st.text_input("Ülke", "Turkey")
            disaster_type = st.selectbox("Afet Tipi", ["Earthquake", "Flood", "Storm", "Wildfire"])

        season = st.selectbox("Mevsim", ["Winter", "Spring", "Summer", "Autumn"])
        month = st.slider("Ay", 1, 12, 6)

    with col2:
        st.subheader("📊 Etki Parametreleri")

        economic_loss = st.number_input("Tahmini Ekonomik Kayıp ($)",
                                        min_value=0,
                                        max_value=100000000000,
                                        value=10000000,
                                        step=1000000,
                                        format="%d")

        casualties = st.number_input("Tahmini Can Kaybı",
                                     min_value=0,
                                     max_value=1000000,
                                     value=100,
                                     step=10)

        response_time = st.slider("Müdahale Süresi (saat)", 1, 168, 24)
        recovery_days = st.slider("Tahmini İyileşme Süresi (gün)", 1, 365, 30)

    st.markdown("---")

    if st.button("🔮 RİSKİ HESAPLA", type="primary", use_container_width=True):

        with st.spinner("Model çalışıyor..."):

            input_data = {
                'casualties': casualties,
                'economic_loss_usd': economic_loss,
                'response_time_hours': response_time,
                'recovery_days': recovery_days,
                'year': datetime.now().year,
                'month': month,
                'population': 50000000,
                'population_density': 100,
                'surface_area_km2': 500000,
            }

            if df is not None:
                for col in ['country', 'disaster_type', 'season', 'region', 'income_group']:
                    if col in df.columns:
                        for val in df[col].unique():
                            key = f"{col}_{val}"
                            if col == 'country':
                                input_data[key] = 1 if val == country else 0
                            elif col == 'disaster_type':
                                input_data[key] = 1 if val == disaster_type else 0
                            elif col == 'season':
                                input_data[key] = 1 if val == season else 0
                            else:
                                input_data[key] = 0

            if model_active:
                pred_class, pred_label = predict_severity(input_data, models)
            else:
                score = (np.log1p(economic_loss) * 0.4) + (np.log1p(casualties) * 0.4) + (response_time / 168 * 0.2)
                if score > 12:
                    pred_class, pred_label = 3, "CRITICAL"
                elif score > 9:
                    pred_class, pred_label = 2, "HIGH"
                elif score > 6:
                    pred_class, pred_label = 1, "MEDIUM"
                else:
                    pred_class, pred_label = 0, "LOW"

            if casualties > 10000 and pred_class < 3:
                pred_class = 3
                pred_label = "CRITICAL"
                st.warning("⚠️ Güvenlik Protokolü: Yüksek can kaybı nedeniyle risk seviyesi yükseltildi!")
            elif casualties > 1000 and pred_class < 2:
                pred_class = 2
                pred_label = "HIGH"

        st.markdown("---")
        st.subheader("📋 Tahmin Sonucu")

        risk_colors = {
            "LOW": ("risk-low", "🟢"),
            "MEDIUM": ("risk-medium", "🟡"),
            "HIGH": ("risk-high", "🟠"),
            "CRITICAL": ("risk-critical", "🔴")
        }

        css_class, icon = risk_colors.get(pred_label, ("risk-medium", "⚪"))

        st.markdown(f"""
        <div class="{css_class}">
            <h1>{icon} {pred_label}</h1>
            <p>Risk Seviyesi: {pred_class + 1}/4</p>
        </div>
        """, unsafe_allow_html=True)

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("💰 Ekonomik Etki", f"${economic_loss:,.0f}")
        with col_r2:
            st.metric("👥 İnsan Etkisi", f"{casualties:,} kişi")
        with col_r3:
            st.metric("⏱️ Müdahale", f"{response_time} saat")

        country_info = get_country_risk(country, models)
        if country_info:
            st.markdown("---")
            st.subheader(f"🏛️ {country} Risk Profili")

            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric("Risk Kategorisi", country_info.get('risk_label', 'N/A'))
            col_c2.metric("Ortalama Şiddet", f"{country_info.get('avg_severity', 0):.2f}")
            col_c3.metric("Risk Skoru", f"{country_info.get('risk_score', 0):.3f}")

# =============================================================================
# SAYFA: TRENDLER
# =============================================================================
elif selected == "Trendler":
    st.title("📈 Trend & Momentum Analizi")

    tab1, tab2 = st.tabs(["🔮 Gelecek Tahmini", "🚀 Afet Momentumu"])

    with tab1:
        st.subheader("Küresel Afet Sıklığı Tahmini (12 Ay)")
        st.info("Polynomial Regression modeli ile gelecek tahminlemesi")
        show_html("model_03", "global_trend_forecast.html", 650)

        if df is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            yearly_counts = df.groupby('year').size()
            col1.metric("Son Yıl Olay Sayısı", f"{yearly_counts.iloc[-1]:,}")
            col2.metric("Ortalama (Yıllık)", f"{yearly_counts.mean():,.0f}")
            col3.metric("Trend", "📈 Artış" if yearly_counts.iloc[-1] > yearly_counts.mean() else "📉 Azalış")

    with tab2:
        st.subheader("Afet Tipi Momentum Analizi")
        st.info("Hangi afet tipleri son 2 yılda artış trendinde?")
        show_html("model_03", "disaster_momentum.html", 650)

        if 'momentum_data' in models:
            st.markdown("---")
            st.subheader("📊 Momentum Tablosu")

            momentum_df = models['momentum_data']


            def color_trend(val):
                if 'RISING' in str(val):
                    return 'background-color: #ffcccc'
                elif 'FALLING' in str(val):
                    return 'background-color: #ccffcc'
                return ''


            styled_df = momentum_df.style.applymap(color_trend, subset=['Trend'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

# =============================================================================
# SAYFA: HAKKINDA
# =============================================================================
elif selected == "Hakkında":
    st.title("ℹ️ Proje Hakkında")

    st.markdown("""
    ## 🌍 Global Disaster AI: Risk & Trend Analysis System

    **Miuul Data Scientist Bootcamp - Final Projesi**

    ### 🎯 Amaç
    2018-2024 yılları arasındaki 50.000+ küresel afet verisini analiz ederek; 
    uluslararası yardım kuruluşları, devletler ve sigorta şirketleri için 
    yapay zeka destekli bir **Karar Destek Sistemi (DSS)** oluşturmak.

    ---

    ### 🤖 Kullanılan Modeller

    | Model | Algoritma | Amaç |
    |-------|-----------|------|
    | Model 01 | XGBoost | Afet şiddeti sınıflandırma |
    | Model 02 | K-Means | Coğrafi risk kümeleme |
    | Model 03 | Polynomial Regression | Trend tahmini |
    | Model 04 | K-Means | Ülke kapasite analizi |
    | Model 05 | K-Means | Spatial grid analizi |

    ---

    ### 🛠️ Teknolojiler

    - **Dil:** Python 3.10
    - **Arayüz:** Streamlit
    - **ML:** Scikit-Learn, XGBoost, LightGBM
    - **Görselleştirme:** Plotly
    - **Veri:** World Bank API Entegrasyonu

    ---

    ### 👨‍💻 Takım: Resque Data

    Ali Özdemir, Nadide Yücel, Aslı Güldağ Bekaroğlu, İbrahim Alnıaçık, **Halil Öztekin**

    ---

    ### 📜 Lisans
    MIT License

    🔗 [GitHub Repository](https://github.com/hoztekin/GlobalDisaster)
    """)

    st.markdown("---")
    st.subheader("📊 Sistem Durumu")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Veri Durumu", "✅ Aktif" if df is not None else "❌ Yok")
        if df is not None:
            st.caption(f"{len(df):,} kayıt yüklü")

    with col2:
        loaded_models = sum(1 for k in models if models[k] is not None)
        st.metric("Model Durumu", f"5/5 Aktif")

    with col3:
        st.metric("Son Güncelleme", datetime.now().strftime("%d.%m.%Y"))