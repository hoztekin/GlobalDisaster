# =============================================================================
# MODEL 01: SEVERITY PREDICTION (Sınıflandırma)
# =============================================================================

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Windows Çökme Engelleyiciler (EN BAŞA EKLENDİ)
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


class Config:
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        PROJECT_ROOT = Path(r"D:\Miuul Final Project\GlobalDisaster")

    INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'disaster_data_final.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'models'
    REPORT_DIR = PROJECT_ROOT / 'reports' / 'model_01'

    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # XGBoost Parametreleri (Tek Çekirdek Zorlaması)
    XGBOOST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'multi:softmax',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'n_jobs': 1
    }

    SMOTE_K_NEIGHBORS = 5

    # Model Eğitimi İçin (Sayısal Etiketler)
    SEVERITY_BINS = [0, 3, 6, 8, 10]
    SEVERITY_LABELS = [0, 1, 2, 3]
    SEVERITY_NAMES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def run_model_pipeline():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🤖 MODEL 01: SEVERITY PREDICTION BAŞLATILIYOR")
    print("=" * 80)

    # --- 1. VERİ YÜKLEME ---
    if not Config.INPUT_FILE.exists():
        raise FileNotFoundError(f"❌ Veri bulunamadı: {Config.INPUT_FILE}")

    df = pd.read_csv(Config.INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Veri yüklendi: {len(df):,} satır")

    # Hedef Değişken
    df['severity_level'] = pd.cut(
        df['severity_index'],
        bins=Config.SEVERITY_BINS,
        labels=Config.SEVERITY_LABELS,
        include_lowest=True
    )

    # --- 2. FEATURE SELECTION ---
    numeric_features = [
        'casualties', 'economic_loss_usd', 'response_time_hours',
        'recovery_days', 'aid_amount_usd', 'year', 'month',
        'population', 'population_density', 'surface_area_km2'
    ]

    # Veri setindeki skor ismine göre ekleme
    if 'response_efficiency_calc' in df.columns:
        numeric_features.append('response_efficiency_calc')
    elif 'response_efficiency_score' in df.columns:
        numeric_features.append('response_efficiency_score')

    categorical_features = ['country', 'disaster_type', 'season', 'region', 'income_group']

    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    print("\n🔄 One-Hot Encoding...")
    df_encoded = pd.get_dummies(df[numeric_features + categorical_features],
                                columns=categorical_features,
                                drop_first=True)

    # --- 3. SPLIT & SMOTE ---
    X = df_encoded
    y = df['severity_level'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    print(f"✂️ Train: {X_train.shape}, Test: {X_test.shape}")

    print("⚖️ SMOTE ile sınıf dengesizliği gideriliyor...")

    # CRASH FIX: n_jobs=1 (Tek işlemci kullanımı)
    smote = SMOTE(k_neighbors=Config.SMOTE_K_NEIGHBORS, random_state=Config.RANDOM_STATE, n_jobs=1)

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"   SMOTE sonrası Train boyutu: {X_train_smote.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. TRAINING ---
    models = {
        "XGBoost": XGBClassifier(**Config.XGBOOST_PARAMS),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1,
                                   n_jobs=1),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    }

    results = {}
    best_score = 0
    best_model_name = ""
    best_model = None

    print("\n🏎️ Modeller Yarıştırılıyor...")
    for name, model in models.items():
        print(f"   ⏳ {name} eğitiliyor...")
        model.fit(X_train_scaled, y_train_smote)

        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)

        results[name] = {"f1": f1, "acc": acc, "model": model, "pred": y_pred}
        print(f"      ✅ {name} -> F1: {f1:.4f} | Acc: {acc:.4f}")

        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_model = model

    print(f"\n🏆 KAZANAN MODEL: {best_model_name} (F1: {best_score:.4f})")

    # --- 5. RAPORLAMA & KAYIT ---
    winner_res = results[best_model_name]
    y_pred_final = winner_res["pred"]

    print("\n📋 Classification Report:")
    # HATA FIX: Config.SEVERITY_NAMES (String Listesi) Kullanıldı
    print(classification_report(y_test, y_pred_final, target_names=Config.SEVERITY_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.SEVERITY_NAMES, yticklabels=Config.SEVERITY_NAMES)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.tight_layout()
    plt.savefig(Config.REPORT_DIR / 'confusion_matrix.png')

    # Kayıtlar
    with open(Config.OUTPUT_DIR / 'severity_prediction_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open(Config.OUTPUT_DIR / 'severity_prediction_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(Config.OUTPUT_DIR / 'model_01_features.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

    print(f"\n💾 Model ve Dosyalar Kaydedildi: {Config.OUTPUT_DIR}")
    print("✅ MODEL 01 SÜRECİ BAŞARIYLA TAMAMLANDI!")


if __name__ == '__main__':
    run_model_pipeline()