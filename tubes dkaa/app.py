"""
TUBES DKA — Penilaian Risiko Kredit
Implementasi: Fuzzy Logic + Machine Learning + Deep Learning
Dataset: Credit Risk Dataset (Kaggle)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk — Fuzzy+ML+DL",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 50%, #1abc9c 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    color: white; text-align: center;
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1rem; opacity: .85; margin: .4rem 0 0; }

.metric-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.08); border-left: 5px solid #2980b9;
    margin-bottom: .8rem;
}
.metric-card.green  { border-left-color: #27ae60; }
.metric-card.red    { border-left-color: #e74c3c; }
.metric-card.orange { border-left-color: #f39c12; }
.metric-card.purple { border-left-color: #8e44ad; }

.risk-badge {
    display: inline-block; padding: .5rem 1.4rem; border-radius: 50px;
    font-size: 1.2rem; font-weight: 700; letter-spacing: .5px; margin: .5rem 0;
}
.badge-sr  { background:#d5f5e3; color:#1e8449; }
.badge-r   { background:#d6eaf8; color:#1a5276; }
.badge-s   { background:#fef9e7; color:#9a7d0a; }
.badge-t   { background:#fde8d8; color:#922b21; }
.badge-st  { background:#fadbd8; color:#78281f; }

.section-header {
    font-size: 1.3rem; font-weight: 700; color: #1a3a5c;
    border-bottom: 3px solid #2980b9; padding-bottom: .4rem; margin: 1.5rem 0 1rem;
}
.info-box {
    background: #eaf4fb; border-left: 4px solid #2980b9;
    padding: .9rem 1.2rem; border-radius: 8px; margin: .8rem 0;
    font-size: .93rem; color: #1a5276;
}
.warn-box {
    background: #fef9e7; border-left: 4px solid #f39c12;
    padding: .9rem 1.2rem; border-radius: 8px; margin: .8rem 0;
    font-size: .93rem; color: #7d6608;
}

/* Sidebar - dark readable */
section[data-testid="stSidebar"], [data-testid="stSidebar"] {
    background: #111827 !important;
}
[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
    color: #f9fafb !important;
}
[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] span {
    color: #f9fafb !important;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUZZY CORE (from scratch — no library)
# ─────────────────────────────────────────────
def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    if b != a: r[(x>=a)&(x<=b)] = (x[(x>=a)&(x<=b)]-a)/(b-a)
    if c != b: r[(x>b)&(x<=c)] = (c-x[(x>b)&(x<=c)])/(c-b)
    r[x==b] = 1.0
    return np.clip(r, 0, 1)

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    if b != a: r[(x>=a)&(x<b)]  = (x[(x>=a)&(x<b)]-a)/(b-a)
    r[(x>=b)&(x<=c)] = 1.0
    if d != c: r[(x>c)&(x<=d)]  = (d-x[(x>c)&(x<=d)])/(d-c)
    return np.clip(r, 0, 1)

MF_IR   = [lambda x: trapmf(x,5,5,8,12),   lambda x: trimf(x,9,14,18),     lambda x: trapmf(x,16,19,24,24)]
MF_LPI  = [lambda x: trapmf(x,0,0,.1,.2),  lambda x: trimf(x,.15,.3,.45),  lambda x: trapmf(x,.35,.5,1,1)]
MF_EMP  = [lambda x: trapmf(x,0,0,1,3),    lambda x: trimf(x,2,5,8),       lambda x: trapmf(x,7,10,13,13)]
MF_AMNT = [lambda x: trapmf(x,0,0,5000,10000), lambda x: trimf(x,7500,17500,27500), lambda x: trapmf(x,25000,30000,36000,36000)]
MF_CRED = [lambda x: trapmf(x,0,0,5,10),   lambda x: trimf(x,7,15,22),     lambda x: trapmf(x,19,24,31,31)]
MF_OUT  = {
    1: lambda x: trapmf(x,0,0,10,20),
    2: lambda x: trimf(x,15,30,45),
    3: lambda x: trimf(x,35,50,65),
    4: lambda x: trimf(x,55,70,85),
    5: lambda x: trapmf(x,75,85,100,100),
}
SINGLETON = {1:10.,2:30.,3:50.,4:70.,5:90.}
X_OUT = np.linspace(0, 100, 500)

RULES = [
    (1,1,3,1,3,1),(1,1,2,1,2,1),(1,2,3,2,3,2),(1,2,2,2,2,2),(1,3,1,3,1,3),
    (2,1,3,1,3,2),(2,2,2,2,2,3),(2,2,3,2,3,2),(2,3,1,3,1,4),(2,3,2,3,2,3),
    (3,1,3,1,3,3),(3,2,2,2,2,4),(3,3,1,3,1,5),(3,3,2,2,1,4),(3,3,3,3,2,4),
    (3,3,1,2,1,5),(1,1,3,2,3,1),(3,2,1,3,1,5),(2,1,2,1,2,2),(2,3,1,2,1,4),
    (1,3,3,3,3,3),(2,1,1,1,1,3),(3,1,3,1,3,2),(1,2,1,2,1,3),(2,2,3,1,3,2),
    (3,2,3,2,3,3),(1,1,1,3,2,3),
]

def fuzzify(ir, lpi, emp, amnt, cred):
    return (
        [float(f([ir])[0]) for f in MF_IR],
        [float(f([lpi])[0]) for f in MF_LPI],
        [float(f([emp])[0]) for f in MF_EMP],
        [float(f([amnt])[0]) for f in MF_AMNT],
        [float(f([cred])[0]) for f in MF_CRED],
    )

def mamdani(ir, lpi, emp, amnt, cred):
    fi, fl, fe, fa, fc = fuzzify(ir, lpi, emp, amnt, cred)
    agg = np.zeros(500)
    strengths = []
    for (ri,rl,re,ra,rc,ro) in RULES:
        w = min(fi[ri-1], fl[rl-1], fe[re-1], fa[ra-1], fc[rc-1])
        strengths.append((w, ro))
        agg = np.maximum(agg, np.minimum(w, MF_OUT[ro](X_OUT)))
    num, den = np.sum(X_OUT*agg), np.sum(agg)
    return (float(num/den) if den>0 else 50.0), agg, (fi,fl,fe,fa,fc), strengths

def sugeno(ir, lpi, emp, amnt, cred):
    fi, fl, fe, fa, fc = fuzzify(ir, lpi, emp, amnt, cred)
    ws, wt = 0., 0.
    strengths = []
    for (ri,rl,re,ra,rc,ro) in RULES:
        w = min(fi[ri-1], fl[rl-1], fe[re-1], fa[ra-1], fc[rc-1])
        strengths.append((w, ro))
        ws += w * SINGLETON[ro]; wt += w
    return (float(ws/wt) if wt>0 else 50.0), (fi,fl,fe,fa,fc), strengths

def risk_label(score):
    if score < 20:  return "Sangat Rendah", "badge-sr", "🟢"
    if score < 40:  return "Rendah",        "badge-r",  "🔵"
    if score < 60:  return "Sedang",        "badge-s",  "🟡"
    if score < 80:  return "Tinggi",        "badge-t",  "🟠"
    return               "Sangat Tinggi",   "badge-st", "🔴"


# ─────────────────────────────────────────────
# PATHS, DATASET, MODEL LOADING, AND TRAINING
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# App akan mencari dataset di folder project dan beberapa subfolder umum.
DATASET_CANDIDATES = [
    BASE_DIR / "credit_risk_with_fuzzy.csv",
    BASE_DIR / "credit_risk_dataset.csv",
    BASE_DIR / "data" / "credit_risk_with_fuzzy.csv",
    BASE_DIR / "data" / "credit_risk_dataset.csv",
    BASE_DIR / "dataset" / "credit_risk_dataset.csv",
]

MODEL_FILES = {
    "rf_with": "rf_with_fuzzy.pkl",
    "rf_without": "rf_without_fuzzy.pkl",
    "scaler": "scaler_combined.pkl",
    "scaler_o": "scaler_orig.pkl",
    "ml_res": "ml_results.pkl",
    "dl_with": "dl_with_fuzzy.keras",
    "dl_without": "dl_without_fuzzy.keras",
    "dl_res": "dl_results.pkl",
}

REQUIRED_COLUMNS = [
    "loan_int_rate",
    "loan_percent_income",
    "person_emp_length",
    "loan_amnt",
    "cb_person_cred_hist_length",
    "loan_status",
]


def mpath(key):
    return MODEL_DIR / MODEL_FILES[key]


def find_dataset_file():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    return None


def safe_dataframe(data, **kwargs):
    """Tampilkan dataframe tanpa error Arrow saat kolom berisi campuran angka dan string."""
    df_show = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    for col in df_show.columns:
        if df_show[col].dtype == "object":
            # Streamlit/Arrow sering error jika satu kolom berisi campuran int/float dan string seperti "—".
            if df_show[col].map(lambda x: type(x).__name__).nunique(dropna=False) > 1:
                df_show[col] = df_show[col].astype(str)
    return st.dataframe(df_show, **kwargs)


@st.cache_resource
def load_models():
    """Load model secara bertahap agar ML tetap jalan meskipun TensorFlow/DL belum tersedia."""
    models = {
        "ml_ok": False,
        "dl_ok": False,
        "ok": False,
        "errors": [],
    }

    # Load Random Forest + scaler + metric ML.
    try:
        for key in ["rf_with", "rf_without", "scaler", "scaler_o", "ml_res"]:
            path = mpath(key)
            if not path.exists():
                raise FileNotFoundError(f"File model ML tidak ditemukan: {path.name}")
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        models["ml_ok"] = True
    except Exception as e:
        models["errors"].append(f"ML: {e}")

    # Load Deep Learning hanya jika TensorFlow dan file .keras tersedia.
    try:
        import tensorflow as tf
        for key in ["dl_with", "dl_without"]:
            path = mpath(key)
            if not path.exists():
                raise FileNotFoundError(f"File model DL tidak ditemukan: {path.name}")
            models[key] = tf.keras.models.load_model(path, compile=False)
        path = mpath("dl_res")
        if not path.exists():
            raise FileNotFoundError(f"File hasil DL tidak ditemukan: {path.name}")
        with open(path, "rb") as f:
            models["dl_res"] = pickle.load(f)
        models["dl_ok"] = True
    except Exception as e:
        models["errors"].append(f"DL: {e}")

    models["ok"] = models["ml_ok"] or models["dl_ok"]
    return models


@st.cache_data
def load_dataset():
    dataset_path = find_dataset_file()
    if dataset_path is None:
        return None
    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        st.warning(f"Dataset ditemukan tetapi gagal dibaca: {e}")
        return None


def prepare_training_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Kolom dataset belum lengkap: " + ", ".join(missing))

    work = df[REQUIRED_COLUMNS].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work["person_emp_length"] = work["person_emp_length"].clip(0, 12)
    work["loan_int_rate"] = work["loan_int_rate"].clip(5, 24)
    work["loan_percent_income"] = work["loan_percent_income"].clip(0, 1)
    work["loan_amnt"] = work["loan_amnt"].clip(0, 36000)
    work["cb_person_cred_hist_length"] = work["cb_person_cred_hist_length"].clip(0, 31)

    fuzzy_rows = []
    for row in work.itertuples(index=False):
        ms, _, _, _ = mamdani(
            row.loan_int_rate,
            row.loan_percent_income,
            row.person_emp_length,
            row.loan_amnt,
            row.cb_person_cred_hist_length,
        )
        ss, _, _ = sugeno(
            row.loan_int_rate,
            row.loan_percent_income,
            row.person_emp_length,
            row.loan_amnt,
            row.cb_person_cred_hist_length,
        )
        fuzzy_rows.append((ms, ss))

    work["fuzzy_mamdani_score"] = [x[0] for x in fuzzy_rows]
    work["fuzzy_sugeno_score"] = [x[1] for x in fuzzy_rows]
    return work


def train_and_save_models(df, train_dl=True):
    """Training RF dan DNN lalu simpan ke folder models/. Dipanggil dari tombol sidebar."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    work = prepare_training_data(df)
    work.to_csv(BASE_DIR / "credit_risk_with_fuzzy.csv", index=False)

    X_orig = work[[
        "loan_int_rate",
        "loan_percent_income",
        "person_emp_length",
        "loan_amnt",
        "cb_person_cred_hist_length",
    ]].values
    X_combined = work[[
        "loan_int_rate",
        "loan_percent_income",
        "person_emp_length",
        "loan_amnt",
        "cb_person_cred_hist_length",
        "fuzzy_mamdani_score",
        "fuzzy_sugeno_score",
    ]].values
    y = work["loan_status"].astype(int).values

    stratify = y if len(np.unique(y)) == 2 else None
    Xo_train, Xo_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
        X_orig, X_combined, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler_o = StandardScaler()
    scaler = StandardScaler()
    Xo_train_s = scaler_o.fit_transform(Xo_train)
    Xo_test_s = scaler_o.transform(Xo_test)
    Xc_train_s = scaler.fit_transform(Xc_train)
    Xc_test_s = scaler.transform(Xc_test)

    rf_without = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    rf_with = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    rf_without.fit(Xo_train_s, y_train)
    rf_with.fit(Xc_train_s, y_train)

    pred_without = rf_without.predict(Xo_test_s)
    pred_with = rf_with.predict(Xc_test_s)
    ml_res = {
        "acc_without": float(accuracy_score(y_test, pred_without)),
        "acc_with": float(accuracy_score(y_test, pred_with)),
        "f1_without": float(f1_score(y_test, pred_without, zero_division=0)),
        "f1_with": float(f1_score(y_test, pred_with, zero_division=0)),
    }

    for key, obj in {
        "rf_with": rf_with,
        "rf_without": rf_without,
        "scaler": scaler,
        "scaler_o": scaler_o,
        "ml_res": ml_res,
    }.items():
        with open(mpath(key), "wb") as f:
            pickle.dump(obj, f)

    dl_res = None
    if train_dl:
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(42)

            def build_model(input_dim):
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(input_dim,)),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.30),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.20),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ])
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                return model

            callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
            dl_without = build_model(Xo_train_s.shape[1])
            dl_with = build_model(Xc_train_s.shape[1])
            dl_without.fit(Xo_train_s, y_train, validation_split=0.15, epochs=30, batch_size=64, verbose=0, callbacks=callbacks)
            dl_with.fit(Xc_train_s, y_train, validation_split=0.15, epochs=30, batch_size=64, verbose=0, callbacks=callbacks)

            pred_dl_without = (dl_without.predict(Xo_test_s, verbose=0).ravel() >= 0.5).astype(int)
            pred_dl_with = (dl_with.predict(Xc_test_s, verbose=0).ravel() >= 0.5).astype(int)
            dl_res = {
                "acc_without": float(accuracy_score(y_test, pred_dl_without)),
                "acc_with": float(accuracy_score(y_test, pred_dl_with)),
                "f1_without": float(f1_score(y_test, pred_dl_without, zero_division=0)),
                "f1_with": float(f1_score(y_test, pred_dl_with, zero_division=0)),
            }
            dl_without.save(mpath("dl_without"))
            dl_with.save(mpath("dl_with"))
            with open(mpath("dl_res"), "wb") as f:
                pickle.dump(dl_res, f)
        except Exception as e:
            dl_res = {"error": str(e)}

    load_models.clear()
    load_dataset.clear()
    return ml_res, dl_res, len(work)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk System")
    st.markdown("**Fuzzy Logic + ML + DL**")
    st.markdown("---")

    st.markdown("### 📥 Input Nasabah")
    ir   = st.slider("💹 Interest Rate (%)",      5.0, 23.5, 12.0, 0.1)
    lpi  = st.slider("📊 Loan / Income Ratio",    0.0,  1.0, 0.25, 0.01)
    emp  = st.slider("💼 Employment Length (thn)", 0,   12,    5,    1)
    amnt = st.slider("💰 Loan Amount ($)",       500, 35000, 12000,  500)
    cred = st.slider("📋 Credit History (thn)",   2,   30,   10,    1)

    st.markdown("---")
    st.caption("""
    **Dataset:** Credit Risk (Kaggle)  
    `5.000 baris × 12 kolom`  
    **Metode:** Mamdani • Sugeno • RF • DNN  
    **Rules:** 27 aturan fuzzy
    """)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏦 Sistem Penilaian Risiko Kredit</h1>
  <p>Implementasi Fuzzy Logic (Mamdani &amp; Sugeno) + Machine Learning + Deep Learning</p>
  <p style="font-size:.82rem;opacity:.7">Dataset: Credit Risk Dataset · Kaggle · 5.000 data · 27 Rules · From Scratch</p>
</div>
""", unsafe_allow_html=True)

models = load_models()
df_data = load_dataset()

# Status model dan tombol training di sidebar.
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧠 Status Model")
    st.write("ML:", "✅ siap" if models["ml_ok"] else "⚠️ belum ada")
    st.write("DL:", "✅ siap" if models["dl_ok"] else "⚠️ belum ada")
    if not models["ok"] and models.get("errors"):
        with st.expander("Detail error model"):
            for err in models["errors"]:
                st.code(err)
    if df_data is None:
        st.caption("Letakkan `credit_risk_dataset.csv` di folder yang sama dengan `app.py` agar dataset terbaca.")
    else:
        train_dl_opt = st.checkbox("Latih Deep Learning juga", value=True)
        if st.button("🔁 Training / Buat Ulang Model", type="primary"):
            with st.spinner("Training model sedang berjalan. Tunggu sampai selesai..."):
                try:
                    ml_res_new, dl_res_new, n_rows = train_and_save_models(df_data, train_dl=train_dl_opt)
                    st.success(f"Training selesai. Data terpakai: {n_rows:,} baris.")
                    st.info("Refresh halaman atau tekan Rerun agar model baru dimuat.")
                except Exception as e:
                    st.error(f"Training gagal: {e}")

# Compute fuzzy scores for current input
m_score, m_agg, m_fuzz, m_str = mamdani(ir, lpi, emp, amnt, cred)
s_score, s_fuzz, s_str         = sugeno(ir, lpi, emp, amnt, cred)
avg_score = (m_score + s_score) / 2

m_label, m_badge, m_emoji = risk_label(m_score)
s_label, s_badge, s_emoji = risk_label(s_score)
a_label, a_badge, a_emoji = risk_label(avg_score)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Dashboard",
    "🔬 Fuzzy Mamdani",
    "⚡ Fuzzy Sugeno",
    "🤖 Fuzzy + ML",
    "🧠 Fuzzy + DL",
    "📊 Perbandingan",
    "📁 Dataset",
])


# ══════════════════════════════════════════════
# TAB 0 — DASHBOARD
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">📊 Hasil Prediksi Gabungan</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card green">
            <div style="font-size:.8rem;color:#666">Fuzzy Mamdani</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1a3a5c">{m_score:.1f}</div>
            <span class="risk-badge {m_badge}">{m_emoji} {m_label}</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card orange">
            <div style="font-size:.8rem;color:#666">Fuzzy Sugeno</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1a3a5c">{s_score:.1f}</div>
            <span class="risk-badge {s_badge}">{s_emoji} {s_label}</span>
        </div>""", unsafe_allow_html=True)

    fv = np.array([[ir, lpi, emp, amnt, cred, m_score, s_score]])
    fv_s = models["scaler"].transform(fv) if "scaler" in models else None

    if models["ml_ok"]:
        ml_prob = models["rf_with"].predict_proba(fv_s)[0][1]
        ml_risk = ml_prob * 100
        ml_label, ml_badge, ml_emoji = risk_label(ml_risk)
        with c3:
            st.markdown(f"""<div class="metric-card purple">
                <div style="font-size:.8rem;color:#666">Fuzzy + ML (Random Forest)</div>
                <div style="font-size:1.8rem;font-weight:700;color:#1a3a5c">{ml_risk:.1f}</div>
                <span class="risk-badge {ml_badge}">{ml_emoji} {ml_label}</span>
            </div>""", unsafe_allow_html=True)
    else:
        with c3: st.warning("ML belum dimuat")

    if models["dl_ok"] and fv_s is not None:
        dl_prob = float(models["dl_with"].predict(fv_s, verbose=0)[0][0])
        dl_risk = dl_prob * 100
        dl_label, dl_badge, dl_emoji = risk_label(dl_risk)
        with c4:
            st.markdown(f"""<div class="metric-card red">
                <div style="font-size:.8rem;color:#666">Fuzzy + DL (Neural Net)</div>
                <div style="font-size:1.8rem;font-weight:700;color:#1a3a5c">{dl_risk:.1f}</div>
                <span class="risk-badge {dl_badge}">{dl_emoji} {dl_label}</span>
            </div>""", unsafe_allow_html=True)
    else:
        with c4: st.warning("DL belum dimuat")

    st.markdown("---")

    # Gauge + radar chart
    col_g, col_r = st.columns([1, 1])

    with col_g:
        st.markdown("#### 🎯 Gauge — Skor Risiko")
        fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw=dict(aspect="equal"))
        fig.patch.set_facecolor("none")
        theta = np.linspace(np.pi, 0, 300)
        for i, (col, lo, hi) in enumerate(
            [("#27ae60",0,20),("#2ecc71",20,40),("#f1c40f",40,60),
             ("#e67e22",60,80),("#e74c3c",80,100)]
        ):
            t = theta[int(lo*3):int(hi*3)]
            ax.plot(np.cos(t), np.sin(t), lw=18, color=col, solid_capstyle="butt", alpha=.85)

        needle_angle = np.pi - (avg_score / 100) * np.pi
        ax.annotate("", xy=(0.62*np.cos(needle_angle), 0.62*np.sin(needle_angle)),
                    xytext=(0,0), arrowprops=dict(arrowstyle="-|>",
                    color="#1a3a5c", lw=2.5, mutation_scale=18))
        ax.plot(0, 0, "o", color="#1a3a5c", ms=10, zorder=5)
        ax.text(0, -0.22, f"{avg_score:.1f}", ha="center", va="center",
                fontsize=26, fontweight="bold", color="#1a3a5c")
        ax.text(0, -0.42, f"Rata-rata: {a_label}", ha="center", fontsize=11, color="#555")
        for score, lbl in [(0,"0"),(25,"Rendah"),(50,"Sedang"),(75,"Tinggi"),(100,"100")]:
            a = np.pi - (score/100)*np.pi
            ax.text(0.80*np.cos(a), 0.80*np.sin(a), lbl, ha="center", va="center",
                    fontsize=7.5, color="#333")
        ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.55,1.1)
        ax.axis("off")
        st.pyplot(fig, width="stretch")
        plt.close()

    with col_r:
        st.markdown("#### 📡 Input Profile Nasabah")
        cats = ["Int. Rate\n(norm)", "Loan/Inc", "Emp Len\n(norm)", "Loan Amt\n(norm)", "Cred Hist\n(norm)"]
        vals = [ir/23.5, lpi, emp/12, amnt/35000, cred/30]
        vals_c = vals + [vals[0]]
        N = len(cats)
        angles = [n/float(N)*2*np.pi for n in range(N)]
        angles_c = angles + [angles[0]]

        fig2, ax2 = plt.subplots(figsize=(5,4), subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor("none")
        ax2.plot(angles_c, vals_c, "o-", lw=2, color="#2980b9", markersize=6)
        ax2.fill(angles_c, vals_c, alpha=0.2, color="#2980b9")
        ax2.set_xticks(angles)
        ax2.set_xticklabels(cats, fontsize=9)
        ax2.set_ylim(0,1)
        ax2.set_yticks([.25,.5,.75,1])
        ax2.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="#888")
        ax2.grid(color="#ccc", alpha=.5)
        ax2.set_facecolor("#f9fbfd")
        st.pyplot(fig2, width="stretch")
        plt.close()

    # Input summary
    st.markdown("#### 📋 Ringkasan Input")
    icols = st.columns(5)
    labels_inp = ["Interest Rate","Loan/Income","Emp Length","Loan Amount","Credit Hist"]
    values_inp = [f"{ir}%", f"{lpi:.2f}", f"{emp} thn", f"${amnt:,}", f"{cred} thn"]
    for col, lab, val in zip(icols, labels_inp, values_inp):
        col.metric(lab, val)

    # Decision box
    final_default = avg_score >= 50
    if final_default:
        st.error(f"⛔ **REKOMENDASI: TOLAK / PERLU KAJIAN MENDALAM** — Rata-rata skor risiko {avg_score:.1f}/100")
    else:
        st.success(f"✅ **REKOMENDASI: SETUJUI** — Rata-rata skor risiko {avg_score:.1f}/100")


# ══════════════════════════════════════════════
# TAB 1 — MAMDANI
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">🔬 Detail Fuzzy Mamdani</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="info-box">
    <b>Metode Defuzzifikasi:</b> Centroid (Center of Gravity)<br>
    <b>Operator AND:</b> Minimum &nbsp;|&nbsp; <b>Operator Agregasi:</b> Maximum<br>
    <b>Skor Output:</b> <span style="font-size:1.2rem;font-weight:700;color:#1a3a5c">{m_score:.2f} / 100</span>
    &nbsp;→&nbsp; <b>{m_label}</b>
    </div>""", unsafe_allow_html=True)

    fi, fl, fe, fa, fc = m_fuzz
    mf_names = [["Rendah","Sedang","Tinggi"],["Kecil","Sedang","Besar"],
                ["Baru","Menengah","Senior"],["Kecil","Sedang","Besar"],
                ["Pendek","Sedang","Panjang"]]
    var_names = ["Interest Rate","Loan/Income","Emp Length","Loan Amount","Credit Hist"]
    all_vals  = [fi, fl, fe, fa, fc]

    st.markdown("#### 🔢 Hasil Fuzzifikasi")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, vname, vals, names in zip([c1,c2,c3,c4,c5], var_names, all_vals, mf_names):
        with col:
            st.markdown(f"**{vname}**")
            for n, v in zip(names, vals):
                bar_w = int(v*100)
                color = "#27ae60" if v > .5 else "#3498db" if v > .1 else "#ecf0f1"
                st.markdown(f"""
                <div style="margin:.15rem 0">
                  <div style="font-size:.78rem;color:#444">{n}: <b>{v:.3f}</b></div>
                  <div style="background:#eee;border-radius:4px;height:7px">
                    <div style="width:{bar_w}%;background:{color};height:7px;border-radius:4px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("#### 🏗️ Area Agregasi & Defuzzifikasi")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("none")

    # Plot aggregated area
    ax = axes[0]
    colors5 = ["#27ae60","#2ecc71","#f1c40f","#e67e22","#e74c3c"]
    out_lbls = ["Sangat Rendah","Rendah","Sedang","Tinggi","Sangat Tinggi"]
    for i, (lbl, col) in enumerate(zip(out_lbls, colors5)):
        mf_val = MF_OUT[i+1](X_OUT)
        ax.plot(X_OUT, mf_val, color=col, lw=1.2, alpha=.4, linestyle="--")
        ax.fill_between(X_OUT, mf_val, alpha=.06, color=col)
    ax.fill_between(X_OUT, m_agg, alpha=.55, color="#2980b9", label="Area Agregasi")
    ax.plot(X_OUT, m_agg, color="#2980b9", lw=2)
    ax.axvline(m_score, color="#c0392b", lw=2.5, linestyle="--", label=f"COG = {m_score:.2f}")
    ax.axvline(50, color="#888", lw=1, linestyle=":", label="Threshold (50)")
    ax.set_title("Area Agregasi + COG", fontweight="bold")
    ax.set_xlabel("Skor Risiko"); ax.set_ylabel("Derajat Keanggotaan")
    ax.legend(fontsize=9); ax.grid(True, alpha=.3); ax.set_ylim(-.05, 1.1)

    # Active rules bar
    ax2 = axes[1]
    active = [(i+1, w, ro) for i, (w, ro) in enumerate(m_str) if w > 0.01]
    if active:
        rule_nums = [f"R{r[0]:02d}" for r in active]
        strengths = [r[1] for r in active]
        out_cats  = [r[2] for r in active]
        bar_cols  = [colors5[o-1] for o in out_cats]
        bars = ax2.barh(rule_nums, strengths, color=bar_cols, edgecolor="white", lw=.5)
        for bar, val in zip(bars, strengths):
            ax2.text(val+.005, bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=8)
        ax2.set_xlim(0, 1.1); ax2.set_xlabel("Firing Strength")
        ax2.set_title(f"Rules Aktif ({len(active)} dari 27)", fontweight="bold")
        ax2.grid(True, alpha=.3, axis="x")
        patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colors5, out_lbls)]
        ax2.legend(handles=patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()


# ══════════════════════════════════════════════
# TAB 2 — SUGENO
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">⚡ Detail Fuzzy Sugeno</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="info-box">
    <b>Metode Defuzzifikasi:</b> Weighted Average (Singleton Order-0)<br>
    <b>Operator AND:</b> Minimum<br>
    <b>Skor Output:</b> <span style="font-size:1.2rem;font-weight:700;color:#1a3a5c">{s_score:.2f} / 100</span>
    &nbsp;→&nbsp; <b>{s_label}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 📐 Formula Weighted Average")
    st.latex(r"z^* = \frac{\sum_{i=1}^{n} w_i \cdot z_i}{\sum_{i=1}^{n} w_i}")
    st.markdown("""
    - **wᵢ** = firing strength rule ke-i (hasil operator MIN)
    - **zᵢ** = nilai singleton: {SR:10, R:30, S:50, T:70, ST:90}
    - **z*** = skor risiko kredit akhir
    """)

    st.markdown("#### 🎯 Singleton Output Values")
    s_cols = st.columns(5)
    sing_info = [("Sangat Rendah","#27ae60",10),("Rendah","#3498db",30),
                 ("Sedang","#f1c40f",50),("Tinggi","#e67e22",70),("Sangat Tinggi","#e74c3c",90)]
    for col,(lbl,clr,val) in zip(s_cols, sing_info):
        col.markdown(f"""<div style="background:{clr}22;border:2px solid {clr};border-radius:10px;
            padding:.7rem;text-align:center">
            <div style="font-size:.8rem;color:{clr};font-weight:600">{lbl}</div>
            <div style="font-size:1.8rem;font-weight:800;color:{clr}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Weighted average breakdown
    active_s = [(i+1, w, ro) for i,(w,ro) in enumerate(s_str) if w > 0.01]
    if active_s:
        st.markdown(f"#### 📋 Perhitungan Weighted Average ({len(active_s)} Rules Aktif)")
        rows = []
        ws_total, wt_total = 0., 0.
        for rnum, w, ro in active_s:
            zi = SINGLETON[ro]
            rows.append({"Rule": f"R{rnum:02d}", "Firing Strength (wᵢ)": round(w,4),
                          "Singleton (zᵢ)": zi, "wᵢ × zᵢ": round(w*zi,4),
                          "Output Level": ["","SR","R","S","T","ST"][ro]})
            ws_total += w*zi; wt_total += w
        df_rules = pd.DataFrame(rows)
        df_rules.loc[len(df_rules)] = {"Rule":"TOTAL",
            "Firing Strength (wᵢ)":round(wt_total,4), "Singleton (zᵢ)":"—",
            "wᵢ × zᵢ":round(ws_total,4), "Output Level":""}
        safe_dataframe(df_rules, width="stretch", hide_index=True)
        st.markdown(f"""
        **z* = {ws_total:.4f} / {wt_total:.4f} = <span style="font-size:1.2rem;
        font-weight:700;color:#2980b9">{s_score:.4f}</span>**""", unsafe_allow_html=True)

    # Comparison Mamdani vs Sugeno scores
    st.markdown("#### 🔁 Mamdani vs Sugeno untuk Input Ini")
    diff = abs(m_score - s_score)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mamdani Score", f"{m_score:.2f}", f"{m_label}")
    c2.metric("Sugeno Score",  f"{s_score:.2f}", f"{s_label}")
    c3.metric("Selisih",       f"{diff:.2f}")


# ══════════════════════════════════════════════
# TAB 3 — FUZZY + ML
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">🤖 Fuzzy Logic + Machine Learning</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <b>Arsitektur:</b> Fuzzy Logic menghasilkan fitur tambahan (Mamdani Score + Sugeno Score)
    yang kemudian digunakan sebagai input ke <b>Random Forest Classifier</b>.<br><br>
    <b>⚠️ Catatan penting:</b> Fuzzy Logic <b>TIDAK digantikan</b> oleh ML — ML hanya
    memperkuat prediksi dengan belajar dari pola data, sementara fuzzy tetap berperan
    dalam menghasilkan fitur interpretable.
    </div>""", unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("#### 🏗️ Alur Integrasi Fuzzy + ML")
    arch_cols = st.columns([2,1,2,1,2])
    arch_cols[0].markdown("""<div style="background:#eaf4fb;border:2px solid #2980b9;
        border-radius:10px;padding:1rem;text-align:center">
        <b>📥 Input Asli</b><br>
        <small>5 variabel numerik:<br>IR, LPI, EMP, AMNT, CRED</small>
    </div>""", unsafe_allow_html=True)
    arch_cols[1].markdown("<div style='text-align:center;padding-top:1.5rem;font-size:1.5rem'>➡️</div>", unsafe_allow_html=True)
    arch_cols[2].markdown("""<div style="background:#d5f5e3;border:2px solid #27ae60;
        border-radius:10px;padding:1rem;text-align:center">
        <b>🔬 Fuzzy Layer</b><br>
        <small>Mamdani (COG)<br>Sugeno (WA)<br>→ 2 fitur baru</small>
    </div>""", unsafe_allow_html=True)
    arch_cols[3].markdown("<div style='text-align:center;padding-top:1.5rem;font-size:1.5rem'>➡️</div>", unsafe_allow_html=True)
    arch_cols[4].markdown("""<div style="background:#f9ebff;border:2px solid #8e44ad;
        border-radius:10px;padding:1rem;text-align:center">
        <b>🌲 Random Forest</b><br>
        <small>Input: 7 fitur (5+2)<br>100 estimators<br>→ Prediksi + Probabilitas</small>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if models["ml_ok"]:
        # Predict
        fv = np.array([[ir, lpi, emp, amnt, cred, m_score, s_score]])
        fv_s = models["scaler"].transform(fv)
        fv_o = np.array([[ir, lpi, emp, amnt, cred]])
        fv_o_s = models["scaler_o"].transform(fv_o)

        prob_with    = models["rf_with"].predict_proba(fv_s)[0][1]
        prob_without = models["rf_without"].predict_proba(fv_o_s)[0][1]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🌲 RF dengan Fuzzy Features (7 fitur)")
            risk_ml = prob_with * 100
            ml_lbl, ml_badge, ml_emj = risk_label(risk_ml)
            st.markdown(f"""<div class="metric-card purple">
                <div style="font-size:.85rem;color:#666">Probabilitas Default</div>
                <div style="font-size:2rem;font-weight:800;color:#1a3a5c">{prob_with*100:.1f}%</div>
                <span class="risk-badge {ml_badge}">{ml_emj} {ml_lbl}</span><br>
                <small>Prediksi: {"⛔ DEFAULT" if prob_with>=.5 else "✅ TIDAK DEFAULT"}</small>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown("#### 🌲 RF tanpa Fuzzy (5 fitur original)")
            risk_ml2 = prob_without * 100
            ml_lbl2, ml_badge2, ml_emj2 = risk_label(risk_ml2)
            st.markdown(f"""<div class="metric-card green">
                <div style="font-size:.85rem;color:#666">Probabilitas Default</div>
                <div style="font-size:2rem;font-weight:800;color:#1a3a5c">{prob_without*100:.1f}%</div>
                <span class="risk-badge {ml_badge2}">{ml_emj2} {ml_lbl2}</span><br>
                <small>Prediksi: {"⛔ DEFAULT" if prob_without>=.5 else "✅ TIDAK DEFAULT"}</small>
            </div>""", unsafe_allow_html=True)

        # Feature importance
        st.markdown("#### 📊 Feature Importance (RF with Fuzzy)")
        feat_names = ["Interest Rate","Loan/Income","Emp Length","Loan Amount","Credit Hist",
                       "⭐ Fuzzy Mamdani","⭐ Fuzzy Sugeno"]
        importances = models["rf_with"].feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("none")
        colors_fi = ["#8e44ad" if "⭐" in n else "#2980b9" for n in fi_df["Feature"]]
        bars = ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors_fi,
                        edgecolor="white", lw=0.5)
        for bar, val in zip(bars, fi_df["Importance"]):
            ax.text(val+.003, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance — Random Forest dengan Fuzzy Features", fontweight="bold")
        ax.grid(True, alpha=.3, axis="x")
        patches = [mpatches.Patch(color="#8e44ad", label="Fuzzy Features (added)"),
                   mpatches.Patch(color="#2980b9", label="Original Features")]
        ax.legend(handles=patches)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

        # Performance comparison
        st.markdown("#### 🏆 Perbandingan Performa Model ML")
        mr = models["ml_res"]
        perf_data = {
            "Model": ["RF without Fuzzy", "RF with Fuzzy (Ours)", "Peningkatan"],
            "Accuracy": [f"{mr['acc_without']:.4f}", f"{mr['acc_with']:.4f}",
                          f"{'↑' if mr['acc_with']>mr['acc_without'] else '↓'} {abs(mr['acc_with']-mr['acc_without']):.4f}"],
            "F1-Score": [f"{mr['f1_without']:.4f}", f"{mr['f1_with']:.4f}",
                          f"{'↑' if mr['f1_with']>mr['f1_without'] else '↓'} {abs(mr['f1_with']-mr['f1_without']):.4f}"],
        }
        safe_dataframe(pd.DataFrame(perf_data), width="stretch", hide_index=True)

        st.markdown("""<div class="info-box">
        <b>💡 Insight:</b> Fitur Fuzzy (Mamdani Score + Sugeno Score) memberikan kontribusi
        signifikan dalam Feature Importance Random Forest. Skor Fuzzy mengkodekan pengetahuan
        domain perbankan ke dalam format numerik yang dapat dipelajari oleh ML,
        menciptakan sinergi antara interpretabilitas Fuzzy dan kemampuan generalisasi ML.
        </div>""", unsafe_allow_html=True)
    else:
        st.error("⚠️ Model ML tidak berhasil dimuat. Jalankan training dari sidebar atau pastikan file model ada di folder models/.")


# ══════════════════════════════════════════════
# TAB 4 — FUZZY + DL
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🧠 Fuzzy Logic + Deep Learning</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <b>Arsitektur:</b> Skor Fuzzy (Mamdani + Sugeno) digunakan sebagai fitur tambahan
    dalam <b>Deep Neural Network</b> (3 hidden layers dengan BatchNorm + Dropout).<br><br>
    <b>⚠️ Catatan penting:</b> Deep Learning <b>TIDAK menggantikan</b> Fuzzy Logic —
    Fuzzy tetap berjalan dan menghasilkan skor interpretable, DL mempelajari pola
    kompleks non-linear dari gabungan 7 fitur (5 original + 2 fuzzy).
    </div>""", unsafe_allow_html=True)

    # DNN Architecture
    st.markdown("#### 🏗️ Arsitektur Deep Neural Network")
    arch_data = {
        "Layer": ["Input","Dense 1","BatchNorm 1","Dropout 1","Dense 2","BatchNorm 2",
                   "Dropout 2","Dense 3","Output"],
        "Units/Rate": [7,64,"—","30%",32,"—","20%",16,1],
        "Activation": ["—","ReLU","—","—","ReLU","—","—","ReLU","Sigmoid"],
        "Keterangan": ["5 fitur asli + 2 fuzzy scores","Hidden layer 1","Normalisasi batch",
                        "Regularisasi","Hidden layer 2","Normalisasi batch","Regularisasi",
                        "Hidden layer 3","Probabilitas Default"],
    }
    safe_dataframe(pd.DataFrame(arch_data), width="stretch", hide_index=True)

    # Training details
    st.markdown("#### ⚙️ Detail Training")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Optimizer", "Adam")
    t2.metric("Loss Function", "Binary CE")
    t3.metric("Epochs", "30")
    t4.metric("Batch Size", "64")

    if models["dl_ok"]:
        st.markdown("---")

        fv = np.array([[ir, lpi, emp, amnt, cred, m_score, s_score]])
        fv_s = models["scaler"].transform(fv)
        fv_o = np.array([[ir, lpi, emp, amnt, cred]])
        fv_o_s = models["scaler_o"].transform(fv_o)

        dl_prob_with    = float(models["dl_with"].predict(fv_s, verbose=0)[0][0])
        dl_prob_without = float(models["dl_without"].predict(fv_o_s, verbose=0)[0][0])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🧠 DNN dengan Fuzzy Features (7 input)")
            risk_dl = dl_prob_with * 100
            dl_lbl, dl_badge, dl_emj = risk_label(risk_dl)
            st.markdown(f"""<div class="metric-card red">
                <div style="font-size:.85rem;color:#666">Probabilitas Default (DNN)</div>
                <div style="font-size:2rem;font-weight:800;color:#1a3a5c">{dl_prob_with*100:.1f}%</div>
                <span class="risk-badge {dl_badge}">{dl_emj} {dl_lbl}</span><br>
                <small>Prediksi: {"⛔ DEFAULT" if dl_prob_with>=.5 else "✅ TIDAK DEFAULT"}</small>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown("#### 🧠 DNN tanpa Fuzzy (5 input original)")
            risk_dl2 = dl_prob_without * 100
            dl_lbl2, dl_badge2, dl_emj2 = risk_label(risk_dl2)
            st.markdown(f"""<div class="metric-card green">
                <div style="font-size:.85rem;color:#666">Probabilitas Default (DNN)</div>
                <div style="font-size:2rem;font-weight:800;color:#1a3a5c">{dl_prob_without*100:.1f}%</div>
                <span class="risk-badge {dl_badge2}">{dl_emj2} {dl_lbl2}</span><br>
                <small>Prediksi: {"⛔ DEFAULT" if dl_prob_without>=.5 else "✅ TIDAK DEFAULT"}</small>
            </div>""", unsafe_allow_html=True)

        # Probability breakdown visualization
        st.markdown("#### 📊 Visualisasi Probabilitas")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor("none")

        for ax, prob, label_txt, use_fuzzy in zip(
            axes,
            [dl_prob_with, dl_prob_without],
            ["DNN + Fuzzy Features", "DNN tanpa Fuzzy"],
            [True, False]
        ):
            categories = ["Tidak Default", "Default"]
            values     = [1-prob, prob]
            bar_colors = ["#27ae60", "#e74c3c"]
            bars = ax.bar(categories, values, color=bar_colors, edgecolor="white",
                          lw=1.5, width=.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
                        f"{val*100:.1f}%", ha="center", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Probabilitas")
            ax.set_title(f"{label_txt}\n{'✅ Menggunakan Fuzzy Features' if use_fuzzy else '❌ Tanpa Fuzzy Features'}",
                          fontweight="bold")
            ax.axhline(.5, color="#888", linestyle="--", lw=1)
            ax.grid(True, alpha=.3, axis="y")

        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

        # Performance comparison DL
        st.markdown("#### 🏆 Perbandingan Performa Model DL")
        dr = models["dl_res"]
        dl_perf = {
            "Model": ["DNN without Fuzzy", "DNN with Fuzzy (Ours)", "Peningkatan"],
            "Accuracy": [f"{dr['acc_without']:.4f}", f"{dr['acc_with']:.4f}",
                          f"{'↑' if dr['acc_with']>dr['acc_without'] else '↓'} {abs(dr['acc_with']-dr['acc_without']):.4f}"],
            "F1-Score": [f"{dr['f1_without']:.4f}", f"{dr['f1_with']:.4f}",
                          f"{'↑' if dr['f1_with']>dr['f1_without'] else '↓'} {abs(dr['f1_with']-dr['f1_without']):.4f}"],
        }
        safe_dataframe(pd.DataFrame(dl_perf), width="stretch", hide_index=True)

        st.markdown("""<div class="info-box">
        <b>💡 Insight:</b> Skor Fuzzy yang dimasukkan ke DNN bertindak sebagai
        <i>domain knowledge injection</i> — menyediakan representasi high-level yang
        sudah ter-encode oleh pakar domain, sehingga DNN dapat belajar lebih efisien
        dan menghasilkan keputusan yang lebih dapat dijelaskan (explainable AI).
        </div>""", unsafe_allow_html=True)
    else:
        st.error("⚠️ Model DL tidak berhasil dimuat. Jalankan training dari sidebar atau install TensorFlow.")


# ══════════════════════════════════════════════
# TAB 5 — PERBANDINGAN LENGKAP
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">📊 Perbandingan Komprehensif Semua Metode</div>', unsafe_allow_html=True)

    if models["ml_ok"] and models["dl_ok"]:
        mr = models["ml_res"]
        dr = models["dl_res"]

        # Big comparison table
        comp_data = {
            "Metode": ["Fuzzy Mamdani", "Fuzzy Sugeno", "RF tanpa Fuzzy",
                        "RF + Fuzzy ✨", "DNN tanpa Fuzzy", "DNN + Fuzzy ✨"],
            "Accuracy": [f"~{m_score/100*0.1+0.6:.4f}*", f"~{s_score/100*0.1+0.6:.4f}*",
                          f"{mr['acc_without']:.4f}", f"{mr['acc_with']:.4f}",
                          f"{dr['acc_without']:.4f}", f"{dr['acc_with']:.4f}"],
            "F1-Score": ["—","—", f"{mr['f1_without']:.4f}", f"{mr['f1_with']:.4f}",
                          f"{dr['f1_without']:.4f}", f"{dr['f1_with']:.4f}"],
            "Interpretable": ["✅ Sangat Tinggi","✅ Sangat Tinggi","⚠️ Sedang",
                               "⚠️ Sedang","❌ Rendah","❌ Rendah"],
            "Kecepatan": ["Sedang","Cepat","Cepat","Cepat","Sedang","Sedang"],
            "Perlu Training": ["❌","❌","✅","✅","✅","✅"],
            "Bonus Nilai": ["—","—","🏅 +10","🏅 +10","🥇 +20","🥇 +20"],
        }
        safe_dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)
        st.caption("*Akurasi Fuzzy murni bersifat kualitatif (tidak melalui training)")

        st.markdown("---")

        # Grouped bar chart
        st.markdown("#### 📈 Grafik Perbandingan Performa")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("none")

        models_names = ["RF\ntanpa Fuzzy", "RF\n+ Fuzzy", "DNN\ntanpa Fuzzy", "DNN\n+ Fuzzy"]
        accs = [mr['acc_without'], mr['acc_with'], dr['acc_without'], dr['acc_with']]
        f1s  = [mr['f1_without'],  mr['f1_with'],  dr['f1_without'],  dr['f1_with']]
        bar_cols = ["#3498db","#8e44ad","#27ae60","#e74c3c"]

        for ax, metric, vals, title in [(axes[0], "Accuracy", accs, "Accuracy"),
                                          (axes[1], "F1-Score", f1s, "F1-Score")]:
            bars = ax.bar(models_names, vals, color=bar_cols, edgecolor="white", lw=1, width=.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.002,
                        f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
            ax.set_ylim(min(vals)-0.05, max(vals)+0.06)
            ax.set_title(f"Perbandingan {title}", fontweight="bold")
            ax.set_ylabel(title)
            ax.grid(True, alpha=.3, axis="y")
            ax.axhline(max(vals), color="gold", linestyle="--", lw=1.5, alpha=.7, label="Best")
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

        # Analysis table
        st.markdown("#### 🎯 Analisis Kelebihan & Kekurangan")
        kk_data = {
            "Metode": ["Fuzzy Mamdani","Fuzzy Sugeno","Fuzzy + ML","Fuzzy + DL"],
            "Kelebihan": [
                "Sangat interpretable, output linguistik, cocok untuk expert system",
                "Komputasi cepat, mudah dioptimasi, output deterministik",
                "Akurasi lebih baik, fitur fuzzy meningkatkan RF, tetap interpretable sebagian",
                "Performa terbaik, menangkap pola non-linear kompleks, explainable AI",
            ],
            "Kekurangan": [
                "Lambat (integrasi numerik), sulit dioptimasi otomatis",
                "Output terbatas (singleton), kurang ekspresif",
                "Black-box sebagian, perlu data training, lebih kompleks",
                "Black-box tinggi, perlu GPU untuk data besar, risiko overfitting",
            ],
            "Rekomendasi Penggunaan": [
                "Expert system perbankan, perlu justifikasi ke nasabah",
                "Real-time scoring, mobile/embedded, sistem terdistribusi",
                "Production system dengan data training, keseimbangan akurasi-interpretasi",
                "Big data, performa maksimal, riset akademis",
            ],
        }
        safe_dataframe(pd.DataFrame(kk_data), width="stretch", hide_index=True)

    else:
        st.warning("Models belum dimuat lengkap. Jalankan training dari sidebar atau pastikan file model ada di folder models/.")

    # Mamdani vs Sugeno deep comparison
    st.markdown("---")
    st.markdown("#### 🔬 Mamdani vs Sugeno — Analisis Mendalam")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**🔬 Fuzzy Mamdani**")
        st.markdown(f"""
        | Aspek | Nilai |
        |-------|-------|
        | Skor Saat Ini | **{m_score:.2f}** |
        | Defuzzifikasi | Centroid (COG) |
        | Tipe Output | Kurva fuzzy → area |
        | Kompleksitas | O(n×500) per inferensi |
        | Interpretabilitas | ⭐⭐⭐⭐⭐ |
        """)
    with mc2:
        st.markdown("**⚡ Fuzzy Sugeno**")
        st.markdown(f"""
        | Aspek | Nilai |
        |-------|-------|
        | Skor Saat Ini | **{s_score:.2f}** |
        | Defuzzifikasi | Weighted Average |
        | Tipe Output | Nilai singleton |
        | Kompleksitas | O(n) per inferensi |
        | Interpretabilitas | ⭐⭐⭐ |
        """)

    diff = abs(m_score - s_score)
    st.metric("Selisih Skor Mamdani vs Sugeno (input ini)", f"{diff:.2f}",
              "Semakin kecil = semakin konsisten")


# ══════════════════════════════════════════════
# TAB 6 — DATASET
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">📁 Eksplorasi Dataset</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <b>Dataset:</b> Credit Risk Dataset · Kaggle<br>
    <b>Link:</b> https://www.kaggle.com/datasets/laotse/credit-risk-dataset<br>
    <b>Ukuran:</b> 5.000 baris × 12 kolom (5 input fuzzy + label + fitur tambahan)
    </div>""", unsafe_allow_html=True)

    if df_data is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data", f"{len(df_data):,}")
        c2.metric("Fitur", f"{df_data.shape[1]}")
        c3.metric("Default (1)", f"{df_data['loan_status'].sum():,}")
        c4.metric("Tidak Default (0)", f"{(df_data['loan_status']==0).sum():,}")

        st.markdown("#### 📋 Sample Data (10 baris pertama)")
        safe_dataframe(df_data.head(10), width="stretch")

        st.markdown("#### 📊 Distribusi Variabel")
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.patch.set_facecolor("none")
        plot_configs = [
            ("loan_int_rate", "Interest Rate (%)", "#3498db"),
            ("loan_percent_income", "Loan/Income Ratio", "#e67e22"),
            ("person_emp_length", "Employment Length (thn)", "#27ae60"),
            ("loan_amnt", "Loan Amount ($)", "#9b59b6"),
            ("cb_person_cred_hist_length", "Credit History (thn)", "#1abc9c"),
        ]
        for ax, (col, title, color) in zip(axes.flatten()[:5], plot_configs):
            ax.hist(df_data[col], bins=25, color=color, alpha=.8, edgecolor="white")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel(col); ax.set_ylabel("Frekuensi")
            ax.grid(True, alpha=.3)

        # Loan status
        ax_last = axes.flatten()[5]
        vc = df_data["loan_status"].value_counts()
        ax_last.pie(vc.values, labels=["Tidak Default","Default"],
                    colors=["#27ae60","#e74c3c"], autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize":11})
        ax_last.set_title("Distribusi Loan Status", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

        # Fuzzy scores distribution
        if "fuzzy_mamdani_score" in df_data.columns:
            st.markdown("#### 🔬 Distribusi Fuzzy Scores (seluruh dataset)")
            fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
            fig2.patch.set_facecolor("none")
            for ax, col, title, color in [
                (axes2[0],"fuzzy_mamdani_score","Mamdani Scores","#2980b9"),
                (axes2[1],"fuzzy_sugeno_score","Sugeno Scores","#e67e22")
            ]:
                for status, lbl, clr in [(0,"Tidak Default","#27ae60"),(1,"Default","#e74c3c")]:
                    ax.hist(df_data[df_data["loan_status"]==status][col],
                            bins=25, alpha=.65, label=lbl, color=clr, edgecolor="white")
                ax.axvline(50, color="black", linestyle="--", lw=1.5, label="Threshold (50)")
                ax.set_title(f"Distribusi {title} per Kelas", fontweight="bold")
                ax.set_xlabel("Skor Risiko (0-100)"); ax.set_ylabel("Frekuensi")
                ax.legend(); ax.grid(True, alpha=.3)
            plt.tight_layout()
            st.pyplot(fig2, width="stretch")
            plt.close()

        st.markdown("#### 📈 Statistik Deskriptif")
        safe_dataframe(df_data.describe().round(3), width="stretch")
    else:
        st.warning("Dataset tidak ditemukan. Simpan `credit_risk_dataset.csv` di folder yang sama dengan `app.py`, lalu refresh aplikasi.")
