"""
app.py — Data Poisoning Attack — Application complète
Fusion de :
  - app.py      : visualisations, stats, courbes, contre-mesures
  - app_models.py : inférence live sur vrais modèles .keras

Lancer : streamlit run app.py

Structure attendue :
    app.py
    models/
        model_clean_best.keras
        model_poisoned_best.keras
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import io, os, time

# ── TensorFlow optionnel ─────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Poison Lab — Data Poisoning Attack",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Bebas+Neue&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #d4d0c8;
}
.stApp {
    background-color: #080a0c;
    background-image:
        radial-gradient(ellipse 60% 40% at 20% 10%, rgba(255,140,0,0.04), transparent),
        radial-gradient(ellipse 40% 60% at 80% 80%, rgba(255,50,50,0.03), transparent),
        repeating-linear-gradient(90deg, transparent, transparent 79px,
            rgba(255,255,255,0.012) 79px, rgba(255,255,255,0.012) 80px),
        repeating-linear-gradient(0deg, transparent, transparent 79px,
            rgba(255,255,255,0.012) 79px, rgba(255,255,255,0.012) 80px);
}
section[data-testid="stSidebar"] {
    background: #060809;
    border-right: 1px solid #1a1d20;
}
section[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; }
h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 3.8rem !important;
     letter-spacing: .06em; line-height: 1; color: #ffffff !important; }
h2 { font-family: 'Bebas Neue', sans-serif !important; font-size: 1.9rem !important;
     letter-spacing: .05em; color: #ffffff !important; }
h3 { font-family: 'IBM Plex Mono', monospace !important; font-size: .88rem !important;
     color: #ff8c00 !important; letter-spacing: .12em; text-transform: uppercase; }
[data-testid="metric-container"] {
    background: #0e1114; border: 1px solid #1a1d20;
    border-radius: 4px; padding: 1rem 1.3rem !important;
}
[data-testid="stMetricLabel"]  { font-family: 'IBM Plex Mono', monospace !important;
    font-size: .68rem !important; color: #5a5a52 !important;
    letter-spacing: .1em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.2rem !important; color: #d4d0c8 !important; }
[data-testid="stMetricDelta"]  { font-family: 'IBM Plex Mono', monospace !important; font-size: .72rem !important; }
[data-testid="stFileUploadDropzone"] {
    background: #0e1114 !important; border: 1px dashed #ff8c00 !important; border-radius: 8px !important;
}
button[data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace !important;
    font-size: .75rem !important; color: #5a5a52 !important;
    letter-spacing: .08em; text-transform: uppercase; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff8c00 !important; border-bottom: 2px solid #ff8c00 !important; }
.stButton > button { background: #ff8c00 !important; color: #080a0c !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 700 !important;
    font-size: .78rem !important; border: none !important; border-radius: 3px !important;
    padding: .55rem 1.6rem !important; letter-spacing: .1em; text-transform: uppercase; }
.stButton > button:hover { background: #e07800 !important; }
[data-testid="stProgress"] > div > div { background: #ff8c00 !important; }
.hazard-banner {
    background: repeating-linear-gradient(-45deg, #ff8c00, #ff8c00 10px, #080a0c 10px, #080a0c 20px);
    height: 4px; margin: 1rem 0;
}
.model-card { background: #0e1114; border: 1px solid #1a1d20; border-radius: 6px; padding: 1.4rem; }
.model-card-clean   { border-top: 3px solid #00d084; }
.model-card-poison  { border-top: 3px solid #ff3232; }
.verdict-clean  { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; color: #00d084; letter-spacing: .06em; }
.verdict-poison { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; color: #ff3232; letter-spacing: .06em; }
.prob-bar-wrap { background: #1a1d20; border-radius: 2px; height: 8px; margin: .2rem 0 .6rem 0; overflow: hidden; }
.mono-tag { font-family: 'IBM Plex Mono', monospace; font-size: .7rem; color: #5a5a52;
            text-transform: uppercase; letter-spacing: .12em; }
code { font-family: 'IBM Plex Mono', monospace !important; background: #0e1114 !important;
       color: #ff8c00 !important; font-size: .78rem !important;
       padding: .15rem .4rem !important; border-radius: 3px !important; }
.info-box { background: rgba(255,140,0,.06); border-left: 3px solid #ff8c00;
            border-radius: 0 6px 6px 0; padding: .8rem 1.2rem;
            font-family: 'IBM Plex Mono', monospace; font-size: .82rem; color: #ff8c00; }
.success-box { background: rgba(0,208,132,.06); border-left: 3px solid #00d084;
               border-radius: 0 6px 6px 0; padding: .8rem 1.2rem;
               font-family: 'IBM Plex Mono', monospace; font-size: .82rem; color: #00d084; }
.danger-box { background: rgba(255,50,50,.06); border-left: 3px solid #ff3232;
              border-radius: 0 6px 6px 0; padding: .8rem 1.2rem;
              font-family: 'IBM Plex Mono', monospace; font-size: .82rem; color: #ff3232; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#080a0c', 'axes.facecolor': '#0e1114',
    'axes.edgecolor': '#1a1d20', 'axes.labelcolor': '#5a5a52',
    'axes.titlecolor': '#d4d0c8', 'xtick.color': '#5a5a52',
    'ytick.color': '#5a5a52', 'grid.color': '#1a1d20', 'grid.linewidth': 0.7,
    'text.color': '#d4d0c8', 'font.family': 'monospace', 'font.size': 8.5,
    'legend.facecolor': '#0e1114', 'legend.edgecolor': '#1a1d20', 'legend.fontsize': 8,
})

GREEN   = '#00d084'
RED     = '#ff3232'
AMBER   = '#ff8c00'
BLUE    = '#4fc3f7'
CLASSES = ['bird', 'cat', 'dog']

# ─────────────────────────────────────────────────────────────────────
#  CHARGEMENT MODÈLES
#  Priorité : 1) Upload via interface  2) Dossier disque
# ─────────────────────────────────────────────────────────────────────
import tempfile

DISK_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
    r"D:\RAG_lois\Empois\models",
]

# Session state pour stocker les modèles uploadés
if 'model_clean'    not in st.session_state: st.session_state.model_clean    = None
if 'model_poisoned' not in st.session_state: st.session_state.model_poisoned = None
if 'upload_errors'  not in st.session_state: st.session_state.upload_errors  = {}
if 'upload_source'  not in st.session_state: st.session_state.upload_source  = {}

def _try_load(path):
    """
    Tente de charger un modèle Keras avec plusieurs stratégies
    pour gérer les incompatibilités de version (batch_shape, etc.)
    """
    errors = []

    # ── Stratégie 1 : chargement standard ────────────────────────────
    try:
        return tf.keras.models.load_model(path), None
    except Exception as e:
        errors.append(f"Standard: {e}")

    # ── Stratégie 2 : custom_objects pour InputLayer ──────────────────
    # Erreur "Unrecognized keyword arguments: batch_shape" = modèle sauvé
    # avec Keras 2 (TF 2.x) chargé dans Keras 3 (TF 2.16+)
    try:
        import keras
        from keras.layers import InputLayer

        class CompatInputLayer(InputLayer):
            def __init__(self, *args, **kwargs):
                # 'batch_shape' → 'shape' dans Keras 3
                if 'batch_shape' in kwargs:
                    bs = kwargs.pop('batch_shape')
                    if bs and bs[0] is None:
                        kwargs['shape'] = bs[1:]
                    else:
                        kwargs['shape'] = bs
                kwargs.pop('ragged', None)
                super().__init__(*args, **kwargs)

        model = tf.keras.models.load_model(
            path,
            custom_objects={'InputLayer': CompatInputLayer}
        )
        return model, None
    except Exception as e:
        errors.append(f"CompatInputLayer: {e}")

    # ── Stratégie 3 : compile=False (ignore les erreurs d'optimizer) ──
    try:
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model, None
    except Exception as e:
        errors.append(f"compile=False: {e}")

    # ── Stratégie 4 : .h5 avec safe_mode=False ────────────────────────
    try:
        model = tf.keras.models.load_model(path, safe_mode=False)
        return model, None
    except Exception as e:
        errors.append(f"safe_mode=False: {e}")

    return None, " | ".join(errors)


def load_keras_from_bytes(uploaded_file):
    """Charge un modèle Keras depuis un fichier uploadé (BytesIO)."""
    if not TF_AVAILABLE:
        return None, "TensorFlow non installé"
    try:
        suffix = '.keras' if uploaded_file.name.endswith('.keras') else '.h5'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        model, err = _try_load(tmp_path)
        try:
            os.unlink(tmp_path)
        except:
            pass
        return model, err
    except Exception as e:
        return None, str(e)

def load_keras_from_disk(path):
    if not os.path.exists(path):
        return None, f"Introuvable : {path}"
    if not TF_AVAILABLE:
        return None, "TensorFlow non installé"
    return _try_load(path)

def get_models():
    """Retourne dict avec clean/poisoned/errors/source."""
    results = {'clean': None, 'poisoned': None, 'errors': {}, 'source': {}}
    # 1) Modèles uploadés via interface
    if st.session_state.model_clean is not None:
        results['clean']           = st.session_state.model_clean
        results['source']['clean'] = st.session_state.upload_source.get('clean','📤 Upload')
    if st.session_state.model_poisoned is not None:
        results['poisoned']           = st.session_state.model_poisoned
        results['source']['poisoned'] = st.session_state.upload_source.get('poisoned','📤 Upload')
    # 2) Fallback disque
    for key, fname in [('clean','model_clean_best.keras'),
                       ('poisoned','model_poisoned_best.keras')]:
        if results[key] is not None:
            continue
        for folder in DISK_DIRS:
            model, err = load_keras_from_disk(os.path.join(folder, fname))
            if model is not None:
                results[key]            = model
                results['source'][key]  = '💾 Disque'
                break
        else:
            results['errors'][key] = f"{fname} — uploadez le fichier dans ⚙ Configuration"
    results['errors'].update(st.session_state.upload_errors)
    return results

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(model, img_array: np.ndarray):
    preds = model.predict(img_array, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]), {c: float(p) for c, p in zip(CLASSES, preds)}

# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='mono-tag' style='color:#ff8c00;margin-bottom:.5rem'>⚗ Poison Lab v3.0</div>",
                unsafe_allow_html=True)
    st.markdown("## Navigation")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠  Introduction",
        "⚗️  Démo Live",
        "🔬  Comparaison Batch",
        "📊  Performances",
        "🧠  Architecture CNN",
        "🛡️  Contre-mesures",
        "📂  Configuration",
        "📖  À propos",
    ], label_visibility="hidden")

    st.markdown("---")

    models_data = get_models()
    clean_ok  = models_data['clean']   is not None
    poison_ok = models_data['poisoned'] is not None

    st.markdown("<div class='mono-tag'>Statut des modèles</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono,monospace;font-size:.75rem;line-height:2.2'>
        {'🟢' if TF_AVAILABLE else '🔴'} TensorFlow {'OK' if TF_AVAILABLE else 'MANQUANT'}<br>
        {'🟢' if clean_ok   else '🔴'} model_clean_best.keras<br>
        {'🟢' if poison_ok  else '🔴'} model_poisoned_best.keras
    </div>""", unsafe_allow_html=True)

    if models_data['errors']:
        st.markdown("---")
        for key, err in models_data['errors'].items():
            st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;"
                        f"color:{RED}'>{key}: {str(err)[:55]}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='mono-tag'>Simulation</div>", unsafe_allow_html=True)
    poison_rate = st.slider("Poison rate", 0.05, 0.60, 0.30, 0.05,
                            format="%d%%", help="Taux simulé pour les graphiques Performances")

    st.markdown("---")
    st.markdown("<div class='mono-tag'>Classes</div>", unsafe_allow_html=True)
    for cls, emoji in [('bird','🐦'), ('cat','🐱'), ('dog','🐶')]:
        st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem'>{emoji} {cls}</div>",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.caption("EfficientNetB0 · Keras 3 · TF 2.x")

# ═════════════════════════════════════════════════════════════════════
#  PAGE : INTRODUCTION
# ═════════════════════════════════════════════════════════════════════
if page == "🏠  Introduction":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>POISON LAB · DATA POISONING ATTACK</div>",
                unsafe_allow_html=True)
    st.markdown("# Attaque par Empoisonnement de Données")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Dataset",         "13 344",  "images")
    with c2: st.metric("Classes",         "3",       "bird / cat / dog")
    with c3: st.metric("Accuracy propre", "94.8%",   "EfficientNetB0")
    with c4: st.metric("Après attaque",   "~48%",    "−46 pts", delta_color="inverse")

    st.markdown("---")
    st.markdown("## Pipeline de l'attaque")
    steps = [
        ("01","Dataset",          "13 344 images bird/cat/dog"),
        ("02","Modèle propre",    "EfficientNetB0 → 94.8%"),
        ("03","Générateur poison","np.roll sur 30% des labels"),
        ("04","Ré-entraînement",  "Même modèle, données corrompues"),
        ("05","Évaluation",       "Comparaison accuracy / loss"),
        ("06","Analyse",          "Impact par classe + courbes"),
    ]
    cols = st.columns(6)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:#0e1114;border:1px solid #1a1d20;border-radius:6px;
                        padding:.8rem;text-align:center'>
                <div style='font-family:Bebas Neue,sans-serif;font-size:1.8rem;color:{AMBER}'>{num}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;
                            color:#d4d0c8;margin:.3rem 0'>{title}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;color:#5a5a52'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Bugs corrigés")
    bugs = [
        ("Bug 1","Compilation commentée",           "model_poisoned.compile() désactivé → ValueError au fit"),
        ("Bug 2","clone_model() sans set_weights()", "Architecture copiée mais pas les poids → from scratch"),
        ("Bug 3","Évaluation sur poisoned_gen",      "Accuracy train sur labels corrompus → non interprétable"),
        ("Bug 4","Indices de poison fixes",          "Toujours les mêmes positions → attaque biaisée"),
    ]
    c1, c2 = st.columns(2)
    for i, (ref, title, desc) in enumerate(bugs):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div style='display:flex;gap:.8rem;padding:.7rem .9rem;margin:.3rem 0;
                        background:#0e1114;border-radius:4px;border-left:2px solid {RED}'>
                <div style='font-family:IBM Plex Mono,monospace;color:{RED};
                            font-size:.72rem;font-weight:700;min-width:3.2rem'>{ref}</div>
                <div>
                    <div style='font-size:.85rem;font-weight:600;margin-bottom:.2rem'>{title}</div>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:.68rem;color:#5a5a52'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
#  PAGE : DÉMO LIVE
# ═════════════════════════════════════════════════════════════════════
elif page == "⚗️  Démo Live":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>DÉMONSTRATION INTERACTIVE</div>",
                unsafe_allow_html=True)
    st.markdown("# Testez les deux modèles")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    if not clean_ok and not poison_ok:
        st.markdown("<div class='info-box'>⚠ Aucun modèle chargé. Allez dans <b>📂 Configuration</b>.</div>",
                    unsafe_allow_html=True)
        st.stop()

    col_upload, col_options = st.columns([2, 1], gap="large")
    with col_upload:
        st.markdown("### Upload d'image")
        uploaded = st.file_uploader("Chat, chien ou oiseau", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
    with col_options:
        st.markdown("### Options")
        show_probs   = st.toggle("Toutes les probabilités", value=True)
        show_compare = st.toggle("Graphique comparatif",    value=True)

    st.markdown("---")

    if uploaded is None:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;background:#0e1114;border-radius:8px;
                    border:1px dashed #1a1d20'>
            <div style='font-family:Bebas Neue,sans-serif;font-size:4rem;color:#1a1d20'>UPLOAD AN IMAGE</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:.75rem;color:#5a5a52;margin-top:.5rem'>
                JPG · PNG · WEBP · Chat / Chien / Oiseau</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    img = Image.open(uploaded)
    img_array = preprocess_image(img) if TF_AVAILABLE else None

    col_img, col_results = st.columns([1, 2], gap="large")

    with col_img:
        st.markdown("### Image")
        st.image(img, use_container_width=True)
        st.markdown(f"<div class='mono-tag' style='text-align:center'>"
                    f"{img.size[0]}×{img.size[1]} px · {uploaded.name}</div>",
                    unsafe_allow_html=True)

    with col_results:
        st.markdown("### Prédictions")
        if not TF_AVAILABLE or img_array is None:
            st.error("TensorFlow requis pour les prédictions.")
            st.stop()

        results = {}
        for key, model_obj in [('clean', models_data['clean']), ('poisoned', models_data['poisoned'])]:
            if model_obj is None:
                results[key] = None
                continue
            t0 = time.time()
            pred_label, confidence, probs = predict(model_obj, img_array)
            results[key] = {'label': pred_label, 'confidence': confidence,
                            'probs': probs, 'time': time.time() - t0}

        c1, c2 = st.columns(2)
        for (key, col), (title, color, card_class) in zip(
            [('clean', c1), ('poisoned', c2)],
            [('MODÈLE PROPRE',    GREEN, 'model-card-clean'),
             ('MODÈLE EMPOISONNÉ', RED,  'model-card-poison')]
        ):
            with col:
                if results.get(key) is None:
                    st.markdown(f"<div class='model-card {card_class}'>"
                                f"<div class='mono-tag'>{title}</div>"
                                f"<div style='color:#5a5a52;margin-top:1rem'>Non chargé</div></div>",
                                unsafe_allow_html=True)
                    continue

                r = results[key]
                emoji_map = {'bird':'🐦','cat':'🐱','dog':'🐶'}
                emoji = emoji_map.get(r['label'], '❓')
                verdict_cls = 'verdict-clean' if key == 'clean' else 'verdict-poison'

                st.markdown(f"""
                <div class='model-card {card_class}'>
                    <div class='mono-tag'>{title}</div>
                    <div class='{verdict_cls}' style='margin-top:.6rem'>{emoji} {r['label'].upper()}</div>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:.82rem;
                                color:{color};margin-bottom:1rem'>{r['confidence']:.1%} de confiance</div>
                """, unsafe_allow_html=True)

                if show_probs:
                    for cls in CLASSES:
                        prob    = r['probs'][cls]
                        is_top  = cls == r['label']
                        bar_col = color if is_top else '#2a2d30'
                        txt_col = '#d4d0c8' if is_top else '#5a5a52'
                        st.markdown(f"""
                        <div style='margin-bottom:.3rem'>
                            <div style='display:flex;justify-content:space-between;
                                        font-family:IBM Plex Mono,monospace;font-size:.7rem;color:{txt_col}'>
                                <span>{emoji_map[cls]} {cls}</span><span>{prob:.1%}</span>
                            </div>
                            <div class='prob-bar-wrap'>
                                <div style='width:{int(prob*100)}%;background:{bar_col};
                                            height:8px;border-radius:2px'></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;"
                            f"color:#2a2d30;margin-top:.4rem'>⏱ {r['time']*1000:.0f} ms</div>"
                            f"</div>", unsafe_allow_html=True)

        # Verdict
        if results.get('clean') and results.get('poisoned'):
            st.markdown("---")
            r_c, r_p = results['clean'], results['poisoned']
            if r_c['label'] == r_p['label']:
                diff = r_c['confidence'] - r_p['confidence']
                st.markdown(f"<div class='success-box'>✓ Même prédiction : <b>{r_c['label'].upper()}</b>. "
                            f"Confiance dégradée de {diff:.1%} ({r_c['confidence']:.1%} → {r_p['confidence']:.1%})"
                            f"</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='danger-box'>☠ DIVERGENCE — Propre : <b>{r_c['label'].upper()}</b> "
                            f"({r_c['confidence']:.1%}) → Empoisonné : <b>{r_p['label'].upper()}</b> "
                            f"({r_p['confidence']:.1%}). Fausse prédiction induite par l'attaque.</div>",
                            unsafe_allow_html=True)

        # Graphique comparatif
        if show_compare and results.get('clean') and results.get('poisoned'):
            fig, ax = plt.subplots(figsize=(7, 2.8))
            x = np.arange(len(CLASSES))
            w = 0.38
            probs_c = [results['clean']['probs'][c]    for c in CLASSES]
            probs_p = [results['poisoned']['probs'][c] for c in CLASSES]
            b1 = ax.bar(x-w/2, probs_c, w, color=GREEN+'55', edgecolor=GREEN, lw=1.2, label='Propre')
            b2 = ax.bar(x+w/2, probs_p, w, color=RED+'55',   edgecolor=RED,   lw=1.2, label='Empoisonné')
            ax.set_xticks(x)
            ax.set_xticklabels(['🐦 bird','🐱 cat','🐶 dog'], fontsize=8.5)
            ax.set_ylabel('Probabilité'); ax.set_ylim(0, 1.15)
            ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.4)
            ax.set_title('Distribution des probabilités : propre vs empoisonné', fontsize=9)
            for bar, prob in zip(b1, probs_c):
                ax.text(bar.get_x()+bar.get_width()/2, prob+.02, f'{prob:.0%}',
                        ha='center', fontsize=7.5, color=GREEN)
            for bar, prob in zip(b2, probs_p):
                ax.text(bar.get_x()+bar.get_width()/2, prob+.02, f'{prob:.0%}',
                        ha='center', fontsize=7.5, color=RED)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════════════════
#  PAGE : COMPARAISON BATCH
# ═════════════════════════════════════════════════════════════════════
elif page == "🔬  Comparaison Batch":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>BATCH TESTING</div>", unsafe_allow_html=True)
    st.markdown("# Comparaison sur plusieurs images")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    if not clean_ok or not poison_ok:
        st.markdown("<div class='info-box'>⚠ Les deux modèles doivent être chargés.</div>",
                    unsafe_allow_html=True)
        st.stop()

    uploaded_batch = st.file_uploader("Plusieurs images", type=["jpg","jpeg","png","webp"],
                                      accept_multiple_files=True, label_visibility="collapsed")

    if not uploaded_batch:
        st.markdown("""
        <div style='text-align:center;padding:3rem;background:#0e1114;border-radius:8px;
                    border:1px dashed #1a1d20'>
            <div style='font-family:Bebas Neue,sans-serif;font-size:3rem;color:#1a1d20'>
                UPLOAD MULTIPLE IMAGES</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#5a5a52;margin-top:.5rem'>
                Ctrl/Cmd + clic pour sélection multiple</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    batch_results = []
    prog = st.progress(0, text="Inférence en cours...")
    for i, f in enumerate(uploaded_batch):
        img = Image.open(f)
        arr = preprocess_image(img)
        l_c, p_c, _ = predict(models_data['clean'],    arr)
        l_p, p_p, _ = predict(models_data['poisoned'], arr)
        batch_results.append({'name': f.name, 'img': img,
                              'clean_label': l_c, 'clean_conf': p_c,
                              'poison_label': l_p, 'poison_conf': p_p,
                              'agree': l_c == l_p})
        prog.progress((i+1)/len(uploaded_batch), text=f"Inférence… {i+1}/{len(uploaded_batch)}")
    prog.empty()

    n_agree    = sum(1 for r in batch_results if r['agree'])
    n_disagree = len(batch_results) - n_agree
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Images testées", len(batch_results))
    with c2: st.metric("Accords",        n_agree,    f"{n_agree/len(batch_results):.0%}")
    with c3: st.metric("Divergences",    n_disagree, f"{n_disagree/len(batch_results):.0%}",
                       delta_color="inverse" if n_disagree > 0 else "off")
    st.markdown("---")

    for i in range(0, len(batch_results), 3):
        cols = st.columns(3)
        for col, r in zip(cols, batch_results[i:i+3]):
            with col:
                border = GREEN if r['agree'] else RED
                st.image(r['img'], use_container_width=True)
                st.markdown(f"""
                <div style='background:#0e1114;border:1px solid {border};border-radius:4px;
                            padding:.6rem .8rem;margin-bottom:.8rem'>
                    <div class='mono-tag'>{r['name'][:22]}</div>
                    <div style='display:flex;justify-content:space-between;margin-top:.4rem'>
                        <div>
                            <div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;color:{GREEN}'>PROPRE</div>
                            <div style='font-family:Bebas Neue,sans-serif;font-size:1.1rem;color:{GREEN}'>
                                {r['clean_label'].upper()} {r['clean_conf']:.0%}</div>
                        </div>
                        <div>
                            <div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;color:{RED}'>EMPOISONNÉ</div>
                            <div style='font-family:Bebas Neue,sans-serif;font-size:1.1rem;color:{RED}'>
                                {r['poison_label'].upper()} {r['poison_conf']:.0%}</div>
                        </div>
                    </div>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:.78rem;
                                color:{border};margin-top:.4rem;text-align:center'>
                        {'✓ ACCORD' if r['agree'] else '☠ DIVERGENCE'}
                    </div>
                </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
#  PAGE : PERFORMANCES
# ═════════════════════════════════════════════════════════════════════
elif page == "📊  Performances":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>ANALYSE · MÉTRIQUES</div>",
                unsafe_allow_html=True)
    st.markdown("# Performances Comparées")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    acc_clean   = 0.948
    acc_poison  = max(0.33, 0.948 - poison_rate * 1.55)
    loss_clean  = 0.156
    loss_poison = min(1.8,  0.156 + poison_rate * 3.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy Propre",     f"{acc_clean:.1%}",  "Référence")
    with c2: st.metric("Accuracy Empoisonné", f"{acc_poison:.1%}",
                       f"−{acc_clean-acc_poison:.1%}", delta_color="inverse")
    with c3: st.metric("Loss Propre",         f"{loss_clean:.3f}", "Convergée", delta_color="off")
    with c4: st.metric("Loss Empoisonnée",    f"{loss_poison:.3f}",
                       f"+{loss_poison-loss_clean:.3f}", delta_color="inverse")

    st.markdown(f"<div class='info-box' style='margin:.8rem 0'>"
                f"Simulation avec poison_rate = {poison_rate:.0%} (ajustable dans la sidebar)</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    EPOCHS = 10
    np.random.seed(42)
    def smooth(a, w=2): return np.convolve(a, np.ones(w)/w, mode='same')
    e = np.arange(1, EPOCHS+1)
    base_p = max(0.28, 0.948 - poison_rate * 1.6)
    c_acc  = smooth(np.clip(.89 + np.linspace(0,.06,EPOCHS) + np.random.normal(0,.008,EPOCHS),0,1))
    p_acc  = smooth(np.clip(base_p + np.linspace(0,.04,EPOCHS) + np.random.normal(0,.025,EPOCHS),0,1))
    c_loss = smooth(np.clip(.33 - np.linspace(0,.17,EPOCHS) + np.random.normal(0,.015,EPOCHS),.05,5))
    p_loss = smooth(np.clip(loss_poison - np.linspace(0,.15,EPOCHS)+np.random.normal(0,.05,EPOCHS),.5,3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(e, c_acc, color=GREEN, lw=2.2, marker='o', ms=4, label='Propre')
    ax1.plot(e, p_acc, color=RED,   lw=2.2, marker='x', ms=5, label='Empoisonné', ls='--')
    ax1.fill_between(e, p_acc, c_acc, alpha=0.07, color=RED)
    ax1.set_title('VALIDATION ACCURACY'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.set_ylim(.2, 1.05); ax1.legend(); ax1.grid(True, alpha=.4)
    ax1.annotate(f'−{float(c_acc[-1]-p_acc[-1]):.1%}',
                 xy=(9.8,(float(c_acc[-1])+float(p_acc[-1]))/2),
                 fontsize=9, color=RED, fontweight='bold', ha='right')
    ax2.plot(e, c_loss, color=GREEN, lw=2.2, marker='o', ms=4, label='Propre')
    ax2.plot(e, p_loss, color=RED,   lw=2.2, marker='x', ms=5, label='Empoisonné', ls='--')
    ax2.fill_between(e, c_loss, p_loss, alpha=0.07, color=RED)
    ax2.set_title('VALIDATION LOSS'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=.4)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### Matrices de confusion")
    CM_CLEAN  = np.array([[338,12,8],[10,342,12],[9,11,326]])
    CM_RAW    = np.array([[180,95,83],[82,165,117],[74,108,174]])
    scale     = min(1.0, poison_rate / 0.3)
    CM_POISON = (CM_CLEAN*(1-scale*0.6) + CM_RAW*scale*0.6).astype(int)
    CM_POISON = np.clip(CM_POISON, 0, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, cm, title, cmap, bc in zip(
        axes, [CM_CLEAN, CM_POISON],
        ['MODÈLE PROPRE', f'MODÈLE EMPOISONNÉ ({poison_rate:.0%})'],
        ['Greens', 'Reds'], [GREEN, RED]
    ):
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=400, aspect='auto')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, fontsize=9); ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_title(title, fontsize=10, color=bc, pad=10)
        ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
        total = cm.sum(axis=1, keepdims=True)
        for i in range(3):
            for j in range(3):
                pct = cm[i,j]/max(total[i,0],1)
                ax.text(j, i, f'{cm[i,j]}\n{pct:.0%}', ha='center', va='center',
                        fontsize=8.5, fontweight='bold',
                        color='#080a0c' if pct>.5 else '#d4d0c8')
        plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════════════════
#  PAGE : ARCHITECTURE CNN
# ═════════════════════════════════════════════════════════════════════
elif page == "🧠  Architecture CNN":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>MODÈLE · EFFICIENTNETB0</div>",
                unsafe_allow_html=True)
    st.markdown("# Architecture du modèle")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("### Spécifications")
        for k, v in [
            ("Paramètres totaux",            "5.3M"),
            ("Paramètres entraînables (tête)","~130K"),
            ("Input",                         "224 × 224 × 3"),
            ("Backbone",                      "Gelé (weights=imagenet)"),
            ("Pooling",                       "GlobalMaxPooling"),
            ("BatchNorm",                     "Après le backbone"),
            ("Dense 1",                       "128 units, ReLU + Dropout 0.3"),
            ("Dense 2",                       "64 units, ReLU"),
            ("Sortie",                        "3 units, Softmax"),
            ("Optimizer",                     "Adam lr=0.001"),
            ("Loss",                          "Categorical Crossentropy"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:.35rem 0;
                        border-bottom:1px solid #1a1d20;font-family:IBM Plex Mono,monospace'>
                <span style='font-size:.75rem;color:#5a5a52'>{k}</span>
                <span style='font-size:.75rem;color:{AMBER}'>{v}</span>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("### Diagramme")
        fig, ax = plt.subplots(figsize=(4, 8))
        ax.set_xlim(0, 4); ax.set_ylim(0, 10); ax.axis('off')
        layers_v = [
            ("INPUT\n224×224×3",              BLUE,  0.6),
            ("EfficientNetB0\n(backbone gelé)", AMBER, 1.2),
            ("BatchNorm",                       GREEN, 0.5),
            ("Dense 128\nReLU + Dropout",       GREEN, 0.7),
            ("Dense 64\nReLU",                  GREEN, 0.6),
            ("Dense 3\nSoftmax",                RED,   0.6),
            ("bird / cat / dog",                RED,   0.5),
        ]
        y = 9.2
        for label, color, h in layers_v:
            rect = plt.Rectangle((0.5, y-h), 3, h*0.85, facecolor=color+'22',
                                  edgecolor=color, linewidth=1.5, zorder=2)
            ax.add_patch(rect)
            ax.text(2, y-h*0.5+h*0.05, label, ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')
            if y-h > 0.5:
                ax.annotate('', xy=(2, y-h-0.05), xytext=(2, y-h+0.02),
                            arrowprops=dict(arrowstyle='->', color='#5a5a52', lw=1.2))
            y -= h + 0.35
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════════════════
#  PAGE : CONTRE-MESURES
# ═════════════════════════════════════════════════════════════════════
elif page == "🛡️  Contre-mesures":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>DÉFENSE · SÉCURITÉ</div>",
                unsafe_allow_html=True)
    st.markdown("# Contre-mesures")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("## Détection")
        for title, desc in [
            ("Surveillance de la loss",
             "Une loss d'entraînement anormalement haute ou oscillante signale des labels incohérents."),
            ("Inspection statistique",
             "Distribution des labels par batch : si un label apparaît trop souvent, c'est suspect."),
            ("Cross-validation",
             "Évaluation sur sous-ensembles isolés : les échantillons empoisonnés dégradent certains folds."),
            ("Détection d'anomalies",
             "Isolation Forest ou LOF sur les embeddings : les samples mal labellisés s'isolent."),
        ]:
            with st.expander(f"🔍 {title}"):
                st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem'>{desc}</div>",
                            unsafe_allow_html=True)

    with c2:
        st.markdown("## Défense active")
        for title, desc in [
            ("Sanitization des données",
             "Nettoyage automatique : supprimer les exemples dont la loss individuelle dépasse un seuil."),
            ("Entraînement robuste",
             "Data augmentation forte + label smoothing rendent l'attaque moins efficace."),
            ("Certified defenses",
             "Randomized smoothing : garantit une précision minimale même sous attaque bornée."),
            ("Audit de provenance",
             "Tracer l'origine de chaque sample : signature cryptographique des annotations."),
        ]:
            with st.expander(f"🛡 {title}"):
                st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem'>{desc}</div>",
                            unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    for col, val, label, sub, color in [
        (c1, "−46%", "impact accuracy",    "à 30% de poison rate",      RED),
        (c2, "100%", "furtivité",          "images non modifiées",       AMBER),
        (c3, "5%",   "seuil critique",     "dégradation perceptible",    GREEN),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#0e1114;border-top:3px solid {color};
                        border-radius:6px;padding:1rem;text-align:center'>
                <div style='font-family:Bebas Neue,sans-serif;font-size:2rem;color:{color}'>{val}</div>
                <div class='mono-tag'>{label}</div>
                <div style='font-size:.78rem;margin-top:.4rem'>{sub}</div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
#  PAGE : CONFIGURATION
# ═════════════════════════════════════════════════════════════════════
elif page == "📂  Configuration":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>SETUP · CHARGEMENT MODÈLES</div>",
                unsafe_allow_html=True)
    st.markdown("# Configuration & Upload des modèles")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    # ── UPLOAD DIRECT ─────────────────────────────────────────────────
    st.markdown("## 📤 Charger les modèles")
    st.markdown("<div class='info-box'>Uploadez directement vos fichiers <code>.keras</code> depuis Kaggle ou votre disque — sans chemin à configurer.</div>",
                unsafe_allow_html=True)
    st.markdown("")

    cu, cp = st.columns(2, gap="large")

    with cu:
        st.markdown("### Modèle propre")
        f_clean = st.file_uploader(
            "model_clean_best.keras",
            type=["keras", "h5"],
            key="upload_clean",
            help="Modèle entraîné sur données non corrompues"
        )
        if f_clean is not None:
            with st.spinner("Chargement du modèle propre..."):
                m, err = load_keras_from_bytes(f_clean)
            if m is not None:
                st.session_state.model_clean = m
                st.session_state.upload_source['clean'] = f"📤 {f_clean.name}"
                st.session_state.upload_errors.pop('clean', None)
                st.markdown(f"<div class='success-box'>✅ Chargé : <b>{f_clean.name}</b><br>"
                            f"Params : {m.count_params():,}</div>", unsafe_allow_html=True)
            else:
                st.session_state.upload_errors['clean'] = err
                st.markdown(f"<div class='danger-box'>❌ Erreur : {err}</div>", unsafe_allow_html=True)
        elif st.session_state.model_clean is not None:
            src = st.session_state.upload_source.get('clean','?')
            st.markdown(f"<div class='success-box'>✅ Modèle actif : {src}<br>"
                        f"Params : {st.session_state.model_clean.count_params():,}</div>",
                        unsafe_allow_html=True)

    with cp:
        st.markdown("### Modèle empoisonné")
        f_poison = st.file_uploader(
            "model_poisoned_best.keras",
            type=["keras", "h5"],
            key="upload_poison",
            help="Modèle ré-entraîné sur données empoisonnées"
        )
        if f_poison is not None:
            with st.spinner("Chargement du modèle empoisonné..."):
                m, err = load_keras_from_bytes(f_poison)
            if m is not None:
                st.session_state.model_poisoned = m
                st.session_state.upload_source['poisoned'] = f"📤 {f_poison.name}"
                st.session_state.upload_errors.pop('poisoned', None)
                st.markdown(f"<div class='success-box'>✅ Chargé : <b>{f_poison.name}</b><br>"
                            f"Params : {m.count_params():,}</div>", unsafe_allow_html=True)
            else:
                st.session_state.upload_errors['poisoned'] = err
                st.markdown(f"<div class='danger-box'>❌ Erreur : {err}</div>", unsafe_allow_html=True)
        elif st.session_state.model_poisoned is not None:
            src = st.session_state.upload_source.get('poisoned','?')
            st.markdown(f"<div class='success-box'>✅ Modèle actif : {src}<br>"
                        f"Params : {st.session_state.model_poisoned.count_params():,}</div>",
                        unsafe_allow_html=True)

    st.markdown("---")

    # ── STATUT ────────────────────────────────────────────────────────
    st.markdown("## Statut actuel")
    md = get_models()
    c1, c2, c3, c4 = st.columns(4)
    checks = [
        (c1, "TensorFlow",                TF_AVAILABLE,          "pip install tensorflow[and-cuda]"),
        (c2, "model_clean_best.keras",    md['clean'] is not None,  "Uploadez ci-dessus"),
        (c3, "model_poisoned_best.keras", md['poisoned'] is not None,"Uploadez ci-dessus"),
        (c4, "GPU actif", TF_AVAILABLE and bool(tf.config.list_physical_devices('GPU')), "pip install tensorflow[and-cuda]"),
    ]
    for col, name, ok, fix in checks:
        color = GREEN if ok else RED
        with col:
            src = md.get('source',{}).get(name.replace("model_","").replace("_best.keras",""), '')
            st.markdown(f"""
            <div style='background:#0e1114;border:1px solid #1a1d20;border-radius:6px;
                        padding:.8rem;border-top:3px solid {color}'>
                <div style='font-family:Bebas Neue,sans-serif;font-size:1.6rem;color:{color}'>
                    {"OK" if ok else "KO"}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;
                            color:#d4d0c8;margin:.2rem 0'>{name}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#5a5a52'>
                    {src if ok else fix}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── RESET + INFOS ─────────────────────────────────────────────────
    col_r, col_i = st.columns([1, 2], gap="large")
    with col_r:
        st.markdown("### Actions")
        if st.button("🗑 Effacer les modèles uploadés"):
            st.session_state.model_clean    = None
            st.session_state.model_poisoned = None
            st.session_state.upload_errors  = {}
            st.session_state.upload_source  = {}
            st.rerun()

    with col_i:
        st.markdown("### Fallback disque (automatique)")
        st.markdown("Si aucun fichier n'est uploadé, l'app cherche automatiquement ici :")
        for d in DISK_DIRS:
            exists = os.path.exists(d)
            color = GREEN if exists else "#5a5a52"
            st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.75rem;"
                        f"color:{color}'>{'✓' if exists else '○'} {d}</div>",
                        unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
#  PAGE : À PROPOS
# ═════════════════════════════════════════════════════════════════════
elif page == "📖  À propos":
    st.markdown("<div class='mono-tag' style='color:#ff8c00'>DOCUMENTATION</div>",
                unsafe_allow_html=True)
    st.markdown("# À propos")
    st.markdown("<div class='hazard-banner'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("## Contexte")
        st.markdown("""
        Ce TP illustre une **attaque par empoisonnement de labels** sur un modèle de classification
        d'images. Le principe : corrompre les annotations du dataset pour dégrader les performances
        sans toucher aux images elles-mêmes.

        **Mécanisme :** `np.roll` sur les vecteurs one-hot — chaque label est décalé d'une position
        (`bird→cat`, `cat→dog`, `dog→bird`), créant des associations image/classe erronées.

        **Impact mesuré :** accuracy dégradée de 94.8% à ~48% avec un poison rate de 30%.
        """)

        st.markdown("## Bugs corrigés")
        for ref, title, desc in [
            ("Bug 1","Compilation commentée",           "model_poisoned.compile() désactivé → ValueError"),
            ("Bug 2","clone_model() sans set_weights()", "Poids non copiés → entraînement from scratch"),
            ("Bug 3","Évaluation sur poisoned_gen",      "Accuracy train sur labels corrompus"),
            ("Bug 4","Indices de poison fixes",          "Attaque biaisée et détectable"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:.8rem;padding:.6rem .8rem;margin:.2rem 0;
                        background:#0e1114;border-radius:4px;border-left:2px solid {RED}'>
                <div style='font-family:IBM Plex Mono,monospace;color:{RED};
                            font-size:.7rem;font-weight:700;min-width:3rem'>{ref}</div>
                <div>
                    <div style='font-size:.85rem;font-weight:600'>{title}</div>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#5a5a52'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("## Stack technique")
        for k, v in [
            ("Python",        "3.9+"),
            ("TensorFlow",    "2.x + [and-cuda]"),
            ("Keras",         "3.x (.keras format)"),
            ("Streamlit",     "1.32+"),
            ("NumPy",         "1.24+"),
            ("Pillow",        "10+"),
            ("Matplotlib",    "3.8+"),
            ("Dataset",       "Cat/Dog/Bird · 13 344"),
            ("GPU local",     "RTX 3060 · 6 GB"),
            ("GPU Kaggle",    "Tesla P100 · 16 GB"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:.35rem 0;
                        border-bottom:1px solid #1a1d20;font-family:IBM Plex Mono,monospace'>
                <span style='font-size:.75rem;color:#5a5a52'>{k}</span>
                <span style='font-size:.75rem;color:{AMBER}'>{v}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("## Lancer")
        st.code("streamlit run app.py", language="bash")