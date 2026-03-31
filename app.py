"""Application Streamlit - Prediction de Severite d'Accidents"""

import streamlit as st
from pathlib import Path
import sys
import os

# Configuration de la page (première commande Streamlit)
st.set_page_config(
    page_title="Severite d'Accidents",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Vérification et entraînement du modèle ---
def check_and_train_model():
    """Vérifier si le modèle existe, sinon l'entraîner"""
    model_path = Path("models/random_forest_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    
    # Si le modèle existe déjà, ne rien faire
    if model_path.exists() and scaler_path.exists():
        return True
    
    # Sinon, entraîner le modèle
    with st.spinner("🔄 Premier lancement: entraînement du modèle en cours (2-3 minutes)..."):
        try:
            # Créer le dossier models s'il n'existe pas
            Path("models").mkdir(exist_ok=True)
            
            # Importer et exécuter l'entraînement
            import train_model
            
            # Utiliser l'échantillon pour un entraînement plus rapide
            train_model.main(use_equilibred=True)
            
            st.success("✅ Modèle entraîné avec succès!")
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement: {e}")
            st.error("Veuillez vérifier que le fichier data/df_with_features_sample.csv existe.")
            st.stop()
            return False

# Exécuter la vérification avant de charger les pages
model_ready = check_and_train_model()

# Si le modèle n'est pas prêt, arrêter l'exécution
if not model_ready:
    st.stop()

# --- CSS ---
st.markdown("""<style>
:root { --blue: #2563eb; --green: #16a34a; --red: #dc2626; --gray: #64748b; }

.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    padding: 2rem 2.5rem; border-radius: 12px; color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero p  { margin: .4rem 0 0; opacity: .85; font-size: 1rem; }

.card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1.3rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.card .label { font-size: .8rem; color: var(--gray); text-transform: uppercase;
               letter-spacing: .5px; margin-bottom: .3rem; }
.card .value { font-size: 1.6rem; font-weight: 700; color: var(--blue); }

.card-green .value { color: var(--green); }
.card-red   .value { color: var(--red); }

.info-box {
    background: #f8fafc; border-left: 4px solid var(--blue);
    padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
}

/* Correction des couleurs de la sidebar */
[data-testid="stSidebar"] {
    background: #f8fafc;
}

[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: #1e293b !important;
    font-weight: 500;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    background: white;
    padding: 0.5rem;
    border-radius: 8px;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    padding: 0.5rem;
    border-radius: 6px;
    transition: background-color 0.2s;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background-color: #e2e8f0;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {
    background-color: var(--blue);
    color: white !important;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] * {
    color: white !important;
}

/* Correction pour les captions */
[data-testid="stSidebar"] .stCaption {
    color: #475569 !important;
}
</style>""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🚗 Navigation")
    # Correction : Ajout d'un label non-vide avec label_visibility="collapsed"
    page = st.radio("Navigation", ["Accueil", "Prediction", "Analyse", "A propos"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption("Contributeur: **AGBEGBO Espoir Jacques Kwassi**")

# --- Pages ---
from page import acceuil, prediction, analyse, apropos

# Afficher la page sélectionnée
if page == "Accueil":
    acceuil.show()  # Notez les parenthèses
elif page == "Prediction":
    prediction.show()  # Notez les parenthèses
elif page == "Analyse":
    analyse.show()  # Notez les parenthèses
elif page == "A propos":
    apropos.show()  # Notez les parenthèses