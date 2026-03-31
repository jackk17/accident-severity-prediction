"""
🚗 Prédicteur de Sévérité d'Accidents
Application Machine Learning - Classification Binaire
Prédire la sévérité des accidents de la route
"""

import streamlit as st
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Prédicteur de Sévérité d'Accidents | Minor vs Severe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**Prédicteur de Sévérité d'Accidents**\n\nVersion 2.0 - Classification Binaire\n\nContributeur: Agbegbo Jacque"
    }
)

# ═══════════════════════════════════════════════════════════════════════════
# STYLE CSS PROFESSIONNEL
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    /* ==================== VARIABLES COULEURS ==================== */
    :root {
        --primary: #1f77b4;
        --secondary: #ff7f0e;
        --success: #2ca02c;
        --danger: #d62728;
        --warning: #ff9800;
        --info: #17a2b8;
        --light: #f8f9fa;
        --dark: #212529;
    }

    /* ==================== STYLES GLOBAUX ==================== */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* ==================== HEADER PRINCIPAL ==================== */
    .header-main {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .header-main h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .header-main p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }

    /* ==================== CARTES D'INFORMATION ==================== */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        padding: 1.8rem;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }

    .info-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }

    .info-card h3 {
        margin: 0 0 0.8rem 0;
        color: #1f77b4;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* ==================== CARTES STATISTIQUES ==================== */
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 0.95rem;
        color: #666;
        font-weight: 500;
    }

    /* ==================== BOUTONS ==================== */
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }

    .stButton > button:hover {
        box-shadow: 0 8px 20px rgba(31, 119, 180, 0.4);
        transform: translateY(-2px);
    }

    /* ==================== SLIDERS ET INPUTS ==================== */
    .stSlider [role="slider"] {
        background: linear-gradient(to right, #1f77b4, #2ca02c);
    }

    .stNumberInput input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem;
        font-size: 1rem;
    }

    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: #666;
        border-bottom: 3px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #1f77b4;
        border-bottom-color: #1f77b4;
    }

    /* ==================== DIVIDER ==================== */
    .stHorizontalBlock {
        margin: 2rem 0;
    }

    /* ==================== SIDEBAR ==================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        padding: 2rem;
        border-radius: 0 15px 15px 0;
        color: white;
    }

    /* ==================== TEXTE ET TITRES ==================== */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    h2 {
        color: #1f77b4;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #e0e0e0;
    }

    h3 {
        color: #333;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* ==================== DATAFRAME ==================== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        border-radius: 8px;
    }

    /* ==================== SUCCESS/WARNING/ERROR ==================== */
    .stSuccess {
        background: #d4edda;
        color: #155724;
        border-radius: 8px;
        border-left: 5px solid #2ca02c;
    }

    .stWarning {
        background: #fff3cd;
        color: #856404;
        border-radius: 8px;
        border-left: 5px solid #ff9800;
    }

    .stError {
        background: #f8d7da;
        color: #721c24;
        border-radius: 8px;
        border-left: 5px solid #d62728;
    }

    .stInfo {
        background: #d1ecf1;
        color: #0c5460;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
    }

    /* ==================== FOOTER ==================== */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid #e0e0e0;
        color: #999;
        font-size: 0.9rem;
    }

    /* ==================== RESPONSIVE ==================== */
    @media (max-width: 768px) {
        .header-main h1 {
            font-size: 2rem;
        }

        .stat-value {
            font-size: 1.8rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; margin: 0;">🚗 Menu Principal</h2>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 0.9rem; margin-top: 0.5rem;">
            Classification Binaire
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page_selection = st.radio(
        "Naviguer vers:",
        ["🏠 Accueil", "🎯 Prédiction", "📊 Analyse", "ℹ️ À Propos"],
        label_visibility="collapsed",
        key="page_nav"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Modèle", "LogReg", "Binaire")
    with col2:
        st.metric("Données", "14.9K", "Équilibré")

    st.markdown("---")

    st.markdown("""
    <div style="margin-top: 2rem;">
        <h4 style="color: #333; margin-top: 1rem;">📝 Informations</h4>
        <p style="font-size: 0.85rem; color: #666; line-height: 1.6;">
        <strong>Version:</strong> 2.0<br>
        <strong>Type:</strong> Classification Binaire<br>
        <strong>Classes:</strong> Minor vs Severe<br>
        <strong>Accuracy:</strong> 58.93%
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">
        <h4 style="color: #333; margin-bottom: 0.5rem;">👤 Contributeur</h4>
        <p style="font-size: 0.9rem; color: #666; margin: 0;">
        <strong>Agbegbo Jacque</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# IMPORT DES PAGES
# ═══════════════════════════════════════════════════════════════════════════

from page import accueil, prediction, analyse, apropos

pages_map = {
    "🏠 Accueil": accueil,
    "🎯 Prédiction": prediction,
    "📊 Analyse": analyse,
    "ℹ️ À Propos": apropos
}

# ═══════════════════════════════════════════════════════════════════════════
# AFFICHAGE DE LA PAGE SÉLECTIONNÉE
# ═══════════════════════════════════════════════════════════════════════════

pages_map[page_selection].show()

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2026 - Prédicteur de Sévérité d'Accidents | Classification Binaire (Minor vs Severe)</p>
    <p>Construit avec <strong>Streamlit</strong> • Modèle <strong>LogisticRegression</strong> • Machine Learning</p>
</div>
""", unsafe_allow_html=True)
