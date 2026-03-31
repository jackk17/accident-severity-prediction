"""
Page Accueil - Page d'accueil professionnelle
"""

import streamlit as st
import pandas as pd


def show():
    """Afficher la page d'accueil"""

    # Header Principal
    st.markdown("""
    <div class="header-main">
        <h1>🚗 Prédicteur de Sévérité d'Accidents</h1>
        <p>Classification Binaire • Machine Learning • Prédiction en Temps Réel</p>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Présentation
    col1, col2 = st.columns([1.2, 0.8], gap="medium")

    with col1:
        st.markdown("""
        ### 📋 À Propos du Projet

        Cette application utilise un **modèle de Machine Learning avancé** pour prédire la sévérité
        des accidents de la route. Notre système analyse les paramètres clés des accidents
        (heure, vitesse, nombre de véhicules, etc.) pour fournir une prédiction précise et fiable.

        **Classification Binaire:**
        - **Minor** 🟢 - Accidents légers (Faible + Grave)
        - **Severe** 🔴 - Accidents graves (Très Grave)
        """)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Modèle Utilisé</div>
            <div class="stat-value">LogisticRegression</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Accuracy</div>
            <div class="stat-value">58.93%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Métriques Clés
    st.markdown("### 📊 Métriques de Performance")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4, gap="medium")

    with metrics_col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Accuracy</div>
            <div class="stat-value">58.93%</div>
            <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Classification correcte</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Precision</div>
            <div class="stat-value">59.41%</div>
            <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Fiabilité positive</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Recall</div>
            <div class="stat-value">56.41%</div>
            <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Détection grave</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">ROC-AUC</div>
            <div class="stat-value">0.6118</div>
            <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Discrimination</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 3: Dataset
    st.markdown("### 📈 Informations sur le Dataset")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Dataset Original</h3>
            <p><strong>Total:</strong> 49,998 accidents</p>
            <p><strong>Classes:</strong> 3 (Faible, Grave, Très Grave)</p>
            <p><strong>Déséquilibre:</strong> 65.87x (très imbalancé)</p>
            <p><strong>Variables:</strong> 10 features pertinentes</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>⚖️ Dataset Équilibré</h3>
            <p><strong>Total:</strong> 14,886 accidents</p>
            <p><strong>Classes:</strong> 2 (Minor, Severe)</p>
            <p><strong>Équilibre:</strong> 1.0x (parfait!)</p>
            <p><strong>Split:</strong> 80% train / 20% test</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 4: Features
    st.markdown("### 🔧 Variables Utilisées (Top 10)")

    features_data = {
        "Rang": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Variable": [
            "1st_Road_Number",
            "Police_Force",
            "Year",
            "heure_num",
            "Day_of_Week",
            "2nd_Road_Number",
            "1st_Road_Class",
            "Number_of_Vehicles",
            "Number_of_Casualties",
            "Speed_limit"
        ],
        "Description": [
            "Numéro de la route principale",
            "Zone de police (juridiction)",
            "Année de l'accident",
            "Heure de l'accident (0-23h)",
            "Jour de la semaine",
            "Numéro de la route secondaire",
            "Classification de la route",
            "Nombre de véhicules impliqués",
            "Nombre de victimes",
            "Limitation de vitesse (km/h)"
        ]
    }

    df_features = pd.DataFrame(features_data)
    st.dataframe(df_features, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 5: Processus
    st.markdown("### 🔄 Processus d'Entraînement")

    col1, col2, col3, col4, col5 = st.columns(5, gap="small")

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h4 style="color: white; margin: 0;">1️⃣</h4>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Chargement</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #f0f0f0; color: #333; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #333; margin: 0;">2️⃣</h4>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Conversion</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: #f0f0f0; color: #333; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #333; margin: 0;">3️⃣</h4>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Nettoyage</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: #f0f0f0; color: #333; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #333; margin: 0;">4️⃣</h4>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Rééquilibrage</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style="background: #f0f0f0; color: #333; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h4 style="color: #333; margin: 0;">5️⃣</h4>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Entraînement</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 6: Guide d'Utilisation
    st.markdown("### 📖 Guide d'Utilisation")

    tab1, tab2, tab3 = st.tabs(["🎯 Prédiction", "📊 Analyse", "ℹ️ Informations"])

    with tab1:
        st.markdown("""
        #### Effectuer une Prédiction

        1. **Accédez à la page "Prédiction"** depuis le menu latéral
        2. **Remplissez le formulaire** avec les paramètres de l'accident:
           - Heure, jour de la semaine
           - Limitation de vitesse
           - Nombre de véhicules et victimes
           - Localisation et classe de route
        3. **Cliquez sur "Prédire la Sévérité"**
        4. **Consultez les résultats:**
           - Classe prédite (Minor ou Severe)
           - Confiance du modèle en pourcentage
           - Probabilités par classe
           - Recommandations de sécurité
        """)

    with tab2:
        st.markdown("""
        #### Analyser les Données

        1. **Accédez à la page "Analyse"** depuis le menu latéral
        2. **Explorez 4 onglets différents:**
           - **Distribution:** Graphiques de distribution
           - **Corrélations:** Matrice de corrélation heatmap
           - **Tendances:** Évolution par année et jour
           - **Données:** Tableau filtrable et téléchargeable
        3. **Filtrez par sévérité ou année** selon vos besoins
        4. **Téléchargez les données** en CSV pour analyse externe
        """)

    with tab3:
        st.markdown("""
        #### Plus d'Informations

        - Consultez la page "À Propos" pour les détails techniques
        - Comprenez la méthodologie et les performances du modèle
        - Découvrez les variables et leur importance
        - Contactez le contributeur pour des questions
        """)

    st.markdown("---")

    # Section 7: Actions Rapides
    st.markdown("### 🚀 Actions Rapides")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        if st.button("🎯 Aller à Prédiction", use_container_width=True):
            st.switch_page("pages/prediction.py")

    with col2:
        if st.button("📊 Aller à Analyse", use_container_width=True):
            st.switch_page("pages/analyse.py")

    with col3:
        if st.button("ℹ️ Aller à À Propos", use_container_width=True):
            st.switch_page("pages/apropos.py")

    st.markdown("---")

    # Section 8: Notes
    st.markdown("""
    <div class="info-card">
        <h3>⚠️ Important</h3>
        <p style="margin: 0;">
        Ce modèle est conçu à des fins éducatives et de démonstration.
        Les prédictions doivent être utilisées comme support décisionnel uniquement,
        et non comme base décisionnelle exclusive pour des questions de sécurité routière.
        </p>
    </div>
    """, unsafe_allow_html=True)
