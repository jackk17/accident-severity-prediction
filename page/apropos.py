"""
Page À Propos - Détails techniques et information sur le projet
"""

import streamlit as st


def show():
    """Afficher la page À Propos"""

    # Header
    st.markdown("""
    <div class="header-main">
        <h1>ℹ️ À Propos du Projet</h1>
        <p>Détails techniques et informations sur cette application</p>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Objectif
    st.markdown("### 🎯 Objectif du Projet")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("""
        Cette application a été développée pour **prédire la sévérité des accidents
        de la route** en utilisant des techniques avancées de **Machine Learning**.

        **Buts Principaux:**
        - 🤖 Fournir des prédictions précises et fiables
        - 📊 Analyser les facteurs de risque routier
        - 💡 Sensibiliser aux conditions dangereuses
        - ⚡ Aider à la prise de décision en temps réel
        """)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Type de Classification</h3>
            <p><strong>Binaire</strong></p>
            <p style="font-size: 0.9rem;">Minor vs Severe</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Méthodologie
    st.markdown("### 🔬 Méthodologie")

    tab1, tab2, tab3 = st.tabs(["📊 Données", "🤖 Modèle", "📈 Performances"])

    with tab1:
        st.markdown("""
        #### 📊 Données Utilisées

        **Source:**
        - Dataset: UK Road Accidents (49,998 accidents originaux)
        - Période: 2000-2015
        - Localisation: Royaume-Uni

        **Processus de Nettoyage:**
        1. Chargement des données brutes (49,998 lignes)
        2. Suppression des valeurs manquantes (0 supprimées)
        3. Conversion à classification binaire:
           - Classes 1+2 (Faible + Grave) → Minor (0)
           - Classe 3 (Très Grave) → Severe (1)
        4. Rééquilibrage des classes:
           - Before: 65.87x imbalancé
           - After: 1.0x équilibré (14,886 lignes)

        **Variables Sélectionnées (Top 10):**
        1. 1st_Road_Number - Numéro route principale
        2. Police_Force - Zone de police
        3. Year - Année
        4. heure_num - Heure (0-23h)
        5. Day_of_Week - Jour de la semaine
        6. 2nd_Road_Number - Numéro route secondaire
        7. 1st_Road_Class - Classe de route
        8. Number_of_Vehicles - Nombre de véhicules
        9. Number_of_Casualties - Nombre de victimes
        10. Speed_limit - Limite de vitesse
        """)

    with tab2:
        st.markdown("""
        #### 🤖 Modèle de Machine Learning

        **Algorithme:** LogisticRegression (Classification Linéaire)

        **Caractéristiques:**
        - Type: Classification Binaire (Minor/Severe)
        - Framework: scikit-learn
        - Normalisation: StandardScaler
        - Test/Train Split: 80/20 (stratifié)
        - Random State: 42 (reproductibilité)

        **Paramètres du Modèle:**
        ```python
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        ```

        **Avantages:**
        - ✅ Interprétabilité élevée
        - ✅ Entraînement rapide
        - ✅ Probabilités natives
        - ✅ Faible consommation mémoire
        - ✅ Idéal pour classification binaire
        """)

    with tab3:
        st.markdown("""
        #### 📈 Métriques de Performance

        | Métrique | Score | Interprétation |
        |----------|-------|----------------|
        | **Accuracy** | 58.93% | Proportion d'exemplestotaux bien classés |
        | **Precision** | 59.41% | Fiabilité des prédictions positives |
        | **Recall** | 56.41% | Capacité à détecter les vrais accidents graves |
        | **F1-Score** | 0.5787 | Moyenne harmonique précision-rappel |
        | **ROC-AUC** | 0.6118 | Capacité de discrimination |

        **Matrice de Confusion:**
        - TN: 915 (Vrais négatifs)
        - FP: 574 (Faux positifs)
        - FN: 649 (Faux négatifs)
        - TP: 840 (Vrais positifs)

        **Analyse:**
        - Le modèle détecte 56% des accidents graves
        - Faible taux de faux positifs (38%)
        - Performance modérée mais acceptable pour une classification binaire
        """)

    st.markdown("---")

    # Section 3: Stack Technique
    st.markdown("### 💻 Stack Technologique")

    tech_col1, tech_col2, tech_col3 = st.columns(3, gap="medium")

    with tech_col1:
        st.markdown("""
        <div class="info-card">
            <h3>🐍 Backend</h3>
            <p><strong>Python 3.8+</strong></p>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>scikit-learn</li>
                <li>pandas</li>
                <li>numpy</li>
                <li>joblib</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tech_col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎨 Frontend</h3>
            <p><strong>Streamlit</strong></p>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>Interface web</li>
                <li>CSS personnalisé</li>
                <li>Visualisations</li>
                <li>Interactive widgets</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tech_col3:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Données</h3>
            <p><strong>Formats</strong></p>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>CSV</li>
                <li>Pickle (modèles)</li>
                <li>Pandas DataFrames</li>
                <li>NumPy arrays</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 4: Architecture
    st.markdown("### 🏗️ Architecture du Projet")

    st.markdown("""
    ```
    accident_severity_prediction/
    │
    ├── app.py                          ← Point d'entrée principal
    ├── nouveau_traitement.py           ← Traitement & entraînement
    ├── train_model.py                  ← Entraînement du modèle
    │
    ├── pages/                          ← Pages Streamlit
    │   ├── accueil.py                  ← 🏠 Accueil
    │   ├── prediction.py               ← 🎯 Prédictions
    │   ├── analyse.py                  ← 📊 Analyse
    │   └── apropos.py                  ← ℹ️ À Propos (cette page)
    │
    ├── models/                         ← Modèles sauvegardés
    │   ├── logistic_regression_model.pkl
    │   └── standard_scaler.pkl
    │
    ├── data/                           ← Données
    │   ├── df_sample.csv               ← Données originales
    │   └── df_equilibre_binaire.csv    ← Données équilibrées
    │
    └── README.md, requirements.txt, etc.
    ```
    """)

    st.markdown("---")

    # Section 5: Utilisation
    st.markdown("### 🚀 Comment Utiliser l'Application")

    with st.expander("📦 Installation", expanded=False):
        st.markdown("""
        #### Prérequis
        - Python 3.8 ou supérieur
        - pip ou conda

        #### Étapes
        1. **Cloner le repository**
           ```bash
           git clone https://github.com/jackk17/accident_severity_prediction.git
           cd accident_severity_prediction
           ```

        2. **Créer un environnement virtuel**
           ```bash
           python -m venv venv
           source venv/bin/activate  # Sur Windows: venv\\Scripts\\activate
           ```

        3. **Installer les dépendances**
           ```bash
           pip install -r requirements.txt
           ```

        4. **Préparer les données et entraîner le modèle**
           ```bash
           python nuevo_traitement.py
           ```

        5. **Lancer l'application**
           ```bash
           streamlit run app.py
           ```
        """)

    with st.expander("🎯 Effectuer une Prédiction", expanded=False):
        st.markdown("""
        1. **Naviguez à la page "Prédiction"**
        2. **Remplissez tous les paramètres:**
           - Route, police, année
           - Heure, jour de la semaine
           - Nombre de véhicules et victimes
           - Classe de route, limite de vitesse
        3. **Cliquez sur "Effectuer la Prédiction"**
        4. **Consultez les résultats:**
           - Classe prédite (Minor/Severe)
           - Confiance du modèle
           - Probabilités détaillées
           - Recommandations de sécurité
        """)

    with st.expander("📊 Analyser les Données", expanded=False):
        st.markdown("""
        1. **Naviguez à la page "Analyse"**
        2. **Explorez 4 onglets:**
           - Distribution: Graphiques des variables
           - Corrélations: Matrice de corrélation
           - Tendances: Évolution temporelle
           - Données: Tableau filtrable et exportable
        3. **Filtrez par sévérité ou année**
        4. **Exportez les résultats en CSV**
        """)

    st.markdown("---")

    # Section 6: Contributeur
    st.markdown("### 👤 Contributeur et Contact")

    st.markdown("""
    <div class="info-card">
        <h3>Agbegbo Jacque</h3>
        <p><strong>Rôle:</strong> Développeur Principal</p>
        <p><strong>Version:</strong> 2.0 (Classification Binaire)</p>
        <p><strong>Date:</strong> Mars 2026</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 7: Limitations et Disclaimer
    st.markdown("### ⚠️ Limitations et Disclaimer")

    st.warning("""
    **IMPORTANT - À LIRE ATTENTIVEMENT**

    Cette application est développée à **fins éducatives et de démonstration** uniquement.

    **Limitations:**
    - Modèle basé sur données UK (2000-2015)
    - Ne prend pas en compte tous les facteurs de risque
    - Performance modérée (58.93% accuracy)
    - Données historiques, pas en temps réel

    **Disclaimer:**
    - Les prédictions ne doivent PAS être utilisées comme seul critère de décision
    - Ne remplace pas le jugement humain et la prudence
    - Utilisez à titre informatif uniquement
    - Les auteurs ne sont pas responsables des usages impropres
    """)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 2px solid #e0e0e0; color: #999;">
        <p>© 2026 - Prédicteur de Sévérité d'Accidents | Version 2.0 - Classification Binaire</p>
        <p>Construit avec <strong>Streamlit</strong> • <strong>scikit-learn</strong> • <strong>Python</strong></p>
    </div>
    """, unsafe_allow_html=True)
