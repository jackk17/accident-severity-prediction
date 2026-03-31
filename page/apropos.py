"""Page A Propos - Version avec Random Forest"""

import json
import streamlit as st
from pathlib import Path


def show():
    st.markdown("""
    <div class="hero">
        <h1>À Propos du Projet</h1>
        <p>Prédiction de la sévérité des accidents routiers avec Random Forest</p>
    </div>""", unsafe_allow_html=True)

    # --- Objectif ---
    st.markdown("### 🎯 Objectif")
    st.markdown("""
    Prédire la **sévérité des accidents de la route** (Mineur / Grave) à partir de données historiques UK (2000-2015) 
    en utilisant le **Machine Learning** avec un modèle **Random Forest** optimisé.
    """)

    st.markdown("---")

    # --- Méthodologie ---
    st.markdown("### 📊 Méthodologie")

    tab1, tab2, tab3 = st.tabs(["📁 Données", "🤖 Modèles", "⚙️ Pipeline"])

    with tab1:
        st.markdown("""
        **Source** : UK Road Accidents — **1.5 million** d'enregistrements, 30 variables initiales

        **Conversion binaire** :
        - Classes 1 + 2 (Fatal + Serious) → **Grave (1)**
        - Classe 3 (Slight) → **Mineur (0)**

        **Feature Engineering avancé** (17 nouvelles features) :

        | Feature | Description |
        |---------|-------------|
        | `casualties_per_vehicle` | Nombre de victimes par véhicule |
        | `is_night` | Accident de nuit (22h-5h) |
        | `is_weekend` | Accident le week-end |
        | `bad_weather` | Conditions météo dangereuses |
        | `bad_road` | État de route dangereux |
        | `poor_light` | Mauvais éclairage |
        | `composite_risk_score` | Score de risque combiné (8 facteurs) |
        | `high_speed_night` | Grande vitesse + nuit |
        | `rush_hour_urban` | Heure de pointe en zone urbaine |

        **Top 5 des variables les plus importantes** :
        """)
        
        # Afficher les meilleures features si disponibles
        p = Path("models/metrics.json")
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            if 'best_features' in metrics:
                st.markdown("| # | Variable | Importance |")
                st.markdown("|---|----------|------------|")
                for i, feat in enumerate(metrics['best_features'][:5], 1):
                    st.markdown(f"| {i} | `{feat['feature']}` | {feat['importance']*100:.1f}% |")
            else:
                st.markdown("""
                | # | Variable | Importance |
                |---|----------|------------|
                | 1 | `casualties_per_vehicle` | 15.5% |
                | 2 | `Number_of_Vehicles` | 15.3% |
                | 3 | `composite_risk_score` | 9.4% |
                | 4 | `Speed_limit` | 7.7% |
                | 5 | `is_night` | 5.9% |
                """)

    with tab2:
        st.markdown("""
        **Modèles comparés** avec gestion du déséquilibre des classes :

        | Modèle | Configuration | Performance |
        |--------|--------------|-------------|
        | **Random Forest** | 100 arbres, max_depth=10, class_weight='balanced' | **Meilleur F1-macro** |
        | Balanced Random Forest | n_estimators=100, équilibrage intégré | Bonne alternative |
        | Logistic Regression | class_weight='balanced', max_iter=1000 | Baseline |
        | Gradient Boosting | 150 estimateurs, max_depth=5 | Compétitif |
        """)

        # Charger métriques dynamiquement
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            st.markdown(f"### 🏆 **Modèle retenu : Random Forest**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            with col2:
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            with col3:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            with col4:
                st.metric("Recall (Grave)", f"{metrics['recall_grave']*100:.1f}%")
            
            st.markdown("""
            **Interprétation des métriques :**
            - **Accuracy** : 62% des prédictions correctes
            - **Recall (Grave)** : 61% des accidents graves sont détectés ✅
            - **ROC-AUC** : 0.65, meilleur que le hasard (0.5)
            """)

    with tab3:
        st.markdown("""
        **Pipeline complet de traitement des données** :

        1. **Chargement** des données brutes (1.5M accidents)
        2. **Nettoyage** : Suppression des valeurs manquantes
        3. **Conversion binaire** : 3 classes → 2 classes (Mineur/Grave)
        4. **Feature Engineering avancé** : Création de 17 nouvelles features
        5. **Sélection** : 21 features les plus pertinentes
        6. **Échantillonnage stratégique** : Ratio 1:2 (Grave:Mineur)
        7. **Normalisation** : StandardScaler
        8. **Entraînement** : Random Forest avec class_weight='balanced'
        9. **Évaluation** : Métriques complètes (Accuracy, F1, Recall, ROC-AUC)
        """)

    st.markdown("---")

    # --- Améliorations apportées ---
    st.markdown("### 📈 Améliorations vs version initiale")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Version initiale (Logistic Regression) :**
        - Accuracy: 58.9%
        - Recall (Grave): 56%
        - ROC-AUC: 0.61
        - 10 variables basiques
        """)
    
    with col2:
        st.markdown("""
        **Version améliorée (Random Forest) :**
        - Accuracy: 62.0% **(+3.1%)**
        - Recall (Grave): 61% **(+5%)**
        - ROC-AUC: 0.65 **(+0.04)**
        - 21 variables + features combinées
        """)

    st.markdown("---")

    # --- Stack technique ---
    st.markdown("### 🛠️ Stack technique")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Backend**
        - Python 3.9+
        - scikit-learn 1.3+
        - imbalanced-learn (SMOTE-Tomek)
        - pandas / numpy
        - joblib (sérialisation)
        """)
    with col2:
        st.markdown("""
        **Frontend**
        - Streamlit 1.28+
        - Matplotlib / Seaborn
        - Plotly (graphiques interactifs)
        - CSS personnalisé
        """)
    with col3:
        st.markdown("""
        **Modèles & Données**
        - Random Forest
        - Balanced Random Forest
        - Logistic Regression
        - 1.5M enregistrements
        - 30+ features
        """)

    st.markdown("---")

    # --- Architecture du projet ---
    st.markdown("### 📁 Architecture du projet")
    st.code("""
projet_fil_rouge/
├── app.py                      # Point d'entrée Streamlit
├── train_model.py              # Script d'entraînement
├── page/
│   ├── acceuil.py              # Métriques du modèle
│   ├── prediction.py           # Formulaire de prédiction
│   ├── analyse.py              # Visualisations interactives
│   └── apropos.py              # Cette page
├── models/
│   ├── random_forest_model.pkl # Modèle entraîné
│   ├── scaler.pkl              # Normalisateur
│   ├── features.json           # Liste des features
│   └── metrics.json            # Performances
└── data/
    ├── df_with_features.csv    # Données avec features
    └── df_sample.csv           # Échantillon original
    """, language=None)

    st.markdown("---")

    # --- Contributeur ---
    st.markdown("### 👨‍💻 Contributeur")
    st.markdown("**AGBEGBO Espoir Jacques Kwassi** — Développeur principal & Data Scientist")

    st.markdown("---")

    # --- Version ---
    st.markdown("### 📌 Version")
    st.markdown("**v2.0** - Modèle Random Forest avec feature engineering avancé (Mars 2025)")

    st.markdown("---")
    
    st.info(
        "⚠️ **Note importante** : Ce modèle est à but éducatif. Les prédictions ne remplacent pas "
        "le jugement humain en matière de sécurité routière. "
        "Le modèle détecte correctement 61% des accidents graves."
    )