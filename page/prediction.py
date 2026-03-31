"""
Page Prédiction - Interface de prédiction professionnelle
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib


def load_model_and_scaler():
    """Charger le modèle et le scaler"""
    try:
        model_path = Path("models/logistic_regression_model.pkl")
        scaler_path = Path("models/standard_scaler.pkl")

        if not model_path.exists() or not scaler_path.exists():
            st.error("❌ Modèle ou scaler non trouvés. Veuillez lancer: python nouveau_traitement.py")
            return None, None

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None, None


def show():
    """Afficher la page Prédiction"""

    # Header
    st.markdown("""
    <div class="header-main">
        <h1>🎯 Prédiction de Sévérité</h1>
        <p>Entrez les paramètres de l'accident pour obtenir une prédiction</p>
    </div>
    """, unsafe_allow_html=True)

    # Charger le modèle
    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.stop()

    # Formulaire de prédiction avec structure améliorée
    st.markdown("### 📝 Paramètres de l'Accident")

    # Diviser en 2 colonnes
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🛣️ Informations de la Route")

        route_number_1 = st.slider(
            "Numéro de route principale",
            min_value=1, max_value=500, value=100,
            help="Identifiant de la route principale (1-500)"
        )

        police_force = st.slider(
            "Zone de police",
            min_value=0, max_value=50, value=25,
            help="Zone de juridiction de la police (0-50)"
        )

        year = st.slider(
            "Année de l'accident",
            min_value=2000, max_value=2025, value=2023,
            help="Année de survenance"
        )

        road_class = st.slider(
            "Classification de la route",
            min_value=1, max_value=10, value=5,
            help="Type de route (1-10)"
        )

    with col2:
        st.markdown("#### ⏰ Conditions Temporelles et Locales")

        heure = st.slider(
            "Heure de l'accident",
            min_value=0, max_value=23, value=12,
            help="Heure en format 24h (0-23)"
        )

        day_of_week = st.slider(
            "Jour de la semaine",
            min_value=0, max_value=6, value=3,
            help="0=Lundi, 1=Mardi, 2=Mercredi, 3=Jeudi, 4=Vendredi, 5=Samedi, 6=Dimanche"
        )

        route_number_2 = st.slider(
            "Numéro de route secondaire",
            min_value=1, max_value=5000, value=500,
            help="Identifiant de la route secondaire (1-5000)"
        )

        speed_limit = st.slider(
            "Limitation de vitesse (km/h)",
            min_value=20, max_value=150, value=50,
            help="Limité en km/h (20-150)"
        )

    st.markdown("---")

    # Section véhicules et victimes
    st.markdown("### 🚗 Véhicules et Victimes")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        nb_vehicules = st.slider(
            "Nombre de véhicules impliqués",
            min_value=1, max_value=10, value=2,
            help="Nombre de véhicules dans l'accident (1-10)"
        )

    with col2:
        nb_victimes = st.slider(
            "Nombre de victimes",
            min_value=0, max_value=10, value=1,
            help="Nombre de personnes blessées ou tuées (0-10)"
        )

    st.markdown("---")

    # Résumé des paramètres
    st.markdown("### 📊 Résumé des Paramètres Saisis")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4, gap="medium")

    with summary_col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">⏰ Heure</div>
            <div class="stat-value">{heure:02d}:00</div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col2:
        day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">📅 Jour</div>
            <div class="stat-value">{day_names[day_of_week]}</div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">🚗 Véhicules</div>
            <div class="stat-value">{nb_vehicules}</div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">👥 Victimes</div>
            <div class="stat-value">{nb_victimes}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Bouton de prédiction
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🔮 Effectuer la Prédiction", use_container_width=True, key="predict_btn"):
            with st.spinner("⏳ Analyse en cours..."):
                # Préparer les données
                input_data = np.array([[
                    route_number_1,
                    police_force,
                    year,
                    heure,
                    day_of_week,
                    route_number_2,
                    road_class,
                    nb_vehicules,
                    nb_victimes,
                    speed_limit
                ]])

                # Normaliser
                input_data_scaled = scaler.transform(input_data)

                # Prédire
                prediction = int(model.predict(input_data_scaled)[0])
                probabilities = model.predict_proba(input_data_scaled)[0]

                # Mapper les prédictions
                severity_map = {
                    0: {
                        "label": "Minor (Accidents Légers)",
                        "emoji": "🟢",
                        "color": "#2ca02c",
                        "icon": "✅"
                    },
                    1: {
                        "label": "Severe (Accidents Graves)",
                        "emoji": "🔴",
                        "color": "#d62728",
                        "icon": "⚠️"
                    },
                }

                sev_info = severity_map.get(prediction, {"label": "Inconnu", "emoji": "❓", "color": "#95a5a6"})
                confiance = max(probabilities) * 100

                # Afficher les résultats
                st.markdown("---")
                st.markdown("### 🎯 Résultats de la Prédiction")

                # Carte de résultat principal
                result_col1, result_col2, result_col3 = st.columns(3, gap="medium")

                with result_col1:
                    st.markdown(f"""
                    <div class="stat-card" style="border-left: 5px solid {sev_info['color']};">
                        <div class="stat-label">Sévérité Prédite</div>
                        <p style="font-size: 2.5rem; text-align: center; margin: 0.5rem 0;">
                            {sev_info['emoji']}
                        </p>
                        <p style="font-size: 1.2rem; font-weight: 700; text-align: center; color: {sev_info['color']}; margin: 0;">
                            {sev_info['label']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with result_col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Confiance du Modèle</div>
                        <div class="stat-value">{confiance:.1f}%</div>
                        <div style="font-size: 0.9rem; color: #666; text-align: center;">
                            Degré de certitude
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with result_col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Classe Binaire</div>
                        <div class="stat-value">{prediction}</div>
                        <div style="font-size: 0.9rem; color: #666; text-align: center;">
                            0 = Minor | 1 = Severe
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Barre de confiance
                st.markdown("#### 📊 Niveau de Confiance")
                st.progress(min(confiance / 100, 1.0))

                # Probabilités détaillées
                st.markdown("#### 📈 Probabilités Détaillées par Classe")

                prob_data = {
                    "Classe": ["Minor (0)", "Severe (1)"],
                    "Probabilité": [f"{probabilities[0]*100:.2f}%", f"{probabilities[1]*100:.2f}%"],
                    "Valeur": [probabilities[0], probabilities[1]]
                }

                col1, col2 = st.columns(2, gap="medium")

                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #2ca02c 0%, rgba(44, 160, 44, 0.2) 100%);
                                padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2ca02c;">
                        <h4 style="color: #2ca02c; margin: 0;">Minor (Accidents Légers)</h4>
                        <p style="font-size: 2rem; font-weight: 700; color: #2ca02c; margin: 0.5rem 0;">
                            {probabilities[0]*100:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #d62728 0%, rgba(214, 39, 40, 0.2) 100%);
                                padding: 1.5rem; border-radius: 12px; border-left: 5px solid #d62728;">
                        <h4 style="color: #d62728; margin: 0;">Severe (Accidents Graves)</h4>
                        <p style="font-size: 2rem; font-weight: 700; color: #d62728; margin: 0.5rem 0;">
                            {probabilities[1]*100:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Recommandations
                st.markdown("---")
                st.markdown("### 💡 Recommandations de Sécurité")

                if prediction == 1:
                    st.error(
                        "⚠️ **PRÉDICTION: ACCIDENT GRAVE**\n\n"
                        "Les conditions analysées indiquent un risque élevé d'accident grave. "
                        "Procédez avec extrême prudence."
                    )

                    st.markdown("""
                    **Actions Recommandées:**
                    - 🔴 Réduisez votre vitesse de manière significative
                    - 📏 Augmentez la distance de sécurité avec les autres véhicules
                    - 🚗 Envisagez de reporter votre trajet si possible
                    - 🚨 Activez vos feux de détresse si nécessaire
                    - 📞 N'hésitez pas à contacter les services d'urgence
                    """)
                else:
                    st.success(
                        "✅ **PRÉDICTION: ACCIDENT MINEUR**\n\n"
                        "Les conditions analysées indiquent un risque moins élevé. "
                        "Restez vigilant comme toujours."
                    )

                    st.markdown("""
                    **Actions Recommandées:**
                    - ✅ Continuez à respecter le code de la route
                    - 👀 Maintenez une attention normale à la conduite
                    - 🛣️ Restez vigilant aux changements de conditions
                    - ⚡ Gardez votre concentration sur la route
                    """)

                st.markdown("---")
                st.info(
                    "📌 **Note Importante:** Cette prédiction est basée sur un modèle de classification binaire. "
                    "Elle doit être utilisée comme aide à la décision uniquement, et non comme seul critère de sécurité."
                )
