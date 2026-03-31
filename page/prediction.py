"""Page de prédiction - Random Forest"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

def load_model():
    """Charger le modèle Random Forest entraîné"""
    model_path = Path("models/random_forest_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    features_path = Path("models/features.json")
    
    if model_path.exists() and scaler_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        return model, scaler, features
    return None, None, None

def prediction():
    """Page de prédiction"""
    st.title("🔮 Prédiction de la sévérité d'un accident")
    st.markdown("Utilisez le formulaire ci-dessous pour prédire si un accident sera **mineur** ou **grave**.")
    
    # Charger le modèle
    model, scaler, features = load_model()
    
    if model is None:
        st.error("""
        ❌ **Modèle non trouvé!**
        
        Veuillez placer les fichiers suivants dans le dossier `models/`:
        - random_forest_model.pkl
        - scaler.pkl
        - features.json
        
        Puis redémarrez l'application.
        """)
        return
    
    st.success("✅ Modèle Random Forest chargé avec succès!")
    
    # Afficher les métriques du modèle
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        with col2:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        with col3:
            st.metric("Recall (Grave)", f"{metrics['recall_grave']*100:.1f}%")
    
    # Formulaire de saisie
    st.markdown("### 📝 Caractéristiques de l'accident")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚗 Informations sur les véhicules")
        num_vehicles = st.number_input(
            "Nombre de véhicules impliqués",
            min_value=1, max_value=20, value=2,
            help="Nombre total de véhicules impliqués"
        )
        
        num_casualties = st.number_input(
            "Nombre de victimes",
            min_value=0, max_value=20, value=1,
            help="Nombre de personnes blessées ou tuées"
        )
        
        speed_limit = st.number_input(
            "Limite de vitesse (km/h)",
            min_value=20, max_value=130, value=50, step=10,
            help="Limite de vitesse sur la route"
        )
        
        urban_rural = st.selectbox(
            "Zone",
            ["Urbaine", "Rurale"],
            help="Type de zone où l'accident s'est produit"
        )
        urban_rural_value = 1 if urban_rural == "Urbaine" else 2
    
    with col2:
        st.markdown("#### ⏰ Conditions temporelles")
        hour = st.slider("Heure de l'accident", 0, 23, 12)
        
        day_of_week = st.selectbox(
            "Jour de la semaine",
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        )
        day_map = {"Lundi": 1, "Mardi": 2, "Mercredi": 3, 
                   "Jeudi": 4, "Vendredi": 5, "Samedi": 6, "Dimanche": 7}
        day_num = day_map[day_of_week]
        
        month = st.selectbox("Mois", list(range(1, 13)), index=0)
        year = st.number_input("Année", min_value=2010, max_value=2025, value=2023)
    
    with col1:
        st.markdown("#### 🌧️ Conditions extérieures")
        weather = st.selectbox(
            "Conditions météo",
            ["Fine", "Raining", "Snowing", "Fog", "High winds"]
        )
        bad_weather = 1 if weather in ["Raining", "Snowing", "Fog", "High winds"] else 0
        
        road_condition = st.selectbox(
            "État de la route",
            ["Dry", "Wet", "Snow", "Ice", "Flood"]
        )
        bad_road = 1 if road_condition in ["Wet", "Snow", "Ice", "Flood"] else 0
        
        light_condition = st.selectbox(
            "Conditions d'éclairage",
            ["Daylight", "Night", "Dark", "Street lights present"]
        )
        poor_light = 1 if light_condition in ["Night", "Dark"] else 0
    
    # Calculer les features avancées
    casualties_per_vehicle = num_casualties / (num_vehicles + 1)
    risk_vehicles_casualties = casualties_per_vehicle
    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    is_weekend = 1 if day_num in [6, 7] else 0
    night_weekend = 1 if (is_night == 1 and is_weekend == 1) else 0
    speed_risk_score = speed_limit / 130
    high_speed = 1 if speed_limit >= 70 else 0
    high_speed_night = 1 if (high_speed == 1 and is_night == 1) else 0
    night_highspeed_risk = high_speed_night
    many_vehicles_night = 1 if (num_vehicles > 2 and is_night == 1) else 0
    rush_hour_urban = 1 if ((hour in [7,8,9,17,18,19]) and urban_rural == "Urbaine") else 0
    
    # Score de risque composite
    composite_risk_score = (
        num_casualties * 0.25 +
        num_vehicles * 0.15 +
        speed_risk_score * 0.15 +
        is_night * 0.10 +
        is_weekend * 0.05 +
        bad_weather * 0.10 +
        bad_road * 0.10 +
        poor_light * 0.10
    )
    
    # Bouton de prédiction
    st.markdown("---")
    
    if st.button("🚗 Prédire la sévérité", type="primary", use_container_width=True):
        # Préparer les données dans l'ordre des features
        input_dict = {
            'risk_vehicles_casualties': risk_vehicles_casualties,
            'casualties_per_vehicle': casualties_per_vehicle,
            'Number_of_Vehicles': num_vehicles,
            'composite_risk_score': composite_risk_score,
            'Speed_limit': speed_limit,
            'speed_risk_score': speed_risk_score,
            'Urban_or_Rural_Area': urban_rural_value,
            'Number_of_Casualties': num_casualties,
            'heure_num': hour,
            'is_night': is_night,
            'Year': year,
            'Month': month,
            'bad_road': bad_road,
            'Day_of_Week': day_num,
            'bad_weather': bad_weather
        }
        
        # Ajouter les features manquantes avec valeurs par défaut
        for feature in features:
            if feature not in input_dict:
                input_dict[feature] = 0
        
        # Créer le DataFrame dans l'ordre des features
        input_df = pd.DataFrame([{f: input_dict[f] for f in features}])
        
        # Normaliser
        input_scaled = scaler.transform(input_df)
        
        # Prédiction - Convertir en entier Python standard
        pred_value = int(model.predict(input_scaled)[0])  # Convertir en int
        proba = model.predict_proba(input_scaled)[0]
        
        # Afficher le résultat
        st.markdown("---")
        st.markdown("## 📊 Résultat de la prédiction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pred_value == 1:
                st.metric("⚠️ Sévérité", "GRAVE", delta="Risque élevé", delta_color="inverse")
            else:
                st.metric("✅ Sévérité", "MINEURE", delta="Risque faible")
        
        with col2:
            # Utiliser l'entier converti pour l'indexation
            proba_value = proba[pred_value] * 100
            st.metric("Probabilité", f"{proba_value:.1f}%")
        
        with col3:
            risk_level = "Élevé" if proba[1] > 0.6 else "Moyen" if proba[1] > 0.4 else "Faible"
            st.metric("Niveau de risque", risk_level)
        
        if pred_value == 1:
            st.error("""
            ### ⚠️ **Accident GRAVE détecté!**
            
            Le modèle prédit que cet accident a un risque élevé d'être **grave**.
            
            **Recommandations:**
            - 🚑 Déclencher une alerte d'urgence immédiate
            - 🚒 Mobiliser des équipes de secours renforcées
            - 🏥 Préparer les services hospitaliers
            """)
        else:
            st.success("""
            ### ✅ **Accident MINEUR détecté!**
            
            Le modèle prédit que cet accident a un risque faible d'être grave.
            
            **Recommandations:**
            - 🚓 Intervention standard sur place
            - 📝 Procédure normale de constatation
            - 🚑 Secours de routine si nécessaire
            """)
        
        # Afficher les facteurs de risque
        with st.expander("📈 Facteurs de risque analysés"):
            st.markdown(f"""
            **Principaux facteurs contributifs:**
            - Victimes par véhicule: {casualties_per_vehicle:.2f}
            - Score de risque composite: {composite_risk_score:.2f}
            - Conditions: {"Dangereuses" if bad_weather or bad_road or poor_light else "Normales"}
            - Heure: {"Nuit" if is_night else "Jour"}
            """)

def show():
    """Alias pour la compatibilité"""
    prediction()