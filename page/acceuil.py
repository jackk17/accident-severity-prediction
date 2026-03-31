"""Page Accueil - Version compatible avec Random Forest"""

import json
import streamlit as st
from pathlib import Path
import pandas as pd


def load_metrics():
    """Charger les métriques du modèle Random Forest"""
    p = Path("models/metrics.json")
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def show():
    """Page d'accueil"""
    st.markdown("""
    <div class="hero">
        <h1>🚗 Predicteur de Severite d'Accidents</h1>
        <p>Classification binaire (Minor / Severe) par Random Forest</p>
    </div>""", unsafe_allow_html=True)

    metrics = load_metrics()

    # --- Metriques du modele Random Forest ---
    if metrics:
        st.markdown("### 🎯 Modele retenu : **Random Forest**")

        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""<div class="card">
                <div class="label">Accuracy</div>
                <div class="value">{metrics['accuracy']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""<div class="card">
                <div class="label">F1-Score</div>
                <div class="value">{metrics['f1_score']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        
        with c3:
            st.markdown(f"""<div class="card">
                <div class="label">ROC-AUC</div>
                <div class="value">{metrics['roc_auc']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        
        with c4:
            st.markdown(f"""<div class="card">
                <div class="label">Recall (Grave)</div>
                <div class="value">{metrics['recall_grave']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # --- Meilleures features ---
        st.markdown("### 🔝 Top 5 des variables les plus importantes")
        
        if 'best_features' in metrics:
            best_features = metrics['best_features']
            
            # Créer un dataframe pour l'affichage
            features_df = pd.DataFrame(best_features).head(5)
            features_df.columns = ['Variable', 'Importance']
            features_df['Importance'] = features_df['Importance'] * 100
            
            st.dataframe(
                features_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Variable": "Variable",
                    "Importance": st.column_config.ProgressColumn(
                        "Importance (%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    )
                }
            )

    else:
        st.warning("Metriques non trouvees. Veuillez vérifier le dossier `models/`.")

    st.markdown("---")

    # --- Pipeline amélioré ---
    st.markdown("### 📊 Pipeline de traitement amélioré")
    st.markdown("""
    1. **Chargement** des données UK Accidents (1.5M accidents)
    2. **Conversion binaire** : Classes 1+2 → Severe (1), Classe 3 → Minor (0)
    3. **Feature engineering avancé** : Création de 17 nouvelles features combinées
    4. **Échantillonnage stratégique** : Équilibrage des classes (ratio 1:2)
    5. **Entraînement** : Random Forest avec gestion du déséquilibre
    6. **Sélection** des 21 features les plus pertinentes
    """)

    st.markdown("---")

    # --- Comparaison des performances ---
    st.markdown("### 📈 Améliorations obtenues")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Avant (Logistic Regression):**
        - Accuracy: 58.9%
        - Recall (Grave): 56%
        - ROC-AUC: 0.61
        """)
    
    with col2:
        if metrics:
            st.markdown(f"""
            **Après (Random Forest):**
            - Accuracy: {metrics['accuracy']*100:.1f}% (+{metrics['accuracy']*100 - 58.9:.1f}%)
            - Recall (Grave): {metrics['recall_grave']*100:.1f}% (+{metrics['recall_grave']*100 - 56:.1f}%)
            - ROC-AUC: {metrics['roc_auc']:.3f} (+{metrics['roc_auc'] - 0.61:.2f})
            """)
        else:
            st.markdown("""
            **Après (Random Forest):**
            - Accuracy: 62.0% (+3.1%)
            - Recall (Grave): 61% (+5%)
            - ROC-AUC: 0.65 (+0.04)
            """)
    
    st.markdown("---")

    # --- Variables clés ---
    st.markdown("### 🔍 Variables clés")
    
    st.markdown("""
    | Variable | Description |
    |----------|-------------|
    | `casualties_per_vehicle` | Nombre de victimes par véhicule (facteur le plus important) |
    | `Number_of_Vehicles` | Nombre de véhicules impliqués |
    | `composite_risk_score` | Score de risque combinant 8 facteurs |
    | `Speed_limit` | Limite de vitesse sur la route |
    | `is_night` | Accident de nuit (22h-5h) |
    """)

    st.markdown("---")
    
    if metrics:
        st.info(
            f"⚠️ **Note importante** : Ce modèle est à but éducatif. Les prédictions ne remplacent pas "
            f"le jugement humain en matière de sécurité routière. Le modèle détecte correctement "
            f"**{metrics['recall_grave']*100:.0f}%** des accidents graves."
        )
    else:
        st.info(
            "⚠️ **Note importante** : Ce modèle est à but éducatif. Les prédictions ne remplacent pas "
            "le jugement humain en matière de sécurité routière."
        )