"""Page Analyse - Version avec données complètes"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@st.cache_data
def load_data():
    """Charger les données avec feature engineering"""
    # Essayer d'abord le fichier complet avec features
    p_full = Path("data/df_with_features.csv")
    if p_full.exists():
        df = pd.read_csv(p_full)
        st.info(f"📊 Données complètes chargées: {len(df):,} accidents")
        return df
    
    # Sinon essayer le fichier original
    p_sample = Path("data/df_sample.csv")
    if p_sample.exists():
        df = pd.read_csv(p_sample)
        st.warning(f"⚠️ Données limitées: {len(df):,} accidents (version complète non trouvée)")
        return df
    
    return None


def show():
    st.markdown("""
    <div class="hero">
        <h1>📊 Analyse des Donnees</h1>
        <p>Visualisations du dataset UK Accidents avec features avancées</p>
    </div>""", unsafe_allow_html=True)

    data = load_data()
    if data is None:
        st.error("Fichier data/df_sample.csv ou data/df_with_features.csv non trouve.")
        return

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total accidents", f"{len(data):,}")
    c2.metric("Variables", len(data.columns))
    
    # Vérifier la présence de la colonne Year
    if 'Year' in data.columns:
        c3.metric("Annees", f"{data['Year'].min()}-{data['Year'].max()}")
    
    # Vérifier la présence de la colonne Speed_limit
    if 'Speed_limit' in data.columns:
        c4.metric("Vitesse moy.", f"{data['Speed_limit'].mean():.0f} km/h")
    else:
        c4.metric("Features avancées", f"{len([c for c in data.columns if c not in ['Accident_Severity', 'Accident_Severity_Binary']])}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribution", "Correlations", "Tendances", "Donnees"])

    # --- Tab 1 : Distribution ---
    with tab1:
        col1, col2 = st.columns(2)

        # Distribution par heure
        with col1:
            if 'heure_num' in data.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(data['heure_num'].dropna(), bins=24, color='#2563eb',
                        edgecolor='white', alpha=.85)
                ax.set_xlabel('Heure')
                ax.set_ylabel('Nombre')
                ax.set_title('Accidents par heure')
                ax.grid(alpha=.2, axis='y')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Colonne 'heure_num' non disponible")

        # Distribution par limite de vitesse
        with col2:
            if 'Speed_limit' in data.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                sc = data['Speed_limit'].value_counts().sort_index()
                ax.bar(sc.index.astype(str), sc.values, color='#16a34a',
                       edgecolor='white', alpha=.85)
                ax.set_xlabel('Limite vitesse')
                ax.set_ylabel('Nombre')
                ax.set_title('Distribution des limites de vitesse')
                ax.grid(alpha=.2, axis='y')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Colonne 'Speed_limit' non disponible")

        # Severite binaire (nouveau modèle)
        st.markdown("#### Severite (Classification binaire)")
        
        # Vérifier quelle colonne de sévérité est disponible
        if 'Accident_Severity_Binary' in data.columns:
            severity_col = 'Accident_Severity_Binary'
            severity_counts = data[severity_col].value_counts()
            labels = {0: 'Mineur', 1: 'Grave'}
            colors = ['#16a34a', '#dc2626']
        elif 'Accident_Severity' in data.columns:
            severity_col = 'Accident_Severity'
            severity_counts = data[severity_col].value_counts().sort_index()
            labels = {1: 'Faible', 2: 'Grave', 3: 'Très Grave'}
            colors = ['#16a34a', '#f59e0b', '#dc2626']
        else:
            st.error("Aucune colonne de sévérité trouvée")
            severity_counts = None

        if severity_counts is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar([labels.get(i, str(i)) for i in severity_counts.index],
                              severity_counts.values, color=colors[:len(severity_counts)], 
                              edgecolor='white')
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                            f'{int(b.get_height()):,}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
                ax.set_title('Distribution')
                ax.grid(alpha=.2, axis='y')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(severity_counts.values,
                       labels=[labels.get(i, str(i)) for i in severity_counts.index],
                       colors=colors[:len(severity_counts)], autopct='%1.1f%%', startangle=90)
                ax.set_title('Proportions')
                st.pyplot(fig)
                plt.close()

        # Nouvelles features avancées
        st.markdown("#### Features avancées créées")
        
        advanced_features = ['casualties_per_vehicle', 'is_night', 'bad_weather', 
                            'bad_road', 'composite_risk_score']
        existing_features = [f for f in advanced_features if f in data.columns]
        
        if existing_features:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'casualties_per_vehicle' in data.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(data['casualties_per_vehicle'].clip(upper=5), 
                           bins=50, color='#2563eb', alpha=.85)
                    ax.set_xlabel('Victimes par véhicule')
                    ax.set_ylabel('Nombre')
                    ax.set_title('Distribution - Victimes par véhicule')
                    ax.grid(alpha=.2, axis='y')
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                if 'composite_risk_score' in data.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(data['composite_risk_score'], bins=50, 
                           color='#f97316', alpha=.85)
                    ax.set_xlabel('Score de risque composite')
                    ax.set_ylabel('Nombre')
                    ax.set_title('Distribution du score de risque')
                    ax.grid(alpha=.2, axis='y')
                    st.pyplot(fig)
                    plt.close()

    # --- Tab 2 : Correlations ---
    with tab2:
        # Sélectionner les colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclure les colonnes d'index si présentes
        numeric_cols = [c for c in numeric_cols if c not in ['Unnamed: 0']]
        
        # Limiter à 20 colonnes pour la lisibilité
        if len(numeric_cols) > 20:
            numeric_cols = numeric_cols[:20]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                    fmt='.2f', square=True, ax=ax, annot_kws={'size': 8})
        ax.set_title('Matrice de correlation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Tab 3 : Tendances ---
    with tab3:
        # Par annee
        if 'Year' in data.columns:
            by_year = data['Year'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(by_year.index.astype(str), by_year.values,
                   color='#2563eb', edgecolor='white', alpha=.85)
            ax.set_title('Accidents par annee')
            ax.set_xlabel('Annee')
            ax.grid(alpha=.2, axis='y')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()

        # Par jour de semaine
        if 'Day_of_Week' in data.columns:
            days = {1: 'Lun', 2: 'Mar', 3: 'Mer', 4: 'Jeu',
                    5: 'Ven', 6: 'Sam', 7: 'Dim'}
            tmp = data.copy()
            tmp['Jour'] = tmp['Day_of_Week'].map(days)
            
            if 'Accident_Severity_Binary' in tmp.columns:
                severity_agg = tmp.groupby('Jour')['Accident_Severity_Binary'].mean()
                fig, ax = plt.subplots(figsize=(10, 4))
                severity_agg.reindex([d for d in days.values() if d in severity_agg.index]).plot(kind='bar', ax=ax, color='#dc2626')
                ax.set_title('Proportion d\'accidents graves par jour')
                ax.set_xlabel('Jour')
                ax.set_ylabel('Proportion de graves (%)')
                ax.set_ylim(0, 1)
                ax.grid(alpha=.2, axis='y')
                plt.xticks(rotation=0)
                st.pyplot(fig)
                plt.close()

    # --- Tab 4 : Donnees ---
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtrer par sévérité
            if 'Accident_Severity_Binary' in data.columns:
                severity_options = {'Mineur': 0, 'Grave': 1}
                sev_filter = st.multiselect(
                    "Severite", 
                    options=list(severity_options.keys()),
                    default=list(severity_options.keys())
                )
                sev_values = [severity_options[s] for s in sev_filter]
            elif 'Accident_Severity' in data.columns:
                sev_filter = st.multiselect(
                    "Severite", 
                    options=sorted(data['Accident_Severity'].unique()),
                    default=sorted(data['Accident_Severity'].unique()),
                    format_func=lambda x: {1: 'Faible', 2: 'Grave', 3: 'Tres Grave'}.get(x, str(x))
                )
                sev_values = sev_filter
        
        with col2:
            if 'Year' in data.columns:
                year_filter = st.multiselect(
                    "Annee", 
                    sorted(data['Year'].unique()),
                    default=sorted(data['Year'].unique())
                )
        
        # Appliquer les filtres
        filtered = data.copy()
        if 'sev_values' in locals() and sev_values:
            if 'Accident_Severity_Binary' in filtered.columns:
                filtered = filtered[filtered['Accident_Severity_Binary'].isin(sev_values)]
            elif 'Accident_Severity' in filtered.columns:
                filtered = filtered[filtered['Accident_Severity'].isin(sev_values)]
        
        if 'year_filter' in locals() and year_filter:
            filtered = filtered[filtered['Year'].isin(year_filter)]
        
        st.caption(f"{len(filtered):,} enregistrements")
        
        # Sélectionner les colonnes à afficher
        display_cols = [c for c in filtered.columns if c not in ['Unnamed: 0']][:15]
        st.dataframe(filtered[display_cols].reset_index(drop=True),
                     use_container_width=True, height=400)
        
        st.download_button("Telecharger CSV", filtered.to_csv(index=False),
                           "accidents_analyses.csv", "text/csv")