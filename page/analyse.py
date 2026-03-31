"""
Page Analyse
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data():
    """Charger les données"""
    try:
        data_path = Path("data/df_sample.csv")
        if not data_path.exists():
            st.error(f"❌ Fichier non trouvé: {data_path}")
            return None

        df = pd.read_csv(data_path)
        return df

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None


def show():
    """Afficher la page Analyse"""

    # Header
    st.markdown("""
    <div class="header-main">
        <h1>📊 Analyse des Données</h1>
        <p>Visualisez les patterns et tendances des accidents de la route</p>
    </div>
    """, unsafe_allow_html=True)

    # Charger les données
    data = load_data()
    if data is None:
        st.stop()

    # Informations sur les données - Cartes statistiques
    st.markdown("### 📈 Informations sur le Dataset")

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total d'accidents</div>
            <div class="stat-value">{len(data):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Nombre de variables</div>
            <div class="stat-value">{len(data.columns)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        severity_3_count = len(data[data['Accident_Severity'] == 3])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Très Grave (3)</div>
            <div class="stat-value">{severity_3_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Dataset Original</div>
            <div class="stat-value">3 Classes</div>
        </div>
        """, unsafe_allow_html=True)

    # Note sur la classification
    st.markdown("---")
    st.info("""
    📌 **À PROPOS DES DONNÉES AFFICHÉES**

    Les données visualisées ci-dessous proviennent du **dataset original (3 classes)**:
    - Classe 1: Faible (646 accidents)
    - Classe 2: Grave (6,797 accidents)
    - Classe 3: Très Grave (42,555 accidents)

    **Le modèle de prédiction** utilise une **version équilibrée et convertie (2 classes)**:
    - Classe 0 (Minor): Faible + Grave (7,443 accidents)
    - Classe 1 (Severe): Très Grave (7,443 accidents)
    """)

    # Onglets pour différentes analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distribution", "🔗 Corrélations", "📉 Tendances", "📋 Données"]
    )

    # Tab 1: Distribution
    with tab1:
        st.markdown("### Distribution des Variables")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Heure de l'accident")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(data['heure_num'].dropna(), bins=24, color='#FF6B6B', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Heure (0-23)', fontsize=12)
            ax.set_ylabel('Nombre de cas', fontsize=12)
            ax.set_title('Distribution des accidents par heure', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.markdown("#### Limitation de vitesse")
            fig, ax = plt.subplots(figsize=(10, 5))
            speed_counts = data['Speed_limit'].value_counts().sort_index()
            ax.bar(speed_counts.index.astype(str), speed_counts.values, color='#4ECDC4', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Limite de vitesse (km/h)', fontsize=12)
            ax.set_ylabel('Nombre de cas', fontsize=12)
            ax.set_title('Distribution de la limite de vitesse', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
            st.pyplot(fig)

        # Distribution de la sévérité
        st.markdown("#### Sévérité des Accidents (Données Originales - 3 Classes)")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            severity_counts = data['Accident_Severity'].value_counts().sort_index()
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            labels_3 = {1: 'Faible', 2: 'Grave', 3: 'Très Grave'}
            bars = ax.bar([labels_3.get(i, str(i)) for i in severity_counts.index],
                          severity_counts.values,
                          color=colors[:len(severity_counts)],
                          edgecolor='black', alpha=0.7)
            ax.set_xlabel('Sévérité', fontsize=12)
            ax.set_ylabel('Nombre de cas', fontsize=12)
            ax.set_title('Distribution de la Sévérité (3 Classes)', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')

            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            severity_counts = data['Accident_Severity'].value_counts().sort_index()
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            labels_3 = {1: 'Faible', 2: 'Grave', 3: 'Très Grave'}
            labels = [labels_3.get(i, str(i)) for i in severity_counts.index]
            ax.pie(severity_counts.values, labels=labels, colors=colors[:len(severity_counts)],
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Proportion de Sévérité (3 Classes)', fontsize=14, fontweight='bold')
            st.pyplot(fig)

        # Distribution binaire convertie
        st.markdown("#### Distribution Convertie pour le Modèle (2 Classes)")
        st.markdown("Le modèle utilise une version équilibrée avec 2 classes:")
        st.markdown("- **Classe 0 (Minor):** Faible + Grave (combinées)")
        st.markdown("- **Classe 1 (Severe):** Très Grave")

        col1, col2 = st.columns(2)

        with col1:
            # Créer la distribution binaire
            minor_count = (data['Accident_Severity'].isin([1, 2])).sum()
            severe_count = (data['Accident_Severity'] == 3).sum()

            fig, ax = plt.subplots(figsize=(8, 6))
            binary_counts = [minor_count, severe_count]
            binary_labels = ['Minor (0)', 'Severe (1)']
            colors_binary = ['#2ecc71', '#e74c3c']
            bars = ax.bar(binary_labels, binary_counts, color=colors_binary, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Classe', fontsize=12)
            ax.set_ylabel('Nombre de cas', fontsize=12)
            ax.set_title('Distribution Binaire Convertie', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            binary_counts = [minor_count, severe_count]
            binary_labels = ['Minor', 'Severe']
            colors_binary = ['#2ecc71', '#e74c3c']
            ax.pie(binary_counts, labels=binary_labels, colors=colors_binary,
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Proportion Binaire Convertie', fontsize=14, fontweight='bold')
            st.pyplot(fig)

    # Tab 2: Corrélations
    with tab2:
        st.markdown("### Analyse des Corrélations")

        # Sélectionner les colonnes numériques
        numeric_cols = [
            'heure_num', 'Number_of_Vehicles', 'Number_of_Casualties',
            'Speed_limit', 'Year', '1st_Road_Number', '2nd_Road_Number'
        ]

        available_cols = [col for col in numeric_cols if col in data.columns]

        if len(available_cols) > 1:
            numeric_data = data[available_cols].copy()
            corr_matrix = numeric_data.corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        fmt='.2f', square=True, ax=ax, cbar_kws={'label': 'Corrélation'})
            ax.set_title('Matrice de Corrélation - Variables Numériques', fontsize=14, fontweight='bold')
            st.pyplot(fig)

            st.markdown("""
            **Interprétation:**
            - Valeurs proches de **+1**: Corrélation positive (augmentation simultanée)
            - Valeurs proches de **-1**: Corrélation négative (variation inverse)
            - Valeurs proches de **0**: Pas de corrélation
            """)

    # Tab 3: Tendances
    with tab3:
        st.markdown("### Tendances par Année")

        # Nombre d'accidents par année
        fig, ax = plt.subplots(figsize=(12, 6))

        accidents_by_year = data['Year'].value_counts().sort_index()
        colors_year = plt.cm.viridis(np.linspace(0, 1, len(accidents_by_year)))

        bars = ax.bar(accidents_by_year.index.astype(str), accidents_by_year.values, color=colors_year, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Année', fontsize=12)
        ax.set_ylabel('Nombre d\'accidents', fontsize=12)
        ax.set_title('Nombre d\'accidents par année', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}',
                   ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig)

        # Sévérité par jour de la semaine
        st.markdown("### Sévérité par Jour de la Semaine (Original - 3 Classes)")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Ordre des jours
        jour_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        day_mapping = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}

        data_copy = data.copy()
        data_copy['Day_Name'] = data_copy['Day_of_Week'].map(day_mapping)

        severite_by_day = pd.crosstab(data_copy['Day_Name'], data_copy['Accident_Severity'])
        severite_by_day = severite_by_day.reindex([d for d in jour_order if d in severite_by_day.index])

        colors_3class = ['#2ecc71', '#f39c12', '#e74c3c']
        severite_by_day.plot(kind='bar', ax=ax, color=colors_3class[:len(severite_by_day.columns)])
        ax.set_xlabel('Jour de la Semaine', fontsize=12)
        ax.set_ylabel('Nombre de cas', fontsize=12)
        ax.set_title('Distribution de la Sévérité par Jour de la Semaine (3 Classes)', fontsize=14, fontweight='bold')
        ax.legend(title='Sévérité', labels=['Faible', 'Grave', 'Très Grave'][:len(severite_by_day.columns)])
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45)

        st.pyplot(fig)

    # Tab 4: Données brutes
    with tab4:
        st.markdown("### Tableau de Données (Original - 3 Classes)")

        # Options de filtrage
        col1, col2 = st.columns(2)

        with col1:
            severity_options = sorted(data['Accident_Severity'].unique())
            severity_labels = {1: '1 - Faible', 2: '2 - Grave', 3: '3 - Très Grave'}
            selected_severity = st.multiselect(
                "Filtrer par Sévérité",
                severity_options,
                default=severity_options,
                format_func=lambda x: severity_labels.get(x, str(x))
            )

        with col2:
            selected_year = st.multiselect(
                "Filtrer par Année",
                sorted(data['Year'].unique()),
                default=sorted(data['Year'].unique())
            )

        # Appliquer les filtres
        filtered_data = data[
            (data['Accident_Severity'].isin(selected_severity)) &
            (data['Year'].isin(selected_year))
        ]

        st.markdown(f"**Total:** {len(filtered_data):,} enregistrements")

        # Afficher le tableau
        st.dataframe(
            filtered_data.reset_index(drop=True),
            use_container_width=True,
            height=400
        )

        # Statistiques résumées
        st.markdown("### Statistiques Résumées")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Heure Moyenne",
                f"{filtered_data['heure_num'].mean():.1f}h"
            )

        with col2:
            st.metric(
                "Véhicules Moyens",
                f"{filtered_data['Number_of_Vehicles'].mean():.1f}"
            )

        with col3:
            st.metric(
                "Victimes Moyennes",
                f"{filtered_data['Number_of_Casualties'].mean():.1f}"
            )

        with col4:
            st.metric(
                "Total Enregistrements",
                f"{len(filtered_data):,}"
            )

        # Télécharger les données
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name="donnees_accidents.csv",
            mime="text/csv"
        )
