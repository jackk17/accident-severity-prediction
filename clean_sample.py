# clean_sample.py
import pandas as pd
import numpy as np

# Charger l'échantillon
df = pd.read_csv("data/df_with_features_sample.csv")

print(f"Shape original: {df.shape}")

# Garder uniquement les colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# S'assurer que la colonne cible est incluse
if 'Accident_Severity_Binary' not in numeric_cols:
    if 'Accident_Severity_Binary' in df.columns:
        numeric_cols.append('Accident_Severity_Binary')

df_clean = df[numeric_cols]

print(f"Shape après nettoyage: {df_clean.shape}")
print(f"Colonnes gardées ({len(df_clean.columns)}):")
for col in df_clean.columns:
    print(f"  - {col}")

# Sauvegarder
df_clean.to_csv("data/df_with_features_sample_clean.csv", index=False)
print("\n✅ Fichier nettoyé sauvegardé: data/df_with_features_sample_clean.csv")