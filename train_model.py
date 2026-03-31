"""
🚗 Script d'entraînement du modèle - Classification Binaire
Charge les données réelles et entraîne le modèle LogisticRegression
pour prédire la sévérité des accidents (Minor vs Severe)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(use_equilibred=True):
    """Charger les données"""

    if use_equilibred:
        data_path = Path("data/df_equilibre_binaire.csv")
        if data_path.exists():
            print("✅ Utilisant le dataset ÉQUILIBRÉ (df_equilibre_binaire.csv)")
        else:
            print("⚠️  Dataset équilibré non trouvé. Lancer: python nouveau_traitement.py")
            print("   Utilisant le dataset original à la place...")
            data_path = Path("data/df_sample.csv")
    else:
        data_path = Path("data/df_sample.csv")
        print("✅ Utilisant le dataset ORIGINAL (df_sample.csv)")

    if not data_path.exists():
        print(f"❌ Fichier non trouvé: {data_path}")
        return None

    df = pd.read_csv(data_path)
    print(f"✅ Données chargées: {df.shape}")
    return df


def prepare_data(df):
    """Préparer les données"""
    print("\n🔧 Préparation des données...")

    # Vérifier les colonnes
    expected_cols = [
        '1st_Road_Number', 'Police_Force', 'Year', 'heure_num',
        'Day_of_Week', '2nd_Road_Number', '1st_Road_Class',
        'Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit',
        'Accident_Severity'
    ]

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
        return None, None

    # Supprimer les NaN
    df_clean = df.dropna()
    print(f"✅ Après suppression NaN: {df_clean.shape}")

    # Séparer X et y
    X = df_clean.drop('Accident_Severity', axis=1)
    y = df_clean['Accident_Severity']

    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"\nDistribution de la cible:")
    severity_labels = {0: "Minor (Accidents légers)", 1: "Severe (Accidents graves)"}
    for cls in sorted(y.unique()):
        count = (y == cls).sum()
        pct = (count / len(y)) * 100
        print(f"   Classe {cls} ({severity_labels.get(cls, 'Unknown')}): {count:>7} ({pct:>6.2f}%)")

    return X, y


def train_model(X, y):
    """Entraîner le modèle"""
    print("\n🤖 Entraînement du modèle...")

    # Diviser train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Train size: {X_train.shape}")
    print(f"✅ Test size: {X_test.shape}")

    # Normaliser les données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Données normalisées (StandardScaler)")

    # Entraîner LogisticRegression (binary classification)
    # Enlever n_jobs pour éviter le warning
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    print("✅ Modèle entraîné (Logistic Regression - Classification Binaire)")

    # Prédictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Métriques de base
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Affichage détaillé
    print("\n" + "="*70)
    print("📊 MÉTRIQUES DE PERFORMANCE")
    print("="*70)

    print(f"\n✅ ACCURACY:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ PRECISION:       {precision:.4f}")
    print(f"✅ RECALL:          {recall:.4f}")
    print(f"✅ F1-SCORE:        {f1:.4f}")
    print(f"✅ ROC-AUC:         {roc_auc:.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Minor (0)', 'Severe (1)']
    ))

    print(f"🔲 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\n   TN (True Negatives):  {cm[0, 0]:>6}")
    print(f"   FP (False Positives): {cm[0, 1]:>6}")
    print(f"   FN (False Negatives): {cm[1, 0]:>6}")
    print(f"   TP (True Positives):  {cm[1, 1]:>6}")

    return model, scaler, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def save_model(model, scaler):
    """Sauvegarder le modèle et le scaler"""
    print("\n💾 Sauvegarde du modèle...")

    # Créer dossier models s'il n'existe pas
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Sauvegarder le modèle
    model_path = models_dir / "logistic_regression_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")

    # Sauvegarder le scaler
    scaler_path = models_dir / "standard_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé: {scaler_path}")

    return model_path, scaler_path


def save_metrics(metrics, output_path="models/metrics.txt"):
    """Sauvegarder les métriques dans un fichier avec UTF-8 encoding"""
    # Utiliser utf-8 encoding pour supporter les emojis
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("📊 MÉTRIQUES DE PERFORMANCE - LOGISTIC REGRESSION (BINAIRE)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:   {metrics['precision']:.4f}\n")
        f.write(f"Recall:      {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:    {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC:     {metrics['roc_auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"{metrics['confusion_matrix']}\n")

    print(f"✅ Métriques sauvegardées: {output_path}")


def main(use_equilibred=True):
    """Fonction principale"""
    print("╔════════════════════════════════════════════════════════╗")
    print("║  🚗 Entraînement du Modèle de Sévérité d'Accident      ║")
    print("║     Classification Binaire (Minor vs Severe)           ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    # Charger les données
    df = load_data(use_equilibred=use_equilibred)
    if df is None:
        return

    # Préparer les données
    X, y = prepare_data(df)
    if X is None:
        return

    # Entraîner le modèle
    model, scaler, metrics = train_model(X, y)

    # Sauvegarder
    model_path, scaler_path = save_model(model, scaler)
    save_metrics(metrics)

    # Résumé final
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║               ✅ ENTRAÎNEMENT RÉUSSI                  ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n📈 Accuracy finale: {metrics['accuracy']*100:.2f}%")
    print(f"📊 F1-Score: {metrics['f1']:.4f}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\n📁 Fichiers sauvegardés:")
    print(f"   • Modèle: {model_path}")
    print(f"   • Scaler: {scaler_path}")
    print(f"   • Métriques: models/metrics.txt")
    print(f"\n🚀 Vous pouvez maintenant lancer: streamlit run app.py")


if __name__ == "__main__":
    import sys

    # Vérifier les arguments
    use_equilibred = True

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "--original":
            use_equilibred = False
            print("📌 Mode: Dataset ORIGINAL\n")
        elif sys.argv[1].lower() == "--equilibred":
            use_equilibred = True
            print("📌 Mode: Dataset ÉQUILIBRÉ\n")
        else:
            print(f"❓ Argument inconnu: {sys.argv[1]}")
            print("   Usage: python train_model.py [--original | --equilibred]")
            print("   Par défaut: --equilibred (dataset équilibré)\n")

    main(use_equilibred=use_equilibred)