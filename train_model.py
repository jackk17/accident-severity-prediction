"""
🚗 Script d'entraînement du modèle - Classification Binaire
Charge les données réelles et entraîne le modèle Random Forest
pour prédire la sévérité des accidents (Minor vs Severe)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # ✅ AJOUTÉ
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


def load_data(use_equilibred=True):
    """Charger les données - priorité à l'échantillon nettoyé"""
    
    # 1. Essayer l'échantillon nettoyé (recommandé)
    clean_sample_path = Path("data/df_with_features_sample_clean.csv")
    if clean_sample_path.exists():
        print("✅ Utilisation de l'ÉCHANTILLON NETTOYÉ (df_with_features_sample_clean.csv)")
        return pd.read_csv(clean_sample_path)
    
    # 2. Essayer l'échantillon original
    sample_path = Path("data/df_with_features_sample.csv")
    if sample_path.exists():
        print("⚠️ Utilisation de l'échantillon original - tentative de nettoyage...")
        df = pd.read_csv(sample_path)
        # Garder uniquement les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Accident_Severity_Binary' in df.columns:
            numeric_cols.append('Accident_Severity_Binary')
        df = df[numeric_cols]
        print(f"✅ Nettoyé: {df.shape}")
        return df
    
    # 3. Essayer le dataset équilibré
    if use_equilibred:
        data_path = Path("data/df_equilibre_binaire.csv")
        if data_path.exists():
            print("✅ Utilisant le dataset ÉQUILIBRÉ")
            return pd.read_csv(data_path)
    
    # 4. Fallback
    data_path = Path("data/df_sample.csv")
    if data_path.exists():
        print("⚠️ Utilisant le dataset original")
        return pd.read_csv(data_path)
    
    print(f"❌ Aucun fichier de données trouvé!")
    return None


def prepare_data(df):
    """Préparer les données pour l'entraînement"""
    print("\n🔧 Préparation des données...")

    # Identifier la colonne cible
    if 'Accident_Severity_Binary' in df.columns:
        target_col = 'Accident_Severity_Binary'
        print(f"✅ Cible: {target_col}")
    elif 'Accident_Severity' in df.columns:
        target_col = 'Accident_Severity'
        print(f"✅ Cible: {target_col}")
    else:
        print("❌ Colonne cible non trouvée!")
        return None, None, None
    
    # Séparer X et y
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Vérifier que toutes les colonnes sont numériques
    cols_to_drop = []
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"⚠️ Colonne non numérique détectée: {col}")
            print(f"   Valeurs uniques: {X[col].unique()[:5]}")
            cols_to_drop.append(col)
    
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        print(f"✅ Colonnes supprimées: {cols_to_drop}")
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"\nDistribution de la cible:")
    for cls in sorted(y.unique()):
        count = (y == cls).sum()
        pct = (count / len(y)) * 100
        print(f"   Classe {cls}: {count:>7} ({pct:>6.2f}%)")
    
    return X, y, X.columns.tolist()  # ✅ Retourne aussi les noms des features


def train_model(X, y):
    """Entraîner le modèle Random Forest"""
    print("\n🤖 Entraînement du modèle Random Forest...")

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

    # Entraîner Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    print("✅ Modèle entraîné (Random Forest - Classification Binaire)")

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
        'recall_grave': recall,  # ✅ Ajout pour compatibilité avec metrics.json
        'f1_score': f1,          # ✅ Ajout pour compatibilité
        'confusion_matrix': cm.tolist(),
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

    # Sauvegarder le modèle (nom compatible avec prediction.py)
    model_path = models_dir / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")

    # Sauvegarder le scaler (nom compatible avec prediction.py)
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé: {scaler_path}")

    return model_path, scaler_path


def save_metrics(metrics, output_path="models/metrics.json"):
    """Sauvegarder les métriques dans un fichier JSON"""
    # Créer dossier models s'il n'existe pas
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Préparer les données pour JSON (sans objets non sérialisables)
    metrics_json = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'recall_grave': metrics['recall_grave'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Métriques sauvegardées: {output_path}")


def save_features(feature_names):
    """Sauvegarder la liste des features"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    features_path = models_dir / "features.json"
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Features sauvegardées: {features_path}")


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

    # Préparer les données (retourne aussi les noms des features)
    X, y, feature_names = prepare_data(df)
    if X is None:
        return

    # Entraîner le modèle
    model, scaler, metrics = train_model(X, y)

    # Sauvegarder
    model_path, scaler_path = save_model(model, scaler)
    save_metrics(metrics)
    save_features(feature_names)  # ✅ AJOUTÉ

    # Résumé final
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║               ✅ ENTRAÎNEMENT RÉUSSI                  ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n📈 Accuracy finale: {metrics['accuracy']*100:.2f}%")
    print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\n📁 Fichiers sauvegardés:")
    print(f"   • Modèle: {model_path}")
    print(f"   • Scaler: {scaler_path}")
    print(f"   • Métriques: models/metrics.json")
    print(f"   • Features: models/features.json")
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