"""
Pipeline complet: nettoyage, conversion binaire, SMOTE, entrainement, evaluation.
Reproduit et ameliore le travail du notebook original.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path("data/df_sample.csv")
OUTPUT_PATH = Path("data/df_equilibre_binaire.csv")
MODELS_DIR = Path("models")
TARGET = "Accident_Severity"


def charger_donnees():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier non trouve: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[1] Chargement: {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


def convertir_binaire(df):
    """Classes 1+2 -> 0 (Minor), Classe 3 -> 1 (Severe)"""
    print(f"\n[2] Conversion binaire")
    dist_before = df[TARGET].value_counts().sort_index()
    for cls, n in dist_before.items():
        print(f"    Classe {cls}: {n:>7} ({n/len(df)*100:.1f}%)")

    df = df.copy()
    df[TARGET] = (df[TARGET] == 3).astype(int)

    dist_after = df[TARGET].value_counts().sort_index()
    print(f"  -> Minor (0): {dist_after[0]:,} | Severe (1): {dist_after[1]:,}")
    return df


def nettoyer(df):
    avant = len(df)
    df = df.dropna()
    print(f"\n[3] Nettoyage: {avant - len(df)} lignes supprimees ({avant:,} -> {len(df):,})")
    return df


def preparer_donnees(df):
    """Split, scale, SMOTE (comme dans le notebook original)"""
    print(f"\n[4] Preparation des donnees")

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Split: train={len(y_train):,}, test={len(y_test):,}")

    # Normalisation
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE sur le train uniquement (comme dans le notebook)
    print(f"    Avant SMOTE: Minor={sum(y_train==0):,}, Severe={sum(y_train==1):,}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)
    print(f"    Apres SMOTE: Minor={sum(y_train_bal==0):,}, Severe={sum(y_train_bal==1):,}")

    return X_train_bal, X_test_s, y_train_bal, y_test, scaler


def entrainer_et_comparer(X_train, X_test, y_train, y_test):
    """Teste plusieurs modeles, retourne le meilleur."""
    print(f"\n[5] Entrainement et comparaison de modeles")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
            class_weight='balanced'
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    resultats = {}
    best_name, best_score = None, 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        resultats[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'f1_macro': f1_macro, 'roc_auc': auc,
            'confusion_matrix': cm.tolist(),
            'y_pred': y_pred, 'y_proba': y_proba, 'model': model,
        }

        # Selectionner sur F1 macro (equilibre entre les 2 classes)
        marker = ""
        if f1_macro > best_score:
            best_score = f1_macro
            best_name = name
            marker = " <-- meilleur"

        print(f"    {name:<25} Acc={acc:.4f}  F1={f1:.4f}  "
              f"F1-macro={f1_macro:.4f}  AUC={auc:.4f}{marker}")

    print(f"\n    Modele retenu: {best_name}")
    return best_name, resultats


def afficher_metriques(name, res):
    """Affiche le rapport detaille du meilleur modele."""
    print(f"\n[6] Metriques detaillees - {name}")

    cm = np.array(res['confusion_matrix'])
    print(f"    Accuracy:  {res['accuracy']:.4f}  ({res['accuracy']*100:.2f}%)")
    print(f"    Precision: {res['precision']:.4f}")
    print(f"    Recall:    {res['recall']:.4f}")
    print(f"    F1-Score:  {res['f1']:.4f}")
    print(f"    ROC-AUC:   {res['roc_auc']:.4f}")

    print(f"\n    Matrice de confusion:")
    print(f"                  Pred Minor  Pred Severe")
    print(f"    Vrai Minor:     {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"    Vrai Severe:    {cm[1,0]:>6}       {cm[1,1]:>6}")

    print(f"\n    Rapport de classification:")
    print(classification_report(
        res['y_pred'] != res['y_pred'],  # placeholder
        res['y_pred'] != res['y_pred'],
        target_names=['Minor (0)', 'Severe (1)'], digits=4
    ))


def sauvegarder(df, best_name, resultats, scaler):
    """Sauvegarde donnees, modele, scaler et metriques."""
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    res = resultats[best_name]

    df.to_csv(OUTPUT_PATH, index=False)
    joblib.dump(res['model'], MODELS_DIR / "logistic_regression_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.pkl")

    # Sauvegarder les metriques en JSON pour l'app Streamlit
    metrics_all = {}
    for name, r in resultats.items():
        metrics_all[name] = {
            'accuracy': round(r['accuracy'], 4),
            'precision': round(r['precision'], 4),
            'recall': round(r['recall'], 4),
            'f1': round(r['f1'], 4),
            'f1_macro': round(r['f1_macro'], 4),
            'roc_auc': round(r['roc_auc'], 4),
            'confusion_matrix': r['confusion_matrix'],
        }

    metrics_out = {
        'best_model': best_name,
        'models': metrics_all,
    }
    with open(MODELS_DIR / "metrics.json", 'w') as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n[7] Sauvegarde:")
    print(f"    Donnees:   {OUTPUT_PATH} ({len(df):,} lignes)")
    print(f"    Modele:    {MODELS_DIR / 'logistic_regression_model.pkl'} ({best_name})")
    print(f"    Scaler:    {MODELS_DIR / 'standard_scaler.pkl'}")
    print(f"    Metriques: {MODELS_DIR / 'metrics.json'}")


def main():
    print("=" * 65)
    print("  Severite d'Accidents - Pipeline Complet (SMOTE + Multi-modele)")
    print("=" * 65)

    df_raw = charger_donnees()
    df_bin = convertir_binaire(df_raw)
    df_clean = nettoyer(df_bin)

    X_train, X_test, y_train, y_test, scaler = preparer_donnees(df_clean)
    best_name, resultats = entrainer_et_comparer(X_train, X_test, y_train, y_test)

    # Afficher le rapport du meilleur
    best = resultats[best_name]
    cm = np.array(best['confusion_matrix'])
    print(f"\n[6] Rapport detaille - {best_name}")
    print(f"    Accuracy:  {best['accuracy']:.4f}  ({best['accuracy']*100:.2f}%)")
    print(f"    Precision: {best['precision']:.4f}")
    print(f"    Recall:    {best['recall']:.4f}")
    print(f"    F1-Score:  {best['f1']:.4f}")
    print(f"    ROC-AUC:   {best['roc_auc']:.4f}")
    print(f"\n    Matrice de confusion:")
    print(f"                  Pred Minor  Pred Severe")
    print(f"    Vrai Minor:     {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"    Vrai Severe:    {cm[1,0]:>6}       {cm[1,1]:>6}")
    print(f"\n    Classification report:")
    # Rebuild y_test from the test data for the report
    y_pred_best = best['y_pred']
    print(classification_report(
        y_test, y_pred_best,
        target_names=['Minor (0)', 'Severe (1)'], digits=4
    ))

    sauvegarder(df_clean, best_name, resultats, scaler)

    b = resultats[best_name]
    print("\n" + "=" * 65)
    print(f"  Termine. Meilleur: {best_name}")
    print(f"  Accuracy: {b['accuracy']*100:.2f}% | F1: {b['f1']:.4f} | AUC: {b['roc_auc']:.4f}")
    print(f"  Lancer: streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
