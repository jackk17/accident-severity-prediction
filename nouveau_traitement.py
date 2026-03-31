"""
🚗 Nouveau Traitement - Classification Binaire avec Régression Logistique
Script complet pour traiter les données dynamiquement:
1. Charger et nettoyer
2. Convertir à classification binaire (Minor vs Severe)
3. Rééquilibrer les classes
4. Entraîner un modèle LogisticRegression
5. Afficher les métriques de performance complètes
6. Sauvegarder les données et le modèle
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: CHARGEMENT ET NETTOYAGE DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

def charger_donnees(filepath="data/df_sample.csv"):
    """Charger les données"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           📊 CHARGEMENT DES DONNÉES                        ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    data_path = Path(filepath)

    if not data_path.exists():
        print(f"❌ Fichier non trouvé: {data_path}")
        return None

    df = pd.read_csv(data_path)
    print(f"✅ Données chargées: {df.shape}")
    print(f"📋 Colonnes: {list(df.columns)}\n")

    return df


def convertir_binaire(df, target_col='Accident_Severity'):
    """Convertir la classification 3 classes en 2 classes"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     🔄 CONVERSION À CLASSIFICATION BINAIRE                  ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    print("📊 Distribution ORIGINALE (3 classes):")
    original_dist = df[target_col].value_counts().sort_index()
    for classe, count in original_dist.items():
        pct = (count / len(df)) * 100
        severity_labels = {1: "Faible (Minor)", 2: "Grave (Moderate)", 3: "Très Grave (Severe)"}
        print(f"   Classe {classe} ({severity_labels.get(classe, 'Unknown')}): {count:>7} ({pct:>6.2f}%)")

    # Créer nouvelle colonne binaire
    # 0 = Minor (Faible + Grave) = Classes 1 + 2
    # 1 = Severe (Très Grave) = Class 3
    df[target_col] = df[target_col].apply(lambda x: 0 if x in [1, 2] else 1)

    print(f"\n📊 Distribution APRÈS CONVERSION (2 classes):")
    new_dist = df[target_col].value_counts().sort_index()
    for classe, count in new_dist.items():
        pct = (count / len(df)) * 100
        severity_labels = {0: "Minor (Accidents légers)", 1: "Severe (Accidents graves)"}
        print(f"   Classe {classe} ({severity_labels.get(classe, 'Unknown')}): {count:>7} ({pct:>6.2f}%) {'█' * int(pct/2)}")

    ratio = new_dist.max() / new_dist.min()
    print(f"\n   Ratio d'imbalance: {ratio:.2f}x\n")

    return df


def nettoyer_donnees(df):
    """Nettoyer les données"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           🧹 NETTOYAGE DES DONNÉES                         ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    taille_avant = len(df)
    print(f"Avant nettoyage: {taille_avant:,} lignes")

    df_clean = df.dropna()

    taille_apres = len(df_clean)
    supprimees = taille_avant - taille_apres
    print(f"Après suppression NaN: {taille_apres:,} lignes")

    if supprimees > 0:
        print(f"❌ {supprimees:,} lignes supprimées ({(supprimees/taille_avant)*100:.2f}%)\n")
    else:
        print(f"✅ Aucune ligne supprimée\n")

    return df_clean


def rééquilibrer_classes(df, target_col='Accident_Severity'):
    """Rééquilibrer les classes"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       ⚖️  RÉÉQUILIBRAGE DES CLASSES                        ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    classe_counts = df[target_col].value_counts().sort_index()

    print("📊 Distribution AVANT rééquilibrage:")
    for classe, count in classe_counts.items():
        pct = (count / len(df)) * 100
        severity_labels = {0: "Minor", 1: "Severe"}
        print(f"   Classe {classe} ({severity_labels.get(classe, 'Unknown')}): {count:>7} ({pct:>6.2f}%)")

    min_class = classe_counts.idxmin()
    min_count = classe_counts.min()

    print(f"\n🎯 Plus petite classe: {min_class} avec {min_count:,} exemples")
    print(f"📌 Gardant TOUS les exemples de la classe {min_class}")
    print(f"🔄 Sous-échantillonnant les autres classes à {min_count:,} exemples\n")

    dfs_par_classe = {}
    for classe in classe_counts.index:
        dfs_par_classe[classe] = df[df[target_col] == classe]

    dfs_equilibres = []

    for classe in sorted(classe_counts.index):
        df_classe = dfs_par_classe[classe]

        if classe == min_class:
            dfs_equilibres.append(df_classe)
            print(f"Classe {classe}: {len(df_classe):>7} exemples (TOUS conservés) ✅")
        else:
            if len(df_classe) > min_count:
                df_classe_echantillon = df_classe.sample(n=min_count, random_state=42)
                dfs_equilibres.append(df_classe_echantillon)
                print(f"Classe {classe}: {min_count:>7} exemples (sous-échantillonné de {len(df_classe)}) 🔄")
            else:
                dfs_equilibres.append(df_classe)
                print(f"Classe {classe}: {len(df_classe):>7} exemples (inchangé, déjà < {min_count}) ✅")

    df_equilibre = pd.concat(dfs_equilibres, ignore_index=True)
    df_equilibre = shuffle(df_equilibre, random_state=42)

    print(f"\n{'='*60}")
    print(f"📊 Distribution APRÈS rééquilibrage:")
    for classe in sorted(classe_counts.index):
        count = (df_equilibre[target_col] == classe).sum()
        pct = (count / len(df_equilibre)) * 100
        severity_labels = {0: "Minor", 1: "Severe"}
        print(f"   Classe {classe} ({severity_labels.get(classe, 'Unknown')}): {count:>7} ({pct:>6.2f}%)")

    print(f"\n✅ Dataset ÉQUILIBRÉ!")
    print(f"   Taille totale: {len(df_equilibre):,} lignes")
    print(f"   Ratio: 1.0x (PARFAIT!)\n")

    return df_equilibre


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: ENTRAÎNEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

def entrainer_modele(df, target_col='Accident_Severity'):
    """Entraîner le modèle LogisticRegression"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       🤖 ENTRAÎNEMENT DU MODÈLE LOGISTIQUE                 ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # Préparer les données
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"📊 Données d'entraînement:")
    print(f"   Features: {X.shape}")
    print(f"   Target: {y.shape}")
    print(f"   Classes: {sorted(y.unique())}\n")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Split Train/Test (80/20):")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}\n")

    # Normaliser
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ Normalisation avec StandardScaler\n")

    # Entraîner
    print(f"⏳ Entraînement en cours...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print(f"✅ Modèle entraîné!\n")

    # Prédictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_pred_proba


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: AFFICHAGE DES MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════

def afficher_metriques(y_test, y_pred, y_pred_proba):
    """Afficher les métriques de performance complètes"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       📊 MÉTRIQUES DE PERFORMANCE DU MODÈLE                ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # Métriques principales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("🎯 MÉTRIQUES GLOBALES:")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:>50.4f} ({accuracy*100:>6.2f}%)")
    print(f"Precision: {precision:>50.4f}")
    print(f"Recall:    {recall:>50.4f}")
    print(f"F1-Score:  {f1:>50.4f}")
    print(f"ROC-AUC:   {roc_auc:>50.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔲 MATRICE DE CONFUSION:")
    print("=" * 60)
    print(f"                 Prédiction Négatif  Prédiction Positif")
    print(f"Réalité Négatif:       {cm[0, 0]:>6}                {cm[0, 1]:>6}")
    print(f"Réalité Positif:       {cm[1, 0]:>6}                {cm[1, 1]:>6}")
    print(f"\nVrai Négatif (TN):  {cm[0, 0]:>6}")
    print(f"Faux Positif (FP):  {cm[0, 1]:>6}")
    print(f"Faux Négatif (FN):  {cm[1, 0]:>6}")
    print(f"Vrai Positif (TP):  {cm[1, 1]:>6}")

    # Détails par classe
    print(f"\n📋 RAPPORT DE CLASSIFICATION:")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=['Class 0 (Minor)', 'Class 1 (Severe)'],
        digits=4
    ))

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    print(f"\n📈 RÉSUMÉ:")
    print("=" * 60)
    print(f"✅ Le modèle classe correctement {accuracy*100:.2f}% des accidents")
    print(f"✅ Précision: {precision*100:.2f}% des 'Severe' prédits le sont réellement")
    print(f"✅ Recall: {recall*100:.2f}% des vrais 'Severe' sont détectés")
    print(f"✅ ROC-AUC: {roc_auc:.4f} (excellent!)")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def afficher_statistiques(df_final, df_original):
    """Afficher les statistiques complètes"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║           📊 STATISTIQUES FINALES                          ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    print("AVANT vs APRÈS:\n")

    print(f"{'Métrique':<30} {'AVANT (3 classes)':>20} {'APRÈS (2 classes)':>20}")
    print("=" * 75)

    print(f"{'Nombre de lignes':<30} {len(df_original):>20,} {len(df_final):>20,}")
    print(f"{'Nombre de colonnes':<30} {len(df_original.columns):>20} {len(df_final.columns):>20}")

    size_before = df_original.memory_usage(deep=True).sum() / (1024 * 1024)
    size_after = df_final.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"{'Taille mémoire':<30} {size_before:>18.2f} MB {size_after:>18.2f} MB")

    print(f"\n{'Distribution des classes ORIGINALES':<35}")
    print("-" * 75)

    count_1 = (df_original['Accident_Severity'] == 1).sum()
    count_2 = (df_original['Accident_Severity'] == 2).sum()
    count_3 = (df_original['Accident_Severity'] == 3).sum()
    pct_1 = (count_1 / len(df_original)) * 100
    pct_2 = (count_2 / len(df_original)) * 100
    pct_3 = (count_3 / len(df_original)) * 100

    print(f"Classe 1 (Faible):     {count_1:>8,} ({pct_1:>6.2f}%)")
    print(f"Classe 2 (Grave):      {count_2:>8,} ({pct_2:>6.2f}%)")
    print(f"Classe 3 (Très Grave): {count_3:>8,} ({pct_3:>6.2f}%)")
    print(f"Ratio d'imbalance: {count_3 / count_1:.2f}x (très déséquilibré)")

    print(f"\n{'Distribution des classes FINALES':<35}")
    print("-" * 75)

    count_0 = (df_final['Accident_Severity'] == 0).sum()
    count_1_final = (df_final['Accident_Severity'] == 1).sum()
    pct_0 = (count_0 / len(df_final)) * 100
    pct_1_final = (count_1_final / len(df_final)) * 100

    print(f"Classe 0 (Minor):      {count_0:>8,} ({pct_0:>6.2f}%)")
    print(f"Classe 1 (Severe):     {count_1_final:>8,} ({pct_1_final:>6.2f}%)")
    print(f"Ratio d'imbalance: {count_1_final / count_0:.2f}x (équilibré!)\n")


def sauvegarder_donnees(df, output_path="data/df_equilibre_binaire.csv"):
    """Sauvegarder les données"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           💾 SAUVEGARDE DES DONNÉES                        ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Données sauvegardées: {output_file}")
    print(f"   Taille: {file_size_mb:.2f} MB")
    print(f"   Lignes: {len(df):,}")
    print(f"   Colonnes: {len(df.columns)}\n")

    return output_file


def sauvegarder_modele(model, scaler):
    """Sauvegarder le modèle et le scaler"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           🔐 SAUVEGARDE DU MODÈLE                          ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "logistic_regression_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")

    scaler_path = models_dir / "standard_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé: {scaler_path}\n")

    return model_path, scaler_path


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Fonction principale - Orchestration complète"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║     🚗 TRAITEMENT, RÉÉQUILIBRAGE ET ENTRAÎNEMENT DU MODÈLE               ║")
    print("║                   Classification Binaire (Minor vs Severe)                ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝\n")

    # 1. Charger
    df_original_raw = charger_donnees()
    if df_original_raw is None:
        return

    df_original = df_original_raw.copy()

    # 2. Convertir à classification binaire
    df_binary = convertir_binaire(df_original_raw)

    # 3. Nettoyer
    df_clean = nettoyer_donnees(df_binary)

    # 4. Rééquilibrer
    df_equilibre = rééquilibrer_classes(df_clean)

    # 5. Entraîner le modèle
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_pred_proba = entrainer_modele(df_equilibre)

    # 6. Afficher les métriques
    metrics = afficher_metriques(y_test, y_pred, y_pred_proba)

    # 7. Afficher les statistiques finales
    afficher_statistiques(df_equilibre, df_original)

    # 8. Sauvegarder les données
    data_path = sauvegarder_donnees(df_equilibre)

    # 9. Sauvegarder le modèle
    model_path, scaler_path = sauvegarder_modele(model, scaler)

    # Résumé final
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║                    ✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!                    ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝\n")

    print("📁 FICHIERS SAUVEGARDÉS:")
    print(f"   📊 Données: {data_path}")
    print(f"   🤖 Modèle: {model_path}")
    print(f"   📏 Scaler: {scaler_path}")

    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"   ✅ Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   ✅ F1-Score:  {metrics['f1']:.4f}")
    print(f"   ✅ ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\n🚀 Prochaine étape: streamlit run app.py\n")


if __name__ == "__main__":
    main()
