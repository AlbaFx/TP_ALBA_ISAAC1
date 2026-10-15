import warnings
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 0. Paramètres
# ----------------------------------------------------------------------
CHEMIN_DONNEES = "SysRec.xlsx" if os.path.exists("SysRec.xlsx") else "ProdData.xlsx"
CIBLE = "RecommendedPlan"
COLS_NUM = ["Grade", "Demand"]
COLS_CAT = ["Loader", "Truck", "Route"]
SEED = 42


# ----------------------------------------------------------------------
# 1. Chargement et contrôle qualité
# ----------------------------------------------------------------------
def charger_donnees(chemin: str) -> pd.DataFrame:
    """Charge le fichier de données Excel."""
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = pd.read_excel(chemin)
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print(df.head())

    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum())
    return df


# ----------------------------------------------------------------------
# 2. Exploration (EDA)
# ----------------------------------------------------------------------
def explorer(df: pd.DataFrame) -> None:
    print("=== 2. EXPLORATION ===")
    print(df[COLS_NUM].describe().round(2), "\n")
    print("Répartition de la cible :")
    print(df[CIBLE].value_counts(normalize=True).round(3), "\n")

    print("Moyennes par plan :")
    print(df.groupby(CIBLE)[COLS_NUM].mean().round(2), "\n")

    for col in ["Route", "Loader"]:
        print(f"Plan recommandé selon {col} (proportions par ligne) :")
        print(pd.crosstab(df[col], df[CIBLE], normalize="index").round(2), "\n")

    print("Corrélation Grade / Demand :")
    print(df[COLS_NUM].corr().round(3), "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    df["Grade"].plot.hist(bins=30, ax=axes[0, 0], title="Distribution de Grade")
    df["Demand"].plot.hist(bins=30, ax=axes[0, 1], title="Distribution de Demand")
    df[CIBLE].value_counts().plot.bar(ax=axes[1, 0], title="Répartition des plans")
    df.boxplot(column="Grade", by=CIBLE, ax=axes[1, 1])
    axes[1, 1].set_title("Grade selon le plan")
    plt.tight_layout()
    plt.savefig("eda.png", dpi=120)
    plt.close()
    print("Graphiques d'exploration sauvegardés : eda.png\n")


# ----------------------------------------------------------------------
# 3. Préparation
# ----------------------------------------------------------------------
def construire_preprocesseur() -> ColumnTransformer:
    """Standardise les numériques, encode en one-hot les catégorielles."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLS_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLS_CAT),
        ]
    )


def separer(df: pd.DataFrame):
    X = df[COLS_NUM + COLS_CAT]
    y = df[CIBLE]
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )


# ----------------------------------------------------------------------
# 4. Modèles et comparaison
# ----------------------------------------------------------------------
def modeles_candidats() -> dict:
    return {
        "Baseline (classe majoritaire)": DummyClassifier(strategy="most_frequent"),
        "Baseline (aléatoire stratifié)": DummyClassifier(
            strategy="stratified", random_state=SEED
        ),
        "Régression logistique": LogisticRegression(max_iter=1000),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=SEED
        ),
    }


def comparer_modeles(X_train, y_train) -> pd.DataFrame:
    print("=== 4. COMPARAISON (validation croisée 5 plis, F1 macro) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    resultats = []
    for nom, modele in modeles_candidats().items():
        pipe = Pipeline([("prep", construire_preprocesseur()), ("clf", modele)])
        res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"acc": "accuracy", "f1": "f1_macro"},
            n_jobs=-1,
        )
        acc, f1 = res["test_acc"], res["test_f1"]
        resultats.append(
            {
                "Modèle": nom,
                "Accuracy (moy)": acc.mean(),
                "Accuracy (écart-type)": acc.std(),
                "F1 macro (moy)": f1.mean(),
            }
        )
        print(f"{nom:<32} acc={acc.mean():.3f} (+/- {acc.std():.3f})  F1={f1.mean():.3f}")
    print(f"\nRéférence du hasard pour 3 classes : {1/3:.3f}\n")
    return pd.DataFrame(resultats).sort_values("F1 macro (moy)", ascending=False)


# ----------------------------------------------------------------------
# 5. Optimisation d'hyperparamètres (Random Forest)
# ----------------------------------------------------------------------
def optimiser(X_train, y_train) -> Pipeline:
    print("=== 5. OPTIMISATION (GridSearchCV sur Random Forest) ===")
    pipe = Pipeline(
        [
            ("prep", construire_preprocesseur()),
            ("clf", RandomForestClassifier(n_jobs=-1, random_state=SEED)),
        ]
    )
    grille = {
        "clf__n_estimators": [200],
        "clf__max_depth": [None, 8],
        "clf__min_samples_leaf": [1, 10, 30],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grille, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Meilleurs paramètres :", gs.best_params_)
    print(f"Meilleur F1 macro (CV 3 plis) : {gs.best_score_:.3f}\n")
    return gs.best_estimator_


# ----------------------------------------------------------------------
# 6. Évaluation finale sur le jeu de test
# ----------------------------------------------------------------------
def evaluer(modele: Pipeline, X_test, y_test) -> None:
    print("=== 6. ÉVALUATION SUR LE JEU DE TEST ===")
    y_pred = modele.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.3f}\n")
    print(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Matrice de confusion (test)")
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=120)
    plt.close()
    print("Matrice de confusion sauvegardée : matrice_confusion.png\n")


def importance_variables(modele: Pipeline) -> None:
    """Importance des variables d'origine (somme des colonnes one-hot)."""
    prep = modele.named_steps["prep"]
    clf = modele.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    noms = prep.get_feature_names_out()
    imp = pd.Series(clf.feature_importances_, index=noms)
    agregees = {}
    for nom, val in imp.items():
        origine = nom.split("__")[1]
        for c in COLS_CAT:
            if origine.startswith(c + "_"):
                origine = c
        agregees[origine] = agregees.get(origine, 0) + val
    print("Importance des variables :")
    print(pd.Series(agregees).sort_values(ascending=False).round(3), "\n")


# ----------------------------------------------------------------------
# 7. Système de recommandation
# ----------------------------------------------------------------------
def recommander(modele: Pipeline, grade: float, demand: int, loader: str, truck: str, route: str) -> dict:
    """Renvoie le plan recommandé et la probabilité de chaque plan."""
    ligne = pd.DataFrame(
        [{
            "Grade": grade,
            "Demand": demand,
            "Loader": loader,
            "Truck": truck,
            "Route": route,
        }]
    )
    probas = modele.predict_proba(ligne)[0]
    classes = modele.classes_
    plan = classes[np.argmax(probas)]
    return {
        "plan_recommande": plan,
        "probabilites": {c: round(float(p), 3) for c, p in zip(classes, probas)},
    }


# ----------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------
def main() -> None:
    df = charger_donnees(CHEMIN_DONNEES)
    explorer(df)

    X_train, X_test, y_train, y_test = separer(df)
    print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes\n")

    tableau = comparer_modeles(X_train, y_train)
    print(tableau.round(3).to_string(index=False), "\n")

    meilleur = optimiser(X_train, y_train)
    evaluer(meilleur, X_test, y_test)
    importance_variables(meilleur)

    joblib.dump(meilleur, "modele_recommandation.joblib")
    print("Modèle sauvegardé : modele_recommandation.joblib\n")

    print("=== 7. EXEMPLE DE RECOMMANDATION ===")
    exemple = recommander(
        meilleur,
        grade=3.5,
        demand=6000,
        loader="L2",
        truck="T15",
        route="North",
    )
    print(exemple)


if __name__ == "__main__":
    main()

