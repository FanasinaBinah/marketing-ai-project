"""
churn_utils.py
Construction du dataset de churn et entraînement des modèles prédictifs.
Reprend la logique du notebook Devoir_ML.ipynb (M6 - Prédiction de churn).

IMPORTANT — compatibilité avec les politiques de sécurité Windows restrictives :
`sklearn.ensemble` (RandomForestClassifier) charge en interne `sklearn.svm`, qui
peut être bloqué par une politique de contrôle d'application Windows avec l'erreur :
    "ImportError: DLL load failed while importing _libsvm_sparse"
On utilise donc `xgboost.XGBRFClassifier` à la place : c'est le même moteur
d'arbres que XGBoost, mais configuré pour se comporter comme une forêt aléatoire
(bagging), et il ne dépend pas de sklearn.ensemble/sklearn.svm (vérifié).
`sklearn.model_selection` et `sklearn.metrics` restent utilisés : ils ne déclenchent
pas le chargement de sklearn.svm.
"""

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)


@st.cache_data(show_spinner="Construction du jeu de données de churn...")
def build_churn_dataset(sales: pd.DataFrame, customers: pd.DataFrame, cutoff_date: str, future_end: str):
    """
    Un client est considéré "actif" (target = 1) s'il a acheté entre cutoff_date et future_end.
    Les features sont calculées sur l'historique avant cutoff_date (RFM + variantes).
    """
    cutoff_date = pd.to_datetime(cutoff_date)
    future_end = pd.to_datetime(future_end)

    sales = sales.copy()
    sales["Date"] = pd.to_datetime(sales["Date"])

    clients_futur = sales[
        (sales["Date"] >= cutoff_date) & (sales["Date"] < future_end)
    ]["Customer_ID"].unique()

    df_target = pd.DataFrame({"Customer_ID": sales["Customer_ID"].unique()})
    df_target["target"] = df_target["Customer_ID"].isin(clients_futur).astype(int)

    df_hist = sales[sales["Date"] < cutoff_date].copy()

    features = df_hist.groupby("Customer_ID").agg(
        recence=("Date", lambda x: (cutoff_date - x.max()).days),
        frequence=("Product_ID", "count"),
        montant_total=("Sale_Price", "sum"),
        panier_moyen=("Sale_Price", "mean"),
        quantite_totale=("Quantity", "sum"),
        produits_uniques=("Product_ID", "nunique"),
        anciennete=("Date", lambda x: (cutoff_date - x.min()).days),
    ).reset_index()

    customers_for_churn = customers.drop(columns=["Age_Group"], errors="ignore")
    dataset = customers_for_churn.merge(features, on="Customer_ID", how="inner")
    dataset = dataset.merge(df_target, on="Customer_ID", how="inner")

    dataset.fillna(
        {
            "frequence": 0,
            "montant_total": 0,
            "panier_moyen": 0,
            "quantite_totale": 0,
            "produits_uniques": 0,
        },
        inplace=True,
    )

    return dataset


@st.cache_resource(show_spinner="Entraînement des modèles (Random Forest / XGBoost)...")
def train_churn_models(dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Entraîne un Random Forest et un XGBoost pour prédire l'activité future du client."""
    cat_cols = [c for c in ["Gender", "Location"] if c in dataset.columns]
    dataset_encoded = pd.get_dummies(dataset, columns=cat_cols, drop_first=True)

    drop_cols = [c for c in ["Customer_ID", "target", "Name", "Join_Date", "Total_Spent"] if c in dataset_encoded.columns]
    X = dataset_encoded.drop(columns=drop_cols)
    y = dataset_encoded["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, test_size=test_size, stratify=y if y.nunique() > 1 else None
    )

    # "Random Forest" via XGBoost : un seul round de boosting (n_estimators=1) avec de
    # nombreux arbres parallèles (num_parallel_tree) + sous-échantillonnage = forêt
    # aléatoire classique (bagging), sans dépendre de sklearn.ensemble.
    rf_model = XGBClassifier(
        n_estimators=1,
        num_parallel_tree=100,
        subsample=0.8,
        colsample_bynode=0.8,
        learning_rate=1,
        reg_lambda=1e-5,
        random_state=random_state,
        eval_metric="logloss",
    )
    rf_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(eval_metric="logloss", random_state=random_state)
    xgb_model.fit(X_train, y_train)

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
    }


def evaluate_models(models: dict):
    """Calcule accuracy, matrices de confusion, courbes ROC et Précision-Rappel pour les 2 modèles."""
    rf_model, xgb_model = models["rf_model"], models["xgb_model"]
    X_test, y_test = models["X_test"], models["y_test"]

    results = {}
    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc = model.score(X_test, y_test)
        cm = confusion_matrix(y_test, preds)

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = auc(recall, precision)

        results[name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "pr_auc": pr_auc,
        }

    return results
