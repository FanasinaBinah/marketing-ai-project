"""
segmentation_utils.py
Segmentation clients (K-means + clustering hiérarchique) avec PCA.
Reprend la logique du notebook Devoir_ML.ipynb (M3 - Segmentation client).

IMPORTANT — compatibilité avec les politiques de sécurité Windows restrictives :
`sklearn.cluster` (KMeans) et `sklearn.compose` (ColumnTransformer) chargent en
interne `sklearn.svm`, qui peut être bloqué par une politique de contrôle
d'application Windows avec l'erreur :
    "ImportError: DLL load failed while importing _libsvm_sparse"
Ce module évite donc volontairement ces deux sous-modules :
 - KMeans -> remplacé par scipy.cluster.vq.kmeans2 (scipy est indépendant de sklearn.svm)
 - ColumnTransformer -> remplacé par une combinaison manuelle numpy des features
   scalées (StandardScaler) et encodées (OneHotEncoder), tous deux sûrs à importer.
 - PCA -> remplacé par une implémentation numpy (SVD), voir pca_2d().
`sklearn.preprocessing` et `sklearn.metrics` restent utilisés : ils ne déclenchent
pas le chargement de sklearn.svm (vérifié).
"""

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans2


def pca_2d(X) -> np.ndarray:
    """
    PCA à 2 composantes ré-implémentée à la main avec numpy (SVD).
    Équivalent mathématique de PCA(n_components=2).fit_transform(X), sans dépendre
    de sklearn.decomposition (qui charge sklearn.svm en interne).
    """
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X_centered = X_dense - X_dense.mean(axis=0)
    U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    return U[:, :2] * S[:2]


def _safe_silhouette_score(X_scaled, labels, sample_size: int = 2000, random_state: int = 42) -> float:
    """
    silhouette_score calcule par défaut une matrice de distances entre TOUTES les
    paires de lignes (O(n²) en mémoire) : sur 10 000 lignes, cela demande un
    tableau de 10 000 x 10 000 float64 (~763 Mo), ce qui peut faire planter
    l'application (MemoryError). On calcule donc le score sur un échantillon
    aléatoire (sample_size lignes) quand le jeu de données est grand : le score
    reste une estimation fiable de la qualité du clustering, sans le coût mémoire.
    """
    n = X_scaled.shape[0] if hasattr(X_scaled, "shape") else len(labels)
    if n > sample_size:
        return float(silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=random_state))
    return float(silhouette_score(X_scaled, labels))


def build_feature_matrix(X: pd.DataFrame) -> np.ndarray:
    """
    Remplace ColumnTransformer : standardise les colonnes numériques et encode en
    one-hot la colonne produit, puis combine le tout en une seule matrice dense.
    """
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(X[["Age", "Total_Spent"]])

    encoder = OneHotEncoder(handle_unknown="ignore")
    product_encoded = encoder.fit_transform(X[["Product_Name"]])
    product_encoded = product_encoded.toarray() if hasattr(product_encoded, "toarray") else product_encoded

    return np.hstack([numeric_scaled, product_encoded])


@st.cache_data(show_spinner="Segmentation des clients en cours...")
def run_segmentation(sales_full: pd.DataFrame, n_clusters: int = 3):
    """
    Segmente les transactions clients sur la base de : Age, Total_Spent, Product_Name.
    Retourne le dataframe enrichi (cluster K-means + cluster hiérarchique) et les
    coordonnées PCA, ainsi que les scores de silhouette.
    """
    customer_features = sales_full.copy()

    features = ["Age", "Total_Spent", "Product_Name"]
    X = sales_full[features]

    X_scaled = build_feature_matrix(X)  # matrice dense numpy

    # PCA (2 composantes) pour la visualisation — implémentation numpy, voir pca_2d()
    X_pca = pca_2d(X_scaled)

    # K-means — implémentation scipy (évite sklearn.cluster)
    centroids, labels = kmeans2(X_scaled, n_clusters, minit="++", seed=42, iter=20)
    customer_features["cluster"] = labels
    score_kmeans = _safe_silhouette_score(X_scaled, labels)

    # Clustering hiérarchique (Ward) — déjà basé sur scipy, inchangé
    linked = linkage(X_scaled, method="ward")
    customer_features["cluster_hierarchical"] = fcluster(linked, n_clusters, criterion="maxclust")
    score_hierarchical = _safe_silhouette_score(X_scaled, customer_features["cluster_hierarchical"])

    return {
        "customer_features": customer_features,
        "X_pca": X_pca,
        "score_kmeans": score_kmeans,
        "score_hierarchical": score_hierarchical,
    }


def build_cluster_profiles(customer_features: pd.DataFrame):
    """Profils / personas moyens par cluster K-means (M4 du notebook)."""
    profiles = customer_features.groupby("cluster").agg(
        Age_moyen=("Age", "mean"),
        Depense_moyenne=("Total_Spent", "mean"),
        Taille_segment=("Customer_ID", "count"),
    ).reset_index()

    # Noms de segments génériques, ordonnés par dépense moyenne
    ordered = profiles.sort_values("Depense_moyenne")
    labels = ["Faible dépense", "Dépense moyenne", "Forte dépense"]
    name_map = {}
    for i, cluster_id in enumerate(ordered["cluster"]):
        name_map[cluster_id] = labels[i] if i < len(labels) else f"Segment {cluster_id}"
    profiles["Persona"] = profiles["cluster"].map(name_map)

    return profiles, name_map


# ============================================================================
# EXTENSIONS — page "Segmentation clients" enrichie (Customer Segmentation
# Analytics). Rien au-dessus de cette ligne n'a été modifié : run_segmentation,
# build_cluster_profiles, pca_2d et build_feature_matrix restent strictement
# identiques pour ne rien casser dans le reste de l'application.
#
# NOTE IMPORTANTE SUR LE NIVEAU D'AGRÉGATION
# -------------------------------------------
# `run_segmentation` clusterise au niveau de la TRANSACTION (une ligne par
# vente, car Product_Name fait partie des features — c'est la logique du
# notebook d'origine). Un même client peut donc, en théorie, avoir des achats
# répartis sur des clusters différents. Pour obtenir "1 client = 1 point" dans
# la cartographie et des KPI réellement exprimés en nombre de clients (et non
# de transactions), `aggregate_customers` ramène le dataframe transactionnel à
# une ligne par client :
#   - le cluster retenu pour un client est son cluster MAJORITAIRE
#     (celui de la plupart de ses transactions) ;
#   - sa position PCA affichée est la MOYENNE de la position PCA de ses
#     transactions.
# Les analyses de chiffre d'affaires (CA total, CA par mois, part du CA...)
# restent en revanche calculées directement sur les transactions
# (customer_features), car un CA doit être attribué à la transaction réelle
# qui l'a généré, quel que soit le cluster majoritaire du client.
# ============================================================================

CLUSTER_COLORS = ["#7C3AED", "#2563EB", "#06B6D4", "#F59E0B", "#EF4444", "#10B981"]


def cluster_color_map(cluster_values) -> dict:
    """Associe une couleur stable à chaque valeur de cluster (cohérence entre tous les graphiques)."""
    uniques = sorted(pd.unique(pd.Series(cluster_values)))
    return {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(uniques)}


@st.cache_data(show_spinner="Agrégation des clients...")
def aggregate_customers(customer_features: pd.DataFrame) -> pd.DataFrame:
    """Ramène le dataframe transactionnel à une ligne par client (voir note ci-dessus)."""

    def _majority(s: pd.Series):
        return s.value_counts().idxmax()

    agg_kwargs = {
        "Age": ("Age", "first"),
        "Total_Spent": ("Total_Spent", "first"),
        "Nb_Transactions": ("Sale_Price", "count"),
        "Nb_Produits_Distincts": ("Product_Name", "nunique"),
        "Panier_Moyen": ("Sale_Price", "mean"),
        "cluster": ("cluster", _majority),
        "cluster_hierarchical": ("cluster_hierarchical", _majority),
    }
    if "PC1" in customer_features.columns and "PC2" in customer_features.columns:
        agg_kwargs["PC1"] = ("PC1", "mean")
        agg_kwargs["PC2"] = ("PC2", "mean")
    if "Gender" in customer_features.columns:
        agg_kwargs["Gender"] = ("Gender", "first")
    if "Location" in customer_features.columns:
        agg_kwargs["Location"] = ("Location", "first")

    per_customer = customer_features.groupby("Customer_ID").agg(**agg_kwargs).reset_index()
    return per_customer


def _assign_personas(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Attribue un nom de persona à chaque cluster, calculé UNIQUEMENT à partir du
    classement relatif des statistiques réelles du cluster (dépense, âge,
    fréquence d'achat). Aucune valeur n'est écrite en dur.
    """
    df = profiles.copy()
    n = len(df)
    spend_rank = df["Depense_moyenne"].rank(ascending=False, method="min")
    age_rank = df["Age_moyen"].rank(ascending=True, method="min")
    freq_rank = df["Frequence_moyenne"].rank(ascending=False, method="min")

    personas = []
    for i in df.index:
        if spend_rank[i] == 1:
            personas.append("Clients premium")
        elif spend_rank[i] == n and n > 1:
            personas.append("Clients à faible engagement")
        elif age_rank[i] == 1 and spend_rank[i] <= max(1, n // 2):
            personas.append("Jeunes clients à potentiel")
        elif freq_rank[i] == n and spend_rank[i] <= max(1, n // 2) and n > 1:
            personas.append("Gros acheteurs occasionnels")
        else:
            personas.append("Clients réguliers")
    df["Persona"] = personas

    # Niveau de valeur client dérivé directement du classement dépense (pas une nouvelle donnée)
    df["Valeur_client"] = np.select(
        [spend_rank == 1, spend_rank == n],
        ["Élevée", "Faible"],
        default="Moyenne",
    )
    return df


@st.cache_data(show_spinner="Calcul des profils de segments...")
def build_full_profiles(per_customer: pd.DataFrame, customer_features: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Table de profils complète par cluster : taille (en clients), % clients,
    âge moyen, dépense moyenne/totale, panier moyen, fréquence, produits
    moyens, CA réel et % du CA — plus le persona dérivé dynamiquement.
    """
    total_clients = per_customer["Customer_ID"].nunique()
    total_ca = customer_features["Sale_Price"].sum()

    profiles = per_customer.groupby(cluster_col).agg(
        Taille_segment=("Customer_ID", "nunique"),
        Age_moyen=("Age", "mean"),
        Depense_moyenne=("Total_Spent", "mean"),
        Depense_totale=("Total_Spent", "sum"),
        Panier_moyen=("Panier_Moyen", "mean"),
        Frequence_moyenne=("Nb_Transactions", "mean"),
        Produits_moyens=("Nb_Produits_Distincts", "mean"),
    ).reset_index()

    ca_reel = customer_features.groupby(cluster_col)["Sale_Price"].sum().reset_index(name="CA_reel")
    profiles = profiles.merge(ca_reel, on=cluster_col, how="left")

    profiles["Pct_clients"] = 100 * profiles["Taille_segment"] / total_clients if total_clients else 0
    profiles["Pct_CA"] = 100 * profiles["CA_reel"] / total_ca if total_ca else 0

    profiles = _assign_personas(profiles)
    return profiles.sort_values(cluster_col).reset_index(drop=True)


def generate_marketing_insights(profiles: pd.DataFrame, cluster_col: str) -> list:
    """
    Génère des phrases d'analyse marketing calculées dynamiquement à partir de
    `profiles` (aucun chiffre écrit en dur).
    """
    if profiles.empty:
        return []

    insights = []

    top_value = profiles.loc[profiles["Pct_CA"].idxmax()]
    insights.append(
        f"Le cluster {top_value[cluster_col]} ({top_value['Persona']}) représente "
        f"{top_value['Pct_clients']:.0f} % des clients mais génère {top_value['Pct_CA']:.0f} % du CA : "
        "c'est le segment à plus forte valeur."
    )

    low_spend = profiles.loc[profiles["Depense_moyenne"].idxmin()]
    insights.append(
        f"Le cluster {low_spend[cluster_col]} ({low_spend['Persona']}) affiche la dépense moyenne la plus "
        f"faible ({low_spend['Depense_moyenne']:.0f} €) : une campagne de réactivation pourrait être pertinente."
    )

    high_spend = profiles.loc[profiles["Depense_moyenne"].idxmax()]
    insights.append(
        f"Le cluster {high_spend[cluster_col]} présente la dépense moyenne la plus élevée "
        f"({high_spend['Depense_moyenne']:.0f} €)."
    )

    if len(profiles) > 1:
        biggest = profiles.loc[profiles["Taille_segment"].idxmax()]
        insights.append(
            f"Le cluster {biggest[cluster_col]} est le plus important en taille avec "
            f"{int(biggest['Taille_segment'])} clients ({biggest['Pct_clients']:.0f} % de la base)."
        )

    return insights


@st.cache_data(show_spinner="Préparation de l'évolution mensuelle du CA...")
def revenue_by_month(customer_features: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """CA mensuel réel par cluster, calculé sur les transactions (voir note en tête de section)."""
    out = (
        customer_features.groupby(["Month", cluster_col])["Sale_Price"]
        .sum()
        .reset_index()
        .sort_values("Month")
    )
    return out


def build_radar_data(profiles: pd.DataFrame, cluster_col: str):
    """Normalise (min-max) les variables de profil pour le radar, afin qu'aucune ne domine les autres."""
    metrics = {
        "Age_moyen": "Âge",
        "Depense_moyenne": "Dépense",
        "Frequence_moyenne": "Fréquence d'achat",
        "Produits_moyens": "Produits",
        "CA_reel": "CA généré",
        "Panier_moyen": "Panier moyen",
    }
    available = [m for m in metrics if m in profiles.columns]
    df = profiles[[cluster_col] + available].copy()
    for m in available:
        mn, mx = df[m].min(), df[m].max()
        df[m] = 0.5 if mx == mn else (df[m] - mn) / (mx - mn)
    return df, [metrics[m] for m in available]


def compare_methods_table(seg_result: dict, per_customer: pd.DataFrame) -> pd.DataFrame:
    """Tableau de comparaison K-means / Hiérarchique (silhouette + nb de clients segmentés)."""
    rows = []
    for label, col, score in [
        ("K-means", "cluster", seg_result["score_kmeans"]),
        ("Hiérarchique", "cluster_hierarchical", seg_result["score_hierarchical"]),
    ]:
        rows.append(
            {
                "Méthode": label,
                "Silhouette": round(float(score), 3),
                "Nb clusters effectifs": int(per_customer[col].nunique()),
                "Clients segmentés": int(per_customer["Customer_ID"].nunique()),
            }
        )
    return pd.DataFrame(rows)