"""
data_utils.py
Chargement, nettoyage et préparation des données pour le dashboard marketing.
Reprend la logique du notebook Devoir_ML.ipynb (M2 - Exploration des données).
"""

from pathlib import Path
import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent / "data" / "input"


@st.cache_data(show_spinner="Chargement des données...")
def load_raw_data(data_dir: str | Path = DATA_DIR):
    """Charge les 4 fichiers sources en CSV (encodage utf-8-sig pour gérer le BOM)."""
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / "customers_data.csv", encoding="utf-8-sig")
    products = pd.read_csv(data_dir / "products_data.csv", encoding="utf-8-sig")
    sales = pd.read_csv(data_dir / "sales_data.csv", encoding="utf-8-sig")
    marketing = pd.read_csv(data_dir / "marketing_data.csv", encoding="utf-8-sig")

    # Typage des dates
    sales["Date"] = pd.to_datetime(sales["Date"])
    customers["Join_Date"] = pd.to_datetime(customers["Join_Date"])
    marketing["Start_Date"] = pd.to_datetime(marketing["Start_Date"])
    marketing["End_Date"] = pd.to_datetime(marketing["End_Date"])

    return customers, products, sales, marketing


@st.cache_data(show_spinner="Préparation des données...")
def build_sales_full(customers: pd.DataFrame, products: pd.DataFrame, sales: pd.DataFrame):
    """
    Reproduit la fusion + feature engineering du notebook (cellule M2 - Nettoyage) :
    - fusion sales + customers + products
    - recalcul du prix de vente = Quantity * Price (prix unitaire produit)
    - dépense totale par client
    """
    sales_full = sales.merge(customers, on="Customer_ID", how="inner")
    sales_full = sales_full.merge(products, on="Product_ID", how="inner")

    sales_full["Sale_Price"] = sales_full["Quantity"] * sales_full["Price"]

    total_spent_per_customer = sales_full.groupby("Customer_ID")["Sale_Price"].sum()
    sales_full["Total_Spent"] = sales_full["Customer_ID"].map(total_spent_per_customer).fillna(0)

    customers = customers.copy()
    customers["Total_Spent"] = customers["Customer_ID"].map(total_spent_per_customer).fillna(0)

    sales_full["Year"] = sales_full["Date"].dt.year
    sales_full["Month"] = sales_full["Date"].dt.to_period("M").astype(str)

    # Tranches d'âge (comme sur le dashboard fourni)
    bins = [17, 24, 34, 44, 54, 200]
    labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
    customers["Age_Group"] = pd.cut(customers["Age"], bins=bins, labels=labels)

    return sales_full, customers


@st.cache_data(show_spinner="Calcul des KPIs marketing...")
def build_marketing_kpis(marketing: pd.DataFrame, sales_full: pd.DataFrame):
    """Reproduit la cellule M5 du notebook : CTR, taux de conversion, CPC, CPA, ROI."""
    marketing = marketing.copy()
    marketing["CTR_pct"] = (marketing["Clicks"] / marketing["Impressions"]) * 100
    marketing["Conversion_Rate_pct"] = (marketing["Conversions"] / marketing["Clicks"]) * 100
    marketing["CPC"] = marketing["Budget"] / marketing["Clicks"]
    marketing["CPA"] = marketing["Budget"] / marketing["Conversions"]

    sales_sum = sales_full["Sale_Price"].sum()
    marketing["ROI_pct"] = ((sales_sum - marketing["Budget"]) / marketing["Budget"]) * 100

    return marketing


def compute_global_kpis(customers: pd.DataFrame, sales_full: pd.DataFrame, marketing: pd.DataFrame) -> dict:
    """KPIs globaux affichés en haut du dashboard (comme sur la capture d'écran)."""
    ca_total = sales_full["Sale_Price"].sum()
    nb_clients = customers["Customer_ID"].nunique()
    nb_transactions = sales_full["Sale_ID"].nunique() if "Sale_ID" in sales_full.columns else len(sales_full)
    panier_moyen = sales_full["Sale_Price"].mean()
    budget_marketing_total = marketing["Budget"].sum()

    return {
        "ca_total": ca_total,
        "nb_clients": nb_clients,
        "nb_transactions": nb_transactions,
        "panier_moyen": panier_moyen,
        "budget_marketing_total": budget_marketing_total,
    }
