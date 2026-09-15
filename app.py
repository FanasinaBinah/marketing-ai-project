"""
app.py — Dashboard Marketing IA (Streamlit)

Basé sur le notebook Devoir_ML.ipynb :
 - M2 : Exploration & KPIs
 - M3/M4 : Segmentation client (K-means / hiérarchique + PCA) et personas
 - M5 : Performance des campagnes marketing
 - M6 : Prédiction du churn (Random Forest via XGBoost / XGBoost)

Lancer en local :
    streamlit run app.py

Déploiement : voir README.md
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap

from data_utils import (
    load_raw_data,
    build_sales_full,
    build_marketing_kpis,
    compute_global_kpis,
)
from segmentation_utils import run_segmentation, build_cluster_profiles
from churn_utils import build_churn_dataset, train_churn_models, evaluate_models


# ============================================================================
# Configuration générale
# ============================================================================
st.set_page_config(
    page_title="MarketIA — Dashboard Marketing",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#6D5AE6"
ACCENT_COLORS = ["#6D5AE6", "#2FB8A6", "#3B82F6", "#F59E0B", "#EF4444", "#EC4899", "#14B8A6"]
PLOTLY_TEMPLATE = "plotly_white"

NAV_ITEMS = [
    ("overview", "🏠", "Vue d'ensemble", "Étape 1 — Exploration des données"),
    ("segmentation", "👥", "Segmentation clients", "Étape 2 — Segmentation clients"),
    ("campaigns", "📣", "Campagnes marketing", "Étape 3 — Analyse marketing"),
    ("churn", "🔮", "Prédiction du churn", "Étape 4 — Modélisation prédictive"),
]
PAGE_META = {key: (icon, label, badge) for key, icon, label, badge in NAV_ITEMS}


# ============================================================================
# Style global — thème sombre pour la sidebar, cartes arrondies pour le contenu
# ============================================================================
st.markdown(
    "\n<style>\n"
    """
        /* ---------- Fond général ---------- */
        .stApp { background: #F6F7FB; }
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px; }

        /* ---------- Sidebar sombre ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B1533 0%, #241C3F 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #E5E7EB; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background: #6D5AE6 !important; }

    .sidebar-brand { display:flex; align-items:center; gap:10px; margin: 2px 0 18px 0; }
    .sidebar-brand-icon {
        width:40px; height:40px; border-radius:12px;
        background:linear-gradient(135deg,#8B7BFF,#4C3BCF);
        display:flex; align-items:center; justify-content:center; font-size:19px; color:#fff;
        flex-shrink:0;
    }
    .sidebar-brand-title { font-size:17px; font-weight:800; color:#fff; line-height:1.15; }
    .sidebar-brand-subtitle { font-size:11.5px; color:#9C93C4; }

    .sidebar-nav-label {
        font-size:11px; font-weight:700; letter-spacing:0.07em; text-transform:uppercase;
        color:#7C74A8; margin: 6px 0 8px 4px;
    }

    [data-testid="stSidebar"] button {
        width: 100%;
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent !important; border: none !important; box-shadow:none !important;
        justify-content:flex-start !important; color:#C7C2E0 !important; font-weight:500 !important;
        border-radius:10px !important; padding: 9px 12px !important; text-align:left !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.07) !important; color:#fff !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background: #6D5AE6 !important; border:none !important; box-shadow:none !important;
        justify-content:flex-start !important; color:#fff !important; font-weight:600 !important;
        border-radius:10px !important; padding: 9px 12px !important; text-align:left !important;
    }
    [data-testid="stSidebar"] .stButton { margin-bottom: 2px; }

    /* ---------- En-tête de page (badge + titre) ---------- */
    .step-badge {
        display:inline-block; background:#EFEBFF; color:#6D5AE6; font-size:12px; font-weight:700;
        padding:5px 14px; border-radius:999px; margin-bottom: 12px; letter-spacing:0.01em;
    }
    .page-title { font-size: 34px; font-weight:800; color:#111827; margin:0 0 8px 0; line-height:1.2; }
    .page-subtitle { font-size:14.5px; color:#6B7280; max-width: 860px; margin-bottom: 22px; line-height:1.5; }

    /* ---------- Cartes KPI ---------- */
    .kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
    .kpi-card {
        flex: 1 1 180px;
        background: #FFFFFF;
        border: 1px solid #ECEBF5;
        border-radius: 18px;
        padding: 20px 20px 18px 20px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
    }
    .kpi-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .kpi-label {
        font-size: 11.5px; font-weight: 700; letter-spacing: 0.05em;
        color: #6B7280; text-transform: uppercase;
    }
    .kpi-icon {
        width: 34px; height: 34px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; flex-shrink: 0;
    }
    .kpi-value { font-size: 27px; font-weight: 800; color: #111827; margin-top: 10px; }
    .kpi-caption { font-size: 12.5px; color: #9CA3AF; margin-top: 2px; }
    .kpi-caption.positive { color: #16A34A; font-weight: 600; }
    .kpi-caption.negative { color: #DC2626; font-weight: 600; }

    /* ---------- Cartes section / graphiques ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important; border: 1px solid #ECEBF5 !important;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04) !important; background: #FFFFFF !important;
    }
    .section-title { font-size: 15px; font-weight: 700; color: #111827; margin-bottom: 2px; }
    .section-caption { font-size: 12.5px; color: #9CA3AF; margin-bottom: 6px; }

    /* ---------- Persona cards ---------- */
    .persona-card {
        background:#FFFFFF; border:1px solid #ECEBF5; border-radius:16px; padding:18px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.04);
    }
    .persona-title { font-size:14px; font-weight:700; color:#111827; margin-bottom:8px; }
    .persona-stat { font-size:12.5px; color:#6B7280; margin-bottom:2px; }
    .persona-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, caption: str, icon: str, color: str, trend: str = None) -> str:
    """Génère le HTML (une seule ligne, sans indentation) d'une carte KPI avec icône."""
    caption_html = f'<div class="kpi-caption">{caption}</div>'
    if trend:
        cls = "positive" if trend.startswith("+") else "negative"
        caption_html = f'<div class="kpi-caption {cls}">{trend}</div>' + caption_html
    return (
        '<div class="kpi-card">'
        '<div class="kpi-top">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-icon" style="background:{color}1A; color:{color};">{icon}</div>'
        "</div>"
        f'<div class="kpi-value">{value}</div>'
        f"{caption_html}"
        "</div>"
    )


def render_kpi_row(cards_html: list[str]):
    st.markdown(f'<div class="kpi-row">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def section_header(title: str, caption: str = ""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-caption">{caption}</div>' if caption else ""),
        unsafe_allow_html=True,
    )


def page_header(key: str):
    icon, label, badge = PAGE_META[key]
    st.markdown(f'<div class="step-badge">{badge}</div>', unsafe_allow_html=True)


def fmt_currency(x):
    return f"{x:,.0f} €".replace(",", " ")


# ============================================================================
# Chargement des données
# ============================================================================
try:
    customers, products, sales, marketing = load_raw_data()
except FileNotFoundError:
    st.error(
        "Fichiers de données introuvables. Vérifiez que `data/input/` contient bien "
        "`customers_data.csv`, `products_data.csv`, `sales_data.csv` et `marketing_data.csv`."
    )
    st.stop()

sales_full, customers = build_sales_full(customers, products, sales)
marketing_kpi = build_marketing_kpis(marketing, sales_full)


# ============================================================================
# Sidebar — marque, navigation, filtres
# ============================================================================
if "page" not in st.session_state:
    st.session_state.page = "overview"

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-icon">✨</div>'
        '<div>'
        '<div class="sidebar-brand-title">MarketIA</div>'
        '<div class="sidebar-brand-subtitle">Marketing Intelligence</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
    for key, icon, label, _ in NAV_ITEMS:
        is_active = st.session_state.page == key
        if st.button(f"{icon}   {label}", key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    st.divider()

    with st.expander("🔎 Filtres du dashboard", expanded=False):
        years = sorted(sales_full["Year"].dropna().unique().tolist())
        selected_years = st.multiselect("Année", years, default=years)

        channels = sorted(sales_full["Channel"].dropna().unique().tolist())
        selected_channels = st.multiselect("Canal de vente", channels, default=channels)

        categories = sorted(sales_full["Category"].dropna().unique().tolist())
        selected_categories = st.multiselect("Catégorie produit", categories, default=categories)

    st.markdown(
        '<div style="font-size:11px; color:#7C74A8; margin-top:24px; line-height:1.5;">'
        "Projet académique — Data Analytics &amp; Machine Learning</div>",
        unsafe_allow_html=True,
    )

mask = (
    sales_full["Year"].isin(selected_years)
    & sales_full["Channel"].isin(selected_channels)
    & sales_full["Category"].isin(selected_categories)
)
sales_filtered = sales_full[mask].copy()

page = st.session_state.page


# ============================================================================
# PAGE 1 — VUE D'ENSEMBLE
# ============================================================================
if page == "overview":
    page_header("overview")
    st.markdown('<div class="page-title">Vue d\'ensemble</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Synthèse de l\'activité commerciale et marketing. '
        "Ces indicateurs servent de base à la segmentation et aux modèles de prédiction.</div>",
        unsafe_allow_html=True,
    )

    kpis = compute_global_kpis(customers, sales_filtered, marketing_kpi)

    # Tendance CA (mois en cours vs mois précédent, sur les données filtrées)
    monthly_ca = sales_filtered.groupby("Month")["Sale_Price"].sum().sort_index()
    trend_txt = None
    if len(monthly_ca) >= 2:
        last, prev = monthly_ca.iloc[-1], monthly_ca.iloc[-2]
        if prev > 0:
            pct = (last - prev) / prev * 100
            trend_txt = f"{'+' if pct >= 0 else ''}{pct:.1f}% vs mois précédent"

    render_kpi_row([
        kpi_card("CA Total", fmt_currency(kpis["ca_total"]), "sur la période filtrée", "📁", ACCENT_COLORS[0], trend_txt),
        kpi_card("Clients", f"{kpis['nb_clients']:,}".replace(",", " "), "clients uniques", "👥", ACCENT_COLORS[1]),
        kpi_card("Transactions", f"{kpis['nb_transactions']:,}".replace(",", " "), "lignes de vente", "🧾", ACCENT_COLORS[2]),
        kpi_card("Panier moyen", f"{kpis['panier_moyen']:.2f} €", "par transaction", "🛍️", ACCENT_COLORS[3]),
        kpi_card("Budget marketing", fmt_currency(kpis["budget_marketing_total"]), "toutes campagnes", "📣", ACCENT_COLORS[4]),
    ])

    col1, col2 = st.columns([1.4, 1])
    with col1:
        with st.container(border=True):
            section_header("Évolution du CA mensuel", "Saisonnalité des ventes sur la période")
            monthly = sales_filtered.groupby("Month")["Sale_Price"].sum().reset_index().sort_values("Month")
            fig = px.area(
                monthly, x="Month", y="Sale_Price", template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig.update_traces(line=dict(width=2.5), fillcolor="rgba(109,90,230,0.12)")
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), xaxis_title=None, yaxis_title=None, height=320)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            section_header("CA par catégorie", "Répartition du chiffre d'affaires")
            cat_ca = sales_filtered.groupby("Category")["Sale_Price"].sum().reset_index()
            fig = px.pie(
                cat_ca, names="Category", values="Sale_Price", hole=0.55,
                template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=320, showlegend=True,
                               legend=dict(orientation="v", font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section_header("Top 20 clients par dépenses")
            top20 = (
                sales_filtered.groupby("Customer_ID")["Sale_Price"].sum()
                .sort_values(ascending=False).head(20).reset_index()
            )
            fig = px.bar(top20, x="Customer_ID", y="Sale_Price", template=PLOTLY_TEMPLATE,
                          color_discrete_sequence=[PRIMARY_COLOR])
            fig.update_xaxes(type="category")
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        with st.container(border=True):
            section_header("Ventes par canal")
            channel_sales = sales_filtered.groupby("Channel")["Sale_Price"].sum().reset_index()
            fig = px.pie(channel_sales, names="Channel", values="Sale_Price", hole=0.55,
                         template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS)
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        with st.container(border=True):
            section_header("Clients par tranche d'âge")
            age_counts = customers["Age_Group"].value_counts().sort_index().reset_index()
            age_counts.columns = ["Tranche d'âge", "Nombre de clients"]
            fig = px.bar(age_counts, x="Tranche d'âge", y="Nombre de clients", template=PLOTLY_TEMPLATE,
                         color_discrete_sequence=[ACCENT_COLORS[1]])
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        with st.container(border=True):
            section_header("Clients par localisation")
            loc_counts = customers["Location"].value_counts().sort_values(ascending=False).reset_index()
            loc_counts.columns = ["Location", "Nombre de clients"]
            fig = px.bar(loc_counts, x="Location", y="Nombre de clients", template=PLOTLY_TEMPLATE,
                         color_discrete_sequence=[ACCENT_COLORS[2]])
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Taux de conversion par canal marketing")
        conv_by_channel = marketing_kpi.groupby("Channel")["Conversion_Rate_pct"].mean().reset_index()
        fig = px.bar(
            conv_by_channel, x="Channel", y="Conversion_Rate_pct", template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[PRIMARY_COLOR],
            labels={"Conversion_Rate_pct": "Taux de conversion (%)"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 2 — SEGMENTATION CLIENTS
# ============================================================================
elif page == "segmentation":
    page_header("segmentation")
    st.markdown('<div class="page-title">Segmentation clients</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Segmentation K-means &amp; hiérarchique basée sur l\'âge, la dépense '
        "totale et les préférences de catégories produit — visualisation en Composantes Principales (PCA).</div>",
        unsafe_allow_html=True,
    )

    n_clusters = st.slider("Nombre de segments (clusters)", min_value=2, max_value=6, value=3)

    with st.spinner("Segmentation des clients en cours..."):
        seg_result = run_segmentation(sales_full, n_clusters=n_clusters)

    customer_features = seg_result["customer_features"]
    X_pca = seg_result["X_pca"]

    render_kpi_row([
        kpi_card("Clients segmentés", f"{len(customer_features):,}".replace(",", " "), "avec historique d'achat", "👥", ACCENT_COLORS[0]),
        kpi_card("Silhouette K-means", f"{seg_result['score_kmeans']:.3f}", "qualité de séparation", "🎯", ACCENT_COLORS[1]),
        kpi_card("Silhouette hiérarchique", f"{seg_result['score_hierarchical']:.3f}", "qualité de séparation", "🌳", ACCENT_COLORS[2]),
    ])

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("Clusters (PCA / K-means)")
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], color=customer_features["cluster"].astype(str),
                labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS,
            )
            fig.update_traces(marker=dict(size=6, opacity=0.75))
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=380)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            section_header("Clusters (PCA / Hiérarchique)")
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], color=customer_features["cluster_hierarchical"].astype(str),
                labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS,
            )
            fig.update_traces(marker=dict(size=6, opacity=0.75))
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=380)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        section_header("Profils des segments (personas)", "Basé sur le clustering K-means")
        profiles, _ = build_cluster_profiles(customer_features)
        profiles = profiles.sort_values("Depense_moyenne").reset_index(drop=True)

        cols = st.columns(len(profiles))
        for i, (_, row) in enumerate(profiles.iterrows()):
            color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
            with cols[i]:
                st.markdown(
                    '<div class="persona-card">'
                    '<div class="persona-title">'
                    f'<span class="persona-dot" style="background:{color};"></span>{row["Persona"]}'
                    "</div>"
                    f'<div class="persona-stat">Âge moyen : <b>{row["Age_moyen"]:.1f} ans</b></div>'
                    f'<div class="persona-stat">Dépense moyenne : <b>{row["Depense_moyenne"]:.2f} €</b></div>'
                    f'<div class="persona-stat">Taille du segment : <b>{int(row["Taille_segment"])} clients</b></div>'
                    "</div>",
                    unsafe_allow_html=True,
                )

    st.caption(
        "ℹ️ Seuls les clients ayant au moins un achat enregistré sont segmentés "
        "(la segmentation est construite à partir de l'historique de transactions)."
    )


# ============================================================================
# PAGE 3 — CAMPAGNES MARKETING
# ============================================================================
elif page == "campaigns":
    page_header("campaigns")
    st.markdown('<div class="page-title">Campagnes marketing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Performance des campagnes par canal : taux de clic, taux de conversion, '
        "coût par clic (CPC) et coût par acquisition (CPA).</div>",
        unsafe_allow_html=True,
    )

    best_channel = marketing_kpi.loc[marketing_kpi["Conversion_Rate_pct"].idxmax(), "Channel"]
    avg_cpa = marketing_kpi["CPA"].mean()
    avg_roi = marketing_kpi["ROI_pct"].mean()

    render_kpi_row([
        kpi_card("Meilleur canal", best_channel, "taux de conversion le + élevé", "🏆", ACCENT_COLORS[0]),
        kpi_card("CPA moyen", f"{avg_cpa:.2f} €", "coût par acquisition", "💸", ACCENT_COLORS[3]),
        kpi_card("ROI moyen", f"{avg_roi:.0f}%", "retour sur investissement", "📈", ACCENT_COLORS[1]),
        kpi_card("Campagnes actives", f"{len(marketing_kpi)}", "sur la période", "📣", ACCENT_COLORS[2]),
    ])

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("CTR (%) par canal")
            fig = px.bar(marketing_kpi, x="Channel", y="CTR_pct", color="Channel",
                        template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS)
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, showlegend=False, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            section_header("Taux de conversion (%) par canal")
            fig = px.bar(marketing_kpi, x="Channel", y="Conversion_Rate_pct", color="Channel",
                        template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS)
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, showlegend=False, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section_header("Coût par acquisition (CPA, €)")
            fig = px.bar(marketing_kpi, x="Channel", y="CPA", color="Channel",
                        template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS)
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, showlegend=False, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        with st.container(border=True):
            section_header("Budget par canal (€)")
            fig = px.bar(marketing_kpi, x="Channel", y="Budget", color="Channel",
                        template=PLOTLY_TEMPLATE, color_discrete_sequence=ACCENT_COLORS)
            fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, showlegend=False, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Détail des campagnes")
        display_cols = {
            "Campaign_ID": "ID", "Channel": "Canal", "Budget": "Budget (€)",
            "Impressions": "Impressions", "Clicks": "Clics", "Conversions": "Conversions",
            "CTR_pct": "CTR (%)", "Conversion_Rate_pct": "Taux conv. (%)",
            "CPC": "CPC (€)", "CPA": "CPA (€)", "ROI_pct": "ROI (%)",
        }
        table = marketing_kpi[list(display_cols.keys())].rename(columns=display_cols)
        st.dataframe(table.round(2), use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 4 — PRÉDICTION DU CHURN
# ============================================================================
elif page == "churn":
    page_header("churn")
    st.markdown('<div class="page-title">Prédiction du churn</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Un client est considéré « actif » s\'il achète pendant la fenêtre future '
        "définie ci-dessous. Les modèles apprennent à partir de l'historique avant la date de coupure (features RFM).</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            cutoff_date = st.date_input("Date de coupure (fin de l'historique)", value=pd.to_datetime("2026-07-01"))
        with col_b:
            future_end = st.date_input("Fin de la fenêtre future à prédire", value=pd.to_datetime("2026-09-01"))

    if future_end <= cutoff_date:
        st.error("La fin de la fenêtre future doit être postérieure à la date de coupure.")
        st.stop()

    dataset = build_churn_dataset(sales, customers, str(cutoff_date), str(future_end))

    if dataset["target"].nunique() < 2:
        st.warning("Une seule classe présente dans la cible sur cette période — impossible d'entraîner un modèle. Modifiez les dates.")
        st.stop()

    models = train_churn_models(dataset)
    results = evaluate_models(models)

    render_kpi_row([
        kpi_card("Clients analysés", f"{len(dataset):,}".replace(",", " "), "avec historique avant coupure", "👥", ACCENT_COLORS[0]),
        kpi_card("Classe positive", f"{dataset['target'].mean():.1%}", "clients actifs sur la fenêtre future", "🎯", ACCENT_COLORS[3]),
        kpi_card("Accuracy — Random Forest", f"{results['Random Forest']['accuracy']:.1%}", "sur le jeu de test", "🌲", ACCENT_COLORS[1]),
        kpi_card("Accuracy — XGBoost", f"{results['XGBoost']['accuracy']:.1%}", "sur le jeu de test", "🚀", ACCENT_COLORS[2]),
    ])

    col1, col2 = st.columns(2)
    for col, name in zip([col1, col2], ["Random Forest", "XGBoost"]):
        cm = results[name]["confusion_matrix"]
        with col:
            with st.container(border=True):
                section_header(f"Matrice de confusion — {name}")
                fig = px.imshow(
                    cm, text_auto=True, color_continuous_scale=[[0, "#F6F7FB"], [1, PRIMARY_COLOR]],
                    labels=dict(x="Prédit", y="Réel", color="Nombre"),
                    x=["0", "1"], y=["0", "1"],
                )
                fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=300, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section_header("Courbe ROC")
            fig = go.Figure()
            for name, color in [("Random Forest", ACCENT_COLORS[1]), ("XGBoost", ACCENT_COLORS[2])]:
                r = results[name]
                fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                          name=f"{name} (AUC={r['roc_auc']:.3f})", line=dict(color=color, width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
                                      line=dict(dash="dash", color="#D1D5DB")))
            fig.update_layout(
                xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                template=PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=6, b=0), height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        with st.container(border=True):
            section_header("Courbe Précision-Rappel")
            fig = go.Figure()
            for name, color in [("Random Forest", ACCENT_COLORS[1]), ("XGBoost", ACCENT_COLORS[2])]:
                r = results[name]
                fig.add_trace(go.Scatter(x=r["recall"], y=r["precision"], mode="lines",
                                          name=f"{name} (PR-AUC={r['pr_auc']:.3f})", line=dict(color=color, width=2.5)))
            fig.update_layout(
                xaxis_title="Rappel", yaxis_title="Précision", template=PLOTLY_TEMPLATE,
                margin=dict(l=0, r=0, t=6, b=0), height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Importance des variables (Random Forest)")
        importances = pd.Series(models["rf_model"].feature_importances_, index=models["feature_names"])
        importances = importances.sort_values(ascending=False).head(15).reset_index()
        importances.columns = ["Variable", "Importance"]
        fig = px.bar(importances, x="Variable", y="Importance", template=PLOTLY_TEMPLATE,
                    color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("⚠️ Note sur la qualité du modèle"):
        st.markdown(
            "Sur ce jeu de données, la classe positive (clients actifs sur la fenêtre future) "
            "est très minoritaire : les modèles ont du mal à la détecter (rappel très faible, "
            "AUC-ROC proche de 0.5). Cela reflète surtout un manque de signal / de volume de données "
            "plutôt qu'un problème de code — à mentionner dans le rapport comme piste d'amélioration "
            "(plus d'historique, rééquilibrage de classes, autres features)."
        )