# streamlit/eda_profiles.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# -------------------- assets --------------------
ASSETS_DIR = Path(__file__).parent / "assets"
IMAGE_MAP = {
    "Young First-Timers":             "Young_First_Timers.jpg",
    "Engaged Re-contacts":            "Engaged_Re_contacts.jpg",
    "Older First-Timers":             "Older_First_Timers.jpg",
    "Over-Contacted Non-Responders":  "Over_Contacted_Non_Responders.jpg",
}
PERSONA_DESC = {
    "Young First-Timers": "Younger or new to campaigns; digital-first and curious.",
    "Engaged Re-contacts": "Previously contacted and engaged; follow-ups work well.",
    "Older First-Timers": "Older audience contacted for the first time; prefer clarity/trust.",
    "Over-Contacted Non-Responders": "Fatigued by many contacts; low conversion.",
}
CLUSTER_COLOR_MAP = {
    "Young First-Timers": "#9ecae1",
    "Older First-Timers": "#2171b5",
    "Engaged Re-contacts": "#fbb4ae",
    "Over-Contacted Non-Responders": "#e41a1c",
}
CLUSTER_ORDER = [
    "Young First-Timers",
    "Older First-Timers",
    "Engaged Re-contacts",
    "Over-Contacted Non-Responders",
]

# -------------------- helpers --------------------
def _derive_helper_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "y" in out.columns and out["y"].dtype != object:
        out["y"] = out["y"].astype(str)

    if "pdays" in out.columns and "No_Previous_contact" not in out.columns:
        out["No_Previous_contact"] = (out["pdays"] < 0).astype(int)
    if "previous" in out.columns and "Not_contacted_for_prior_campaign" not in out.columns:
        out["Not_contacted_for_prior_campaign"] = (out["previous"] == 0).astype(int)
    if ("pdays" in out.columns) and ("previous" in out.columns) and (
        "Never_contacted_before_No_Camapign_no_contact" not in out.columns
    ):
        out["Never_contacted_before_No_Camapign_no_contact"] = (
            (out["pdays"] < 0) & (out["previous"] == 0)
        ).astype(int)

    if "yearly_balance" not in out.columns and "balance" in out.columns:
        out["yearly_balance"] = pd.cut(
            out["balance"], bins=[-np.inf, 20000, 60000, np.inf], labels=["Low", "Medium", "High"]
        )
    return out

@st.cache_resource(show_spinner=False)
def _fit_kmeans_pipeline(df: pd.DataFrame):
    df = _derive_helper_cols(df)

    num_features = [c for c in [
        "age", "balance", "day", "campaign", "pdays", "previous",
        "No_Previous_contact",
        "Not_contacted_for_prior_campaign",
        "Never_contacted_before_No_Camapign_no_contact",
    ] if c in df.columns]

    cat_features = [c for c in [
        "job","marital","education","default","housing","loan","contact","month","poutcome"
    ] if c in df.columns]

    if len(num_features) < 2 and len(cat_features) == 0:
        return None, [], []

    # scikit-learn 1.2+ uses 'sparse_output'; older uses 'sparse'
    try:
        cat_ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    except TypeError:
        cat_ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", cat_ohe, cat_features),
        ],
        remainder="drop"
    )

    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    pipe = Pipeline(steps=[("preprocessor", pre), ("clusterer", kmeans)])
    return pipe, num_features, cat_features

def _assign_clusters_farah(df: pd.DataFrame) -> pd.DataFrame:
    df = _derive_helper_cols(df)
    if "cluster_label" in df.columns:
        return df

    pipe, num_features, cat_features = _fit_kmeans_pipeline(df)
    if pipe is None:
        st.error("Not enough columns to build clusters. Please ensure the dataset has the expected features.")
        return df

    use_cols = num_features + cat_features
    X = df[use_cols].copy()
    for c in num_features:
        X[c] = X[c].fillna(X[c].median())

    df["cluster"] = pipe.fit_predict(X)
    label_map = {
        0: "Young First-Timers",
        1: "Engaged Re-contacts",
        2: "Older First-Timers",
        3: "Over-Contacted Non-Responders",
    }
    df["cluster_label"] = df["cluster"].map(label_map).astype(str)
    return df

# -------------------- chart builders --------------------
def _fig_outcome(d: pd.DataFrame, title="Subscription Outcome (y)"):
    if "y" not in d.columns:
        return go.Figure()
    vc = d["y"].value_counts(normalize=True).reindex(["no", "yes"], fill_value=0) * 100
    plot_df = pd.DataFrame({"y": vc.index, "percent": vc.values})
    fig = px.bar(
        plot_df, x="y", y="percent", text="percent", color="y",
        color_discrete_map={"yes":"#66c2a5", "no":"#fc8d62"},
        title=title
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_yaxes(range=[0, 100], title="Percent")
    fig.update_layout(showlegend=False)
    return fig

def _fig_age_box(d: pd.DataFrame):
    if "age" not in d.columns:
        return go.Figure()
    return px.box(d, y="age", points=False, title="Age Distribution", color_discrete_sequence=["#7db5ff"])

def _fig_pie(d: pd.DataFrame, col: str, title: str):
    if col not in d.columns:
        return go.Figure()
    vc = d[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    return px.pie(vc, names=col, values="count", hole=0.35, title=title)

def _fig_top_jobs(d: pd.DataFrame, top_n=10):
    if "job" not in d.columns:
        return go.Figure()
    vc = d["job"].value_counts().head(top_n)
    plot_df = vc.sort_values(ascending=True).reset_index()
    plot_df.columns = ["job", "count"]
    fig = px.bar(
        plot_df,
        x="count", y="job", orientation="h",
        title=f"Top {top_n} Jobs", color_discrete_sequence=["#3fc1c9"]
    )
    return fig

def _fig_edu_percent(d: pd.DataFrame):
    if "education" not in d.columns:
        return go.Figure()
    vc = (d["education"].value_counts(normalize=True) * 100).round(1)
    plot_df = vc.sort_values(ascending=True).reset_index()
    plot_df.columns = ["education", "percent"]
    fig = px.bar(
        plot_df,
        x="percent", y="education", orientation="h",
        title="Education — %", color_discrete_sequence=["#a1c45a"]
    )
    fig.update_xaxes(range=[0, 100], title="Percent")
    return fig

def _fig_month(d: pd.DataFrame):
    if "month" not in d.columns:
        return go.Figure()
    month_counts = d["month"].value_counts().sort_index()
    fig = px.bar(
        month_counts,
        x=month_counts.index,
        y=month_counts.values,
        title="Number of Contacts by Month",
        labels={"x": "Month", "y": "Contacts"},
        color=month_counts.values,
        color_continuous_scale="Blues"
    )
    return fig

def _fig_compare_bars(dcomp: pd.DataFrame) -> go.Figure:
    if "y" not in dcomp.columns or "cluster_label" not in dcomp.columns:
        return go.Figure()
    counts = dcomp.groupby(["cluster_label", "y"]).size().reset_index(name="count")
    totals = counts.groupby("cluster_label")["count"].transform("sum")
    counts["percent"] = (100.0 * counts["count"] / totals).round(1)

    fig = px.bar(
        counts,
        x="cluster_label",
        y="percent",
        color="y",
        category_orders={"cluster_label": CLUSTER_ORDER},
        barmode="group",
        text="percent",
        title="Yes/No distribution by cluster (%)",
        color_discrete_map={"yes": "#66c2a5", "no": "#fc8d62"}
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_yaxes(range=[0, 100], title="Percent")
    fig.update_layout(legend_title="Outcome")
    return fig

def _fig_cluster_proportion(d: pd.DataFrame) -> go.Figure:
    if "cluster_label" not in d.columns:
        return go.Figure()
    counts = d["cluster_label"].value_counts(normalize=True).reset_index()
    counts.columns = ["cluster_label", "percent"]
    counts["percent"] = (counts["percent"] * 100).round(1)

    counts["cluster_label"] = pd.Categorical(
        counts["cluster_label"],
        categories=CLUSTER_ORDER,
        ordered=True
    )
    counts = counts.sort_values("cluster_label")

    fig = px.pie(
        counts,
        names="cluster_label",
        values="percent",
        hole=0.35,
        title="Overall cluster proportions (%)",
        color="cluster_label",
        color_discrete_map=CLUSTER_COLOR_MAP
    )
    return fig  # return required

def _fig_compare_sunburst(d: pd.DataFrame) -> go.Figure:
    if "cluster_label" not in d.columns or "y" not in d.columns:
        return go.Figure()
    counts = d.groupby(["cluster_label", "y"], as_index=False).size()
    totals = counts.groupby("cluster_label")["size"].transform("sum")
    counts["percent"] = (100.0 * counts["size"] / totals).round(1)

    labels, parents, values = [], [], []
    root = "All clusters"
    labels.append(root); parents.append(""); values.append(0)

    for cl in CLUSTER_ORDER:
        if cl not in counts["cluster_label"].unique():
            continue
        labels.append(cl); parents.append(root); values.append(100)
        sub = counts[counts["cluster_label"] == cl]
        for _, row in sub.iterrows():
            labels.append(row["y"])
            parents.append(cl)
            values.append(float(row["percent"]))

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        maxdepth=2,
        insidetextorientation="radial",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Compare clusters — y (yes/no) % per cluster",
                      margin=dict(t=60, l=0, r=0, b=0))
    return fig

# -------------------- PUBLIC ENTRY --------------------
def show_profiles(df: pd.DataFrame):
    st.title("👥 Profiles Explorer")

    # Personas (images avec use_container_width)
    with st.expander("Show Profiles", expanded=True):
        cols = st.columns(2)
        for i, name in enumerate([
            "Young First-Timers",
            "Engaged Re-contacts",
            "Older First-Timers",
            "Over-Contacted Non-Responders",
        ]):
            with cols[i % 2]:
                img = IMAGE_MAP.get(name)
                p = ASSETS_DIR / img if img else None
                if p and p.exists():
                    st.image(str(p), caption=name, use_container_width=True)
                else:
                    st.info(f"Add image at: assets/{img}")
                st.write(PERSONA_DESC.get(name, ""))

    # Ensure clusters
    if df is None or df.empty:
        st.error("No data to display.")
        return

    df2 = _assign_clusters_farah(df)
    if "cluster_label" not in df2.columns:
        st.error("Could not create clusters. Please verify the dataset has the required columns.")
        return

    st.divider()

    # -------- Controls --------
    mode = st.radio("Mode", ["Compare clusters", "Single cluster"], horizontal=True, index=0)
    labels = sorted(df2["cluster_label"].unique().tolist())

    if mode == "Compare clusters":
        st.subheader("Cluster comparison (all 4 shown)")
        dcomp = df2.copy()
        st.plotly_chart(_fig_cluster_proportion(dcomp), use_container_width=True)
        st.plotly_chart(_fig_compare_bars(dcomp), use_container_width=True)
        # Optionnel :
        # st.plotly_chart(_fig_compare_sunburst(dcomp), use_container_width=True)
        return

    # ----- Single cluster mode -----
    chosen = st.selectbox("Choose a cluster", labels, index=0)
    dsel = df2[df2["cluster_label"] == chosen].copy()
    if dsel.empty:
        st.info("No data for this selection.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Outcome", "Age", "Loans", "Housing", "Jobs & Education", "Month"]
    )
    with tab1:
        st.plotly_chart(_fig_outcome(dsel), use_container_width=True)
    with tab2:
        st.plotly_chart(_fig_age_box(dsel), use_container_width=True)
    with tab3:
        st.plotly_chart(_fig_pie(dsel, "loan", "Loan Status"), use_container_width=True)
    with tab4:
        st.plotly_chart(_fig_pie(dsel, "housing", "Housing Loan Status"), use_container_width=True)
    with tab5:
        top_n = st.slider("Show top N jobs", 3, 15, 10)
        st.plotly_chart(_fig_top_jobs(dsel, top_n=top_n), use_container_width=True)
        st.plotly_chart(_fig_edu_percent(dsel), use_container_width=True)
    with tab6:
        st.plotly_chart(_fig_month(dsel), use_container_width=True)
