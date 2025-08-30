# streamlit/eda_profiles.py
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
    "Young First-Timers":    "Young_First_Timers.jpg",
    "Engaged Re-contacts":   "Engaged_Re_contacts.jpg",
    "Older First-Timers":    "Older_First_Timers.jpg",
    "Senior Professionals":  "Senior_Professionals.jpg",
}

PERSONA_DESC = {
    "Young First-Timers":   "New contacts with high housing loan rates; difficult to convert.",
    "Engaged Re-contacts":  "Previously contacted, highly educated with top conversion rate.",
    "Older First-Timers":   "Affluent, older clients contacted for the first time.",
    "Senior Professionals": "Older, management-heavy profiles with decent response rate.",
}

CLUSTER_COLOR_MAP = {
    "Young First-Timers":   "#9ecae1",
    "Engaged Re-contacts":  "#fbb4ae",
    "Older First-Timers":   "#2171b5",
    "Senior Professionals": "#a7c957",
}

CLUSTER_ORDER = [
    "Young First-Timers",
    "Engaged Re-contacts",
    "Older First-Timers",
    "Senior Professionals",
]

# Palette Outcome (réutilisée dans les graphiques y=yes/no)
OUTCOME_COLOR_MAP = {
    "yes": "#66c2a5",
    "no":  "#fc8d62",
}

# Palette pour ROI / Insights
INSIGHT_COLOR_POS = "#66c2a5"  # vert
INSIGHT_COLOR_NEG = "#fc8d62"  # orange/rouge


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
            out["balance"],
            bins=[-np.inf, 20000, 60000, np.inf],
            labels=["Low", "Medium", "High"]
        )
    return out


# -------------------- Pipeline KMeans (Without duration) --------------------
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
        "job","marital","education","default","housing","loan",
        "contact","month","poutcome"
    ] if c in df.columns]

    if len(num_features) < 2 and len(cat_features) == 0:
        return None, [], []

    try:
        cat_ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    except TypeError:
        # compat scikit-learn < 1.2
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


# -------------------  Clusters Attribution -------------------
@st.cache_data(show_spinner="Running K-Means Clustering...")
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
        3: "Senior Professionals",
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
        color_discrete_map=OUTCOME_COLOR_MAP, title=title
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_yaxes(range=[0, 100], title="Percent")
    fig.update_layout(showlegend=False)
    return fig


def _fig_age_box(d: pd.DataFrame):
    """
    Age en histogramme (percent) groupé par y (palette Outcome).
    """
    if "age" not in d.columns or "y" not in d.columns:
        return go.Figure()
    fig = px.histogram(
        d, x="age", color="y",
        nbins=30, histnorm="percent", barmode="group",
        color_discrete_map=OUTCOME_COLOR_MAP,
        title="Age vs Target (y)"
    )
    fig.update_yaxes(title="Percent")
    return fig


def _fig_pie(d: pd.DataFrame, col: str, title: str):
    """
    Pie générique. Si valeurs yes/no -> palette Outcome.
    """
    if col not in d.columns:
        return go.Figure()
    vc = d[col].astype(str).value_counts().reset_index()
    vc.columns = [col, "count"]
    fig = px.pie(
        vc, names=col, values="count",
        color=col, color_discrete_map=OUTCOME_COLOR_MAP,
        hole=0.35, title=title
    )
    return fig


def _fig_top_jobs(d: pd.DataFrame, top_n=10, color_by_y: bool = True):
    """
    Top jobs (counts). Par défaut: split yes/no (barmode=group) palette Outcome.
    """
    if "job" not in d.columns:
        return go.Figure()

    vc = d["job"].value_counts().head(top_n)
    top_jobs = vc.index
    dd = d[d["job"].isin(top_jobs)].copy()

    if color_by_y and "y" in dd.columns:
        g = dd.groupby(["job", "y"]).size().reset_index(name="count")
        g["job"] = pd.Categorical(g["job"], categories=top_jobs, ordered=True)
        fig = px.bar(
            g, x="count", y="job", color="y",
            orientation="h", barmode="group",
            color_discrete_map=OUTCOME_COLOR_MAP,
            title=f"Top {top_n} Jobs — Counts by y"
        )
        fig.update_layout(xaxis_title="Count", yaxis_title="job")
        return fig

    # Fallback sans y
    plot_df = vc.sort_values(ascending=True).reset_index()
    plot_df.columns = ["job", "count"]
    fig = px.bar(
        plot_df, x="count", y="job", orientation="h",
        title=f"Top {top_n} Jobs", color_discrete_sequence=["#3fc1c9"]
    )
    return fig


def _fig_edu_percent(d: pd.DataFrame, color_by_y: bool = True):
    """
    Education en % par niveau. Si color_by_y: % par niveau (row-normalized).
    """
    if "education" not in d.columns:
        return go.Figure()

    if color_by_y and "y" in d.columns:
        g = d.groupby(["education", "y"]).size().reset_index(name="count")
        g["total_edu"] = g.groupby("education")["count"].transform("sum")
        g["percent"] = (g["count"] / g["total_edu"] * 100).round(1)

        fig = px.bar(
            g, x="percent", y="education", color="y",
            orientation="h", barmode="group",
            color_discrete_map=OUTCOME_COLOR_MAP,
            title="Education — % by y (row-normalized)"
        )
        fig.update_xaxes(range=[0, 100], title="Percent")
        fig.update_yaxes(title="education")
        return fig

    # Fallback sans y
    vc = (d["education"].value_counts(normalize=True) * 100).round(1)
    plot_df = vc.sort_values(ascending=True).reset_index()
    plot_df.columns = ["education", "percent"]
    fig = px.bar(
        plot_df, x="percent", y="education", orientation="h",
        title="Education — %", color_discrete_sequence=["#a1c45a"]
    )
    fig.update_xaxes(range=[0, 100], title="Percent")
    return fig


def _fig_month(d: pd.DataFrame):
    """
    Month en % par mois, split par y, groupé, palette Outcome.
    """
    if "month" not in d.columns:
        return go.Figure()

    if "y" not in d.columns:
        month_counts = d["month"].value_counts().sort_index()
        return px.bar(
            month_counts,
            x=month_counts.index,
            y=month_counts.values,
            title="Number of Contacts by Month",
            labels={"x": "Month", "y": "Contacts"},
            color=month_counts.values,
            color_continuous_scale="Blues"
        )

    counts = d.groupby(["month", "y"]).size().reset_index(name="count")
    totals = counts.groupby("month")["count"].transform("sum")
    counts["percent"] = (counts["count"] / totals * 100).round(1)

    month_order = None
    unique_months = counts["month"].astype(str).str.lower().unique().tolist()
    known = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    if all(m in known for m in [m.lower() for m in unique_months]):
        month_order = [m for m in known if m in [mm.lower() for mm in unique_months]]
        mapping = {m.lower(): m for m in counts["month"].astype(str).unique()}
        month_order = [mapping[m] for m in month_order]

    fig = px.bar(
        counts, x="month", y="percent", color="y",
        barmode="group",
        color_discrete_map=OUTCOME_COLOR_MAP,
        category_orders={"month": month_order} if month_order else None,
        title="Contacts by Month — % within month"
    )
    fig.update_yaxes(title="Percent", range=[0, 100])
    return fig


def _fig_compare_bars(dcomp: pd.DataFrame) -> go.Figure:
    if "y" not in dcomp.columns or "cluster_label" not in dcomp.columns:
        return go.Figure()
    counts = dcomp.groupby(["cluster_label", "y"]).size().reset_index(name="count")
    totals = counts.groupby("cluster_label")["count"].transform("sum")
    counts["percent"] = (100.0 * counts["count"] / totals).round(1)

    fig = px.bar(
        counts, x="cluster_label", y="percent", color="y",
        category_orders={"cluster_label": CLUSTER_ORDER},
        barmode="group", text="percent",
        title="Yes/No distribution by cluster (%)",
        color_discrete_map=OUTCOME_COLOR_MAP
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
        counts["cluster_label"], categories=CLUSTER_ORDER, ordered=True
    )
    counts = counts.sort_values("cluster_label")

    fig = px.pie(
        counts, names="cluster_label", values="percent",
        hole=0.35, title="Overall profile proportions (%)",
        color="cluster_label", color_discrete_map=CLUSTER_COLOR_MAP
    )
    return fig


# -------------------- ROI / INSIGHTS HELPERS (NEW) --------------------
def _add_age_bucket(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if "age" in out.columns and "age_bucket" not in out.columns:
        out["age_bucket"] = pd.cut(
            out["age"],
            bins=[-np.inf, 25, 35, 55, np.inf],
            labels=["<=25", "26-35", "36-55", "56+"]
        )
    return out

def _compute_lifts(d: pd.DataFrame, cols=None, min_count=100):
    """
    Compute simple % point lift: (P(yes|modality) - P(yes overall)) * 100
    Returns a tidy DF with columns: feature, value, count, p_yes, pp_lift
    """
    if "y" not in d.columns:
        return pd.DataFrame(columns=["feature","value","count","p_yes","pp_lift"])

    dd = d.copy()
    dd = _add_age_bucket(dd)

    candidate_cols = [
        "housing", "loan", "contact", "education", "marital", "month",
        "age_bucket"  # derived
    ]
    if cols:
        candidate_cols = [c for c in cols if c in dd.columns]
    else:
        candidate_cols = [c for c in candidate_cols if c in dd.columns]

    # overall yes-rate
    p_overall = (dd["y"].astype(str) == "yes").mean()

    rows = []
    for col in candidate_cols:
        tmp = dd[[col, "y"]].dropna()
        if tmp.empty:
            continue
        grp = tmp.groupby(col)["y"].apply(lambda s: (s.astype(str)=="yes").mean())
        cnt = tmp.groupby(col)["y"].size()
        for val in grp.index:
            count = int(cnt.loc[val])
            if count < min_count:
                continue
            p_yes = float(grp.loc[val])
            pp_lift = (p_yes - p_overall) * 100.0
            rows.append([col, str(val), count, p_yes*100.0, pp_lift])

    lifts = pd.DataFrame(rows, columns=["feature","value","count","p_yes","pp_lift"])
    if lifts.empty:
        return lifts
    lifts = lifts.sort_values("pp_lift", ascending=False)
    return lifts

# streamlit/eda_profiles.py
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
    "Young First-Timers":    "Young_First_Timers.jpg",
    "Engaged Re-contacts":   "Engaged_Re_contacts.jpg",
    "Older First-Timers":    "Older_First_Timers.jpg",
    "Senior Professionals":  "Senior_Professionals.jpg",
}

PERSONA_DESC = {
    "Young First-Timers":   "New contacts with high housing loan rates; difficult to convert.",
    "Engaged Re-contacts":  "Previously contacted, highly educated with top conversion rate.",
    "Older First-Timers":   "Affluent, older clients contacted for the first time.",
    "Senior Professionals": "Older, management-heavy profiles with decent response rate.",
}

CLUSTER_COLOR_MAP = {
    "Young First-Timers":   "#9ecae1",
    "Engaged Re-contacts":  "#fbb4ae",
    "Older First-Timers":   "#2171b5",
    "Senior Professionals": "#a7c957",
}

CLUSTER_ORDER = [
    "Young First-Timers",
    "Engaged Re-contacts",
    "Older First-Timers",
    "Senior Professionals",
]

# Palette Outcome (réutilisée dans les graphiques y=yes/no)
OUTCOME_COLOR_MAP = {
    "yes": "#66c2a5",
    "no":  "#fc8d62",
}

# Palette pour ROI / Insights
INSIGHT_COLOR_POS = "#66c2a5"  # vert
INSIGHT_COLOR_NEG = "#fc8d62"  # orange/rouge


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
            out["balance"],
            bins=[-np.inf, 20000, 60000, np.inf],
            labels=["Low", "Medium", "High"]
        )
    return out


# -------------------- Pipeline KMeans (Without duration) --------------------
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
        "job","marital","education","default","housing","loan",
        "contact","month","poutcome"
    ] if c in df.columns]

    if len(num_features) < 2 and len(cat_features) == 0:
        return None, [], []

    try:
        cat_ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    except TypeError:
        # compat scikit-learn < 1.2
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


# -------------------  Clusters Attribution -------------------
@st.cache_data(show_spinner="Running K-Means Clustering...")
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
        3: "Senior Professionals",
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
        color_discrete_map=OUTCOME_COLOR_MAP, title=title
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_yaxes(range=[0, 100], title="Percent")
    fig.update_layout(showlegend=False)
    return fig


def _fig_age_box(d: pd.DataFrame):
    """
    Age en histogramme (percent) groupé par y (palette Outcome).
    """
    if "age" not in d.columns or "y" not in d.columns:
        return go.Figure()
    fig = px.histogram(
        d, x="age", color="y",
        nbins=30, histnorm="percent", barmode="group",
        color_discrete_map=OUTCOME_COLOR_MAP,
        title="Age vs Target (y)"
    )
    fig.update_yaxes(title="Percent")
    return fig


def _fig_pie(d: pd.DataFrame, col: str, title: str):
    """
    Pie générique. Si valeurs yes/no -> palette Outcome.
    """
    if col not in d.columns:
        return go.Figure()
    vc = d[col].astype(str).value_counts().reset_index()
    vc.columns = [col, "count"]
    fig = px.pie(
        vc, names=col, values="count",
        color=col, color_discrete_map=OUTCOME_COLOR_MAP,
        hole=0.35, title=title
    )
    return fig


def _fig_top_jobs(d: pd.DataFrame, top_n=10, color_by_y: bool = True):
    """
    Top jobs (counts). Par défaut: split yes/no (barmode=group) palette Outcome.
    """
    if "job" not in d.columns:
        return go.Figure()

    vc = d["job"].value_counts().head(top_n)
    top_jobs = vc.index
    dd = d[d["job"].isin(top_jobs)].copy()

    if color_by_y and "y" in dd.columns:
        g = dd.groupby(["job", "y"]).size().reset_index(name="count")
        g["job"] = pd.Categorical(g["job"], categories=top_jobs, ordered=True)
        fig = px.bar(
            g, x="count", y="job", color="y",
            orientation="h", barmode="group",
            color_discrete_map=OUTCOME_COLOR_MAP,
            title=f"Top {top_n} Jobs — Counts by y"
        )
        fig.update_layout(xaxis_title="Count", yaxis_title="job")
        return fig

    # Fallback sans y
    plot_df = vc.sort_values(ascending=True).reset_index()
    plot_df.columns = ["job", "count"]
    fig = px.bar(
        plot_df, x="count", y="job", orientation="h",
        title=f"Top {top_n} Jobs", color_discrete_sequence=["#3fc1c9"]
    )
    return fig


def _fig_edu_percent(d: pd.DataFrame, color_by_y: bool = True):
    """
    Education en % par niveau. Si color_by_y: % par niveau (row-normalized).
    """
    if "education" not in d.columns:
        return go.Figure()

    if color_by_y and "y" in d.columns:
        g = d.groupby(["education", "y"]).size().reset_index(name="count")
        g["total_edu"] = g.groupby("education")["count"].transform("sum")
        g["percent"] = (g["count"] / g["total_edu"] * 100).round(1)

        fig = px.bar(
            g, x="percent", y="education", color="y",
            orientation="h", barmode="group",
            color_discrete_map=OUTCOME_COLOR_MAP,
            title="Education — % by y (row-normalized)"
        )
        fig.update_xaxes(range=[0, 100], title="Percent")
        fig.update_yaxes(title="education")
        return fig

    # Fallback sans y
    vc = (d["education"].value_counts(normalize=True) * 100).round(1)
    plot_df = vc.sort_values(ascending=True).reset_index()
    plot_df.columns = ["education", "percent"]
    fig = px.bar(
        plot_df, x="percent", y="education", orientation="h",
        title="Education — %", color_discrete_sequence=["#a1c45a"]
    )
    fig.update_xaxes(range=[0, 100], title="Percent")
    return fig


def _fig_month(d: pd.DataFrame):
    """
    Month en % par mois, split par y, groupé, palette Outcome.
    """
    if "month" not in d.columns:
        return go.Figure()

    if "y" not in d.columns:
        month_counts = d["month"].value_counts().sort_index()
        return px.bar(
            month_counts,
            x=month_counts.index,
            y=month_counts.values,
            title="Number of Contacts by Month",
            labels={"x": "Month", "y": "Contacts"},
            color=month_counts.values,
            color_continuous_scale="Blues"
        )

    counts = d.groupby(["month", "y"]).size().reset_index(name="count")
    totals = counts.groupby("month")["count"].transform("sum")
    counts["percent"] = (counts["count"] / totals * 100).round(1)

    month_order = None
    unique_months = counts["month"].astype(str).str.lower().unique().tolist()
    known = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    if all(m in known for m in [m.lower() for m in unique_months]):
        month_order = [m for m in known if m in [mm.lower() for mm in unique_months]]
        mapping = {m.lower(): m for m in counts["month"].astype(str).unique()}
        month_order = [mapping[m] for m in month_order]

    fig = px.bar(
        counts, x="month", y="percent", color="y",
        barmode="group",
        color_discrete_map=OUTCOME_COLOR_MAP,
        category_orders={"month": month_order} if month_order else None,
        title="Contacts by Month — % within month"
    )
    fig.update_yaxes(title="Percent", range=[0, 100])
    return fig


def _fig_compare_bars(dcomp: pd.DataFrame) -> go.Figure:
    if "y" not in dcomp.columns or "cluster_label" not in dcomp.columns:
        return go.Figure()
    counts = dcomp.groupby(["cluster_label", "y"]).size().reset_index(name="count")
    totals = counts.groupby("cluster_label")["count"].transform("sum")
    counts["percent"] = (100.0 * counts["count"] / totals).round(1)

    fig = px.bar(
        counts, x="cluster_label", y="percent", color="y",
        category_orders={"cluster_label": CLUSTER_ORDER},
        barmode="group", text="percent",
        title="Yes/No distribution by profile (%)",
        color_discrete_map=OUTCOME_COLOR_MAP
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
        counts["cluster_label"], categories=CLUSTER_ORDER, ordered=True
    )
    counts = counts.sort_values("cluster_label")

    fig = px.pie(
        counts, names="cluster_label", values="percent",
        hole=0.35, title="Overall cluster proportions (%)",
        color="cluster_label", color_discrete_map=CLUSTER_COLOR_MAP
    )
    return fig


# -------------------- ROI / INSIGHTS HELPERS (NEW) --------------------
def _add_age_bucket(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if "age" in out.columns and "age_bucket" not in out.columns:
        out["age_bucket"] = pd.cut(
            out["age"],
            bins=[-np.inf, 25, 35, 55, np.inf],
            labels=["<=25", "26-35", "36-55", "56+"]
        )
    return out

def _compute_lifts(d: pd.DataFrame, cols=None, min_count=100):
    """
    Compute simple % point lift: (P(yes|modality) - P(yes overall)) * 100
    Returns a tidy DF with columns: feature, value, count, p_yes, pp_lift
    """
    if "y" not in d.columns:
        return pd.DataFrame(columns=["feature","value","count","p_yes","pp_lift"])

    dd = d.copy()
    dd = _add_age_bucket(dd)

    candidate_cols = [
        "housing", "loan", "contact", "education", "marital", "month",
        "age_bucket"  # derived
    ]
    if cols:
        candidate_cols = [c for c in cols if c in dd.columns]
    else:
        candidate_cols = [c for c in candidate_cols if c in dd.columns]

    # overall yes-rate
    p_overall = (dd["y"].astype(str) == "yes").mean()

    rows = []
    for col in candidate_cols:
        tmp = dd[[col, "y"]].dropna()
        if tmp.empty:
            continue
        grp = tmp.groupby(col)["y"].apply(lambda s: (s.astype(str)=="yes").mean())
        cnt = tmp.groupby(col)["y"].size()
        for val in grp.index:
            count = int(cnt.loc[val])
            if count < min_count:
                continue
            p_yes = float(grp.loc[val])
            pp_lift = (p_yes - p_overall) * 100.0
            rows.append([col, str(val), count, p_yes*100.0, pp_lift])

    lifts = pd.DataFrame(rows, columns=["feature","value","count","p_yes","pp_lift"])
    if lifts.empty:
        return lifts
    lifts = lifts.sort_values("pp_lift", ascending=False)
    return lifts

def _fig_top_lifts(lifts: pd.DataFrame, top_k=8) -> go.Figure:
    if lifts.empty:
        return go.Figure()

    # pick top positive and negative to balance view
    pos = lifts[lifts["pp_lift"] > 0].head(top_k//2)
    neg = lifts[lifts["pp_lift"] < 0].tail(top_k//2)  # already sorted desc
    plot = pd.concat([pos, neg]).copy()

    plot["label"] = plot["feature"] + " = " + plot["value"]
    plot = plot.sort_values("pp_lift", ascending=True)  # for horizontal bar

    colors = [INSIGHT_COLOR_NEG if v < 0 else INSIGHT_COLOR_POS for v in plot["pp_lift"]]

    fig = px.bar(
        plot, x="pp_lift", y="label", orientation="h",
        color=plot["label"],  # ensures unique color per bar
        color_discrete_sequence=colors,
        title="Top drivers ( yes-rate vs overall, percentage points)"
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Lift (pp)",
        yaxis_title="",
        showlegend=False
    )
    return fig


# -------------------- PUBLIC ENTRY --------------------
def show_profiles(df: pd.DataFrame):
    st.title("Profiles Explorer")

    with st.expander("Show Profiles", expanded=True):
        cols = st.columns(2)
        for i, name in enumerate(CLUSTER_ORDER):
            with cols[i % 2]:
                img = IMAGE_MAP.get(name)
                p = ASSETS_DIR / img if img else None
                if p and p.exists():
                    st.image(str(p), caption=name, use_container_width=True)
                else:
                    st.info(f"Add image at: assets/{img}")
                st.write(PERSONA_DESC.get(name, ""))

    if df is None or df.empty:
        st.error("No data to display.")
        return

    # Build clusters (cached)
    df2 = _assign_clusters_farah(df)
    if "cluster_label" not in df2.columns:
        st.error("Could not create profiles. Please verify the dataset has the required columns.")
        return

    st.divider()
    # 🔴 CHANGED: add a 3rd mode
    mode = st.radio(
        "Select a view",
        ["Compare Profiles", "Single Profile", "Conversion Insights"],
        horizontal=True,
        index=0
    )

    labels = sorted(df2["cluster_label"].unique().tolist())

    # ---------------- Compare clusters ----------------
    if mode == "Compare Profiles":
        st.subheader("All Profiles")
        dcomp = df2.copy()
        st.plotly_chart(_fig_cluster_proportion(dcomp), use_container_width=True)
        st.plotly_chart(_fig_compare_bars(dcomp), use_container_width=True)
        return

    # ---------------- Single cluster ----------------
    if mode == "Single Profile":
        st.subheader("Profile Analysis")
        chosen = st.selectbox("Choose a cluster", labels, index=0)
        dsel = df2[df2["cluster_label"] == chosen].copy()
        if dsel.empty:
            st.info("No data for this selection.")
            return

        # ------ 7 tabs ------
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Outcome", "Age", "Loans", "Housing", "Jobs", "Education", "Month"]
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
            top_n = st.slider("Show top N jobs", 3, 20, 10)
            st.plotly_chart(_fig_top_jobs(dsel, top_n=top_n, color_by_y=True), use_container_width=True)

        with tab6:
            st.plotly_chart(_fig_edu_percent(dsel, color_by_y=True), use_container_width=True)

        with tab7:
            st.plotly_chart(_fig_month(dsel), use_container_width=True)
        return

    # ---------------- ROI / Insights (NEW) ----------------
    if mode == "Conversion Insights":
        st.subheader("Conversion Insights")

        scope = st.radio("Select a detailed view", ["Overall", "By cluster"], horizontal=True, index=0)

        if scope == "Overall":
            base = df2.copy()
            lifts = _compute_lifts(base, cols=None, min_count=100)
            if lifts.empty:
                st.info("Not enough data to compute insights.")
                return
            st.plotly_chart(_fig_top_lifts(lifts, top_k=8), use_container_width=True)

        else:
            chosen = st.selectbox("Choose a profile", labels, index=0)
            base = df2[df2["cluster_label"] == chosen].copy()
            lifts = _compute_lifts(base, cols=None, min_count=50)  # lower threshold inside cluster
            if lifts.empty:
                st.info("Not enough data in this profile to compute insights.")
                return
            st.plotly_chart(_fig_top_lifts(lifts, top_k=8), use_container_width=True)

        # Short, business-first narrative
        st.markdown(
            """
	- Green (right) = improves conversion (good signals)
	- Orange (left) = hurts conversion (risk signals)


            """
        )
