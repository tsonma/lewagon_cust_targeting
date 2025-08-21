import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from theme import set_theme
set_theme()

def show_profiles(data: pd.DataFrame):
    """
    Default: donut of profiles among y=yes (toggle to y=no).
    If a 'Filter by' variable is selected, show the appropriate chart
    (hist/box/stacked) for the selected Profile (All or one profile).
    """
    COLOR_MAP = {"yes": "#66c2a5", "no": "#fc8d62"}  # same palette as EDA
    SKEWED_NUM = {"balance", "duration", "campaign", "pdays", "previous"}

    # ---------- Preconditions ----------
    assert "y" in data.columns, "Expected target column 'y'."
    if "profile" not in data.columns:
        st.error("No 'profile' column found. Please add profiles (e.g., via KMeans) before using this page.")
        return

    # ---------- Header / Guide ----------
    st.markdown("""
    ### 👥 Profiles Explorer
    **Default view:** a donut (beigne) showing the distribution of **profiles** among subscribers (`y = yes`).
    - Toggle to switch to `y = no`
    - Select a **Profile** (All or a specific one)
    - Use **Filter by** to visualize a variable against the target (`y`) within the chosen profile
    """)

    # ---------- Controls ----------
    profile_opts = ["All"] + sorted(data["profile"].astype(str).unique())
    exclude = {"y", "y_bin", "cluster", "profile"}
    var_opts = ["— Default donut —"] + [c for c in data.columns if c not in exclude]

    c1, c2, c3 = st.columns([1, 1, 1])
    profile = c1.selectbox("Profile", profile_opts, index=0)
    var     = c2.selectbox("Filter by", var_opts, index=0)
    y_yes   = c3.toggle("Show y = yes (default)", value=True)

    # Subset by profile
    dplot = data if profile == "All" else data[data["profile"].astype(str) == str(profile)]
    if dplot.empty:
        st.info("No data for this selection.")
        return

    # ---------- DEFAULT DONUT ----------
    if var == "— Default donut —":
        y_val = "yes" if y_yes else "no"
        d0 = dplot[dplot["y"] == y_val]
        if d0.empty:
            st.info(f"No rows for y = {y_val} in the selected profile.")
            return

        # observed=True avoids the FutureWarning about pandas groupby default change
        counts = (
            d0.groupby("profile", observed=True)
              .size()
              .reset_index(name="count")
              .sort_values("count", ascending=False)
        )

        fig = px.pie(
            counts, values="count", names="profile",
            hole=0.5,
            title=f"Profiles among y = {y_val} — Profile scope: {profile}",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        return

    # ---------- VARIABLE VIEW ----------
    # Treat 'day' as categorical if chosen
    xcol = var
    if var == "day" and var in dplot.columns:
        dplot = dplot.assign(_x=dplot["day"].astype(str))
        xcol = "_x"

    is_numeric = pd.api.types.is_numeric_dtype(dplot[var]) and var != "day"
    title = f"{var.capitalize()} vs Target (y) — Profile: {profile}"

    if is_numeric:
        if var in SKEWED_NUM:
            fig = px.box(
                dplot, x="y", y=var, color="y",
                color_discrete_map=COLOR_MAP, points="outliers", title=title
            )
        else:
            fig = px.histogram(
                dplot, x=var, color="y",
                nbins=30, histnorm="percent", barmode="group",
                color_discrete_map=COLOR_MAP, title=title
            )
    else:
        fig = px.histogram(
            dplot, x=xcol, color="y",
            barmode="stack", barnorm="percent",
            color_discrete_map=COLOR_MAP, title=title
        )
        if xcol == var:
            fig.update_layout(xaxis_tickangle=-45)

    # Legend formatting and y-axis label for percent
    fig.for_each_trace(lambda t: t.update(
        name="Yes — They Subscribed" if t.name == "yes" else "No — They Didn't"
    ))
    fig.update_layout(legend_title_text="Subscription Status",
                      legend=dict(traceorder="reversed"))
    if any(getattr(t, "histnorm", None) == "percent" for t in fig.data) or \
       any(getattr(t, "barnorm", None) == "percent" for t in fig.data):
        fig.update_yaxes(title="Percent")

    st.plotly_chart(fig, use_container_width=True)
