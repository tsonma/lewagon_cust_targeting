import streamlit as st
import plotly.express as px
import pandas as pd

def show_eda(data):
    COLOR_MAP = {"yes": "#66c2a5", "no": "#fc8d62"}  # pleasant Set2-like colors

    # --- User guide ---
    st.markdown("""
    ### 📊 Interactive Variable Explorer
    Use the dropdowns below to explore how different variables relate to the subscription target (`y`).

    **Groups:**
    - **Demography**: Customer age, job type, marital status, and education.
    - **Finance & Credit**: Balance, housing loan, and personal loan status.
    - **Current contact**: Contact method, day, month, and last call duration.
    - **History**: Previous campaign results, number of contacts, and time since last contact.

    _The chart and explanation will update instantly when you change selections._
    """)

    # --- Dropdowns ---
    group = st.selectbox(
        "Select Variable Group:",
        ["Demography", "Finance & Credit", "Current contact", "History"]
    )

    group_vars = {
        "Demography": ["age", "job", "marital", "education"],
        "Finance & Credit": ["balance", "housing", "loan"],
        "Current contact": ["contact", "day", "month", "duration"],
        "History": ["campaign", "pdays", "previous", "poutcome"]
    }
    var = st.selectbox("Select Variable:", group_vars[group])

    # --- Variable type detection ---
    force_categorical = {"day"} if group == "Current contact" else set()
    is_numeric = (pd.api.types.is_numeric_dtype(data[var])) and (var not in force_categorical)
    title = f"{var.capitalize()} vs Target (y)"

    # --- Dynamic explanation ---
    def get_chart_explanation(group, var, is_numeric):
        if group == "Demography" and var == "age":
            return "This histogram shows the age distribution of customers, comparing subscribers (`y = yes`) and non-subscribers (`y = no`)."
        elif is_numeric and group != "Finance & Credit":
            return f"This histogram shows how `{var}` values are distributed for subscribers and non-subscribers."
        elif is_numeric and group == "Finance & Credit":
            return f"This boxplot compares `{var}` values between subscribers and non-subscribers, showing medians, quartiles, and outliers."
        else:
            return f"This stacked bar chart shows the proportion of subscribers and non-subscribers for each `{var}` category."

    # --- Chart logic ---
    if group == "Demography":
        if var == "age":
            fig = px.histogram(
                data, x="age", color="y", nbins=30, histnorm="percent", barmode="group",
                color_discrete_map=COLOR_MAP, title=title
            )
        else:
            fig = px.histogram(
                data, x=var, color="y", barmode="stack",
                color_discrete_map=COLOR_MAP, title=title
            )
            fig.update_layout(xaxis_tickangle=-45)

    elif group == "Finance & Credit":
        if is_numeric:
            fig = px.box(
                data, x="y", y=var, color="y",
                color_discrete_map=COLOR_MAP, points="outliers", title=title
            )
        else:
            fig = px.histogram(
                data, x=var, color="y", barmode="stack",
                color_discrete_map=COLOR_MAP, title=title
            )
            fig.update_layout(xaxis_tickangle=-30)

    elif group == "Current contact":
        if var == "duration":
            fig = px.histogram(
                data, x="duration", color="y",
                nbins=40, histnorm="percent", barmode="group",
                color_discrete_map=COLOR_MAP, title=title
            )
        else:
            x_series = data[var].astype(str) if var == "day" else data[var]
            fig = px.histogram(
                data.assign(_x=x_series), x="_x", color="y", barmode="stack",
                color_discrete_map=COLOR_MAP, title=title
            )
            fig.update_layout(xaxis_title=var, xaxis_tickangle=-45)

    else:  # History
        if var in {"campaign", "pdays", "previous"} and is_numeric:
            fig = px.histogram(
                data, x=var, color="y",
                nbins=30, histnorm="percent", barmode="group",
                color_discrete_map=COLOR_MAP, title=title
            )
        else:
            fig = px.histogram(
                data, x=var, color="y", barmode="stack",
                color_discrete_map=COLOR_MAP, title="Poutcome vs Target (y)"
            )
            fig.update_layout(xaxis_tickangle=-30)

    # --- Format legend ---
    fig.for_each_trace(lambda t: t.update(
        name="Yes — They Subscribed" if t.name == "yes" else "No — They Didn't"
    ))
    fig.update_layout(
        legend_title_text="Subscription Status",
        legend=dict(traceorder="reversed")
    )

    # --- Y-axis label if percentage ---
    if any(getattr(t, "histnorm", None) == "percent" for t in fig.data):
        fig.update_yaxes(title="Percent")

    # --- Show explanation & chart ---
    st.markdown(f"**What you’re seeing:** {get_chart_explanation(group, var, is_numeric)}")
    st.plotly_chart(fig, use_container_width=True)
