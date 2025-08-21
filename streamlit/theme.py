# theme.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st

def set_theme():
    st.markdown(
        """
        <style>
        /* Import Racing Sans One from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Racing+Sans+One&display=swap');

        /* App background (Anthropic-style) */
        .stApp {
            background-color: #F5F5F8;  /* Anthropic light background */
            color: #A288F6;             /* galactic purple text */
            font-family: 'Racing Sans One', sans-serif !important;
        }

        /* Headings */
        h1, h2, h3 {
            color: #C1A3FF;  /* lighter galactic glow for headings */
            font-family: 'Racing Sans One', sans-serif !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;  /* Anthropic-style sidebar */
            color: #A288F6;             /* galactic purple text */
            font-family: 'Racing Sans One', sans-serif !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #7E7E9B !important;
            color: #FFFFFF !important;
            font-family: 'Racing Sans One', sans-serif !important;
        }

        /* Other text elements */
        .css-1d391kg, .css-1v0mbdj.edgvbvh3 {
            font-family: 'Racing Sans One', sans-serif !important;
            color: #A288F6 !important;
        }

        /* Horizontal separators */
        hr {
            border-top: 1px solid #C1A3FF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
