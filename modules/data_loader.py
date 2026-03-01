"""
Shared data loader — cached so all modules share the same DataFrame.
"""
import pandas as pd
import streamlit as st
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "OmniFlow_D2D_India_Unified_5200.csv")


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Order_Date"])
    df = df[df["Order_Status"] != "Cancelled"].copy()
    df["YearMonth"] = df["Order_Date"].dt.to_period("M")
    return df
