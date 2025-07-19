import pandas as pd
import streamlit as st
from utils import merge_dfs

@st.cache_data
def load_data():
    """Load data from local CSV files"""
    customer_data = pd.read_csv("customer_data.csv")
    banker_data = pd.read_csv("banker_data.csv")
    branch_data = pd.read_csv("branch_data.csv")
    return customer_data, banker_data, branch_data

def get_merged_data():
    """Load and merge all data"""
    customer_data, banker_data, branch_data = load_data()
    data = merge_dfs(customer_data, banker_data, branch_data)
    return customer_data, banker_data, branch_data, data
