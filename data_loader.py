import streamlit as st
import pandas as pd
from utils import merge_dfs, clean_portfolio_data

def load_data():
    """Load customer, banker, and branch data from files"""
    # This function should contain your existing data loading logic
    # Replace with your actual data loading implementation
    
    # Example implementation - replace with your actual file paths and loading logic
    try:
        customer_data = pd.read_csv('customer_data.csv')
        banker_data = pd.read_csv('banker_data.csv') 
        branch_data = pd.read_csv('branch_data.csv')
        
        return customer_data, banker_data, branch_data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def clean_initial_data(customer_data):
    """Clean initial data by removing Unassigned/Unmanaged rows for customers who also have In-Market/Centralized assignments"""
    
    if customer_data.empty:
        return customer_data
    
    original_count = len(customer_data)
    
    # Step 1: Priority Conflict Resolution
    # Find customers (ECNs) who have In-Market/Centralized assignments
    priority_customers = customer_data[
        customer_data['TYPE'].str.lower().str.strip().isin(['in-market', 'inmarket', 'centralized'])
    ]['CG_ECN'].unique()
    
    # Remove Unassigned/Unmanaged rows for these priority customers
    mask_to_remove = (
        customer_data['CG_ECN'].isin(priority_customers) & 
        customer_data['TYPE'].str.lower().str.strip().isin(['unassigned', 'unmanaged'])
    )
    
    cleaned_data = customer_data[~mask_to_remove].copy()
    priority_removed = original_count - len(cleaned_data)
    
    # Step 2: Comprehensive deduplication using new function
    final_data = clean_portfolio_data(cleaned_data)
    duplicate_removed = len(cleaned_data) - len(final_data)
    
    # Log cleanup results
    total_removed = original_count - len(final_data)
    if total_removed > 0:
        print(f"Data cleanup: Removed {priority_removed} priority conflicts and {duplicate_removed} duplicate ECNs. Total removed: {total_removed}")
    
    return final_data

@st.cache_data
def get_merged_data():
    """Load and merge all data with initial cleanup - CACHED VERSION"""
    customer_data, banker_data, branch_data = load_data()
    
    # Initial data cleanup - remove conflicting portfolio assignments
    customer_data = clean_initial_data(customer_data)
    
    data = merge_dfs(customer_data, banker_data, branch_data)
    return customer_data, banker_data, branch_data, data
