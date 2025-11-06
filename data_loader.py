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

@st.cache_data
def load_hh_data():
    """Load HH_DF.csv and map columns to match standard format - CACHED VERSION
    
    Column Mapping:
    - HH_ECN → CG_ECN
    - NEW_SEGMENT → CS_NEW_NS
    - Keep all other columns as-is
    """
    try:
        # Load HH_DF.csv
        hh_df = pd.read_csv('HH_DF.csv')
        
        # Column mapping
        column_mapping = {
            'HH_ECN': 'CG_ECN',
            'NEW_SEGMENT': 'CS_NEW_NS'
        }
        
        # Rename columns
        hh_df = hh_df.rename(columns=column_mapping)
        
        # Verify required columns exist
        required_columns = ['CG_ECN', 'CS_NEW_NS', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE', 
                          'BILLINGPOSTALCODE', 'DEPOSIT_BAL', 'CG_GROSS_SALES', 'BANK_REVENUE', 
                          'LON_NUM', 'LAT_NUM']
        
        missing_columns = [col for col in required_columns if col not in hh_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in HH_DF.csv: {missing_columns}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Clean HH data
        hh_df = clean_portfolio_data(hh_df)
        
        # Load branch data (needed for portfolio creation)
        _, _, branch_data = load_data()
        
        st.success(f"Loaded {len(hh_df):,} customers from HH_DF.csv for Q1 2026 Move")
        
        return hh_df, branch_data
        
    except FileNotFoundError:
        st.error("HH_DF.csv file not found. Please ensure the file exists in the application directory.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading HH_DF.csv: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def validate_hh_data(hh_df):
    """Validate HH_DF data quality"""
    issues = []
    
    # Check for missing ECNs
    if hh_df['CG_ECN'].isna().any():
        issues.append(f"{hh_df['CG_ECN'].isna().sum()} rows have missing ECNs")
    
    # Check for missing coordinates
    missing_coords = hh_df[hh_df['LAT_NUM'].isna() | hh_df['LON_NUM'].isna()]
    if not missing_coords.empty:
        issues.append(f"{len(missing_coords)} rows have missing coordinates")
    
    # Check for duplicate ECNs
    duplicates = hh_df[hh_df.duplicated(subset=['CG_ECN'], keep=False)]
    if not duplicates.empty:
        issues.append(f"{len(duplicates)} duplicate ECNs found")
    
    if issues:
        st.warning("Data quality issues detected:\n" + "\n".join([f"- {issue}" for issue in issues]))
    
    return len(issues) == 0
