import pandas as pd
import streamlit as st
from utils import clean_portfolio_data

@st.cache_data
def load_data():
    """Load customer, banker, and branch data from CSV files"""
    try:
        customer_data = pd.read_csv('customer_data.csv')
        banker_data = pd.read_csv('banker_data.csv')
        branch_data = pd.read_csv('branch_data.csv')
        return customer_data, banker_data, branch_data
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error reading CSV files: {e}")
        st.stop()

def clean_initial_data(customer_data):
    """Clean initial customer data by removing conflicting portfolio assignments"""
    # Remove customers assigned to multiple portfolios
    duplicate_mask = customer_data.duplicated(subset=['CG_ECN'], keep=False)
    duplicates = customer_data[duplicate_mask]
    
    if len(duplicates) > 0:
        unique_customers = duplicates['CG_ECN'].nunique()
        total_removed = len(duplicates)
        
        # Keep only first occurrence of each customer
        customer_data = customer_data.drop_duplicates(subset=['CG_ECN'], keep='first')
        
        st.info(f"ℹ️ Data Cleanup: Removed {total_removed} conflicting portfolio assignments for {unique_customers} customers. Kept first assignment for each.")
    
    return customer_data

def merge_dfs(customer_data, banker_data, branch_data):
    """Merge customer, banker, and branch dataframes"""
    # Merge customer and banker data
    data = customer_data.merge(
        banker_data[['PORT_CODE', 'EM', 'BANKER_NAME', 'AU']], 
        left_on='CG_PORTFOLIO_CD', 
        right_on='PORT_CODE', 
        how='left'
    )
    
    # Merge with branch data
    data = data.merge(
        branch_data[['AU', 'LAT', 'LON', 'BRANCH_NAME']], 
        on='AU', 
        how='left',
        suffixes=('', '_BRANCH')
    )
    
    return data

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
        
        # Verify ORIGINAL columns exist in HH_DF.csv
        required_original_columns = ['HH_ECN', 'NEW_SEGMENT', 'BILLINGSTREET', 'BILLINGCITY', 
                                    'BILLINGSTATE', 'BILLINGPOSTALCODE', 'DEPOSIT_BAL', 
                                    'CG_GROSS_SALES', 'BANK_REVENUE', 'LON_NUM', 'LAT_NUM']
        
        missing_columns = [col for col in required_original_columns if col not in hh_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in HH_DF.csv: {missing_columns}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Column mapping to match customer_data.csv format
        column_mapping = {
            'HH_ECN': 'CG_ECN',
            'NEW_SEGMENT': 'CS_NEW_NS'
        }
        
        # Rename columns
        hh_df = hh_df.rename(columns=column_mapping)
        
        # Simple HH data cleanup - NO TYPE column needed
        # Remove duplicates based on CG_ECN
        hh_df = hh_df.drop_duplicates(subset=['CG_ECN'], keep='first')
        
        # Remove rows with missing ECN or coordinates
        hh_df = hh_df.dropna(subset=['CG_ECN', 'LAT_NUM', 'LON_NUM'])
        
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
