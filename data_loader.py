import pandas as pd

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

# Note: get_merged_data() function moved back to main.py to match original structure
