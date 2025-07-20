# ===========================
# COMPLETE PORTFOLIO CREATION TOOL
# With Portfolio Assignment + Portfolio Mapping
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import math
from math import sin, cos, atan2, radians, sqrt
import plotly.graph_objects as go

# ===========================
# PORTFOLIO MAPPING IMPORTS
# ===========================
try:
    from portfolio_mapping import (
        enhanced_customer_au_assignment_with_streamlit_progress,
        convert_mapping_results_to_portfolio_format
    )
    MAPPING_AVAILABLE = True
except ImportError:
    MAPPING_AVAILABLE = False
    st.warning("Portfolio Mapping module not found. Only Portfolio Assignment will be available.")

# ===========================
# UTILITY FUNCTIONS
# ===========================

def haversine_distance(clat, clon, blat, blon):
    """Calculate distance between two points using Haversine formula"""
    if math.isnan(clat) or math.isnan(clon) or math.isnan(blat) or math.isnan(blon):
        return 0
        
    delta_lat = radians(clat - blat)
    delta_lon = radians(clon - blon)
    
    a = sin(delta_lat/2)**2 + cos(radians(clat))*cos(radians(blat))*sin(delta_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    distance = 3959*c  # Earth's radius in miles
    return distance

def merge_dfs(customer_data, banker_data, branch_data):
    """Merge customer, banker, and branch dataframes"""
    customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    final_table = customer_data.merge(banker_data, on="PORT_CODE", how="left")
    final_table.fillna(0, inplace=True)
    return final_table

# ===========================
# PORTFOLIO LOGIC (SIMPLIFIED)
# ===========================

def filter_customers_for_au(customer_data, banker_data, selected_au, branch_data, role, cust_state, cust_portcd, max_dist, min_rev, min_deposit):
    """Filter customers for a specific AU based on criteria"""
    
    # Get AU data
    AU_row = branch_data[branch_data['AU'] == int(selected_au)].iloc[0]
    AU_lat = AU_row['BRANCH_LAT_NUM']
    AU_lon = AU_row['BRANCH_LON_NUM']
    
    # Filter customers by distance box
    box_lat = max_dist/69
    box_lon = max_dist/ (69 * np.cos(np.radians(AU_lat)))
    
    customer_data_boxed = customer_data[(customer_data['LAT_NUM'] >= AU_lat - box_lat) &
                                        (customer_data['LAT_NUM'] <= AU_lat + box_lat) &
                                        (customer_data['LON_NUM'] <= AU_lon + box_lon) &
                                        (customer_data['LON_NUM'] >= AU_lon - box_lon)]
    
    # Calculate distances
    customer_data_boxed['Distance'] = customer_data_boxed.apply(
        lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], AU_lat, AU_lon), axis=1
    )
    
    customer_data_boxed = customer_data_boxed.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    filtered_data = customer_data_boxed.merge(banker_data, on="PORT_CODE", how='left')
    
    # Apply filters
    if role is None or (role is not None and not any(r.lower().strip() == 'centralized' for r in role)):
        filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
    
    if cust_state is not None:
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
    
    if role is not None or cust_portcd is not None:
        role_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        portfolio_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        
        if role is not None:
            filtered_data['TYPE_CLEAN'] = filtered_data['TYPE'].fillna('').str.strip().str.lower()
            role_clean = [r.strip().lower() for r in role]
            role_condition = filtered_data['TYPE_CLEAN'].isin(role_clean)
            filtered_data = filtered_data.drop('TYPE_CLEAN', axis=1)
        
        if cust_portcd is not None:
            portfolio_condition = filtered_data['PORT_CODE'].isin(cust_portcd)
        
        combined_condition = role_condition | portfolio_condition
        filtered_data = filtered_data[combined_condition]
    
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    # Add AU information
    filtered_data['AU'] = selected_au
    filtered_data['BRANCH_LAT_NUM'] = AU_lat
    filtered_data['BRANCH_LON_NUM'] = AU_lon
    
    return filtered_data, AU_row

def apply_portfolio_selection_changes(portfolios_created, portfolio_controls, selected_aus, branch_data):
    """Apply the selection changes from portfolio controls to filter customers"""
    
    updated_portfolios = {}
    
    for au_id in selected_aus:
        if au_id not in portfolios_created or au_id not in portfolio_controls:
            continue
            
        original_data = portfolios_created[au_id].copy()
        control_data = portfolio_controls[au_id]
        
        selected_customers = []
        
        # Process each portfolio selection
        for _, row in control_data.iterrows():
            portfolio_id = row['Portfolio ID']
            select_count = int(row['Select'])
            include = bool(row.get('Include', False))  # Default to False (not included)
            
            # Skip if not included or select count is 0
            if not include or select_count <= 0:
                continue
                
            if portfolio_id == 'UNMANAGED':
                # Handle unmanaged customers
                unmanaged_customers = original_data[
                    (original_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                    (original_data['PORT_CODE'].isna())
                ].copy()
                
                if not unmanaged_customers.empty:
                    # Sort by distance (closest first) and take the requested count
                    unmanaged_sorted = unmanaged_customers.sort_values('Distance').head(select_count)
                    selected_customers.append(unmanaged_sorted)
                    
            elif 'INMARKET' in portfolio_id or 'CENTRALIZED' in portfolio_id:
                # Handle mapping results (INMARKET_123, CENTRALIZED_123)
                portfolio_type = portfolio_id.split('_')[0]  # Get INMARKET or CENTRALIZED
                type_customers = original_data[original_data['TYPE'] == portfolio_type].copy()
                
                if not type_customers.empty:
                    # Sort by distance and take requested count
                    type_sorted = type_customers.sort_values('Distance').head(select_count)
                    selected_customers.append(type_sorted)
                    
            else:
                # Handle regular portfolios
                portfolio_customers = original_data[original_data['PORT_CODE'] == portfolio_id].copy()
                
                if not portfolio_customers.empty:
                    # Sort by distance (closest first) and take the requested count
                    portfolio_sorted = portfolio_customers.sort_values('Distance').head(select_count)
                    selected_customers.append(portfolio_sorted)
        
        # Combine all selected customers for this AU
        if selected_customers:
            au_final_customers = pd.concat(selected_customers, ignore_index=True)
            # Remove duplicates if any
            au_final_customers = au_final_customers.drop_duplicates(subset=['CG_ECN'], keep='first')
            updated_portfolios[au_id] = au_final_customers
        else:
            # No customers selected for this AU - create empty DataFrame
            updated_portfolios[au_id] = pd.DataFrame()
    
    return updated_portfolios

# ===========================
# MAP VISUALIZATION
# ===========================

def create_combined_map(all_portfolios, branch_data):
    """Create a combined map showing all portfolios with different colors"""
    
    if not all_portfolios:
        return None
    
    fig = go.Figure()
    
    portfolio_colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightblue', 
                       'pink', 'darkgreen', 'brown', 'gray', 'cyan', 'magenta', 'yellow', 'red', 'lime']
    
    au_locations = set()
    for portfolio_id, df in all_portfolios.items():
        if not df.empty:
            au_id = df['AU'].iloc[0]
            au_row = branch_data[branch_data['AU'] == au_id]
            if not au_row.empty:
                au_lat = au_row.iloc[0]['BRANCH_LAT_NUM']
                au_lon = au_row.iloc[0]['BRANCH_LON_NUM']
                au_locations.add((au_id, au_lat, au_lon))
    
    for portfolio_idx, (portfolio_id, df) in enumerate(all_portfolios.items()):
        if df.empty:
            continue
        
        au_id = portfolio_id.split('_')[1] if '_' in portfolio_id else portfolio_id
        color = portfolio_colors[portfolio_idx % len(portfolio_colors)]
        
        hover_text = []
        for _, customer in df.iterrows():
            hover_text.append(f"""
            <b>{customer.get('CG_ECN', 'N/A')}</b><br>
            AU Portfolio: {au_id}<br>
            Portfolio ID: {customer.get('PORT_CODE', 'N/A')}<br>
            Distance: {customer.get('Distance', 0):.1f} miles<br>
            Revenue: ${customer.get('BANK_REVENUE', 0):,.0f}<br>
            Deposit: ${customer.get('DEPOSIT_BAL', 0):,.0f}<br>
            State: {customer.get('BILLINGSTATE', 'N/A')}<br>
            Type: {customer.get('TYPE', 'N/A')}
            """)
        
        fig.add_trace(go.Scattermapbox(
            lat=df['LAT_NUM'],
            lon=df['LON_NUM'],
            mode='markers',
            marker=dict(size=8, color=color, symbol='circle'),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            name=f"AU {au_id} Portfolio ({len(df)} customers)",
            showlegend=True
        ))
    
    for au_id, au_lat, au_lon in au_locations:
        au_details = branch_data[branch_data['AU'] == au_id]
        au_name = au_details['CITY'].iloc[0] if not au_details.empty else f"AU {au_id}"
        
        fig.add_trace(go.Scattermapbox(
            lat=[au_lat],
            lon=[au_lon],
            mode='markers',
            marker=dict(size=12, color='black', symbol='circle'),
            text=f"AU {au_id}",
            hovertemplate=f"""
            <b>AU {au_id}</b><br>
            Location: {au_name}<br>
            Coordinates: {au_lat:.4f}, {au_lon:.4f}
            <extra></extra>
            """,
            name=f"AU {au_id}",
            showlegend=True
        ))
    
    all_lats = []
    all_lons = []
    for portfolio_id, df in all_portfolios.items():
        if not df.empty:
            all_lats.extend(df['LAT_NUM'].tolist())
            all_lons.extend(df['LON_NUM'].tolist())
    
    if all_lats:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        zoom = 6
    else:
        center_lat = 39.8283
        center_lon = -98.5795
        zoom = 4
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

# ===========================
# DATA LOADING
# ===========================

@st.cache_data
def load_data():
    """Load data from local CSV files"""
    customer_data = pd.read_csv("customer_data.csv")
    banker_data = pd.read_csv("banker_data.csv")
    branch_data = pd.read_csv("branch_data.csv")
    return customer_data, banker_data, branch_data

# ===========================
# STREAMLIT APP
# ===========================

def main():
    """Main application function"""
    st.set_page_config("Portfolio Creation Tool", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        [data-testid="stHeader"] {
            background-color: rgb(215, 30, 40) !important;
        }
        div[data-testid="column"] button[kind="secondary"] {
            background: none !important;
            border: none !important;
            padding: 0 !important;
            color: #1f77b4 !important;
            text-decoration: underline !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            box-shadow: none !important;
            height: auto !important;
            margin-top: 0.5rem !important;
        }
        div[data-testid="column"] button[kind="secondary"]:hover {
            color: #0d47a1 !important;
            background: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add logo
    try:
        st.logo("logo.svg")
    except:
        try:
            st.logo("logo.png")
        except:
            st.info("To display a logo, place logo.svg or logo.png in your project directory")
    
    # Create header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Portfolio Creation Tool")
    with col2:
        st.markdown("")
        page = st.radio("", ["Portfolio Assignment", "Portfolio Mapping"], horizontal=True, key="page_nav")
    
    # Initialize session state
    session_vars = {
        'all_portfolios': {},
        'portfolio_controls': {},
        'recommend_reassignment': {},
        'should_create_portfolios': False,
        'mapping_portfolio_controls': {},
        'should_generate_mapping': False,
        'filter_states': [],
        'filter_cities': [],
        'filter_selected_aus': [],
        'filter_cust_state': [],
        'filter_role': [],
        'filter_cust_portcd': [],
        'filter_max_dist': 20,
        'filter_min_rev': 5000,
        'filter_min_deposit': 100000
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
    
    # Load data
    customer_data, banker_data, branch_data = load_data()
    data = merge_dfs(customer_data, banker_data, branch_data)
    
    if page == "Portfolio Assignment":
        portfolio_assignment_page(customer_data, banker_data, branch_data)
    elif page == "Portfolio Mapping":
        if MAPPING_AVAILABLE:
            portfolio_mapping_page(customer_data, banker_data, branch_data)
        else:
            st.error("Portfolio Mapping functionality requires the portfolio_mapping.py module.")
            st.info("Please ensure portfolio_mapping.py is in the same directory as this file.")

def portfolio_assignment_page(customer_data, banker_data, branch_data):
    """Portfolio Assignment page logic"""
    
    # AU Selection filters
    col_header1, col_clear1 = st.columns([9, 1])
    with col_header1:
        st.subheader("Select AUs for Portfolio Creation")
    with col_clear1:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_au_filters", help="Clear AU selection filters", type="secondary"):
            st.session_state.filter_states = []
            st.session_state.filter_cities = []
            st.session_state.filter_selected_aus = []
            if 'portfolios_created' in st.session_state:
                del st.session_state.portfolios_created
            if 'portfolio_summaries' in st.session_state:
                del st.session_state.portfolio_summaries
            st.session_state.portfolio_controls = {}
            st.experimental_rerun()
    
    with st.expander("Select AUs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_states = list(branch_data['STATECODE'].dropna().unique())
            default_states = [s for s in st.session_state.filter_states if s in available_states]
            states = st.multiselect("State", available_states, default=default_states, key="states")
            st.session_state.filter_states = states
        
        if states:
            filtered_branch_data = branch_data[branch_data['STATECODE'].isin(states)]
        else:
            filtered_branch_data = branch_data
            
        with col2:
            available_cities = list(filtered_branch_data['CITY'].dropna().unique())
            default_cities = [c for c in st.session_state.filter_cities if c in available_cities]
            cities = st.multiselect("City", available_cities, default=default_cities, key="cities")
            st.session_state.filter_cities = cities
        
        if cities:
            filtered_branch_data = filtered_branch_data[filtered_branch_data['CITY'].isin(cities)]
        
        with col3:
            available_aus = list(filtered_branch_data['AU'].dropna().unique())
            default_aus = [a for a in st.session_state.filter_selected_aus if a in available_aus]
            selected_aus = st.multiselect("AU", available_aus, default=default_aus, key="selected_aus")
            st.session_state.filter_selected_aus = selected_aus
    
    # Customer Selection Criteria (same function for both pages)
    cust_state, role, cust_portcd, max_dist, min_rev, min_deposit = create_customer_filters(customer_data, "assignment")
    
    # Create Portfolios button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")
    with col2:
        if st.button("Create Portfolios", key="create_portfolios", type="primary"):
            if not selected_aus:
                st.error("Please select at least one AU")
            else:
                st.session_state.should_create_portfolios = True
    
    # Process portfolio creation
    if st.session_state.should_create_portfolios:
        if not selected_aus:
            st.error("Please select at least one AU")
            st.session_state.should_create_portfolios = False
        else:
            process_portfolio_assignment(selected_aus, customer_data, banker_data, branch_data,
                                       role, cust_state, cust_portcd, max_dist, min_rev, min_deposit)
    
    # Display results
    display_assignment_results(branch_data)

def portfolio_mapping_page(customer_data, banker_data, branch_data):
    """Portfolio Mapping page logic with advanced clustering"""
    
    st.subheader("Advanced Portfolio Mapping")
    st.info("🧠 This uses AI-powered clustering to automatically optimize portfolio assignments based on geographic proximity and business rules.")
    
    # Customer Selection Criteria (no AU selection)
    cust_state, role, cust_portcd, max_dist, min_rev, min_deposit = create_customer_filters(customer_data, "mapping")
    
    # Generate Portfolio Map button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")
    with col2:
        if st.button("Generate Portfolio Map", key="generate_mapping", type="primary"):
            st.session_state.should_generate_mapping = True
    
    # Process portfolio mapping
    if st.session_state.should_generate_mapping:
        process_portfolio_mapping(customer_data, banker_data, branch_data,
                                role, cust_state, cust_portcd, max_dist, min_rev, min_deposit)
    
    # Display mapping results
    display_mapping_results(branch_data)

def create_customer_filters(customer_data, page_type):
    """Create customer selection criteria filters"""
    col_header2, col_clear2 = st.columns([9, 1])
    with col_header2:
        st.subheader("Customer Selection Criteria")
    with col_clear2:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        clear_key = f"clear_{page_type}_filters"
        if st.button("Clear filters", key=clear_key, help="Clear customer selection filters", type="secondary"):
            st.session_state.filter_cust_state = []
            st.session_state.filter_role = []
            st.session_state.filter_cust_portcd = []
            st.session_state.filter_max_dist = 20
            st.session_state.filter_min_rev = 5000
            st.session_state.filter_min_deposit = 100000
            # Clear created portfolios
            if f'{page_type}_portfolios_created' in st.session_state:
                del st.session_state[f'{page_type}_portfolios_created']
            if f'{page_type}_portfolio_summaries' in st.session_state:
                del st.session_state[f'{page_type}_portfolio_summaries']
            getattr(st.session_state, f'{page_type}_portfolio_controls', {}).clear()
            st.experimental_rerun()
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col2_or, col3 = st.columns([1, 1, 0.1, 1])
        
        with col1:
            cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
            default_cust_states = [s for s in st.session_state.filter_cust_state if s in cust_state_options]
            cust_state = st.multiselect("Customer State", cust_state_options, 
                                      default=default_cust_states, key=f"{page_type}_cust_state")
            st.session_state.filter_cust_state = cust_state
            if not cust_state:
                cust_state = None
        
        with col2:
            role_options = list(customer_data['TYPE'].dropna().unique())
            default_roles = [r for r in st.session_state.filter_role if r in role_options]
            role = st.multiselect("Role", role_options, 
                                default=default_roles, key=f"{page_type}_role")
            st.session_state.filter_role = role
            if not role:
                role = None
        
        with col2_or:
            st.markdown("<div style='text-align: center; padding-top: 8px; font-weight: bold;'>-OR-</div>", unsafe_allow_html=True)
        
        with col3:
            customer_data_temp = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            portfolio_options = list(customer_data_temp['PORT_CODE'].dropna().unique())
            default_portfolios = [p for p in st.session_state.filter_cust_portcd if p in portfolio_options]
            cust_portcd = st.multiselect("Portfolio Code", portfolio_options, 
                                       default=default_portfolios, key=f"{page_type}_cust_portcd")
            st.session_state.filter_cust_portcd = cust_portcd
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5, col6 = st.columns(3)
        with col4:
            max_dist = st.slider("Max Distance (miles)", 1, 100, value=st.session_state.filter_max_dist, key=f"{page_type}_max_distance")
            st.session_state.filter_max_dist = max_dist
        with col5:
            min_rev = st.slider("Minimum Revenue", 0, 20000, value=st.session_state.filter_min_rev, step=1000, key=f"{page_type}_min_revenue")
            st.session_state.filter_min_rev = min_rev
        with col6:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, value=st.session_state.filter_min_deposit, step=5000, key=f"{page_type}_min_deposit")
            st.session_state.filter_min_deposit = min_deposit
    
    return cust_state, role, cust_portcd, max_dist, min_rev, min_deposit

def apply_customer_filters(customer_data, role, cust_state, cust_portcd, min_rev, min_deposit):
    """Apply customer filters"""
    filtered_data = customer_data.copy()
    
    if cust_state is not None:
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
    
    if role is not None or cust_portcd is not None:
        role_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        portfolio_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        
        if role is not None:
            filtered_data['TYPE_CLEAN'] = filtered_data['TYPE'].fillna('').str.strip().str.lower()
            role_clean = [r.strip().lower() for r in role]
            role_condition = filtered_data['TYPE_CLEAN'].isin(role_clean)
            filtered_data = filtered_data.drop('TYPE_CLEAN', axis=1)
        
        if cust_portcd is not None:
            customer_data_temp = filtered_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            portfolio_condition = customer_data_temp['PORT_CODE'].isin(cust_portcd)
        
        combined_condition = role_condition | portfolio_condition
        filtered_data = filtered_data[combined_condition]
    
    if 'BANK_REVENUE' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    if 'DEPOSIT_BAL' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    return filtered_data

def process_portfolio_assignment(selected_aus, customer_data, banker_data, branch_data, role, cust_state, cust_portcd, max_dist, min_rev, min_deposit):
    """Process portfolio assignment logic"""
    portfolios_created = {}
    portfolio_summaries = {}
    
    for au_id in selected_aus:
        filtered_data, au_row = filter_customers_for_au(
            customer_data, banker_data, au_id, branch_data, 
            role, cust_state, cust_portcd, max_dist, min_rev, min_deposit
        )
        
        if not filtered_data.empty:
            portfolio_summary = create_portfolio_summary_simple(filtered_data, au_id, customer_data)
            portfolios_created[au_id] = filtered_data
            portfolio_summaries[au_id] = portfolio_summary
    
    if portfolios_created:
        st.success(f"Portfolios created for {len(portfolios_created)} AUs")
        st.session_state.portfolios_created = portfolios_created
        st.session_state.portfolio_summaries = portfolio_summaries
    else:
        st.warning("No customers found for the selected AUs with current filters.")
    
    st.session_state.should_create_portfolios = False

def process_portfolio_mapping(customer_data, banker_data, branch_data, role, cust_state, cust_portcd, max_dist, min_rev, min_deposit):
    """Process portfolio mapping logic"""
    filtered_customers = apply_customer_filters(
        customer_data, role, cust_state, cust_portcd, min_rev, min_deposit
    )
    
    if len(filtered_customers) == 0:
        st.error("No customers found with the selected filters.")
        st.session_state.should_generate_mapping = False
        return
    
    st.info(f"Generating optimized portfolio map for {len(filtered_customers)} customers...")
    
    try:
        result_df, final_unassigned = enhanced_customer_au_assignment_with_streamlit_progress(
            filtered_customers, branch_data
        )
        
        if len(result_df) > 0:
            portfolios_created, portfolio_summaries = convert_mapping_results_to_portfolio_format(
                result_df, branch_data
            )
            
            st.session_state.mapping_portfolios_created = portfolios_created
            st.session_state.mapping_portfolio_summaries = portfolio_summaries
            
            st.success(f"🎯 Portfolio mapping completed! Created {len(portfolios_created)} optimized portfolios.")
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                inmarket_count = len(result_df[result_df['TYPE'] == 'INMARKET'])
                st.metric("INMARKET Customers", inmarket_count)
            with col2:
                centralized_count = len(result_df[result_df['TYPE'] == 'CENTRALIZED'])
                st.metric("CENTRALIZED Customers", centralized_count)
            with col3:
                avg_distance = result_df['DISTANCE_TO_AU'].mean()
                st.metric("Avg Distance", f"{avg_distance:.1f} miles")
            with col4:
                unique_aus = result_df['ASSIGNED_AU'].nunique()
                st.metric("Optimized AUs", unique_aus)
        else:
            st.warning("No optimized portfolios could be created with the selected customers.")
    
    except Exception as e:
        st.error(f"Error during portfolio mapping: {str(e)}")
        st.info("Please check that the portfolio_mapping.py file is properly configured.")
    
    st.session_state.should_generate_mapping = False

def create_portfolio_summary_simple(filtered_data, au_id, customer_data):
    """Create a simple portfolio summary"""
    portfolio_summary = []
    
    grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
    
    for pid, group in grouped:
        total_customer = len(customer_data[customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})["PORT_CODE"] == pid])
        
        portfolio_type = "Unknown"
        if not group.empty:
            types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
            if not types.empty:
                portfolio_type = types.index[0]
        
        portfolio_summary.append({
            'Include': True,
            'Portfolio ID': pid,
            'Portfolio Type': portfolio_type,
            'Total Customers': total_customer,
            'Available for this portfolio': len(group),
            'Select': len(group)
        })
    
    unmanaged_customers = filtered_data[
        (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
        (filtered_data['PORT_CODE'].isna())
    ]
    
    if not unmanaged_customers.empty:
        portfolio_summary.append({
            'Include': True,
            'Portfolio ID': 'UNMANAGED',
            'Portfolio Type': 'Unmanaged',
            'Total Customers': len(customer_data[
                (customer_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                (customer_data['PORT_CODE'].isna())
            ]),
            'Available for this portfolio': len(unmanaged_customers),
            'Select': len(unmanaged_customers)
        })
    
    return portfolio_summary

def display_assignment_results(branch_data):
    """Display portfolio assignment results"""
    if 'portfolios_created' in st.session_state and st.session_state.portfolios_created:
        portfolios_created = st.session_state.portfolios_created
        portfolio_summaries = st.session_state.get('portfolio_summaries', {})
        
        st.markdown("----")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Portfolio Summary Tables")
            display_portfolio_tables(portfolios_created, portfolio_summaries, branch_data, "assignment")
        
        with col2:
            st.subheader("Geographic Distribution")
            display_geographic_map(portfolios_created, branch_data)

def display_mapping_results(branch_data):
    """Display portfolio mapping results"""
    if 'mapping_portfolios_created' in st.session_state and st.session_state.mapping_portfolios_created:
        portfolios_created = st.session_state.mapping_portfolios_created
        portfolio_summaries = st.session_state.get('mapping_portfolio_summaries', {})
        
        st.markdown("----")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Optimized Portfolio Summary")
            display_portfolio_tables(portfolios_created, portfolio_summaries, branch_data, "mapping")
        
        with col2:
            st.subheader("Geographic Distribution")
            display_geographic_map(portfolios_created, branch_data)

def display_portfolio_tables(portfolios_created, portfolio_summaries, branch_data, page_type):
    """Display portfolio summary tables"""
    if len(portfolios_created) > 1:
        au_tabs = st.tabs([f"AU {au_id}" for au_id in portfolios_created.keys()])
        
        for tab_idx, (au_id, tab) in enumerate(zip(portfolios_created.keys(), au_tabs)):
            with tab:
                display_single_portfolio_table(au_id, portfolio_summaries, portfolios_created, branch_data, page_type, True)
    else:
        au_id = list(portfolios_created.keys())[0]
        display_single_portfolio_table(au_id, portfolio_summaries, portfolios_created, branch_data, page_type, False)

def display_single_portfolio_table(au_id, portfolio_summaries, portfolios_created, branch_data, page_type, is_multi_au):
    """Display table for a single AU"""
    if au_id in portfolio_summaries:
        portfolio_df = pd.DataFrame(portfolio_summaries[au_id])
        
        if len(portfolio_df) == 0:
            st.info("No portfolios for this AU")
            return
        
        portfolio_df = portfolio_df.sort_values('Available for this portfolio', ascending=False).reset_index(drop=True)
        
        column_config = {
            "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio"),
            "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
            "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
            "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
            "Available for this portfolio": st.column_config.NumberColumn("Available for this portfolio", disabled=True),
            "Select": st.column_config.NumberColumn(
                "Select",
                help="Number of customers to select from this portfolio",
                min_value=0,
                step=1
            )
        }
        
        edited_df = st.data_editor(
            portfolio_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            key=f"{page_type}_portfolio_editor_{au_id}_{len(portfolio_df)}"
        )
        
        # Store the edited data
        control_key = f'{page_type}_portfolio_controls'
        if control_key not in st.session_state:
            st.session_state[control_key] = {}
        st.session_state[control_key][au_id] = edited_df
        
        # Apply Changes button
        suffix = "_single" if not is_multi_au else ""
        if st.button(f"Apply Changes for AU {au_id}", key=f"apply_{page_type}_changes_{au_id}{suffix}"):
            if page_type == "assignment":
                apply_assignment_changes(au_id, branch_data)
            else:
                # Import and use mapping changes function
                from portfolio_mapping import apply_mapping_portfolio_changes
                apply_mapping_portfolio_changes(au_id, branch_data)
        
        # Display summary statistics
        au_filtered_data = portfolios_created[au_id]
        display_au_summary_statistics(au_filtered_data)

def apply_assignment_changes(au_id, branch_data):
    """Apply portfolio assignment changes"""
    with st.spinner("Applying selection changes..."):
        if 'portfolios_created' in st.session_state and au_id in st.session_state.portfolios_created:
            # Get original and control data
            original_count = len(st.session_state.portfolios_created[au_id])
            
            updated_portfolios = apply_portfolio_selection_changes(
                st.session_state.portfolios_created, 
                st.session_state.portfolio_controls, 
                [au_id], 
                branch_data
            )
            
            if au_id in updated_portfolios:
                st.session_state.portfolios_created[au_id] = updated_portfolios[au_id]
                new_count = len(updated_portfolios[au_id])
                
                # Show detailed feedback
                if new_count < original_count:
                    removed_count = original_count - new_count
                    st.success(f"✅ Portfolio updated! Removed {removed_count} customers. Now showing {new_count} customers.")
                elif new_count == original_count:
                    st.success(f"✅ Portfolio updated! {new_count} customers selected.")
                else:
                    st.success(f"✅ Portfolio updated! Now showing {new_count} customers.")
                
                st.experimental_rerun()
            else:
                st.warning("No customers selected with current settings.")
        else:
            st.error("No portfolio data found to update.")

def display_au_summary_statistics(au_filtered_data):
    """Display summary statistics for an AU"""
    if not au_filtered_data.empty:
        st.subheader("AU Summary Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Customers", len(au_filtered_data))
        with col_b:
            if 'Distance' in au_filtered_data.columns:
                st.metric("Avg Distance (Miles)", f"{au_filtered_data['Distance'].mean():.1f}")
            else:
                st.metric("Avg Distance (Miles)", "N/A")
        with col_c:
            if 'BANK_REVENUE' in au_filtered_data.columns:
                avg_revenue_k = au_filtered_data['BANK_REVENUE'].mean() / 1000
                st.metric("Average Revenue", f"{avg_revenue_k:.1f}K")
            else:
                st.metric("Average Revenue", "N/A")
        with col_d:
            if 'DEPOSIT_BAL' in au_filtered_data.columns:
                avg_deposit_mm = au_filtered_data['DEPOSIT_BAL'].mean() / 1000000
                st.metric("Average Deposits", f"{avg_deposit_mm:.1f}MM")
            else:
                st.metric("Average Deposits", "N/A")

def display_geographic_map(portfolios_created, branch_data):
    """Display the geographic distribution map"""
    preview_portfolios = {}
    
    for au_id, filtered_data in portfolios_created.items():
        if not filtered_data.empty:
            preview_portfolios[f"AU_{au_id}_Portfolio"] = filtered_data
    
    if preview_portfolios:
        combined_map = create_combined_map(preview_portfolios, branch_data)
        if combined_map:
            st.plotly_chart(combined_map, use_container_width=True)
    else:
        st.info("No customers selected for map display")

if __name__ == "__main__":
    main()
