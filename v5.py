from io import BytesIO
import pandas as pd
import math
from math import sin, cos, atan2, radians, sqrt
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def haversine_distance(clat, clon, blat, blon):
    if math.isnan(clat) or math.isnan(clon) or math.isnan(blat) or math.isnan(blon):
        return 0
        
    delta_lat = radians(clat - blat)
    delta_lon = radians(clon - blon)
    
    a = sin(delta_lat/2)**2 + cos(radians(clat))*cos(radians(blat))*sin(delta_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    distance = 3959*c  # Changed from 6371 km to 3959 miles (Earth's radius in miles)
    return distance

def merge_dfs(customer_data, banker_data, branch_data):
    customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    final_table = customer_data.merge(banker_data, on = "PORT_CODE", how = "left")
    final_table.fillna(0, inplace = True)
    return final_table

def create_distance_circle(center_lat, center_lon, radius_miles, num_points=100):
    """Create points for a circle around a center point"""
    angles = np.linspace(0, 2*np.pi, num_points)
    circle_lats = []
    circle_lons = []
    
    for angle in angles:
        # Convert miles to degrees (rough approximation)
        lat_offset = radius_miles / 69.0  # 1 degree lat ≈ 69 miles
        lon_offset = radius_miles / (69.0 * math.cos(math.radians(center_lat)))
        
        lat = center_lat + lat_offset * math.cos(angle)
        lon = center_lon + lon_offset * math.sin(angle)
        
        circle_lats.append(lat)
        circle_lons.append(lon)
    
    # Close the circle
    circle_lats.append(circle_lats[0])
    circle_lons.append(circle_lons[0])
    
    return circle_lats, circle_lons

def create_combined_map(all_portfolios, branch_data):
    """Create a combined map showing all portfolios with different colors - one color per AU"""
    
    if not all_portfolios:
        return None
    
    fig = go.Figure()
    
    # Color scheme for different AU portfolios
    portfolio_colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightblue', 'pink', 'darkgreen', 'brown', 'gray', 'cyan', 'magenta', 'yellow', 'red', 'lime']
    
    # Get all unique AU locations from branch_data
    au_locations = set()
    for portfolio_id, df in all_portfolios.items():
        if not df.empty:
            au_id = df['AU'].iloc[0]
            # Get AU coordinates from branch_data
            au_row = branch_data[branch_data['AU'] == au_id]
            if not au_row.empty:
                au_lat = au_row.iloc[0]['BRANCH_LAT_NUM']
                au_lon = au_row.iloc[0]['BRANCH_LON_NUM']
                au_locations.add((au_id, au_lat, au_lon))
    
    # Add customers from each AU portfolio with unique colors
    for portfolio_idx, (portfolio_id, df) in enumerate(all_portfolios.items()):
        if df.empty:
            continue
        
        # Extract AU ID from portfolio name (e.g., "AU_101_Portfolio" -> "101")
        au_id = portfolio_id.split('_')[1] if '_' in portfolio_id else portfolio_id
        
        color = portfolio_colors[portfolio_idx % len(portfolio_colors)]
        
        # Create hover text for this AU portfolio
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
            marker=dict(
                size=8,
                color=color,
                symbol='circle'
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            name=f"AU {au_id} Portfolio ({len(df)} customers)",
            showlegend=True
        ))
    
    # Add AU markers (on top) - CHANGED: symbol from 'triangle-up' to 'circle' and color remains black
    for au_id, au_lat, au_lon in au_locations:
        au_details = branch_data[branch_data['AU'] == au_id]
        au_name = au_details['CITY'].iloc[0] if not au_details.empty else f"AU {au_id}"
        
        fig.add_trace(go.Scattermapbox(
            lat=[au_lat],
            lon=[au_lon],
            mode='markers',
            marker=dict(
                size=12,
                color='black',
                symbol='circle'  # Changed from 'triangle-up' to 'circle'
            ),
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
    
    # Calculate center point
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
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

def to_excel(all_portfolios):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for portfolio_id, df in all_portfolios.items():
            sheet_name = f"Portfolio_{portfolio_id}"[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

def apply_portfolio_selection_changes(portfolios_created, portfolio_controls, selected_aus, branch_data):
    """Apply the selection changes from portfolio controls to filter customers"""
    
    updated_portfolios = {}
    
    for au_id in selected_aus:
        if au_id not in portfolios_created or au_id not in portfolio_controls:
            continue
            
        original_data = portfolios_created[au_id].copy()
        control_data = portfolio_controls[au_id]
        
        # Start with empty list for this AU
        selected_customers = []
        
        # Process each portfolio selection
        for _, row in control_data.iterrows():
            portfolio_id = row['Portfolio ID']
            select_count = row['Select']
            include = row.get('Include', True)  # Default to True (included)
            
            # Only include portfolios that are checked (include=True) and have select_count > 0
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
            updated_portfolios[au_id] = au_final_customers
        else:
            # No customers selected for this AU
            updated_portfolios[au_id] = pd.DataFrame()
    
    return updated_portfolios

def filter_customers_for_au(customer_data, banker_data, selected_au, branch_data, role, cust_state, cust_portcd, max_dist, min_rev, min_deposit):
    """Filter customers for a specific AU based on criteria"""
    
    # Get AU data
    AU_row = branch_data[branch_data['AU'] == int(selected_au)].iloc[0]
    AU_lat = AU_row['BRANCH_LAT_NUM']
    AU_lon = AU_row['BRANCH_LON_NUM']
    
    # Filter customers by distance box (convert miles to degrees)
    box_lat = max_dist/69  # Changed from 111 km to 69 miles per degree
    box_lon = max_dist/ (69 * np.cos(np.radians(AU_lat)))  # Changed from 111 km to 69 miles per degree
    
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
    
    # Apply distance filter for all roles except CENTRALIZED (distance now in miles)
    if role is None or (role is not None and not any(r.lower().strip() == 'centralized' for r in role)):
        filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
    
    # Apply Customer State filter
    if cust_state is not None:
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
    
    # Apply Role OR Portfolio Code filter (combined with OR logic)
    if role is not None or cust_portcd is not None:
        role_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        portfolio_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        
        # Check role condition
        if role is not None:
            filtered_data['TYPE_CLEAN'] = filtered_data['TYPE'].fillna('').str.strip().str.lower()
            role_clean = [r.strip().lower() for r in role]
            role_condition = filtered_data['TYPE_CLEAN'].isin(role_clean)
            filtered_data = filtered_data.drop('TYPE_CLEAN', axis=1)
        
        # Check portfolio code condition
        if cust_portcd is not None:
            portfolio_condition = filtered_data['PORT_CODE'].isin(cust_portcd)
        
        # Apply OR logic: keep rows that match either role OR portfolio code
        combined_condition = role_condition | portfolio_condition
        filtered_data = filtered_data[combined_condition]
    
    # Apply other filters
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    # Add AU information
    filtered_data['AU'] = selected_au
    filtered_data['BRANCH_LAT_NUM'] = AU_lat
    filtered_data['BRANCH_LON_NUM'] = AU_lon
    
    return filtered_data, AU_row

def reassign_to_nearest_au(portfolios_created, selected_aus, branch_data):
    """Reassign customers to their nearest AU from the selected AUs"""
    
    if not portfolios_created or not selected_aus:
        return portfolios_created, {}
    
    # Combine all customers from all portfolios
    all_customers = []
    for au_id, customers_df in portfolios_created.items():
        if not customers_df.empty:
            customers_copy = customers_df.copy()
            customers_copy['original_au'] = au_id
            all_customers.append(customers_copy)
    
    if not all_customers:
        return portfolios_created, {}
    
    combined_customers = pd.concat(all_customers, ignore_index=True)
    
    # Get AU coordinates for selected AUs
    au_coordinates = {}
    for au_id in selected_aus:
        au_row = branch_data[branch_data['AU'] == au_id]
        if not au_row.empty:
            au_coordinates[au_id] = (au_row.iloc[0]['BRANCH_LAT_NUM'], au_row.iloc[0]['BRANCH_LON_NUM'])
    
    # Reassign each customer to nearest AU
    reassignment_summary = []
    
    for idx, customer in combined_customers.iterrows():
        if pd.isna(customer['LAT_NUM']) or pd.isna(customer['LON_NUM']):
            continue
            
        nearest_au = None
        min_distance = float('inf')
        
        # Check distance to all selected AUs
        for au_id, (au_lat, au_lon) in au_coordinates.items():
            distance = haversine_distance(customer['LAT_NUM'], customer['LON_NUM'], au_lat, au_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_au = au_id
        
        # Update customer data
        original_au = customer['original_au']
        if nearest_au and nearest_au != original_au:
            # Update AU information
            combined_customers.at[idx, 'AU'] = nearest_au
            combined_customers.at[idx, 'BRANCH_LAT_NUM'] = au_coordinates[nearest_au][0]
            combined_customers.at[idx, 'BRANCH_LON_NUM'] = au_coordinates[nearest_au][1]
            combined_customers.at[idx, 'Distance'] = min_distance
            
            # Track reassignment
            reassignment_summary.append({
                'Customer': customer.get('CG_ECN', 'N/A'),
                'Original AU': original_au,
                'New AU': nearest_au,
                'New Distance': min_distance,
                'Portfolio': customer.get('PORT_CODE', 'N/A'),
                'Revenue': customer.get('BANK_REVENUE', 0)
            })
        elif nearest_au:
            # Update distance even if AU doesn't change
            combined_customers.at[idx, 'Distance'] = min_distance
    
    # Rebuild portfolios_created with reassigned customers
    new_portfolios_created = {}
    for au_id in selected_aus:
        au_customers = combined_customers[combined_customers['AU'] == au_id]
        if not au_customers.empty:
            # Remove the temporary 'original_au' column
            au_customers = au_customers.drop('original_au', axis=1)
            new_portfolios_created[au_id] = au_customers.reset_index(drop=True)
    
    return new_portfolios_created, reassignment_summary

def recommend_reassignment(all_portfolios: dict) -> pd.DataFrame:
    if not all_portfolios:
        return pd.DataFrame()
    
    combine_df = pd.concat([df.assign(original_portfolio=portfolio_id) for portfolio_id, df in all_portfolios.items()], ignore_index=True)
    
    au_map = {portfolio_id: (df["BRANCH_LAT_NUM"].iloc[0], df["BRANCH_LON_NUM"].iloc[0])
              for portfolio_id, df in all_portfolios.items()
              if not df.empty}
                
    records = []
    for _, row in combine_df.iterrows():
        best_portfolio = None
        min_dist = float("inf")
        for portfolio_id, (au_lat, au_lon) in au_map.items():
            dist = haversine_distance(row['LAT_NUM'], row['LON_NUM'], au_lat, au_lon)
            if dist < min_dist:
                best_portfolio = portfolio_id
                min_dist = dist
                
        row_data = row.to_dict()
        row_data['recommended_portfolio'] = best_portfolio
        row_data['recommended_dist'] = min_dist
        records.append(row_data)
        
    return pd.DataFrame(records)

#------------------------Streamlit App---------------------------------------------------------------
st.set_page_config("Portfolio Creation tool", layout="wide")

# Custom CSS for red header
st.markdown("""
<style>
    [data-testid="stHeader"] {
        background-color: rgb(215, 30, 40) !important;
    }
</style>
""", unsafe_allow_html=True)

# Add logo using Streamlit's logo function
try:
    st.logo("logo.svg")
except:
    try:
        st.logo("logo.png")
    except:
        st.info("To display a logo, place logo.svg or logo.png in your project directory")

# Header with title and navigation
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Portfolio Creation Tool")
with col2:
    # Add navigation buttons in the header
    st.markdown("") # Add some spacing
    page = st.radio("", ["Portfolio Assignment", "Portfolio Mapping"], horizontal=True, key="page_nav")

# Initialize session state
if 'all_portfolios' not in st.session_state:
    st.session_state.all_portfolios = {}
    
if 'portfolio_controls' not in st.session_state:
    st.session_state.portfolio_controls = {}
    
if 'recommend_reassignment' not in st.session_state:
    st.session_state.recommend_reassignment = {}

# Add flag to track when portfolio creation should run
if 'should_create_portfolios' not in st.session_state:
    st.session_state.should_create_portfolios = False

# Initialize filter states
if 'filter_states' not in st.session_state:
    st.session_state.filter_states = {}
    
if 'filter_cities' not in st.session_state:
    st.session_state.filter_cities = []
    
if 'filter_selected_aus' not in st.session_state:
    st.session_state.filter_selected_aus = []
    
if 'filter_cust_state' not in st.session_state:
    st.session_state.filter_cust_state = []
    
if 'filter_role' not in st.session_state:
    st.session_state.filter_role = []
    
if 'filter_cust_portcd' not in st.session_state:
    st.session_state.filter_cust_portcd = []
    
if 'filter_max_dist' not in st.session_state:
    st.session_state.filter_max_dist = 20
    
if 'filter_min_rev' not in st.session_state:
    st.session_state.filter_min_rev = 5000
    
if 'filter_min_deposit' not in st.session_state:
    st.session_state.filter_min_deposit = 100000

# Load data from local CSV files
@st.cache_data
def load_data():
    customer_data = pd.read_csv("customer_data.csv")
    banker_data = pd.read_csv("banker_data.csv")
    branch_data = pd.read_csv("branch_data.csv")
    return customer_data, banker_data, branch_data

# Load data on app startup
customer_data, banker_data, branch_data = load_data()
data = merge_dfs(customer_data, banker_data, branch_data)

if page == "Portfolio Assignment":
    
    # AU Selection Section
    # Custom CSS for header-sized clear filters buttons
    st.markdown("""
    <style>
    /* Style for clear filters buttons to look like header text */
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
    
    col_header1, col_clear1 = st.columns([9, 1])
    with col_header1:
        st.subheader("Select AUs for Portfolio Creation")
    with col_clear1:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)  # Small spacing to align with subheader
        if st.button("Clear filters", key="clear_au_filters", help="Clear AU selection filters", type="secondary"):
            # Clear AU filters
            st.session_state.filter_states = []
            st.session_state.filter_cities = []
            st.session_state.filter_selected_aus = []
            # Clear created portfolios
            if 'portfolios_created' in st.session_state:
                del st.session_state.portfolios_created
            if 'portfolio_summaries' in st.session_state:
                del st.session_state.portfolio_summaries
            st.session_state.portfolio_controls = {}
            st.experimental_rerun()
    
    # Multi-select for AUs with expander
    with st.expander("Select AUs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_states = list(branch_data['STATECODE'].dropna().unique())
            # Filter default states to only include available options
            default_states = [s for s in st.session_state.filter_states if s in available_states]
            states = st.multiselect("State", available_states, 
                                  default=default_states, key="states")
            st.session_state.filter_states = states
        
        # Filter branch data based on selected states
        if states:
            filtered_branch_data = branch_data[branch_data['STATECODE'].isin(states)]
        else:
            filtered_branch_data = branch_data
            
        with col2:
            available_cities = list(filtered_branch_data['CITY'].dropna().unique())
            # Filter default cities to only include available options
            default_cities = [c for c in st.session_state.filter_cities if c in available_cities]
            cities = st.multiselect("City", available_cities, 
                                  default=default_cities, key="cities")
            st.session_state.filter_cities = cities
        
        # Filter further based on selected cities
        if cities:
            filtered_branch_data = filtered_branch_data[filtered_branch_data['CITY'].isin(cities)]
        
        with col3:
            available_aus = list(filtered_branch_data['AU'].dropna().unique())
            # Filter default AUs to only include available options
            default_aus = [a for a in st.session_state.filter_selected_aus if a in available_aus]
            selected_aus = st.multiselect("AU", available_aus, 
                                        default=default_aus, key="selected_aus")
            st.session_state.filter_selected_aus = selected_aus
    
    # Customer Selection Criteria
    col_header2, col_clear2 = st.columns([9, 1])
    with col_header2:
        st.subheader("Customer Selection Criteria")
    with col_clear2:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)  # Small spacing to align with subheader
        if st.button("Clear filters", key="clear_customer_filters", help="Clear customer selection filters", type="secondary"):
            # Clear customer filters
            st.session_state.filter_cust_state = []
            st.session_state.filter_role = []
            st.session_state.filter_cust_portcd = []
            st.session_state.filter_max_dist = 20
            st.session_state.filter_min_rev = 5000
            st.session_state.filter_min_deposit = 100000
            # Clear created portfolios
            if 'portfolios_created' in st.session_state:
                del st.session_state.portfolios_created
            if 'portfolio_summaries' in st.session_state:
                del st.session_state.portfolio_summaries
            st.session_state.portfolio_controls = {}
            st.experimental_rerun()
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col2_or, col3 = st.columns([1, 1, 0.1, 1])
        
        with col1:
            cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
            # Filter default customer states to only include available options
            default_cust_states = [s for s in st.session_state.filter_cust_state if s in cust_state_options]
            cust_state = st.multiselect("Customer State", cust_state_options, 
                                      default=default_cust_states, key="cust_state")
            st.session_state.filter_cust_state = cust_state
            if not cust_state:
                cust_state = None
        
        with col2:
            role_options = list(customer_data['TYPE'].dropna().unique())
            # Filter default roles to only include available options
            default_roles = [r for r in st.session_state.filter_role if r in role_options]
            role = st.multiselect("Role", role_options, 
                                default=default_roles, key="role")
            st.session_state.filter_role = role
            if not role:
                role = None
        
        with col2_or:
            st.markdown("<div style='text-align: center; padding-top: 8px; font-weight: bold;'>-OR-</div>", unsafe_allow_html=True)
        
        with col3:
            customer_data_temp = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            portfolio_options = list(customer_data_temp['PORT_CODE'].dropna().unique())
            # Filter default portfolio codes to only include available options
            default_portfolios = [p for p in st.session_state.filter_cust_portcd if p in portfolio_options]
            cust_portcd = st.multiselect("Portfolio Code", portfolio_options, 
                                       default=default_portfolios, key="cust_portcd")
            st.session_state.filter_cust_portcd = cust_portcd
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5, col6 = st.columns(3)
        with col4:
            max_dist = st.slider("Max Distance (miles)", 1, 100, value=st.session_state.filter_max_dist, key="max_distance")
            st.session_state.filter_max_dist = max_dist
        with col5:
            min_rev = st.slider("Minimum Revenue", 0, 20000, value=st.session_state.filter_min_rev, step=1000, key="min_revenue")
            st.session_state.filter_min_rev = min_rev
        with col6:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, value=st.session_state.filter_min_deposit, step=5000, key="min_deposit")
            st.session_state.filter_min_deposit = min_deposit
    
    # Process button
        # Process button - positioned to extreme right
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")  # Empty space
    with col2:
        if st.button("Create Portfolios", key="create_portfolios", type="primary"):
        if not selected_aus:
            st.error("Please select at least one AU")
        else:
                # Create portfolios for each selected AU
                portfolios_created = {}
                portfolio_summaries = {}
                
                for au_id in selected_aus:
                    # Filter customers for this AU
                    filtered_data, au_row = filter_customers_for_au(
                        customer_data, banker_data, au_id, branch_data, 
                        role, cust_state, cust_portcd, max_dist, min_rev, min_deposit
                    )
                    
                    if not filtered_data.empty:
                        # Create portfolio summary for this AU
                        portfolio_summary = []
                        
                        # Group by portfolio for assigned customers
                        grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
                        
                        for pid, group in grouped:
                            total_customer = len(customer_data[customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})["PORT_CODE"] == pid])
                            
                            # Determine portfolio type
                            portfolio_type = "Unknown"
                            if not group.empty:
                                types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
                                if not types.empty:
                                    portfolio_type = types.index[0]
                            
                            portfolio_summary.append({
                                'AU': au_id,
                                'Portfolio ID': pid,
                                'Portfolio Type': portfolio_type,
                                'Total Customers': total_customer,
                                'Available': len(group),
                                'Select': len(group)
                            })
                        
                        # Add unmanaged customers
                        unmanaged_customers = filtered_data[
                            (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                            (filtered_data['PORT_CODE'].isna())
                        ]
                        
                        customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
                        
                        if not unmanaged_customers.empty:
                            portfolio_summary.append({
                                'AU': au_id,
                                'Portfolio ID': 'UNMANAGED',
                                'Portfolio Type': 'Unmanaged',
                                'Total Customers': len(customer_data[
                                    (customer_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                                    (customer_data['PORT_CODE'].isna())
                                ]),
                                'Available': len(unmanaged_customers),
                                'Select': len(unmanaged_customers)
                            })
                        
                        portfolios_created[au_id] = filtered_data
                        portfolio_summaries[au_id] = portfolio_summary
                
                if portfolios_created:
                    # Apply nearest AU reassignment
                    with st.spinner("Reassigning customers to nearest AUs..."):
                        portfolios_created, reassignment_summary = reassign_to_nearest_au(
                            portfolios_created, selected_aus, branch_data
                        )
                    
                    st.success(f"Portfolios created for {len(portfolios_created)} AUs")
                    
                    # Recalculate portfolio summaries after reassignment
                    portfolio_summaries = {}
                    
                    # First, calculate totals across all AUs for each portfolio (for "Available for all new portfolios")
                    all_portfolio_counts = {}
                    for au_id, filtered_data in portfolios_created.items():
                        # Count regular portfolios
                        grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
                        for pid, group in grouped:
                            if pid not in all_portfolio_counts:
                                all_portfolio_counts[pid] = 0
                            all_portfolio_counts[pid] += len(group)
                        
                        # Count unmanaged customers
                        unmanaged_customers = filtered_data[
                            (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                            (filtered_data['PORT_CODE'].isna())
                        ]
                        if not unmanaged_customers.empty:
                            if 'UNMANAGED' not in all_portfolio_counts:
                                all_portfolio_counts['UNMANAGED'] = 0
                            all_portfolio_counts['UNMANAGED'] += len(unmanaged_customers)
                    
                    for au_id, filtered_data in portfolios_created.items():
                        portfolio_summary = []
                        
                        # Group by portfolio for assigned customers
                        grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
                        
                        for pid, group in grouped:
                            total_customer = len(customer_data[customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})["PORT_CODE"] == pid])
                            
                            # Determine portfolio type
                            portfolio_type = "Unknown"
                            if not group.empty:
                                types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
                                if not types.empty:
                                    portfolio_type = types.index[0]
                            
                            summary_item = {
                                'Include': True,
                                'Portfolio ID': pid,
                                'Portfolio Type': portfolio_type,
                                'Total Customers': total_customer,
                                'Available for this portfolio': len(group),
                                'Select': len(group)
                            }
                            
                            # Add "Available for all new portfolios" column only if multiple AUs
                            if len(portfolios_created) > 1:
                                summary_item['Available for all new portfolios'] = all_portfolio_counts.get(pid, 0)
                                # Reorder columns
                                summary_item = {
                                    'Include': True,
                                    'Portfolio ID': summary_item['Portfolio ID'],
                                    'Portfolio Type': summary_item['Portfolio Type'],
                                    'Total Customers': summary_item['Total Customers'],
                                    'Available for all new portfolios': summary_item['Available for all new portfolios'],
                                    'Available for this portfolio': summary_item['Available for this portfolio'],
                                    'Select': summary_item['Select']
                                }
                            
                            portfolio_summary.append(summary_item)
                        
                        # Add unmanaged customers
                        unmanaged_customers = filtered_data[
                            (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                            (filtered_data['PORT_CODE'].isna())
                        ]
                        
                        customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
                        
                        if not unmanaged_customers.empty:
                            summary_item = {
                                'Include': True,
                                'Portfolio ID': 'UNMANAGED',
                                'Portfolio Type': 'Unmanaged',
                                'Total Customers': len(customer_data[
                                    (customer_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                                    (customer_data['PORT_CODE'].isna())
                                ]),
                                'Available for this portfolio': len(unmanaged_customers),
                                'Select': len(unmanaged_customers)
                            }
                            
                            # Add "Available for all new portfolios" column only if multiple AUs
                            if len(portfolios_created) > 1:
                                summary_item['Available for all new portfolios'] = all_portfolio_counts.get('UNMANAGED', 0)
                                # Reorder columns
                                summary_item = {
                                    'Include': True,
                                    'Portfolio ID': summary_item['Portfolio ID'],
                                    'Portfolio Type': summary_item['Portfolio Type'],
                                    'Total Customers': summary_item['Total Customers'],
                                    'Available for all new portfolios': summary_item['Available for all new portfolios'],
                                    'Available for this portfolio': summary_item['Available for this portfolio'],
                                    'Select': summary_item['Select']
                                }
                            
                            portfolio_summary.append(summary_item)
                        
                        portfolios_created[au_id] = filtered_data
                        portfolio_summaries[au_id] = portfolio_summary
                    
                    # Store the created portfolios data
                    st.session_state.portfolios_created = portfolios_created
                    st.session_state.portfolio_summaries = portfolio_summaries
                    
                    # Reset the flag after portfolios are created
                    st.session_state.should_create_portfolios = False
                else:
                    st.warning("No customers found for the selected AUs with current filters.")
                    st.session_state.should_create_portfolios = False
                    
    # Display results if portfolios exist in session state
    if 'portfolios_created' in st.session_state and st.session_state.portfolios_created:
        portfolios_created = st.session_state.portfolios_created
        portfolio_summaries = st.session_state.get('portfolio_summaries', {})
        
        # Show Portfolio Summary Tables and Geographic Distribution
        st.markdown("----")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Portfolio Summary Tables")
            
            # Create tabs for each AU
            if len(portfolios_created) > 1:
                au_tabs = st.tabs([f"AU {au_id}" for au_id in portfolios_created.keys()])
                
                for tab_idx, (au_id, tab) in enumerate(zip(portfolios_created.keys(), au_tabs)):
                    with tab:
                        if au_id in portfolio_summaries:
                            portfolio_df = pd.DataFrame(portfolio_summaries[au_id])
                            portfolio_df = portfolio_df.sort_values('Available for this portfolio', ascending=False).reset_index(drop=True)
                            
                            # Create column config based on number of AUs
                            if len(portfolios_created) > 1:
                                column_config = {
                                    "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio in selection"),
                                    "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
                                    "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
                                    "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
                                    "Available for all new portfolios": st.column_config.NumberColumn("Available for all new portfolios", disabled=True),
                                    "Available for this portfolio": st.column_config.NumberColumn("Available for this portfolio", disabled=True),
                                    "Select": st.column_config.NumberColumn(
                                        "Select",
                                        help="Number of customers to select from this portfolio",
                                        min_value=0,
                                        step=1
                                    )
                                }
                            else:
                                column_config = {
                                    "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio in selection"),
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
                            
                            # Create editable dataframe with unique key
                            edited_df = st.data_editor(
                                portfolio_df,
                                column_config=column_config,
                                hide_index=True,
                                use_container_width=True,
                                key=f"portfolio_editor_{au_id}_{len(portfolio_df)}"
                            )
                            
                            # Store the edited data
                            st.session_state.portfolio_controls[au_id] = edited_df
                            
                            # Add Apply Changes button
                            if st.button(f"Apply Changes for AU {au_id}", key=f"apply_changes_{au_id}"):
                                with st.spinner("Applying selection changes..."):
                                    # Apply the portfolio selection changes using original data
                                    if 'portfolios_created' in st.session_state and au_id in st.session_state.portfolios_created:
                                        updated_portfolios = apply_portfolio_selection_changes(
                                            st.session_state.portfolios_created, 
                                            st.session_state.portfolio_controls, 
                                            [au_id], 
                                            branch_data
                                        )
                                        
                                        # Update only this AU's portfolio
                                        if au_id in updated_portfolios:
                                            st.session_state.portfolios_created[au_id] = updated_portfolios[au_id]
                                        
                                        st.success("Portfolio selection updated!")
                                        st.experimental_rerun()
                            
                            # Summary statistics for this AU
                            au_filtered_data = st.session_state.portfolios_created[au_id]
                            if not au_filtered_data.empty:
                                st.subheader("AU Summary Statistics")
                                # Use 4 columns for metrics in a single row
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("Total Customers", len(au_filtered_data))
                                with col_b:
                                    st.metric("Avg Distance (Miles)", f"{au_filtered_data['Distance'].mean():.1f}")
                                with col_c:
                                    avg_revenue_k = au_filtered_data['BANK_REVENUE'].mean() / 1000
                                    st.metric("Average Revenue", f"{avg_revenue_k:.1f}K")
                                with col_d:
                                    avg_deposit_mm = au_filtered_data['DEPOSIT_BAL'].mean() / 1000000
                                    st.metric("Average Deposits", f"{avg_deposit_mm:.1f}MM")
            else:
                # Single AU case
                au_id = list(portfolios_created.keys())[0]
                
                if au_id in portfolio_summaries:
                    portfolio_df = pd.DataFrame(portfolio_summaries[au_id])
                    portfolio_df = portfolio_df.sort_values('Available for this portfolio', ascending=False).reset_index(drop=True)
                    
                    # Create editable dataframe (single AU case) with unique key
                    edited_df = st.data_editor(
                        portfolio_df,
                        column_config={
                            "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio in selection"),
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
                        },
                        hide_index=True,
                        use_container_width=True,
                        key=f"portfolio_editor_{au_id}_{len(portfolio_df)}"
                    )
                    
                    # Store the edited data
                    st.session_state.portfolio_controls[au_id] = edited_df
                    
                    # Add Apply Changes button
                    if st.button(f"Apply Changes for AU {au_id}", key=f"apply_changes_{au_id}_single"):
                        with st.spinner("Applying selection changes..."):
                            # Apply the portfolio selection changes using original data
                            if 'portfolios_created' in st.session_state and au_id in st.session_state.portfolios_created:
                                updated_portfolios = apply_portfolio_selection_changes(
                                    st.session_state.portfolios_created, 
                                    st.session_state.portfolio_controls, 
                                    [au_id], 
                                    branch_data
                                )
                                
                                # Update only this AU's portfolio
                                if au_id in updated_portfolios:
                                    st.session_state.portfolios_created[au_id] = updated_portfolios[au_id]
                                
                                st.success("Portfolio selection updated!")
                                st.experimental_rerun()
                    
                    # Summary statistics for this AU
                    au_filtered_data = st.session_state.portfolios_created[au_id]
                    if not au_filtered_data.empty:
                        st.subheader("AU Summary Statistics")
                        # Use 4 columns for metrics in a single row
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Total Customers", len(au_filtered_data))
                        with col_b:
                            st.metric("Avg Distance (Miles)", f"{au_filtered_data['Distance'].mean():.1f}")
                        with col_c:
                            avg_revenue_k = au_filtered_data['BANK_REVENUE'].mean() / 1000
                            st.metric("Average Revenue", f"{avg_revenue_k:.1f}K")
                        with col_d:
                            avg_deposit_mm = au_filtered_data['DEPOSIT_BAL'].mean() / 1000000
                            st.metric("Average Deposits", f"{avg_deposit_mm:.1f}MM")
        
        with col2:
            st.subheader("Geographic Distribution")
            
            # Create preview portfolios for map display - one portfolio per AU
            preview_portfolios = {}
            
            for au_id, filtered_data in st.session_state.portfolios_created.items():
                if not filtered_data.empty:
                    preview_portfolios[f"AU_{au_id}_Portfolio"] = filtered_data
            
            # Display the map with preview portfolios
            if preview_portfolios:
                combined_map = create_combined_map(preview_portfolios, branch_data)
                if combined_map:
                    st.plotly_chart(combined_map, use_container_width=True)
            else:
                st.info("No customers selected for map display")
        
    else:
        # Show message when no portfolios exist
        if st.session_state.get('portfolios_created') is not None:
            st.warning("No customers found for the selected AUs with current filters.")
    
    # Show recommendation reassignment table if it exists
    if 'recommend_reassignment' in st.session_state and isinstance(st.session_state.recommend_reassignment, pd.DataFrame) and not st.session_state.recommend_reassignment.empty:
        st.markdown("----")
        st.subheader("Recommended Reassignments")
        st.dataframe(st.session_state.recommend_reassignment, use_container_width=True)

elif page == "Portfolio Mapping":
    st.subheader("Portfolio Mapping")
    
    # Portfolio mapping functionality
    st.info("Portfolio Mapping functionality coming soon...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution by Type")
        if not data.empty and 'TYPE' in data.columns:
            type_counts = data['TYPE'].value_counts()
            st.bar_chart(type_counts)
    
    with col2:
        st.subheader("Customer Distribution by State")
        if not data.empty and 'BILLINGSTATE' in data.columns:
            state_counts = data['BILLINGSTATE'].value_counts().head(10)
            st.bar_chart(state_counts)
    
    if not data.empty:
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(data))
        with col2:
            if 'BANK_REVENUE' in data.columns:
                st.metric("Total Revenue", f"${data['BANK_REVENUE'].sum():,.0f}")
        with col3:
            if 'DEPOSIT_BAL' in data.columns:
                st.metric("Total Deposits", f"${data['DEPOSIT_BAL'].sum():,.0f}")
        with col4:
            if 'PORT_CODE' in data.columns:
                st.metric("Unique Portfolios", data['PORT_CODE'].nunique())
