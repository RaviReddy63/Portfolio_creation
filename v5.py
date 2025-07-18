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
        return float('inf')  # Return infinity for invalid coordinates
        
    delta_lat = radians(clat - blat)
    delta_lon = radians(clon - blon)
    
    a = sin(delta_lat/2)**2 + cos(radians(clat))*cos(radians(blat))*sin(delta_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    distance = 6371*c
    return distance

def find_nearest_au(customer_lat, customer_lon, branch_data):
    """Find the nearest AU for a customer based on their coordinates"""
    min_distance = float('inf')
    nearest_au = None
    
    for _, branch in branch_data.iterrows():
        if pd.isna(branch['BRANCH_LAT_NUM']) or pd.isna(branch['BRANCH_LON_NUM']):
            continue
            
        distance = haversine_distance(
            customer_lat, customer_lon, 
            branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
        )
        
        if distance < min_distance:
            min_distance = distance
            nearest_au = branch['AU']
    
    return nearest_au, min_distance

def assign_customers_to_nearest_au(customer_data, branch_data):
    """Assign each customer to their nearest AU and calculate distance"""
    customer_data_with_au = customer_data.copy()
    
    # Initialize new columns
    customer_data_with_au['NEAREST_AU'] = None
    customer_data_with_au['DISTANCE_TO_AU'] = None
    customer_data_with_au['AU_LAT'] = None
    customer_data_with_au['AU_LON'] = None
    
    # Process each customer
    for idx, customer in customer_data_with_au.iterrows():
        if pd.isna(customer['LAT_NUM']) or pd.isna(customer['LON_NUM']):
            continue
            
        nearest_au, distance = find_nearest_au(
            customer['LAT_NUM'], customer['LON_NUM'], branch_data
        )
        
        if nearest_au is not None:
            customer_data_with_au.at[idx, 'NEAREST_AU'] = nearest_au
            customer_data_with_au.at[idx, 'DISTANCE_TO_AU'] = distance
            
            # Get AU coordinates
            au_info = branch_data[branch_data['AU'] == nearest_au].iloc[0]
            customer_data_with_au.at[idx, 'AU_LAT'] = au_info['BRANCH_LAT_NUM']
            customer_data_with_au.at[idx, 'AU_LON'] = au_info['BRANCH_LON_NUM']
    
    return customer_data_with_au

def merge_dfs(customer_data, banker_data, branch_data):
    customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    final_table = customer_data.merge(banker_data, on = "PORT_CODE", how = "left")
    final_table.fillna(0, inplace = True)
    return final_table

def create_distance_circle(center_lat, center_lon, radius_km, num_points=100):
    """Create points for a circle around a center point"""
    angles = np.linspace(0, 2*np.pi, num_points)
    circle_lats = []
    circle_lons = []
    
    for angle in angles:
        # Convert km to degrees (rough approximation)
        lat_offset = radius_km / 111.0  # 1 degree lat ≈ 111 km
        lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
        
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
    
    # Get all unique AU locations
    au_locations = set()
    for portfolio_id, df in all_portfolios.items():
        if not df.empty:
            au_locations.add((df['NEAREST_AU'].iloc[0], df['AU_LAT'].iloc[0], df['AU_LON'].iloc[0]))
    
    # Add AU markers
    for au_id, au_lat, au_lon in au_locations:
        au_details = branch_data[branch_data['AU'] == au_id]
        au_name = au_details['CITY'].iloc[0] if not au_details.empty else f"AU {au_id}"
        
        fig.add_trace(go.Scattermapbox(
            lat=[au_lat],
            lon=[au_lon],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='building'
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
            Nearest AU: {customer.get('NEAREST_AU', 'N/A')}<br>
            Portfolio ID: {customer.get('PORT_CODE', 'N/A')}<br>
            Distance to AU: {customer.get('DISTANCE_TO_AU', 0):.1f} km<br>
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

def filter_customers_by_criteria(customer_data_with_au, banker_data, selected_aus, role, cust_state, cust_portcd, max_dist, min_rev, min_deposit):
    """Filter customers based on criteria and their nearest AU assignment"""
    
    # Start with customers assigned to selected AUs
    if selected_aus:
        filtered_data = customer_data_with_au[customer_data_with_au['NEAREST_AU'].isin(selected_aus)].copy()
    else:
        filtered_data = customer_data_with_au.copy()
    
    # Rename column for merge
    filtered_data = filtered_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    
    # Merge with banker data
    filtered_data = filtered_data.merge(banker_data, on="PORT_CODE", how='left')
    
    # Apply distance filter (except for CENTRALIZED roles)
    if role is None or (role is not None and not any(r.lower().strip() == 'centralized' for r in role)):
        filtered_data = filtered_data[filtered_data['DISTANCE_TO_AU'] <= max_dist]
    
    # Apply role-specific filters
    if role is not None:
        filtered_data['TYPE_CLEAN'] = filtered_data['TYPE'].fillna('').str.strip().str.lower()
        role_clean = [r.strip().lower() for r in role]
        filtered_data = filtered_data[filtered_data['TYPE_CLEAN'].isin(role_clean)]
        filtered_data = filtered_data.drop('TYPE_CLEAN', axis=1)
    
    # Apply other filters
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    if cust_state is not None:
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
    
    if cust_portcd is not None:
        filtered_data = filtered_data[filtered_data['PORT_CODE'].isin(cust_portcd)]
    
    return filtered_data

def create_portfolios_by_au(filtered_data, branch_data):
    """Create portfolios grouped by AU"""
    portfolios_by_au = {}
    portfolio_summaries = {}
    
    # Group customers by their nearest AU
    for au_id in filtered_data['NEAREST_AU'].unique():
        if pd.isna(au_id):
            continue
            
        au_customers = filtered_data[filtered_data['NEAREST_AU'] == au_id].copy()
        
        if au_customers.empty:
            continue
        
        # Create portfolio summary for this AU
        portfolio_summary = []
        
        # Group by portfolio for assigned customers
        grouped = au_customers[au_customers['PORT_CODE'].notna()].groupby("PORT_CODE")
        
        for pid, group in grouped:
            # Find total customers in this portfolio (across all AUs)
            total_customer = len(filtered_data[filtered_data["PORT_CODE"] == pid])
            
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
        unmanaged_customers = au_customers[
            (au_customers['TYPE'].str.lower().str.strip() == 'unmanaged') |
            (au_customers['PORT_CODE'].isna())
        ]
        
        if not unmanaged_customers.empty:
            portfolio_summary.append({
                'AU': au_id,
                'Portfolio ID': 'UNMANAGED',
                'Portfolio Type': 'Unmanaged',
                'Total Customers': len(filtered_data[
                    (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                    (filtered_data['PORT_CODE'].isna())
                ]),
                'Available': len(unmanaged_customers),
                'Select': len(unmanaged_customers)
            })
        
        portfolios_by_au[au_id] = au_customers
        portfolio_summaries[au_id] = portfolio_summary
    
    return portfolios_by_au, portfolio_summaries

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

# Header with title
st.title("Portfolio Creation Tool - Nearest AU Assignment")

page = st.selectbox("Select Page", ["Portfolio Assignment", "Portfolio Mapping"])

# Initialize session state
if 'all_portfolios' not in st.session_state:
    st.session_state.all_portfolios = {}
    
if 'portfolio_controls' not in st.session_state:
    st.session_state.portfolio_controls = {}
    
if 'customer_data_with_au' not in st.session_state:
    st.session_state.customer_data_with_au = None

# Load data from local CSV files
@st.cache_data
def load_data():
    customer_data = pd.read_csv("customer_data.csv")
    banker_data = pd.read_csv("banker_data.csv")
    branch_data = pd.read_csv("branch_data.csv")
    return customer_data, banker_data, branch_data

# Load data on app startup
customer_data, banker_data, branch_data = load_data()

# Assign customers to nearest AU (cached to avoid recomputation)
@st.cache_data
def get_customer_data_with_au():
    return assign_customers_to_nearest_au(customer_data, branch_data)

if st.session_state.customer_data_with_au is None:
    with st.spinner("Assigning customers to nearest AUs..."):
        st.session_state.customer_data_with_au = get_customer_data_with_au()

customer_data_with_au = st.session_state.customer_data_with_au

if page == "Portfolio Assignment":
    
    # Show assignment summary
    st.subheader("Customer Assignment Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_customers = len(customer_data_with_au)
        assigned_customers = len(customer_data_with_au[customer_data_with_au['NEAREST_AU'].notna()])
        st.metric("Total Customers", total_customers)
    
    with col2:
        st.metric("Assigned to AUs", assigned_customers)
    
    with col3:
        unassigned_customers = total_customers - assigned_customers
        st.metric("Unassigned", unassigned_customers)
    
    # Show AU distribution
    if assigned_customers > 0:
        st.subheader("Customer Distribution by AU")
        au_distribution = customer_data_with_au['NEAREST_AU'].value_counts().head(10)
        st.bar_chart(au_distribution)
    
    # AU Selection Section
    st.subheader("Select AUs for Portfolio Creation")
    
    # Multi-select for AUs based on assigned customers
    available_aus = customer_data_with_au['NEAREST_AU'].dropna().unique()
    available_aus = sorted([int(au) for au in available_aus if not pd.isna(au)])
    
    with st.expander("Select AUs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter by state
            au_state_info = []
            for au in available_aus:
                au_info = branch_data[branch_data['AU'] == au]
                if not au_info.empty:
                    au_state_info.append((au, au_info['STATECODE'].iloc[0]))
            
            available_states = list(set([state for _, state in au_state_info if pd.notna(state)]))
            states = st.multiselect("State", sorted(available_states), key="states")
        
        with col2:
            # Filter AUs by selected states
            if states:
                filtered_aus = [au for au, state in au_state_info if state in states]
            else:
                filtered_aus = available_aus
            
            cities = []
            for au in filtered_aus:
                au_info = branch_data[branch_data['AU'] == au]
                if not au_info.empty:
                    cities.append(au_info['CITY'].iloc[0])
            
            unique_cities = sorted(list(set([city for city in cities if pd.notna(city)])))
            selected_cities = st.multiselect("City", unique_cities, key="cities")
        
        with col3:
            # Filter AUs by selected cities
            if selected_cities:
                city_filtered_aus = []
                for au in filtered_aus:
                    au_info = branch_data[branch_data['AU'] == au]
                    if not au_info.empty and au_info['CITY'].iloc[0] in selected_cities:
                        city_filtered_aus.append(au)
                final_aus = city_filtered_aus
            else:
                final_aus = filtered_aus
            
            selected_aus = st.multiselect("AU", final_aus, key="selected_aus")
    
    # Customer Selection Criteria
    st.subheader("Customer Selection Criteria")
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            role_options = list(customer_data_with_au['TYPE'].dropna().unique())
            role = st.multiselect("Role", role_options, key="role")
            if not role:
                role = None
        
        with col2:
            cust_state_options = list(customer_data_with_au['BILLINGSTATE'].dropna().unique())
            cust_state = st.multiselect("Customer State", cust_state_options, key="cust_state")
            if not cust_state:
                cust_state = None
        
        with col3:
            customer_data_temp = customer_data_with_au.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            cust_portcd = st.multiselect("Portfolio Code", customer_data_temp['PORT_CODE'].dropna().unique(), key="cust_portcd")
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5, col6 = st.columns(3)
        with col4:
            max_dist = st.slider("Max Distance from AU (km)", 1, 200, 50, key="max_distance")
        with col5:
            min_rev = st.slider("Minimum Revenue", 0, 20000, 5000, step=1000, key="min_revenue")
        with col6:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, 100000, step=5000, key="min_deposit")
    
    # Process button
    if st.button("Create Portfolios", key="create_portfolios"):
        if not selected_aus:
            st.error("Please select at least one AU")
        else:
            with st.spinner("Creating portfolios..."):
                # Filter customers based on criteria
                filtered_data = filter_customers_by_criteria(
                    customer_data_with_au, banker_data, selected_aus, 
                    role, cust_state, cust_portcd, max_dist, min_rev, min_deposit
                )
                
                if not filtered_data.empty:
                    # Create portfolios by AU
                    portfolios_by_au, portfolio_summaries = create_portfolios_by_au(filtered_data, branch_data)
                    
                    if portfolios_by_au:
                        st.success(f"Portfolios created for {len(portfolios_by_au)} AUs with {len(filtered_data)} customers")
                        
                        # Store the created portfolios data
                        st.session_state.portfolios_created = portfolios_by_au
                        st.session_state.portfolio_summaries = portfolio_summaries
                        
                        # Show Portfolio Summary Tables and Geographic Distribution
                        st.markdown("----")
                        
                        # Create horizontal layout with equal width columns
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Portfolio Summary Tables")
                            
                            # Create tabs for each AU
                            if len(portfolios_by_au) > 1:
                                au_tabs = st.tabs([f"AU {au_id}" for au_id in portfolios_by_au.keys()])
                                
                                for tab_idx, (au_id, tab) in enumerate(zip(portfolios_by_au.keys(), au_tabs)):
                                    with tab:
                                        if au_id in portfolio_summaries:
                                            portfolio_df = pd.DataFrame(portfolio_summaries[au_id])
                                            portfolio_df = portfolio_df.sort_values('Available', ascending=False).reset_index(drop=True)
                                            
                                            # Create editable dataframe
                                            edited_df = st.data_editor(
                                                portfolio_df,
                                                column_config={
                                                    "AU": st.column_config.NumberColumn("AU", disabled=True),
                                                    "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
                                                    "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
                                                    "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
                                                    "Available": st.column_config.NumberColumn("Available", disabled=True),
                                                    "Select": st.column_config.NumberColumn(
                                                        "Select",
                                                        help="Number of customers to select from this portfolio",
                                                        min_value=0,
                                                        step=1
                                                    )
                                                },
                                                hide_index=True,
                                                use_container_width=True,
                                                key=f"portfolio_editor_{au_id}_main"
                                            )
                                            
                                            # Store the edited data
                                            st.session_state.portfolio_controls[au_id] = edited_df
                                            
                                            # Summary statistics for this AU
                                            au_filtered_data = portfolios_by_au[au_id]
                                            if not au_filtered_data.empty:
                                                st.subheader("AU Summary Statistics")
                                                col_a, col_b, col_c, col_d = st.columns(4)
                                                with col_a:
                                                    st.metric("Total Customers", len(au_filtered_data))
                                                with col_b:
                                                    st.metric("Avg Distance", f"{au_filtered_data['DISTANCE_TO_AU'].mean():.1f} km")
                                                with col_c:
                                                    st.metric("Average Revenue", f"${au_filtered_data['BANK_REVENUE'].mean():,.0f}")
                                                with col_d:
                                                    st.metric("Average Deposits", f"${au_filtered_data['DEPOSIT_BAL'].mean():,.0f}")
                            else:
                                # Single AU case
                                au_id = list(portfolios_by_au.keys())[0]
                                
                                if au_id in portfolio_summaries:
                                    portfolio_df = pd.DataFrame(portfolio_summaries[au_id])
                                    portfolio_df = portfolio_df.sort_values('Available', ascending=False).reset_index(drop=True)
                                    
                                    # Create editable dataframe
                                    edited_df = st.data_editor(
                                        portfolio_df,
                                        column_config={
                                            "AU": st.column_config.NumberColumn("AU", disabled=True),
                                            "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
                                            "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
                                            "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
                                            "Available": st.column_config.NumberColumn("Available", disabled=True),
                                            "Select": st.column_config.NumberColumn(
                                                "Select",
                                                help="Number of customers to select from this portfolio",
                                                min_value=0,
                                                step=1
                                            )
                                        },
                                        hide_index=True,
                                        use_container_width=True,
                                        key=f"portfolio_editor_{au_id}_main"
                                    )
                                    
                                    # Store the edited data
                                    st.session_state.portfolio_controls[au_id] = edited_df
                                    
                                    # Summary statistics for this AU
                                    au_filtered_data = portfolios_by_au[au_id]
                                    if not au_filtered_data.empty:
                                        st.subheader("AU Summary Statistics")
                                        col_a, col_b, col_c, col_d = st.columns(4)
                                        with col_a:
                                            st.metric("Total Customers", len(au_filtered_data))
                                        with col_b:
                                            st.metric("Avg Distance", f"{au_filtered_data['DISTANCE_TO_AU'].mean():.1f} km")
                                        with col_c:
                                            st.metric("Average Revenue", f"${au_filtered_data['BANK_REVENUE'].mean():,.0f}")
                                        with col_d:
                                            st.metric("Average Deposits", f"${au_filtered_data['DEPOSIT_BAL'].mean():,.0f}")
                        
                        with col2:
                            st.subheader("Geographic Distribution")
                            
                            # Create preview portfolios for map display
                            preview_portfolios = {}
                            
                            for au_id, au_customers in portfolios_by_au.items():
                                if au_id in portfolio_summaries:
                                    portfolio_summary = portfolio_summaries[au_id]
                                    
                                    # Collect all selected customers for this AU
                                    selected_customers = []
                                    
                                    for portfolio_info in portfolio_summary:
                                        pid = portfolio_info['Portfolio ID']
                                        select_count = portfolio_info['Select']
                                        
                                        if select_count > 0:
                                            if pid == 'UNMANAGED':
                                                unmanaged_customers = au_customers[
                                                    (au_customers['TYPE'].str.lower().str.strip() == 'unmanaged') |
                                                    (au_customers['PORT_CODE'].isna())
                                                ]
                                                if not unmanaged_customers.empty:
                                                    selected = unmanaged_customers.sort_values(by='DISTANCE_TO_AU').head(select_count)
                                                    selected_customers.append(selected)
                                            else:
                                                portfolio_customers = au_customers[au_customers['PORT_CODE'] == pid]
                                                if not portfolio_customers.empty:
                                                    selected = portfolio_customers.sort_values(by='DISTANCE_TO_AU').head(select_count)
                                                    selected_customers.append(selected)
                                    
                                    # Combine all customers for this AU
                                    if selected_customers:
                                        au_portfolio = pd.concat(selected_customers, ignore_index=True)
                                        preview_portfolios[f"AU_{au_id}_Portfolio"] = au_portfolio
                            
                            # Display the map
                            if preview_portfolios:
                                combined_map = create_combined_map(preview_portfolios, branch_data)
                                if combined_map:
                                    st.plotly_chart(combined_map, use_container_width=True)
                            else:
                                st.info("No customers selected for map display")
                                
                    else:
                        st.warning("No portfolios could be created with the current filters.")
                else:
                    st.warning("No customers found matching the selected criteria.")

elif page == "Portfolio Mapping":
    st.subheader("Portfolio Mapping - Nearest AU Analysis")
    
    if customer_data_with_au is not None:
        # Show overall assignment statistics
        st.subheader("AU Assignment Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Distribution by AU")
            au_counts = customer_data_with_au['NEAREST_AU'].value_counts().head(15)
            st.bar_chart(au_counts)
        
        with col2:
            st.subheader("Average Distance to Nearest AU")
            avg_distances = customer_data_with_au.groupby('NEAREST_AU')['DISTANCE_TO_AU'].mean().head(15)
            st.bar_chart(avg_distances)
        
        # Distance analysis
        st.subheader("Distance Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_distance = customer_data_with_au['DISTANCE_TO_AU'].mean()
            st.metric("Average Distance to AU", f"{avg_distance:.1f} km")
        
        with col2:
            median_distance = customer_data_with_au['DISTANCE_TO_AU'].median()
            st.metric("Median Distance to AU", f"{median_distance:.1f} km")
        
        with col3:
            max_distance = customer_data_with_au['DISTANCE_TO_AU'].max()
            st.metric("Maximum Distance", f"{max_distance:.1f} km")
        
        with col4:
            customers_within_50km = len(customer_data_with_au[customer_data_with_au['DISTANCE_TO_AU'] <= 50])
            percentage_within_50km = (customers_within_50km / len(customer_data_with_au)) * 100
            st.metric("Within 50km", f"{percentage_within_50km:.1f}%")
        
        # Show detailed AU information
        st.subheader("AU Details with Customer Assignments")
        
        # Create AU summary table
        au_summary_data = []
        for au_id in customer_data_with_au['NEAREST_AU'].dropna().unique():
            au_customers = customer_data_with_au[customer_data_with_au['NEAREST_AU'] == au_id]
            au_info = branch_data[branch_data['AU'] == au_id]
            
            if not au_info.empty:
                au_summary_data.append({
                    'AU': au_id,
                    'City': au_info['CITY'].iloc[0],
                    'State': au_info['STATECODE'].iloc[0],
                    'Customers Assigned': len(au_customers),
                    'Avg Distance (km)': au_customers['DISTANCE_TO_AU'].mean(),
                    'Max Distance (km)': au_customers['DISTANCE_TO_AU'].max(),
                    'Total Revenue': au_customers['BANK_REVENUE'].sum(),
                    'Total Deposits': au_customers['DEPOSIT_BAL'].sum()
                })
        
        if au_summary_data:
            au_summary_df = pd.DataFrame(au_summary_data)
            au_summary_df = au_summary_df.sort_values('Customers Assigned', ascending=False)
            
            st.dataframe(
                au_summary_df,
                column_config={
                    "AU": st.column_config.NumberColumn("AU"),
                    "City": st.column_config.TextColumn("City"),
                    "State": st.column_config.TextColumn("State"),
                    "Customers Assigned": st.column_config.NumberColumn("Customers Assigned"),
                    "Avg Distance (km)": st.column_config.NumberColumn("Avg Distance (km)", format="%.1f"),
                    "Max Distance (km)": st.column_config.NumberColumn("Max Distance (km)", format="%.1f"),
                    "Total Revenue": st.column_config.NumberColumn("Total Revenue", format="$%.0f"),
                    "Total Deposits": st.column_config.NumberColumn("Total Deposits", format="$%.0f")
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Distance distribution histogram
        st.subheader("Distance Distribution")
        
        # Create histogram data
        distances = customer_data_with_au['DISTANCE_TO_AU'].dropna()
        
        import plotly.express as px
        fig_hist = px.histogram(
            x=distances,
            nbins=50,
            title="Distribution of Customer Distances to Nearest AU",
            labels={'x': 'Distance to Nearest AU (km)', 'y': 'Number of Customers'}
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Show customers that might need reassignment (very far from their nearest AU)
        st.subheader("Customers Far from Nearest AU")
        
        distance_threshold = st.slider("Distance Threshold (km)", 50, 500, 100, step=25)
        far_customers = customer_data_with_au[customer_data_with_au['DISTANCE_TO_AU'] > distance_threshold]
        
        if not far_customers.empty:
            st.warning(f"Found {len(far_customers)} customers more than {distance_threshold}km from their nearest AU")
            
            # Show sample of far customers
            display_cols = ['CG_ECN', 'BILLINGSTATE', 'NEAREST_AU', 'DISTANCE_TO_AU', 'BANK_REVENUE', 'DEPOSIT_BAL']
            far_customers_display = far_customers[display_cols].head(20)
            
            st.dataframe(
                far_customers_display,
                column_config={
                    "CG_ECN": st.column_config.TextColumn("Customer ID"),
                    "BILLINGSTATE": st.column_config.TextColumn("State"),
                    "NEAREST_AU": st.column_config.NumberColumn("Nearest AU"),
                    "DISTANCE_TO_AU": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
                    "BANK_REVENUE": st.column_config.NumberColumn("Revenue", format="$%.0f"),
                    "DEPOSIT_BAL": st.column_config.NumberColumn("Deposits", format="$%.0f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.success(f"All customers are within {distance_threshold}km of their nearest AU")
        
        # Geographic visualization of all assignments
        st.subheader("Geographic View of All AU Assignments")
        
        # Create a map showing all customers and their AU assignments
        if st.button("Generate Complete Assignment Map"):
            with st.spinner("Creating comprehensive map..."):
                # Sample data if too large
                map_data = customer_data_with_au.dropna(subset=['LAT_NUM', 'LON_NUM', 'NEAREST_AU'])
                
                if len(map_data) > 5000:
                    st.info(f"Sampling 5000 customers from {len(map_data)} total for map performance")
                    map_data = map_data.sample(n=5000, random_state=42)
                
                # Create portfolios for mapping (group by AU)
                all_au_portfolios = {}
                for au_id in map_data['NEAREST_AU'].unique():
                    au_customers = map_data[map_data['NEAREST_AU'] == au_id]
                    if not au_customers.empty:
                        all_au_portfolios[f"AU_{int(au_id)}_All"] = au_customers
                
                if all_au_portfolios:
                    complete_map = create_combined_map(all_au_portfolios, branch_data)
                    if complete_map:
                        st.plotly_chart(complete_map, use_container_width=True)
    else:
        st.error("Customer data with AU assignments not available. Please check data loading.")

# Add download functionality for AU assignments
if customer_data_with_au is not None and not customer_data_with_au.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Data")
    
    if st.sidebar.button("Download AU Assignments"):
        # Prepare data for download
        download_data = customer_data_with_au[[
            'CG_ECN', 'BILLINGSTATE', 'LAT_NUM', 'LON_NUM', 
            'NEAREST_AU', 'DISTANCE_TO_AU', 'AU_LAT', 'AU_LON',
            'BANK_REVENUE', 'DEPOSIT_BAL', 'TYPE'
        ]].copy()
        
        # Convert to CSV
        csv_data = download_data.to_csv(index=False)
        
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="customer_au_assignments.csv",
            mime="text/csv"
        )
