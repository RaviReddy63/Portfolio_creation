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
    distance = 6371*c
    return distance

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
            au_locations.add((df['AU'].iloc[0], df['BRANCH_LAT_NUM'].iloc[0], df['BRANCH_LON_NUM'].iloc[0]))
    
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
            AU Portfolio: {au_id}<br>
            Portfolio ID: {customer.get('PORT_CODE', 'N/A')}<br>
            Distance: {customer.get('Distance', 0):.1f} km<br>
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

def filter_customers_for_au(customer_data, banker_data, selected_au, branch_data, role, cust_state, cust_portcd, max_dist, min_rev, min_deposit):
    """Filter customers for a specific AU based on criteria"""
    
    # Get AU data
    AU_row = branch_data[branch_data['AU'] == int(selected_au)].iloc[0]
    AU_lat = AU_row['BRANCH_LAT_NUM']
    AU_lon = AU_row['BRANCH_LON_NUM']
    
    # Filter customers by distance box
    box_lat = max_dist/111
    box_lon = max_dist/ (111 * np.cos(np.radians(AU_lat)))
    
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
    
    # Apply distance filter for all roles except CENTRALIZED
    if role is None or (role is not None and not any(r.lower().strip() == 'centralized' for r in role)):
        filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
    
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
    
    # Add AU information
    filtered_data['AU'] = selected_au
    filtered_data['BRANCH_LAT_NUM'] = AU_lat
    filtered_data['BRANCH_LON_NUM'] = AU_lon
    
    return filtered_data, AU_row

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

# Header with title
st.title("Portfolio Creation Tool")

page = st.selectbox("Select Page", ["Portfolio Assignment", "Portfolio Mapping"])

# Initialize session state
if 'all_portfolios' not in st.session_state:
    st.session_state.all_portfolios = {}
    
if 'portfolio_controls' not in st.session_state:
    st.session_state.portfolio_controls = {}
    
if 'recommend_reassignment' not in st.session_state:
    st.session_state.recommend_reassignment = {}

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
    st.subheader("Select AUs for Portfolio Creation")
    
    # Multi-select for AUs
    with st.expander("Select AUs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            states = st.multiselect("State", branch_data['STATECODE'].dropna().unique(), key="states")
        
        # Filter branch data based on selected states
        if states:
            filtered_branch_data = branch_data[branch_data['STATECODE'].isin(states)]
        else:
            filtered_branch_data = branch_data
            
        with col2:
            cities = st.multiselect("City", filtered_branch_data['CITY'].dropna().unique(), key="cities")
        
        # Filter further based on selected cities
        if cities:
            filtered_branch_data = filtered_branch_data[filtered_branch_data['CITY'].isin(cities)]
        
        with col3:
            selected_aus = st.multiselect("AU", filtered_branch_data['AU'].dropna().unique(), key="selected_aus")
    
    # Customer Selection Criteria
    st.subheader("Customer Selection Criteria")
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            role_options = list(customer_data['TYPE'].dropna().unique())
            role = st.multiselect("Role", role_options, key="role")
            if not role:
                role = None
        
        with col2:
            cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
            cust_state = st.multiselect("Customer State", cust_state_options, key="cust_state")
            if not cust_state:
                cust_state = None
        
        with col3:
            customer_data_temp = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            cust_portcd = st.multiselect("Portfolio Code", customer_data_temp['PORT_CODE'].dropna().unique(), key="cust_portcd")
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5, col6 = st.columns(3)
        with col4:
            max_dist = st.slider("Max Distance (km)", 1, 100, 20, key="max_distance")
        with col5:
            min_rev = st.slider("Minimum Revenue", 0, 20000, 5000, step=1000, key="min_revenue")
        with col6:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, 100000, step=5000, key="min_deposit")
    
    # Process button
    if st.button("Create Portfolios", key="create_portfolios"):
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
                st.success(f"Portfolios created for {len(portfolios_created)} AUs")
                
                # Store the created portfolios data
                st.session_state.portfolios_created = portfolios_created
                
                # Show map immediately after creating portfolios
                st.markdown("----")
                
                # Create layout with table and map side by side
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
                                    au_filtered_data = portfolios_created[au_id]
                                    if not au_filtered_data.empty:
                                        st.subheader("AU Summary Statistics")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Total Customers", len(au_filtered_data))
                                        with col2:
                                            st.metric("Avg Distance", f"{au_filtered_data['Distance'].mean():.1f} km")
                                        with col3:
                                            st.metric("Average Revenue", f"${au_filtered_data['BANK_REVENUE'].mean():,.0f}")
                                        with col4:
                                            st.metric("Average Deposits", f"${au_filtered_data['DEPOSIT_BAL'].mean():,.0f}")
                    else:
                        # Single AU case
                        au_id = list(portfolios_created.keys())[0]
                        
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
                            au_filtered_data = portfolios_created[au_id]
                            if not au_filtered_data.empty:
                                st.subheader("AU Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Customers", len(au_filtered_data))
                                with col2:
                                    st.metric("Avg Distance", f"{au_filtered_data['Distance'].mean():.1f} km")
                                with col3:
                                    st.metric("Average Revenue", f"${au_filtered_data['BANK_REVENUE'].mean():,.0f}")
                                with col4:
                                    st.metric("Average Deposits", f"${au_filtered_data['DEPOSIT_BAL'].mean():,.0f}")
                
                with col2:
                    st.subheader("Geographic Distribution")
                    
                    # Create preview portfolios for map display - one portfolio per AU
                    preview_portfolios = {}
                    
                    for au_id, filtered_data in portfolios_created.items():
                        if au_id in portfolio_summaries:
                            portfolio_summary = portfolio_summaries[au_id]
                            
                            # Collect all selected customers for this AU (this becomes one portfolio)
                            au_customers = []
                            
                            for portfolio_info in portfolio_summary:
                                pid = portfolio_info['Portfolio ID']
                                select_count = portfolio_info['Select']
                                
                                if select_count > 0:
                                    if pid == 'UNMANAGED':
                                        # Handle unmanaged customers
                                        unmanaged_customers = filtered_data[
                                            (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                                            (filtered_data['PORT_CODE'].isna())
                                        ]
                                        if not unmanaged_customers.empty:
                                            selected_customers = unmanaged_customers.sort_values(by='Distance').head(select_count)
                                            au_customers.append(selected_customers)
                                    else:
                                        # Handle regular portfolios
                                        portfolio_customers = filtered_data[filtered_data['PORT_CODE'] == pid]
                                        if not portfolio_customers.empty:
                                            selected_customers = portfolio_customers.sort_values(by='Distance').head(select_count)
                                            au_customers.append(selected_customers)
                            
                            # Combine all customers for this AU into one portfolio
                            if au_customers:
                                au_portfolio = pd.concat(au_customers, ignore_index=True)
                                preview_portfolios[f"AU_{au_id}_Portfolio"] = au_portfolio
                    
                    # Display the map with preview portfolios
                    if preview_portfolios:
                        combined_map = create_combined_map(preview_portfolios, branch_data)
                        if combined_map:
                            st.plotly_chart(combined_map, use_container_width=True)
                    else:
                        st.info("No customers selected for map display")
                    
                    # Also show final portfolio distribution in the same column
                    if 'portfolios_created' in st.session_state and st.session_state.portfolios_created:
                        st.markdown("---")
                        st.subheader("Final Portfolio Distribution")
                        
                        # Create final portfolios for display
                        final_portfolios = {}
                        
                        for au_id, filtered_data in st.session_state.portfolios_created.items():
                            if au_id in st.session_state.portfolio_controls:
                                edited_df = st.session_state.portfolio_controls[au_id]
                                
                                # Collect all selected customers for this AU (this becomes one portfolio)
                                au_customers = []
                                
                                for _, row in edited_df.iterrows():
                                    if row['Select'] > 0:
                                        pid = row['Portfolio ID']
                                        select_count = int(row['Select'])
                                        
                                        if pid == 'UNMANAGED':
                                            # Handle unmanaged customers
                                            unmanaged_customers = filtered_data[
                                                (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                                                (filtered_data['PORT_CODE'].isna())
                                            ]
                                            if not unmanaged_customers.empty:
                                                selected_customers = unmanaged_customers.sort_values(by='Distance').head(select_count)
                                                au_customers.append(selected_customers)
                                        else:
                                            # Handle regular portfolios
                                            portfolio_customers = filtered_data[filtered_data['PORT_CODE'] == pid]
                                            if not portfolio_customers.empty:
                                                selected_customers = portfolio_customers.sort_values(by='Distance').head(select_count)
                                                au_customers.append(selected_customers)
                                
                                # Combine all customers for this AU into one portfolio
                                if au_customers:
                                    au_portfolio = pd.concat(au_customers, ignore_index=True)
                                    final_portfolios[f"AU_{au_id}_Portfolio"] = au_portfolio
                        
                        if final_portfolios:
                            combined_map_final = create_combined_map(final_portfolios, branch_data)
                            if combined_map_final:
                                st.plotly_chart(combined_map_final, use_container_width=True)
                            
                            # Summary statistics in the map column
                            st.subheader("Summary Statistics")
                            total_customers = sum(len(df) for df in final_portfolios.values())
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Portfolios", len(final_portfolios))
                                st.metric("Total Customers", total_customers)
                            with col2:
                                if total_customers > 0:
                                    all_data = pd.concat(final_portfolios.values())
                                    st.metric("Average Revenue", f"${all_data['BANK_REVENUE'].mean():,.0f}")
                                    st.metric("Average Deposits", f"${all_data['DEPOSIT_BAL'].mean():,.0f}")
                            
                            # Recommendation and export buttons in the map column
                            st.markdown("---")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.button("Recommended Reassignment"):
                                    rec_df = recommend_reassignment(final_portfolios)
                                    st.session_state.recommend_reassignment = rec_df
                                    st.subheader("Recommended Reassignments")
                                    st.dataframe(st.session_state.recommend_reassignment, use_container_width=True)
                            
                            with col2:
                                if st.button("Export to Excel"):
                                    excel_buffer = to_excel(final_portfolios)
                                    st.download_button(
                                        label="Download Excel Report",
                                        data=excel_buffer,
                                        file_name="portfolio_assignments.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                
            else:
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
