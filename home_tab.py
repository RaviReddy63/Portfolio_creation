import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def show_home_tab_content(customer_data, banker_data, branch_data):
    """Main function to display Home tab content with current portfolio state"""
    
    st.markdown("### 📊 Portfolio Dashboard - Current State")
    st.markdown("---")
    
    # Merge data for analysis
    portfolio_data = prepare_portfolio_data(customer_data, banker_data, branch_data)
    
    # Create filters
    filtered_data = create_home_filters(portfolio_data)
    
    if filtered_data.empty:
        st.warning("No data found for the selected filters.")
        return
    
    # Display summary statistics with visuals
    display_portfolio_metrics(filtered_data)
    
    st.markdown("---")
    
    # Display map
    st.subheader("🗺️ Portfolio Geographic Distribution")
    create_portfolio_map(filtered_data, branch_data)

def prepare_portfolio_data(customer_data, banker_data, branch_data):
    """Prepare and merge all data for portfolio analysis"""
    
    # Start with customer data
    portfolio_data = customer_data.copy()
    
    # Rename portfolio column for consistency
    if 'CG_PORTFOLIO_CD' in portfolio_data.columns:
        portfolio_data = portfolio_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    
    # Merge with banker data
    portfolio_data = portfolio_data.merge(banker_data, on="PORT_CODE", how='left')
    
    # Create banker hierarchy fields if they don't exist
    if 'DIRECTOR_NAME' not in portfolio_data.columns:
        portfolio_data['DIRECTOR_NAME'] = portfolio_data.get('BANKER_FIRSTNAME', 'Unknown') + ' ' + portfolio_data.get('BANKER_LASTNAME', 'Director')
    
    if 'MANAGER_NAME' not in portfolio_data.columns:
        portfolio_data['MANAGER_NAME'] = portfolio_data.get('BANKER_FIRSTNAME', 'Unknown') + ' ' + portfolio_data.get('BANKER_LASTNAME', 'Manager')
    
    if 'BANKER_NAME' not in portfolio_data.columns:
        portfolio_data['BANKER_NAME'] = portfolio_data.get('BANKER_FIRSTNAME', 'Unknown') + ' ' + portfolio_data.get('BANKER_LASTNAME', 'Banker')
    
    # Clean and categorize portfolio types
    portfolio_data['COVERAGE'] = portfolio_data['TYPE'].fillna('Unassigned').str.strip().str.title()
    
    # Standardize coverage categories
    coverage_mapping = {
        'In-Market': 'In-Market',
        'Inmarket': 'In-Market', 
        'In Market': 'In-Market',
        'Centralized': 'Centralized',
        'Unassigned': 'Unassigned',
        'Unmanaged': 'Unmanaged'
    }
    
    portfolio_data['COVERAGE'] = portfolio_data['COVERAGE'].map(coverage_mapping).fillna('Other')
    
    # Fill missing values
    portfolio_data['BANK_REVENUE'] = pd.to_numeric(portfolio_data['BANK_REVENUE'], errors='coerce').fillna(0)
    portfolio_data['DEPOSIT_BAL'] = pd.to_numeric(portfolio_data['DEPOSIT_BAL'], errors='coerce').fillna(0)
    portfolio_data['CG_GROSS_SALES'] = pd.to_numeric(portfolio_data.get('CG_GROSS_SALES', 0), errors='coerce').fillna(0)
    
    return portfolio_data

def create_home_filters(portfolio_data):
    """Create filter section for Home tab with default director filter"""
    
    st.subheader("🔍 Portfolio Filters")
    
    with st.expander("Filter Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Director filter - single select with default
            director_options = ['All'] + sorted(portfolio_data['DIRECTOR_NAME'].dropna().unique().tolist())
            
            # Set default to first director (not 'All') for better performance
            default_director = director_options[1] if len(director_options) > 1 else director_options[0]
            
            selected_director = st.selectbox(
                "Director Name", 
                director_options, 
                index=director_options.index(default_director) if default_director in director_options else 0,
                key="home_director",
                help="Default filter applied for better performance"
            )
            
            # Manager filter - multi select
            manager_options = sorted(portfolio_data['MANAGER_NAME'].dropna().unique().tolist())
            selected_managers = st.multiselect("Manager Name", manager_options, key="home_managers")
        
        with col2:
            # Banker filter - multi select
            banker_options = sorted(portfolio_data['BANKER_NAME'].dropna().unique().tolist())
            selected_bankers = st.multiselect("Banker Name", banker_options, key="home_bankers")
            
            # Portfolio Code filter - multi select
            portfolio_options = sorted(portfolio_data['PORT_CODE'].dropna().unique().tolist())
            selected_portfolios = st.multiselect("Portfolio Code", portfolio_options, key="home_portfolios")
        
        with col3:
            # Billing State filter - multi select
            state_options = sorted(portfolio_data['BILLINGSTATE'].dropna().unique().tolist())
            selected_states = st.multiselect("Billing State", state_options, key="home_states")
            
            # Type filter - multi select (replaced Coverage)
            type_options = sorted(portfolio_data['TYPE'].dropna().unique().tolist())
            selected_types = st.multiselect("Type", type_options, key="home_types")
    
    # Apply filters with early filtering for performance
    filtered_data = portfolio_data.copy()
    
    # Apply director filter first for performance (most restrictive)
    if selected_director != 'All':
        filtered_data = filtered_data[filtered_data['DIRECTOR_NAME'] == selected_director]
    
    # Apply multi-select filters (only if selections are made)
    if selected_managers:
        filtered_data = filtered_data[filtered_data['MANAGER_NAME'].isin(selected_managers)]
    
    if selected_bankers:
        filtered_data = filtered_data[filtered_data['BANKER_NAME'].isin(selected_bankers)]
    
    if selected_portfolios:
        filtered_data = filtered_data[filtered_data['PORT_CODE'].isin(selected_portfolios)]
    
    if selected_states:
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(selected_states)]
    
    if selected_types:
        filtered_data = filtered_data[filtered_data['TYPE'].isin(selected_types)]
    
    # Show filter summary for user feedback
    if selected_director != 'All':
        filter_summary = [f"Director: **{selected_director}**"]
        
        if selected_managers:
            filter_summary.append(f"Managers: {len(selected_managers)} selected")
        if selected_bankers:
            filter_summary.append(f"Bankers: {len(selected_bankers)} selected")
        if selected_portfolios:
            filter_summary.append(f"Portfolios: {len(selected_portfolios)} selected")
        if selected_states:
            filter_summary.append(f"States: {len(selected_states)} selected")
        if selected_types:
            filter_summary.append(f"Types: {len(selected_types)} selected")
        
        filter_text = " | ".join(filter_summary)
        st.info(f"📊 Active Filters: {filter_text} | **{len(filtered_data):,} customers**")
    
    return filtered_data

def display_portfolio_metrics(filtered_data):
    """Display portfolio metrics with visualizations in specified order"""
    
    st.subheader("📈 Portfolio Metrics")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(filtered_data)
    
    # Row 1: Portfolio counts
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Portfolios", f"{metrics['total_portfolios']:,}")
    
    with col2:
        st.metric("🎯 In-Market Portfolios", f"{metrics['inmarket_portfolios']:,}")
    
    with col3:
        st.metric("🏢 Centralized Portfolios", f"{metrics['centralized_portfolios']:,}")
    
    with col4:
        st.metric("❓ Unassigned Portfolios", f"{metrics['unassigned_portfolios']:,}")
    
    # Row 2: Customer counts
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("👥 Total Customers", f"{metrics['total_customers']:,}")
    
    with col6:
        st.metric("📍 In-Market Customers", f"{metrics['inmarket_customers']:,}")
    
    with col7:
        st.metric("🏢 Centralized Customers", f"{metrics['centralized_customers']:,}")
    
    with col8:
        st.metric("❓ Unassigned Customers", f"{metrics['unassigned_customers']:,}")
    
    # Row 3: Financial metrics and unmanaged
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        st.metric("💰 Avg Bank Revenue", f"${metrics['avg_revenue']:,.0f}")
    
    with col10:
        st.metric("🏦 Avg Deposit Balance", f"${metrics['avg_deposits']:,.0f}")
    
    with col11:
        st.metric("📊 Avg Gross Sales", f"${metrics['avg_gross_sales']:,.0f}")
    
    with col12:
        st.metric("⚪ Total Unmanaged Customers", f"{metrics['unmanaged_customers']:,}")
    
    # Create visualizations
    create_metrics_charts(filtered_data, metrics)

def calculate_portfolio_metrics(filtered_data):
    """Calculate all portfolio metrics with safety checks"""
    
    # Safety check for empty data
    if filtered_data.empty:
        return {
            'total_customers': 0,
            'total_portfolios': 0,
            'inmarket_portfolios': 0,
            'centralized_portfolios': 0,
            'unassigned_portfolios': 0,
            'inmarket_customers': 0,
            'centralized_customers': 0,
            'unassigned_customers': 0,
            'unmanaged_customers': 0,
            'avg_revenue': 0,
            'avg_deposits': 0,
            'avg_gross_sales': 0,
            'utilization_rate': 0
        }
    
    # Basic counts
    total_customers = len(filtered_data)
    
    # Safe portfolio counting
    total_portfolios = filtered_data['PORT_CODE'].nunique() if 'PORT_CODE' in filtered_data.columns else 0
    
    # Coverage breakdown with safety checks
    if 'COVERAGE' in filtered_data.columns:
        coverage_counts = filtered_data['COVERAGE'].value_counts()
        inmarket_customers = coverage_counts.get('In-Market', 0)
        centralized_customers = coverage_counts.get('Centralized', 0)
        unassigned_customers = coverage_counts.get('Unassigned', 0)
        unmanaged_customers = coverage_counts.get('Unmanaged', 0)
    else:
        inmarket_customers = centralized_customers = unassigned_customers = unmanaged_customers = 0
    
    # Portfolio type counts with safety checks
    if 'PORT_CODE' in filtered_data.columns and 'COVERAGE' in filtered_data.columns:
        inmarket_portfolios = len(filtered_data[filtered_data['COVERAGE'] == 'In-Market']['PORT_CODE'].unique())
        centralized_portfolios = len(filtered_data[filtered_data['COVERAGE'] == 'Centralized']['PORT_CODE'].unique())
        unassigned_portfolios = len(filtered_data[filtered_data['COVERAGE'] == 'Unassigned']['PORT_CODE'].unique())
    else:
        inmarket_portfolios = centralized_portfolios = unassigned_portfolios = 0
    
    # Financial metrics with safety checks
    avg_revenue = filtered_data['BANK_REVENUE'].mean() if 'BANK_REVENUE' in filtered_data.columns else 0
    avg_deposits = filtered_data['DEPOSIT_BAL'].mean() if 'DEPOSIT_BAL' in filtered_data.columns else 0
    avg_gross_sales = filtered_data['CG_GROSS_SALES'].mean() if 'CG_GROSS_SALES' in filtered_data.columns else 0
    
    # Handle NaN values
    avg_revenue = avg_revenue if not pd.isna(avg_revenue) else 0
    avg_deposits = avg_deposits if not pd.isna(avg_deposits) else 0
    avg_gross_sales = avg_gross_sales if not pd.isna(avg_gross_sales) else 0
    
    # Utilization rate (managed customers vs total)
    managed_customers = inmarket_customers + centralized_customers
    utilization_rate = (managed_customers / total_customers * 100) if total_customers > 0 else 0
    
    return {
        'total_customers': total_customers,
        'total_portfolios': total_portfolios,
        'inmarket_portfolios': inmarket_portfolios,
        'centralized_portfolios': centralized_portfolios,
        'unassigned_portfolios': unassigned_portfolios,
        'inmarket_customers': inmarket_customers,
        'centralized_customers': centralized_customers,
        'unassigned_customers': unassigned_customers,
        'unmanaged_customers': unmanaged_customers,
        'avg_revenue': avg_revenue,
        'avg_deposits': avg_deposits,
        'avg_gross_sales': avg_gross_sales,
        'utilization_rate': utilization_rate
    }

def create_metrics_charts(filtered_data, metrics):
    """Create visualization charts for metrics with safety checks"""
    
    st.markdown("### 📊 Visual Analytics")
    
    # Safety check for empty data
    if filtered_data.empty:
        st.info("No data available for charts")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Coverage distribution pie chart with safety checks
        if 'COVERAGE' in filtered_data.columns:
            coverage_data = filtered_data['COVERAGE'].value_counts()
            if not coverage_data.empty:
                fig_pie = px.pie(
                    values=coverage_data.values,
                    names=coverage_data.index,
                    title="Customer Distribution by Coverage Type",
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No coverage data available for pie chart")
        else:
            st.info("Coverage column not found")
    
    with col2:
        # Revenue by coverage type with safety checks
        if 'COVERAGE' in filtered_data.columns and 'BANK_REVENUE' in filtered_data.columns:
            revenue_by_coverage = filtered_data.groupby('COVERAGE')['BANK_REVENUE'].mean().reset_index()
            if not revenue_by_coverage.empty:
                fig_bar = px.bar(
                    revenue_by_coverage,
                    x='COVERAGE',
                    y='BANK_REVENUE',
                    title="Average Revenue by Coverage Type",
                    color='COVERAGE',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                )
                fig_bar.update_layout(height=400, showlegend=False, yaxis_title="Average Revenue ($)")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No revenue data available for bar chart")
        else:
            st.info("Required columns not found for revenue chart")
    
    # State-wise distribution with safety checks
    if ('BILLINGSTATE' in filtered_data.columns and 
        'COVERAGE' in filtered_data.columns and 
        len(filtered_data['BILLINGSTATE'].unique()) > 1):
        
        st.markdown("#### State-wise Portfolio Distribution")
        state_coverage = filtered_data.groupby(['BILLINGSTATE', 'COVERAGE']).size().reset_index(name='Count')
        
        if not state_coverage.empty:
            fig_state = px.bar(
                state_coverage,
                x='BILLINGSTATE',
                y='Count',
                color='COVERAGE',
                title="Customer Count by State and Coverage Type",
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
            fig_state.update_layout(height=400)
            st.plotly_chart(fig_state, use_container_width=True)
        else:
            st.info("No state-wise data available")

def create_portfolio_map(filtered_data, branch_data):
    """Create interactive map showing current portfolio state"""
    
    # Safety check for empty data
    if filtered_data.empty:
        st.warning("No data to display on map")
        return
    
    # Check for valid coordinates
    valid_customer_data = filtered_data.dropna(subset=['LAT_NUM', 'LON_NUM'])
    if valid_customer_data.empty:
        st.warning("No valid coordinates found for mapping")
        return
    
    # Debug info
    st.write(f"🔍 Debug: {len(valid_customer_data)} customers with valid coordinates")
    unique_portfolios_debug = valid_customer_data['PORT_CODE'].dropna().unique()
    st.write(f"🔍 Debug: {len(unique_portfolios_debug)} unique portfolios found: {list(unique_portfolios_debug)[:5]}...")
    
    fig = go.Figure()
    
    # Color palette for portfolios - Fixed approach
    unique_portfolios = valid_customer_data['PORT_CODE'].dropna().unique()
    
    # Safety check for empty portfolios
    if len(unique_portfolios) == 0:
        st.warning("No portfolios found with current filters")
        return
    
    # Create color mapping with more colors
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Extend colors if needed
    while len(colors) < len(unique_portfolios):
        colors.extend(colors)
    
    portfolio_colors = dict(zip(unique_portfolios, colors[:len(unique_portfolios)]))
    
    # Add customer dots by portfolio (different colors) - with debugging
    customers_plotted = 0
    for i, portfolio in enumerate(unique_portfolios):
        portfolio_customers = valid_customer_data[valid_customer_data['PORT_CODE'] == portfolio]
        
        # Safety check - skip empty portfolios
        if portfolio_customers.empty:
            continue
        
        # Use coordinates directly without additional filtering
        plot_data = portfolio_customers[['LAT_NUM', 'LON_NUM']].dropna()
        if plot_data.empty:
            continue
        
        # Create hover text
        hover_text = []
        for _, customer in portfolio_customers.iterrows():
            hover_text.append(f"""
            <b>{customer.get('CG_ECN', 'N/A')}</b><br>
            Portfolio: {portfolio}<br>
            Coverage: {customer.get('COVERAGE', 'N/A')}<br>
            Banker: {customer.get('BANKER_NAME', 'N/A')}<br>
            Revenue: ${customer.get('BANK_REVENUE', 0):,.0f}<br>
            Deposits: ${customer.get('DEPOSIT_BAL', 0):,.0f}<br>
            State: {customer.get('BILLINGSTATE', 'N/A')}
            """)
        
        # Get color for this portfolio
        portfolio_color = portfolio_colors[portfolio]
        
        # Add trace with explicit color
        fig.add_trace(go.Scattermapbox(
            lat=portfolio_customers['LAT_NUM'],
            lon=portfolio_customers['LON_NUM'],
            mode='markers',
            marker=dict(
                size=8,
                color=portfolio_color,
                opacity=0.8
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            name=f"Portfolio {portfolio}",
            showlegend=False
        ))
        
        customers_plotted += len(portfolio_customers)
    
    st.write(f"🔍 Debug: {customers_plotted} customers plotted on map")
    
    # Skip AU plotting for now since current state data might not have AU assignments
    # Focus on getting customer colors working first
    
    # Add legend traces (invisible, just for legend)
    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=8, color='blue', symbol='circle'),
        name="🔵 Customers",
        showlegend=True
    ))
    
    # Calculate map center
    if not valid_customer_data.empty:
        center_lat = valid_customer_data['LAT_NUM'].mean()
        center_lon = valid_customer_data['LON_NUM'].mean()
        
        # Calculate zoom based on data spread
        lat_range = valid_customer_data['LAT_NUM'].max() - valid_customer_data['LAT_NUM'].min()
        lon_range = valid_customer_data['LON_NUM'].max() - valid_customer_data['LON_NUM'].min()
        max_range = max(lat_range, lon_range) if lat_range > 0 and lon_range > 0 else 5
        
        if max_range > 20:
            zoom = 4
        elif max_range > 10:
            zoom = 5
        elif max_range > 5:
            zoom = 6
        else:
            zoom = 7
    else:
        # Default US center if no valid coordinates
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
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
