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
    """Create filter section for Home tab"""
    
    st.subheader("🔍 Portfolio Filters")
    
    with st.expander("Filter Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Director filter
            director_options = ['All'] + sorted(portfolio_data['DIRECTOR_NAME'].dropna().unique().tolist())
            selected_director = st.selectbox("Director Name", director_options, key="home_director")
            
            # Manager filter  
            manager_options = ['All'] + sorted(portfolio_data['MANAGER_NAME'].dropna().unique().tolist())
            selected_manager = st.selectbox("Manager Name", manager_options, key="home_manager")
        
        with col2:
            # Banker filter
            banker_options = ['All'] + sorted(portfolio_data['BANKER_NAME'].dropna().unique().tolist())
            selected_banker = st.selectbox("Banker Name", banker_options, key="home_banker")
            
            # Portfolio Code filter
            portfolio_options = ['All'] + sorted(portfolio_data['PORT_CODE'].dropna().unique().tolist())
            selected_portfolio = st.selectbox("Portfolio Code", portfolio_options, key="home_portfolio")
        
        with col3:
            # Billing State filter
            state_options = ['All'] + sorted(portfolio_data['BILLINGSTATE'].dropna().unique().tolist())
            selected_state = st.selectbox("Billing State", state_options, key="home_state")
            
            # Coverage filter
            coverage_options = ['All'] + sorted(portfolio_data['COVERAGE'].dropna().unique().tolist())
            selected_coverage = st.selectbox("Coverage", coverage_options, key="home_coverage")
    
    # Apply filters
    filtered_data = portfolio_data.copy()
    
    if selected_director != 'All':
        filtered_data = filtered_data[filtered_data['DIRECTOR_NAME'] == selected_director]
    
    if selected_manager != 'All':
        filtered_data = filtered_data[filtered_data['MANAGER_NAME'] == selected_manager]
    
    if selected_banker != 'All':
        filtered_data = filtered_data[filtered_data['BANKER_NAME'] == selected_banker]
    
    if selected_portfolio != 'All':
        filtered_data = filtered_data[filtered_data['PORT_CODE'] == selected_portfolio]
    
    if selected_state != 'All':
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'] == selected_state]
    
    if selected_coverage != 'All':
        filtered_data = filtered_data[filtered_data['COVERAGE'] == selected_coverage]
    
    return filtered_data

def display_portfolio_metrics(filtered_data):
    """Display portfolio metrics with visualizations"""
    
    st.subheader("📈 Portfolio Metrics")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(filtered_data)
    
    # Create metrics display in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Portfolios", f"{metrics['total_portfolios']:,}")
        st.metric("🎯 In-Market Portfolios", f"{metrics['inmarket_portfolios']:,}")
    
    with col2:
        st.metric("🏢 Centralized Portfolios", f"{metrics['centralized_portfolios']:,}")
        st.metric("💰 Avg Bank Revenue", f"${metrics['avg_revenue']:,.0f}")
    
    with col3:
        st.metric("🏦 Avg Deposit Balance", f"${metrics['avg_deposits']:,.0f}")
        st.metric("📊 Avg Gross Sales", f"${metrics['avg_gross_sales']:,.0f}")
    
    with col4:
        st.metric("👥 Total Customers", f"{metrics['total_customers']:,}")
        st.metric("📍 In-Market Customers", f"{metrics['inmarket_customers']:,}")
    
    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("🏢 Centralized Customers", f"{metrics['centralized_customers']:,}")
    
    with col6:
        st.metric("❓ Unassigned Customers", f"{metrics['unassigned_customers']:,}")
    
    with col7:
        st.metric("⚪ Unmanaged Customers", f"{metrics['unmanaged_customers']:,}")
    
    with col8:
        st.metric("📈 Portfolio Utilization", f"{metrics['utilization_rate']:.1f}%")
    
    # Create visualizations
    create_metrics_charts(filtered_data, metrics)

def calculate_portfolio_metrics(filtered_data):
    """Calculate all portfolio metrics"""
    
    # Basic counts
    total_customers = len(filtered_data)
    total_portfolios = filtered_data['PORT_CODE'].nunique()
    
    # Coverage breakdown
    coverage_counts = filtered_data['COVERAGE'].value_counts()
    inmarket_customers = coverage_counts.get('In-Market', 0)
    centralized_customers = coverage_counts.get('Centralized', 0)
    unassigned_customers = coverage_counts.get('Unassigned', 0)
    unmanaged_customers = coverage_counts.get('Unmanaged', 0)
    
    # Portfolio type counts
    inmarket_portfolios = len(filtered_data[filtered_data['COVERAGE'] == 'In-Market']['PORT_CODE'].unique())
    centralized_portfolios = len(filtered_data[filtered_data['COVERAGE'] == 'Centralized']['PORT_CODE'].unique())
    
    # Financial metrics
    avg_revenue = filtered_data['BANK_REVENUE'].mean()
    avg_deposits = filtered_data['DEPOSIT_BAL'].mean()
    avg_gross_sales = filtered_data['CG_GROSS_SALES'].mean()
    
    # Utilization rate (managed customers vs total)
    managed_customers = inmarket_customers + centralized_customers
    utilization_rate = (managed_customers / total_customers * 100) if total_customers > 0 else 0
    
    return {
        'total_customers': total_customers,
        'total_portfolios': total_portfolios,
        'inmarket_portfolios': inmarket_portfolios,
        'centralized_portfolios': centralized_portfolios,
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
    """Create visualization charts for metrics"""
    
    st.markdown("### 📊 Visual Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Coverage distribution pie chart
        coverage_data = filtered_data['COVERAGE'].value_counts()
        fig_pie = px.pie(
            values=coverage_data.values,
            names=coverage_data.index,
            title="Customer Distribution by Coverage Type",
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Revenue by coverage type
        revenue_by_coverage = filtered_data.groupby('COVERAGE')['BANK_REVENUE'].mean().reset_index()
        fig_bar = px.bar(
            revenue_by_coverage,
            x='COVERAGE',
            y='BANK_REVENUE',
            title="Average Revenue by Coverage Type",
            color='COVERAGE',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
        fig_bar.update_layout(height=400, showlegend=False)
        fig_bar.update_layout(yaxis_title="Average Revenue ($)")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # State-wise distribution
    if len(filtered_data['BILLINGSTATE'].unique()) > 1:
        st.markdown("#### State-wise Portfolio Distribution")
        state_coverage = filtered_data.groupby(['BILLINGSTATE', 'COVERAGE']).size().reset_index(name='Count')
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

def create_portfolio_map(filtered_data, branch_data):
    """Create interactive map showing current portfolio state"""
    
    if filtered_data.empty:
        st.warning("No data to display on map")
        return
    
    fig = go.Figure()
    
    # Color palette for portfolios
    unique_portfolios = filtered_data['PORT_CODE'].unique()
    colors = px.colors.qualitative.Set3[:len(unique_portfolios)]
    portfolio_colors = dict(zip(unique_portfolios, colors))
    
    # Add customer dots by portfolio (different colors)
    for portfolio in unique_portfolios:
        portfolio_customers = filtered_data[filtered_data['PORT_CODE'] == portfolio]
        
        if portfolio_customers.empty:
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
        
        fig.add_trace(go.Scattermapbox(
            lat=portfolio_customers['LAT_NUM'],
            lon=portfolio_customers['LON_NUM'],
            mode='markers',
            marker=dict(
                size=6,
                color=portfolio_colors[portfolio],
                opacity=0.7
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            name=f"Customers",
            showlegend=False  # Don't show portfolio colors in legend
        ))
    
    # Add In-Market AU triangles
    inmarket_data = filtered_data[filtered_data['COVERAGE'] == 'In-Market']
    if not inmarket_data.empty:
        # Get unique AUs for in-market customers
        inmarket_aus = inmarket_data.groupby(['AU', 'COVERAGE']).first().reset_index()
        
        for _, au_data in inmarket_aus.iterrows():
            # Get AU coordinates from branch_data
            au_info = branch_data[branch_data['AU'] == au_data['AU']]
            if not au_info.empty:
                au_lat = au_info.iloc[0]['BRANCH_LAT_NUM']
                au_lon = au_info.iloc[0]['BRANCH_LON_NUM']
                au_city = au_info.iloc[0].get('CITY', f"AU {au_data['AU']}")
                
                customer_count = len(inmarket_data[inmarket_data['AU'] == au_data['AU']])
                
                fig.add_trace(go.Scattermapbox(
                    lat=[au_lat],
                    lon=[au_lon],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='blue',
                        symbol='triangle-up',
                        line=dict(width=2, color='darkblue')
                    ),
                    hovertemplate=f"""
                    <b>In-Market AU {au_data['AU']}</b><br>
                    Location: {au_city}<br>
                    Customers: {customer_count}<br>
                    Type: In-Market Portfolio
                    <extra></extra>
                    """,
                    name="In-Market AUs",
                    showlegend=False
                ))
    
    # Add Centralized portfolio centroids (stars)
    centralized_data = filtered_data[filtered_data['COVERAGE'] == 'Centralized']
    if not centralized_data.empty:
        # Calculate centroids for each centralized portfolio
        centralized_portfolios = centralized_data.groupby('PORT_CODE').agg({
            'LAT_NUM': 'mean',
            'LON_NUM': 'mean',
            'CG_ECN': 'count',
            'BANK_REVENUE': 'mean'
        }).reset_index()
        
        for _, portfolio in centralized_portfolios.iterrows():
            fig.add_trace(go.Scattermapbox(
                lat=[portfolio['LAT_NUM']],
                lon=[portfolio['LON_NUM']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate=f"""
                <b>Centralized Portfolio {portfolio['PORT_CODE']}</b><br>
                Centroid Location<br>
                Customers: {portfolio['CG_ECN']}<br>
                Avg Revenue: ${portfolio['BANK_REVENUE']:,.0f}<br>
                Type: Centralized Portfolio
                <extra></extra>
                """,
                name="Centralized Centroids",
                showlegend=False
            ))
    
    # Add legend traces (invisible, just for legend)
    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=8, color='gray', symbol='circle'),
        name="🔵 Customers",
        showlegend=True
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='triangle-up'),
        name="🔺 In-Market AUs",
        showlegend=True
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name="⭐ Centralized Centroids",
        showlegend=True
    ))
    
    # Calculate map center
    if not filtered_data['LAT_NUM'].isna().all():
        center_lat = filtered_data['LAT_NUM'].mean()
        center_lon = filtered_data['LON_NUM'].mean()
        
        # Calculate zoom based on data spread
        lat_range = filtered_data['LAT_NUM'].max() - filtered_data['LAT_NUM'].min()
        lon_range = filtered_data['LON_NUM'].max() - filtered_data['LON_NUM'].min()
        max_range = max(lat_range, lon_range)
        
        if max_range > 20:
            zoom = 4
        elif max_range > 10:
            zoom = 5
        elif max_range > 5:
            zoom = 6
        else:
            zoom = 7
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
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
