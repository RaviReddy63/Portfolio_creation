import plotly.graph_objects as go

def create_combined_map(all_portfolios, branch_data):
    """Create a combined map showing all portfolios with different colors - one color per AU"""
    
    if not all_portfolios:
        return None
    
    fig = go.Figure()
    
    # Color scheme for different AU portfolios
    portfolio_colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightblue', 
                       'pink', 'darkgreen', 'brown', 'gray', 'cyan', 'magenta', 'yellow', 'red', 'lime']
    
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
    
    # Add AU markers (on top)
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
                symbol='circle'
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
