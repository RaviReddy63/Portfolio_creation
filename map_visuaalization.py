import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import format_currency, format_number

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
            revenue_formatted = format_currency(customer.get('BANK_REVENUE', 0))
            deposit_formatted = format_currency(customer.get('DEPOSIT_BAL', 0))
            
            hover_text.append(f"""
            <b>{customer.get('CG_ECN', 'N/A')}</b><br>
            AU Portfolio: {au_id}<br>
            Portfolio ID: {customer.get('PORT_CODE', 'N/A')}<br>
            Distance: {customer.get('Distance', 0):.1f} miles<br>
            Revenue: {revenue_formatted}<br>
            Deposit: {deposit_formatted}<br>
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
            name=f"AU {au_id} Portfolio ({format_number(len(df))} customers)",
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

def create_smart_portfolio_map(results_df, branch_data):
    """Create a map showing smart portfolio assignments with different symbols for INMARKET vs CENTRALIZED"""
    
    if results_df.empty:
        return None
    
    fig = go.Figure()
    
    # Color scheme for different AUs
    au_list = sorted(results_df['ASSIGNED_AU'].unique())
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 
              'pink', 'darkgreen', 'brown', 'gray', 'cyan', 'magenta', 'yellow', 'lime']
    
    au_color_map = {au: colors[i % len(colors)] for i, au in enumerate(au_list)}
    
    # Group customers by AU and Type
    for au in au_list:
        au_data = results_df[results_df['ASSIGNED_AU'] == au]
        color = au_color_map[au]
        
        # Get AU location from branch_data
        au_branch = branch_data[branch_data['AU'] == au]
        if au_branch.empty:
            continue
        
        au_lat = au_branch.iloc[0]['BRANCH_LAT_NUM']
        au_lon = au_branch.iloc[0]['BRANCH_LON_NUM']
        au_city = au_branch.iloc[0].get('CITY', f'AU {au}')
        
        # Split by portfolio type
        inmarket_data = au_data[au_data['TYPE'] == 'INMARKET']
        centralized_data = au_data[au_data['TYPE'] == 'CENTRALIZED']
        
        # Add INMARKET customers (circles)
        if not inmarket_data.empty:
            inmarket_hover = []
            for _, customer in inmarket_data.iterrows():
                inmarket_hover.append(f"""
                <b>{customer.get('ECN', 'N/A')}</b><br>
                Type: INMARKET<br>
                Assigned AU: {au}<br>
                Distance: {customer.get('DISTANCE_TO_AU', 0):.1f} miles<br>
                City: {customer.get('BILLINGCITY', 'N/A')}<br>
                State: {customer.get('BILLINGSTATE', 'N/A')}
                """)
            
            fig.add_trace(go.Scattermapbox(
                lat=inmarket_data['LAT_NUM'],
                lon=inmarket_data['LON_NUM'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle',
                    opacity=0.7
                ),
                hovertemplate='%{text}<extra></extra>',
                text=inmarket_hover,
                name=f"AU {au} - INMARKET ({format_number(len(inmarket_data))})",
                showlegend=True
            ))
        
        # Add CENTRALIZED customers (squares)
        if not centralized_data.empty:
            centralized_hover = []
            for _, customer in centralized_data.iterrows():
                cluster_id = customer.get('CLUSTER_ID', 'N/A')
                centralized_hover.append(f"""
                <b>{customer.get('ECN', 'N/A')}</b><br>
                Type: CENTRALIZED<br>
                Assigned AU: {au}<br>
                Distance: {customer.get('DISTANCE_TO_AU', 0):.1f} miles<br>
                Cluster ID: {cluster_id}<br>
                City: {customer.get('BILLINGCITY', 'N/A')}<br>
                State: {customer.get('BILLINGSTATE', 'N/A')}
                """)
            
            fig.add_trace(go.Scattermapbox(
                lat=centralized_data['LAT_NUM'],
                lon=centralized_data['LON_NUM'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='square',
                    opacity=0.7
                ),
                hovertemplate='%{text}<extra></extra>',
                text=centralized_hover,
                name=f"AU {au} - CENTRALIZED ({format_number(len(centralized_data))})",
                showlegend=True
            ))
        
        # Add AU marker (star)
        fig.add_trace(go.Scattermapbox(
            lat=[au_lat],
            lon=[au_lon],
            mode='markers',
            marker=dict(
                size=15,
                color='black',
                symbol='star',
                line=dict(width=2, color='white')
            ),
            hovertemplate=f"""
            <b>AU {au}</b><br>
            Location: {au_city}<br>
            Coordinates: {au_lat:.4f}, {au_lon:.4f}<br>
            Total Customers: {format_number(len(au_data))}<br>
            INMARKET: {format_number(len(inmarket_data))}<br>
            CENTRALIZED: {format_number(len(centralized_data))}
            <extra></extra>
            """,
            name=f"AU {au} Branch",
            showlegend=True
        ))
    
    # Calculate center point for map
    all_lats = results_df['LAT_NUM'].tolist()
    all_lons = results_df['LON_NUM'].tolist()
    
    if all_lats:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Calculate zoom level based on data spread
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
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
        height=700,
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
        ),
        title=dict(
            text="Smart Portfolio Geographic Distribution",
            x=0.5,
            font=dict(size=16)
        )
    )
    
    return fig
