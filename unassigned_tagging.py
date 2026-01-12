import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two points"""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def get_banker_locations(active_bankers, branch_data, portfolio_centroids):
    """Get banker locations based on ROLE_TYPE"""
    bankers = active_bankers.copy()
    
    # Merge IN MARKET bankers with branch locations
    in_market = bankers[bankers['ROLE_TYPE'] == 'IN MARKET'].merge(
        branch_data[['AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
        on='AU', 
        how='left'
    )
    in_market['LAT'] = in_market['BRANCH_LAT_NUM']
    in_market['LON'] = in_market['BRANCH_LON_NUM']
    
    # Merge CENTRALIZED bankers with portfolio centroids
    centralized = bankers[bankers['ROLE_TYPE'] == 'CENTRALIZED'].merge(
        portfolio_centroids, 
        on='PORT_CODE', 
        how='left'
    )
    centralized['LAT'] = centralized['PORT_CENTROID_LAT']
    centralized['LON'] = centralized['PORT_CENTROID_LON']
    
    # Combine both
    banker_locations = pd.concat([
        in_market[['PORT_CODE', 'BANKER_TYPE', 'ROLE_TYPE', 'LAT', 'LON', 'COVERAGE']],
        centralized[['PORT_CODE', 'BANKER_TYPE', 'ROLE_TYPE', 'LAT', 'LON', 'COVERAGE']]
    ])
    
    # Remove rows with missing coordinates
    banker_locations = banker_locations.dropna(subset=['LAT', 'LON'])
    
    return banker_locations

def assign_by_state(customers_no_coords, active_bankers, banker_type):
    """Assign customers without coordinates based on state coverage"""
    
    if len(customers_no_coords) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
    
    # Filter bankers by type
    bankers = active_bankers[active_bankers['BANKER_TYPE'] == banker_type].copy()
    
    # Expand COVERAGE column (comma-separated states) into multiple rows
    bankers['COVERAGE'] = bankers['COVERAGE'].fillna('')
    bankers_expanded = []
    
    for _, banker in bankers.iterrows():
        states = [s.strip() for s in str(banker['COVERAGE']).split(',') if s.strip()]
        for state in states:
            banker_row = banker.copy()
            banker_row['STATE'] = state
            bankers_expanded.append(banker_row)
    
    if len(bankers_expanded) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
    
    bankers_by_state = pd.DataFrame(bankers_expanded)
    
    assignments = []
    
    for _, customer in customers_no_coords.iterrows():
        customer_state = str(customer['BILLINGSTATE']).strip()
        
        # Find bankers covering this state
        matching_bankers = bankers_by_state[bankers_by_state['STATE'] == customer_state]
        
        if len(matching_bankers) > 0:
            # Assign to first matching banker (you can randomize or load balance here)
            banker = matching_bankers.iloc[0]
            assignments.append({
                'CG_ECN': customer['CG_ECN'],
                'HH_ECN': customer['HH_ECN'],
                'PORT_CODE': banker['PORT_CODE'],
                'PROM_SEG_RAW': customer['PROM_SEG_RAW'],
                'BANKER_TYPE': banker_type
            })
    
    return pd.DataFrame(assignments)

def assign_customers(unassigned_pop, banker_locations, prom_seg, banker_type, 
                     max_radius, in_market_radius=None, centralized_radius=600):
    """Assign customers to nearest bankers within radius"""
    
    # Filter customers by segment
    customers = unassigned_pop[unassigned_pop['PROM_SEG_RAW'] == prom_seg].copy()
    
    # Separate customers with and without coordinates
    customers_with_coords = customers.dropna(subset=['ECN_LAT', 'ECN_LON'])
    customers_no_coords = customers[customers['ECN_LAT'].isna() | customers['ECN_LON'].isna()]
    
    bankers = banker_locations[banker_locations['BANKER_TYPE'] == banker_type].copy()
    
    if len(customers_with_coords) == 0 and len(customers_no_coords) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']), customers_no_coords
    
    assignments = []
    
    # Process customers WITH coordinates
    if len(customers_with_coords) > 0 and len(bankers) > 0:
        
        # For RM (PROM_SEG_RAW=3), try IN MARKET first
        if in_market_radius is not None:
            in_market_bankers = bankers[bankers['ROLE_TYPE'] == 'IN MARKET'].copy()
