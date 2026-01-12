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
        in_market[['PORT_CODE', 'BANKER_TYPE', 'ROLE_TYPE', 'LAT', 'LON']],
        centralized[['PORT_CODE', 'BANKER_TYPE', 'ROLE_TYPE', 'LAT', 'LON']]
    ])
    
    # Remove rows with missing coordinates
    banker_locations = banker_locations.dropna(subset=['LAT', 'LON'])
    
    return banker_locations

def assign_customers(unassigned_pop, banker_locations, prom_seg, banker_type, 
                     max_radius, in_market_radius=None):
    """Assign customers to nearest bankers within radius"""
    
    # Filter customers and bankers
    customers = unassigned_pop[unassigned_pop['PROM_SEG_RAW'] == prom_seg].copy()
    
    # Remove customers with missing coordinates
    customers = customers.dropna(subset=['ECN_LAT', 'ECN_LON'])
    
    bankers = banker_locations[banker_locations['BANKER_TYPE'] == banker_type].copy()
    
    if len(customers) == 0 or len(bankers) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
    
    assignments = []
    
    # For RM (PROM_SEG_RAW=3), try IN MARKET first
    if in_market_radius is not None:
        in_market_bankers = bankers[bankers['ROLE_TYPE'] == 'IN MARKET'].copy()
        
        if len(in_market_bankers) > 0:
            # Build BallTree for IN MARKET bankers
            banker_coords = in_market_bankers[['LAT', 'LON']].values.astype(float)
            banker_coords_rad = np.radians(banker_coords)
            tree = BallTree(banker_coords_rad, metric='haversine')
            
            customer_coords = customers[['ECN_LAT', 'ECN_LON']].values.astype(float)
            customer_coords_rad = np.radians(customer_coords)
            distances, indices = tree.query(customer_coords_rad, k=1)
            distances_miles = distances.flatten() * 3959  # Convert to miles
            
            # Assign customers within IN MARKET radius
            mask = distances_miles <= in_market_radius
            assigned_indices = np.where(mask)[0]
            
            for idx in assigned_indices:
                banker_idx = indices[idx][0]
                assignments.append({
                    'CG_ECN': customers.iloc[idx]['CG_ECN'],
                    'HH_ECN': customers.iloc[idx]['HH_ECN'],
                    'PORT_CODE': in_market_bankers.iloc[banker_idx]['PORT_CODE'],
                    'PROM_SEG_RAW': prom_seg,
                    'BANKER_TYPE': banker_type
                })
            
            # Remove assigned customers
            customers = customers.iloc[~mask].copy()
    
    # Assign remaining customers to any banker (CENTRALIZED or all bankers)
    if len(customers) > 0 and len(bankers) > 0:
        banker_coords = bankers[['LAT', 'LON']].values.astype(float)
        banker_coords_rad = np.radians(banker_coords)
        tree = BallTree(banker_coords_rad, metric='haversine')
        
        customer_coords = customers[['ECN_LAT', 'ECN_LON']].values.astype(float)
        customer_coords_rad = np.radians(customer_coords)
        distances, indices = tree.query(customer_coords_rad, k=1)
        distances_miles = distances.flatten() * 3959
        
        # Assign customers within max radius
        mask = distances_miles <= max_radius
        assigned_indices = np.where(mask)[0]
        
        for idx in assigned_indices:
            banker_idx = indices[idx][0]
            assignments.append({
                'CG_ECN': customers.iloc[idx]['CG_ECN'],
                'HH_ECN': customers.iloc[idx]['HH_ECN'],
                'PORT_CODE': bankers.iloc[banker_idx]['PORT_CODE'],
                'PROM_SEG_RAW': prom_seg,
                'BANKER_TYPE': banker_type
            })
    
    return pd.DataFrame(assignments)

# Main execution
def main(df1, df2, df3, df4):
    """Main function to assign all customers"""
    
    # Get banker locations
    banker_locations = get_banker_locations(df2, df3, df4)
    
    print(f"Total bankers with valid locations: {len(banker_locations)}")
    print(f"RC bankers: {len(banker_locations[banker_locations['BANKER_TYPE']=='RC'])}")
    print(f"RM bankers: {len(banker_locations[banker_locations['BANKER_TYPE']=='RM'])}")
    
    # Step 1: Assign PROM_SEG_RAW=4 to RC bankers (400 miles max)
    rc_assignments = assign_customers(
        df1, banker_locations, 
        prom_seg=4, 
        banker_type='RC', 
        max_radius=400
    )
    print(f"RC assignments: {len(rc_assignments)}")
    
    # Step 2: Assign PROM_SEG_RAW=3 to RM bankers (40 miles IN MARKET, then 400 miles CENTRALIZED)
    rm_assignments = assign_customers(
        df1, banker_locations, 
        prom_seg=3, 
        banker_type='RM', 
        max_radius=400,
        in_market_radius=40
    )
    print(f"RM assignments: {len(rm_assignments)}")
    
    # Combine all assignments
    final_assignments = pd.concat([rc_assignments, rm_assignments], ignore_index=True)
    
    return final_assignments[['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']]

# Run the assignment
result = main(df1, df2, df3, df4)
print(f"\nTotal assignments: {len(result)}")
print(result.head())
