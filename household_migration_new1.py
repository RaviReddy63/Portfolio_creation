import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import random
from sklearn.neighbors import BallTree
from sklearn.impute import KNNImputer
import builtins

# ==================== CONFIGURATION ====================
# Portfolio constraints
RM_MIN = 270
RM_MAX = 350
RC_MIN = 220
RC_MAX = 250

# Radius parameters
IN_MARKET_START_RADIUS = 20
IN_MARKET_MAX_RADIUS = 200
IN_MARKET_RADIUS_INCREMENT = 20

CENTRALIZED_START_RADIUS = 200
CENTRALIZED_MAX_RADIUS = 1000
CENTRALIZED_RADIUS_INCREMENT = 100

SBB_RADIUS = 10

# Eastern US bounds for missing centroids
EASTERN_US_LAT_RANGE = (35, 42)
EASTERN_US_LON_RANGE = (-85, -75)

# Earth's radius in miles
EARTH_RADIUS_MILES = 3959


# ==================== UTILITY FUNCTIONS ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth in miles.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return EARTH_RADIUS_MILES * c


def impute_missing_coordinates(hh_df):
    """
    Impute missing LAT_NUM and LON_NUM using KNN based on BILLINGCITY, BILLINGSTATE.
    Much faster than iterative approach.
    """
    hh_df = hh_df.copy()
    
    # Find rows with missing coordinates
    missing_mask = hh_df['LAT_NUM'].isna() | hh_df['LON_NUM'].isna()
    
    if missing_mask.sum() == 0:
        print("No missing coordinates to impute.")
        return hh_df
    
    print(f"Imputing {missing_mask.sum()} missing coordinates using KNN...")
    
    # Create location groups based on BILLINGCITY and BILLINGSTATE
    hh_df['location_group'] = hh_df['BILLINGCITY'].astype(str) + '_' + hh_df['BILLINGSTATE'].astype(str)
    
    # Impute by location group
    for group in hh_df['location_group'].unique():
        group_mask = hh_df['location_group'] == group
        group_data = hh_df[group_mask].copy()
        
        group_missing = group_data['LAT_NUM'].isna() | group_data['LON_NUM'].isna()
        
        if group_missing.sum() > 0 and group_missing.sum() < len(group_data):
            # Has both missing and non-missing in this group
            coords = group_data[['LAT_NUM', 'LON_NUM']].values
            
            # Calculate number of valid neighbors
            num_valid = len(group_data) - group_missing.sum()
            n_neighbors = 5 if num_valid >= 5 else num_valid
            
            # Use KNN imputer
            imputer = KNNImputer(n_neighbors=n_neighbors)
            coords_imputed = imputer.fit_transform(coords)
            
            # Update the main dataframe
            hh_df.loc[group_mask, 'LAT_NUM'] = coords_imputed[:, 0]
            hh_df.loc[group_mask, 'LON_NUM'] = coords_imputed[:, 1]
        elif group_missing.sum() == len(group_data):
            # All missing in this group, use state-level imputation
            state = group_data['BILLINGSTATE'].iloc[0]
            state_data = hh_df[
                (hh_df['BILLINGSTATE'] == state) & 
                hh_df['LAT_NUM'].notna()
            ]
            
            if len(state_data) > 0:
                median_lat = state_data['LAT_NUM'].median()
                median_lon = state_data['LON_NUM'].median()
                hh_df.loc[group_mask, 'LAT_NUM'] = median_lat
                hh_df.loc[group_mask, 'LON_NUM'] = median_lon
            else:
                # Use Eastern US random as last resort
                hh_df.loc[group_mask, 'LAT_NUM'] = random.uniform(*EASTERN_US_LAT_RANGE)
                hh_df.loc[group_mask, 'LON_NUM'] = random.uniform(*EASTERN_US_LON_RANGE)
    
    # Clean up temporary column
    hh_df.drop('location_group', axis=1, inplace=True)
    
    print("Coordinate imputation complete.")
    return hh_df


def create_balltree(lat_lon_array):
    """
    Create a BallTree for efficient distance queries.
    Input: array of [lat, lon] in degrees
    Returns: BallTree object
    """
    # Convert to radians for haversine metric
    lat_lon_array = np.array(lat_lon_array, dtype = np.float64)
    lat_lon_radians = np.radians(lat_lon_array)
    return BallTree(lat_lon_radians, metric='haversine')


def query_balltree(tree, query_points, radius_miles):
    """
    Query BallTree for points within radius.
    
    Args:
        tree: BallTree object
        query_points: array of [lat, lon] in degrees
        radius_miles: search radius in miles
    
    Returns:
        indices, distances (in miles)
    """
    query_radians = np.radians(query_points)
    radius_radians = radius_miles / EARTH_RADIUS_MILES
    
    indices, distances = tree.query_radius(
        query_radians, 
        r=radius_radians, 
        return_distance=True,
        sort_results=True
    )
    
    # Convert distances back to miles
    distances_miles = [d * EARTH_RADIUS_MILES for d in distances]
    
    return indices, distances_miles


def create_portfolio_location_map(rbrm_data, branch_data, portfolio_centroids):
    """
    Create a mapping of portfolio code to location (lat, lon).
    IN MARKET portfolios use branch locations, CENTRALIZED use centroids.
    """
    portfolio_locations = {}
    
    # Convert branch_data AU to int for matching
    branch_data = branch_data.copy()
    branch_data['AU'] = branch_data['AU'].astype(int)
    
    for _, row in rbrm_data.iterrows():
        portfolio_cd = row['CG_PORTFOLIO_CD']
        placement = row['PLACEMENT']
        
        if placement == 'IN MARKET':
            # Use branch location
            au = int(row['AU'])
            branch = branch_data[branch_data['AU'] == au]
            if len(branch) > 0:
                portfolio_locations[portfolio_cd] = {
                    'lat': branch.iloc[0]['BRANCH_LAT_NUM'],
                    'lon': branch.iloc[0]['BRANCH_LON_NUM'],
                    'placement': 'IN MARKET',
                    'banker_type': row['BANKER_TYPE']
                }
        else:  # CENTRALIZED
            # Use centroid or random Eastern US location
            centroid = portfolio_centroids[portfolio_centroids['CG_PORTFOLIO_CD'] == portfolio_cd]
            if len(centroid) > 0 and pd.notna(centroid.iloc[0]['CENTROID_LAT_NUM']):
                portfolio_locations[portfolio_cd] = {
                    'lat': centroid.iloc[0]['CENTROID_LAT_NUM'],
                    'lon': centroid.iloc[0]['CENTROID_LON_NUM'],
                    'placement': 'CENTRALIZED',
                    'banker_type': row['BANKER_TYPE']
                }
            else:
                # Generate random Eastern US location
                portfolio_locations[portfolio_cd] = {
                    'lat': random.uniform(*EASTERN_US_LAT_RANGE),
                    'lon': random.uniform(*EASTERN_US_LON_RANGE),
                    'placement': 'CENTRALIZED',
                    'banker_type': row['BANKER_TYPE']
                }
    
    return portfolio_locations


# ==================== STEP 1: SBB ASSIGNMENT ====================

def assign_sbb_bankers(hh_df, sbb_data, branch_data):
    """
    Assign Segment 2 HH_ECNs with RULE='SBB/RETAIN' to SBB bankers if available.
    Uses BallTree for efficient distance calculations.
    Returns updated hh_df and a separate dataframe of SBB assignments.
    """
    hh_df = hh_df.copy()
    sbb_assignments = []
    
    # Filter Segment 2 with SBB/RETAIN rule
    sbb_retain_mask = (hh_df['NEW_SEGMENT'] == 2) & (hh_df['RULE'] == 'SBB/RETAIN')
    sbb_candidates = hh_df[sbb_retain_mask].copy()
    
    print(f"Processing {len(sbb_candidates)} Segment 2 SBB/RETAIN households...")
    
    if len(sbb_candidates) == 0:
        return hh_df, pd.DataFrame()
    
    # Convert branch_data AU to int for matching
    branch_data = branch_data.copy()
    branch_data['AU'] = branch_data['AU'].astype(int)
    
    # Create SBB banker location mapping
    sbb_locations = []
    sbb_info = []
    
    for _, sbb in sbb_data.iterrows():
        sbb_au = int(sbb['AU'])
        branch = branch_data[branch_data['AU'] == sbb_au]
        
        if len(branch) > 0:
            sbb_locations.append([
                branch.iloc[0]['BRANCH_LAT_NUM'],
                branch.iloc[0]['BRANCH_LON_NUM']
            ])
            sbb_info.append({
                'full_name': sbb['FULL NAME'],
                'au': sbb_au
            })
    
    if len(sbb_locations) == 0:
        print("No SBB bankers with valid branch locations found.")
        # Convert all to RETAIN
        hh_df.loc[sbb_retain_mask, 'RULE'] = 'RETAIN'
        return hh_df, pd.DataFrame()
    
    # Create BallTree for SBB locations
    sbb_tree = create_balltree(np.array(sbb_locations))
    
    # Query for each household
    hh_locations = sbb_candidates[['LAT_NUM', 'LON_NUM']].values
    
    for idx, (hh_idx, hh) in enumerate(sbb_candidates.iterrows()):
        assigned = False
        hh_au = int(hh['PATR_AU_STR']) if pd.notna(hh['PATR_AU_STR']) else 0
        
        # Check if SBB banker exists in same AU
        sbb_in_au = [i for i, info in enumerate(sbb_info) if info['au'] == hh_au]
        
        if len(sbb_in_au) > 0:
            # Assign to first SBB banker in AU
            sbb_idx = sbb_in_au[0]
            sbb_banker = sbb_info[sbb_idx]
            sbb_assignments.append({
                'BHH_SKEY': hh['BHH_SKEY'],
                'HH_ECN': hh['HH_ECN'],
                'SBB_BANKER': sbb_banker['full_name'],
                'SBB_AU': sbb_banker['au'],
                'ASSIGNMENT_TYPE': 'Same AU'
            })
            # Remove from HH_DF (unassign from portfolio)
            hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = None
            hh_df.loc[hh_idx, 'BANKER_TYPE'] = 'SBB'
            assigned = True
        else:
            # Check within 10-mile radius using BallTree
            query_point = hh_locations[idx:idx+1]
            indices, distances = query_balltree(sbb_tree, query_point, SBB_RADIUS)
            
            if len(indices[0]) > 0:
                # Assign to closest SBB banker within radius
                closest_idx = indices[0][0]
                closest_distance = distances[0][0]
                sbb_banker = sbb_info[closest_idx]
                
                sbb_assignments.append({
                    'BHH_SKEY': hh['BHH_SKEY'],
                    'HH_ECN': hh['HH_ECN'],
                    'SBB_BANKER': sbb_banker['full_name'],
                    'SBB_AU': sbb_banker['au'],
                    'ASSIGNMENT_TYPE': f'Within {closest_distance:.1f} miles'
                })
                # Remove from HH_DF (unassign from portfolio)
                hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = None
                hh_df.loc[hh_idx, 'BANKER_TYPE'] = 'SBB'
                assigned = True
        
        # If not assigned, convert to RETAIN
        if not assigned:
            hh_df.loc[hh_idx, 'RULE'] = 'RETAIN'
    
    sbb_assignments_df = pd.DataFrame(sbb_assignments)
    print(f"Assigned {len(sbb_assignments_df)} households to SBB bankers")
    print(f"Converted {len(sbb_candidates) - len(sbb_assignments_df)} to RETAIN")
    
    return hh_df, sbb_assignments_df


# ==================== STEP 2: CALCULATE REQUIREMENTS ====================

def calculate_portfolio_requirements(hh_df, portfolio_locations):
    """
    Calculate current counts and requirements (MIN/MAX) for each portfolio.
    Returns a dictionary with portfolio statistics.
    """
    portfolio_stats = {}
    
    for portfolio_cd, location_info in portfolio_locations.items():
        banker_type = location_info['banker_type']
        placement = location_info['placement']
        
        # Get households in this portfolio
        portfolio_hhs = hh_df[hh_df['CG_PORTFOLIO_CD'] == portfolio_cd]
        
        if banker_type == 'RM':
            # Count Segment 3 households
            target_segment = 3
            segment_count = len(portfolio_hhs[portfolio_hhs['NEW_SEGMENT'] == target_segment])
            min_required = RM_MIN
            max_allowed = RM_MAX
        else:  # RC
            # Count Segment 4 households
            target_segment = 4
            segment_count = len(portfolio_hhs[portfolio_hhs['NEW_SEGMENT'] == target_segment])
            min_required = RC_MIN
            max_allowed = RC_MAX
        
        deficit_val = min_required - segment_count
        excess_val = segment_count - max_allowed
        
        portfolio_stats[portfolio_cd] = {
            'banker_type': banker_type,
            'placement': placement,
            'target_segment': target_segment,
            'current_count': segment_count,
            'total_count': len(portfolio_hhs),
            'min_required': min_required,
            'max_allowed': max_allowed,
            'deficit': builtins.max(0, deficit_val),
            'excess': builtins.max(0, excess_val),
            'lat': location_info['lat'],
            'lon': location_info['lon']
        }
    
    return portfolio_stats


# ==================== STEP 3: FIND NEAREST PORTFOLIO FOR HOUSEHOLD ====================

def find_nearest_portfolio_for_household(hh_lat, hh_lon, portfolio_list, portfolio_stats, radius):
    """
    Find the nearest portfolio to a household within given radius.
    Returns (portfolio_cd, distance) or (None, None) if none found.
    """
    if len(portfolio_list) == 0:
        return None, None
    
    # Create BallTree for portfolios
    portfolio_locations = np.array([
        [portfolio_stats[p]['lat'], portfolio_stats[p]['lon']] 
        for p in portfolio_list
    ])
    portfolio_tree = create_balltree(portfolio_locations)
    
    # Query for portfolios within radius
    query_point = np.array([[hh_lat, hh_lon]])
    indices, distances = query_balltree(portfolio_tree, query_point, radius)
    
    if len(indices[0]) > 0:
        nearest_idx = indices[0][0]
        nearest_distance = distances[0][0]
        nearest_portfolio = portfolio_list[nearest_idx]
        return nearest_portfolio, nearest_distance
    
    return None, None


# ==================== STEP 4: TRIM OVERSIZED PORTFOLIOS ====================

def trim_oversized_portfolios(hh_df, portfolio_stats):
    """
    For portfolios exceeding MAX, remove the farthest households and return them to POOL.
    Uses BallTree for efficient distance calculation.
    Returns updated hh_df.
    """
    hh_df = hh_df.copy()
    
    for portfolio_cd, stats in portfolio_stats.items():
        if stats['excess'] > 0:
            # Get all households of target segment in this portfolio
            portfolio_hhs = hh_df[
                (hh_df['CG_PORTFOLIO_CD'] == portfolio_cd) & 
                (hh_df['NEW_SEGMENT'] == stats['target_segment']) &
                (hh_df['RULE'] == 'POOL')  # Only trim POOL households
            ]
            
            if len(portfolio_hhs) == 0:
                continue
            
            # Calculate distances using vectorized haversine
            hh_locations = portfolio_hhs[['LAT_NUM', 'LON_NUM']].values
            portfolio_location = np.array([[stats['lat'], stats['lon']]])
            
            # Create tree and query
            hh_tree = create_balltree(hh_locations)
            indices, distances = query_balltree(
                hh_tree, 
                portfolio_location, 
                radius_miles=10000  # Large radius to get all
            )
            
            # Create distance mapping
            distances_list = []
            original_indices = portfolio_hhs.index.values
            for i, dist in zip(indices[0], distances[0]):
                distances_list.append((original_indices[i], dist))
            
            # Sort by distance (farthest first)
            distances_list.sort(key=lambda x: x[1], reverse=True)
            
            # Remove excess households (farthest ones)
            num_to_remove = builtins.min(stats['excess'], len(distances_list))
            for i in range(num_to_remove):
                idx, dist = distances_list[i]
                hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = None
                print(f"  Trimmed HH_ECN {hh_df.loc[idx, 'HH_ECN']} from {portfolio_cd} (distance: {dist:.1f} miles)")
    
    return hh_df


# ==================== STEP 5: FIND UNDERSIZED PORTFOLIOS ====================

def find_undersized_portfolios(portfolio_stats):
    """
    Returns list of portfolio codes that are below MIN requirement.
    """
    undersized = []
    for portfolio_cd, stats in portfolio_stats.items():
        if stats['deficit'] > 0:
            undersized.append(portfolio_cd)
    
    return undersized


# ==================== STEP 6: OPTIMIZE IN MARKET PORTFOLIOS ====================

def optimize_in_market_portfolios(hh_df, portfolio_stats, banker_type, segment):
    """
    Optimize IN MARKET portfolios using iterative radius expansion (20-200 miles).
    Assigns each household to nearest undersized portfolio.
    """
    hh_df = hh_df.copy()
    
    # Filter IN MARKET portfolios of given banker type
    in_market_portfolios = [
        p for p, s in portfolio_stats.items() 
        if s['placement'] == 'IN MARKET' and s['banker_type'] == banker_type
    ]
    
    print(f"\nOptimizing {len(in_market_portfolios)} {banker_type} IN MARKET portfolios (Segment {segment})...")
    
    for radius in range(IN_MARKET_START_RADIUS, IN_MARKET_MAX_RADIUS + 1, IN_MARKET_RADIUS_INCREMENT):
        print(f"\n  Radius: {radius} miles")
        
        # Recalculate portfolio stats
        portfolio_stats = calculate_portfolio_requirements(hh_df, 
            {p: {'lat': portfolio_stats[p]['lat'], 
                 'lon': portfolio_stats[p]['lon'],
                 'placement': portfolio_stats[p]['placement'],
                 'banker_type': portfolio_stats[p]['banker_type']} 
             for p in portfolio_stats})
        
        # Get undersized portfolios
        undersized = [p for p in in_market_portfolios if portfolio_stats[p]['deficit'] > 0]
        
        if not undersized:
            print(f"  No undersized portfolios. Breaking loop.")
            break
        
        print(f"  Found {len(undersized)} undersized portfolios")
        
        # Get all unassigned households of target segment
        unassigned_hhs = hh_df[
            (hh_df['NEW_SEGMENT'] == segment) & 
            (hh_df['RULE'] == 'POOL') & 
            (hh_df['CG_PORTFOLIO_CD'].isna())
        ]
        
        assigned_count = 0
        
        # For each unassigned household, find nearest undersized portfolio
        for hh_idx, hh in unassigned_hhs.iterrows():
            hh_lat = hh['LAT_NUM']
            hh_lon = hh['LON_NUM']
            
            # Find nearest undersized portfolio within radius
            nearest_portfolio, distance = find_nearest_portfolio_for_household(
                hh_lat, hh_lon, undersized, portfolio_stats, radius
            )
            
            if nearest_portfolio is not None:
                # Assign household to nearest portfolio
                hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
                hh_df.loc[hh_idx, 'BANKER_TYPE'] = banker_type
                assigned_count += 1
                
                # Update portfolio stats for this portfolio
                portfolio_stats[nearest_portfolio]['current_count'] += 1
                deficit_val = portfolio_stats[nearest_portfolio]['min_required'] - portfolio_stats[nearest_portfolio]['current_count']
                portfolio_stats[nearest_portfolio]['deficit'] = builtins.max(0, deficit_val)
                
                # Remove from undersized list if now at MIN
                if portfolio_stats[nearest_portfolio]['deficit'] == 0:
                    undersized.remove(nearest_portfolio)
                
                # Break if no more undersized portfolios
                if len(undersized) == 0:
                    break
        
        print(f"  Assigned {assigned_count} households to nearest portfolios")
        
        # Trim oversized portfolios
        portfolio_stats = calculate_portfolio_requirements(hh_df, 
            {p: {'lat': portfolio_stats[p]['lat'], 
                 'lon': portfolio_stats[p]['lon'],
                 'placement': portfolio_stats[p]['placement'],
                 'banker_type': portfolio_stats[p]['banker_type']} 
             for p in portfolio_stats})
        
        oversized = [p for p in in_market_portfolios if portfolio_stats[p]['excess'] > 0]
        if oversized:
            print(f"  Trimming {len(oversized)} oversized portfolios")
            hh_df = trim_oversized_portfolios(hh_df, portfolio_stats)
    
    return hh_df


# ==================== STEP 6.5: FILL IN MARKET PORTFOLIOS TO MAX ====================

def fill_in_market_to_max(hh_df, portfolio_stats, banker_type, segment, radius=20):
    """
    Fill IN MARKET portfolios up to MAX by assigning nearby households to nearest portfolio.
    Each household is assigned to the nearest portfolio that has capacity.
    """
    hh_df = hh_df.copy()
    
    # Filter IN MARKET portfolios of given banker type
    in_market_portfolios = [
        p for p, s in portfolio_stats.items() 
        if s['placement'] == 'IN MARKET' and s['banker_type'] == banker_type
    ]
    
    print(f"\nFilling {len(in_market_portfolios)} {banker_type} IN MARKET portfolios to MAX (Segment {segment})...")
    print(f"  Using {radius} mile radius")
    
    # Recalculate portfolio stats
    portfolio_stats = calculate_portfolio_requirements(hh_df, 
        {p: {'lat': portfolio_stats[p]['lat'], 
             'lon': portfolio_stats[p]['lon'],
             'placement': portfolio_stats[p]['placement'],
             'banker_type': portfolio_stats[p]['banker_type']} 
         for p in portfolio_stats})
    
    # Get portfolios with capacity (below MAX)
    portfolios_with_capacity = [
        p for p in in_market_portfolios 
        if portfolio_stats[p]['current_count'] < portfolio_stats[p]['max_allowed']
    ]
    
    if not portfolios_with_capacity:
        print("  All portfolios are at MAX capacity.")
        return hh_df
    
    # Get all unassigned households of target segment
    unassigned_hhs = hh_df[
        (hh_df['NEW_SEGMENT'] == segment) & 
        (hh_df['RULE'] == 'POOL') & 
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]
    
    assigned_count = 0
    
    # For each unassigned household, find nearest portfolio with capacity
    for hh_idx, hh in unassigned_hhs.iterrows():
        if len(portfolios_with_capacity) == 0:
            break
        
        hh_lat = hh['LAT_NUM']
        hh_lon = hh['LON_NUM']
        
        # Find nearest portfolio with capacity within radius
        nearest_portfolio, distance = find_nearest_portfolio_for_household(
            hh_lat, hh_lon, portfolios_with_capacity, portfolio_stats, radius
        )
        
        if nearest_portfolio is not None:
            # Assign household to nearest portfolio
            hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
            hh_df.loc[hh_idx, 'BANKER_TYPE'] = banker_type
            assigned_count += 1
            
            # Update portfolio stats
            portfolio_stats[nearest_portfolio]['current_count'] += 1
            
            # Remove from capacity list if now at MAX
            if portfolio_stats[nearest_portfolio]['current_count'] >= portfolio_stats[nearest_portfolio]['max_allowed']:
                portfolios_with_capacity.remove(nearest_portfolio)
    
    print(f"  Assigned {assigned_count} additional households to nearest portfolios with capacity")
    
    return hh_df


# ==================== STEP 7 & 8: OPTIMIZE CENTRALIZED PORTFOLIOS ====================

def optimize_centralized_portfolios(hh_df, portfolio_stats, banker_type, segment):
    """
    Optimize CENTRALIZED portfolios using iterative radius expansion (200-1000 miles).
    Assigns each household to nearest undersized portfolio.
    """
    hh_df = hh_df.copy()
    
    # Filter CENTRALIZED portfolios of given banker type
    centralized_portfolios = [
        p for p, s in portfolio_stats.items() 
        if s['placement'] == 'CENTRALIZED' and s['banker_type'] == banker_type
    ]
    
    print(f"\nOptimizing {len(centralized_portfolios)} {banker_type} CENTRALIZED portfolios (Segment {segment})...")
    
    for radius in range(CENTRALIZED_START_RADIUS, CENTRALIZED_MAX_RADIUS + 1, CENTRALIZED_RADIUS_INCREMENT):
        print(f"\n  Radius: {radius} miles")
        
        # Recalculate portfolio stats
        portfolio_stats = calculate_portfolio_requirements(hh_df, 
            {p: {'lat': portfolio_stats[p]['lat'], 
                 'lon': portfolio_stats[p]['lon'],
                 'placement': portfolio_stats[p]['placement'],
                 'banker_type': portfolio_stats[p]['banker_type']} 
             for p in portfolio_stats})
        
        # Get undersized portfolios
        undersized = [p for p in centralized_portfolios if portfolio_stats[p]['deficit'] > 0]
        
        if not undersized:
            print(f"  No undersized portfolios. Breaking loop.")
            break
        
        print(f"  Found {len(undersized)} undersized portfolios")
        
        # Get all unassigned households of target segment
        unassigned_hhs = hh_df[
            (hh_df['NEW_SEGMENT'] == segment) & 
            (hh_df['RULE'] == 'POOL') & 
            (hh_df['CG_PORTFOLIO_CD'].isna())
        ]
        
        assigned_count = 0
        
        # For each unassigned household, find nearest undersized portfolio
        for hh_idx, hh in unassigned_hhs.iterrows():
            hh_lat = hh['LAT_NUM']
            hh_lon = hh['LON_NUM']
            
            # Find nearest undersized portfolio within radius
            nearest_portfolio, distance = find_nearest_portfolio_for_household(
                hh_lat, hh_lon, undersized, portfolio_stats, radius
            )
            
            if nearest_portfolio is not None:
                # Assign household to nearest portfolio
                hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
                hh_df.loc[hh_idx, 'BANKER_TYPE'] = banker_type
                assigned_count += 1
                
                # Update portfolio stats for this portfolio
                portfolio_stats[nearest_portfolio]['current_count'] += 1
                deficit_val = portfolio_stats[nearest_portfolio]['min_required'] - portfolio_stats[nearest_portfolio]['current_count']
                portfolio_stats[nearest_portfolio]['deficit'] = builtins.max(0, deficit_val)
                
                # Remove from undersized list if now at MIN
                if portfolio_stats[nearest_portfolio]['deficit'] == 0:
                    undersized.remove(nearest_portfolio)
                
                # Break if no more undersized portfolios
                if len(undersized) == 0:
                    break
        
        print(f"  Assigned {assigned_count} households to nearest portfolios")
        
        # Trim oversized portfolios
        portfolio_stats = calculate_portfolio_requirements(hh_df, 
            {p: {'lat': portfolio_stats[p]['lat'], 
                 'lon': portfolio_stats[p]['lon'],
                 'placement': portfolio_stats[p]['placement'],
                 'banker_type': portfolio_stats[p]['banker_type']} 
             for p in portfolio_stats})
        
        oversized = [p for p in centralized_portfolios if portfolio_stats[p]['excess'] > 0]
        if oversized:
            print(f"  Trimming {len(oversized)} oversized portfolios")
            hh_df = trim_oversized_portfolios(hh_df, portfolio_stats)
    
    return hh_df


# ==================== STEP 9 & 10: FINAL CLEANUP ====================

def assign_remaining_households(hh_df, portfolio_stats, segment, banker_type):
    """
    Final cleanup: Assign all remaining unassigned households of given segment.
    First fills undersized CENTRALIZED portfolios to MIN, then assigns rest to nearest.
    Each household is assigned to nearest portfolio.
    """
    hh_df = hh_df.copy()
    
    # Get CENTRALIZED portfolios of given banker type
    centralized_portfolios = [
        p for p, s in portfolio_stats.items() 
        if s['placement'] == 'CENTRALIZED' and s['banker_type'] == banker_type
    ]
    
    print(f"\nFinal cleanup for Segment {segment} ({banker_type} CENTRALIZED)...")
    
    # Phase A: Fill undersized portfolios to MIN
    portfolio_stats = calculate_portfolio_requirements(hh_df, 
        {p: {'lat': portfolio_stats[p]['lat'], 
             'lon': portfolio_stats[p]['lon'],
             'placement': portfolio_stats[p]['placement'],
             'banker_type': portfolio_stats[p]['banker_type']} 
         for p in portfolio_stats})
    
    undersized = [p for p in centralized_portfolios if portfolio_stats[p]['deficit'] > 0]
    
    if undersized:
        print(f"  Phase A: Filling {len(undersized)} undersized portfolios to MIN")
        
        # Get all unassigned households of target segment
        unassigned = hh_df[
            (hh_df['NEW_SEGMENT'] == segment) & 
            (hh_df['RULE'] == 'POOL') & 
            (hh_df['CG_PORTFOLIO_CD'].isna())
        ]
        
        assigned_count = 0
        
        # For each unassigned household, find nearest undersized portfolio
        for hh_idx, hh in unassigned.iterrows():
            if len(undersized) == 0:
                break
            
            hh_lat = hh['LAT_NUM']
            hh_lon = hh['LON_NUM']
            
            # Find nearest undersized portfolio (no radius limit)
            nearest_portfolio, distance = find_nearest_portfolio_for_household(
                hh_lat, hh_lon, undersized, portfolio_stats, radius=10000
            )
            
            if nearest_portfolio is not None:
                # Assign household
                hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
                hh_df.loc[hh_idx, 'BANKER_TYPE'] = banker_type
                assigned_count += 1
                
                # Update portfolio stats
                portfolio_stats[nearest_portfolio]['current_count'] += 1
                deficit_val = portfolio_stats[nearest_portfolio]['min_required'] - portfolio_stats[nearest_portfolio]['current_count']
                portfolio_stats[nearest_portfolio]['deficit'] = builtins.max(0, deficit_val)
                
                # Remove from undersized if reached MIN
                if portfolio_stats[nearest_portfolio]['deficit'] == 0:
                    undersized.remove(nearest_portfolio)
        
        print(f"    Assigned {assigned_count} households to reach MIN")
    
    # Phase B: Assign all remaining to nearest portfolio
    remaining = hh_df[
        (hh_df['NEW_SEGMENT'] == segment) & 
        (hh_df['RULE'] == 'POOL') & 
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]
    
    print(f"  Phase B: Assigning {len(remaining)} remaining households to nearest portfolio")
    
    assigned_count = 0
    
    # For each remaining household, find nearest portfolio (no capacity limit)
    for hh_idx, hh in remaining.iterrows():
        hh_lat = hh['LAT_NUM']
        hh_lon = hh['LON_NUM']
        
        # Find nearest portfolio (no radius limit)
        nearest_portfolio, distance = find_nearest_portfolio_for_household(
            hh_lat, hh_lon, centralized_portfolios, portfolio_stats, radius=10000
        )
        
        if nearest_portfolio is not None:
            hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
            hh_df.loc[hh_idx, 'BANKER_TYPE'] = banker_type
            assigned_count += 1
    
    print(f"    Assigned {assigned_count} households to nearest portfolio")
    
    # Verify no unassigned left
    final_unassigned = hh_df[
        (hh_df['NEW_SEGMENT'] == segment) & 
        (hh_df['RULE'] == 'POOL') & 
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]
    
    print(f"  Final unassigned Segment {segment}: {len(final_unassigned)}")
    
    return hh_df


# ==================== MAIN ORCHESTRATOR ====================

def run_portfolio_reconstruction(hh_df, branch_data, rbrm_data, sbb_data, portfolio_centroids):
    """
    Main function to orchestrate the entire portfolio reconstruction process.
    """
    print("="*70)
    print("PORTFOLIO RECONSTRUCTION - START")
    print("="*70)
    
    # ========== INITIALIZATION ==========
    print("\n[INITIALIZATION]")
    print("Imputing missing coordinates using KNN...")
    hh_df = impute_missing_coordinates(hh_df)
    
    print("Creating portfolio location map...")
    portfolio_locations = create_portfolio_location_map(rbrm_data, branch_data, portfolio_centroids)
    print(f"Created location map for {len(portfolio_locations)} portfolios")
    
    # ========== STEP 1: SBB ASSIGNMENT ==========
    print("\n[STEP 1: SBB/RETAIN ASSIGNMENT]")
    hh_df, sbb_assignments_df = assign_sbb_bankers(hh_df, sbb_data, branch_data)
    
    # ========== STEP 2: CALCULATE REQUIREMENTS ==========
    print("\n[STEP 2: CALCULATE PORTFOLIO REQUIREMENTS]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    
    # Print summary
    rm_in_market = [p for p, s in portfolio_stats.items() if s['banker_type'] == 'RM' and s['placement'] == 'IN MARKET']
    rm_centralized = [p for p, s in portfolio_stats.items() if s['banker_type'] == 'RM' and s['placement'] == 'CENTRALIZED']
    rc_centralized = [p for p, s in portfolio_stats.items() if s['banker_type'] == 'RC' and s['placement'] == 'CENTRALIZED']
    
    print(f"RM IN MARKET portfolios: {len(rm_in_market)}")
    print(f"RM CENTRALIZED portfolios: {len(rm_centralized)}")
    print(f"RC CENTRALIZED portfolios: {len(rc_centralized)}")
    
    # ========== STEPS 3-6: RM IN MARKET OPTIMIZATION ==========
    print("\n[STEPS 3-6: RM IN MARKET OPTIMIZATION]")
    hh_df = optimize_in_market_portfolios(hh_df, portfolio_stats, 'RM', 3)
    
    # ========== STEP 6.5: FILL RM IN MARKET TO MAX ==========
    print("\n[STEP 6.5: FILL RM IN MARKET TO MAX]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = fill_in_market_to_max(hh_df, portfolio_stats, 'RM', 3, radius=20)
    
    # ========== STEP 7: RM CENTRALIZED OPTIMIZATION ==========
    print("\n[STEP 7: RM CENTRALIZED OPTIMIZATION]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = optimize_centralized_portfolios(hh_df, portfolio_stats, 'RM', 3)
    
    # ========== STEP 8: RC CENTRALIZED OPTIMIZATION ==========
    print("\n[STEP 8: RC CENTRALIZED OPTIMIZATION]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = optimize_centralized_portfolios(hh_df, portfolio_stats, 'RC', 4)
    
    # ========== STEP 9: FINAL CLEANUP - SEGMENT 3 ==========
    print("\n[STEP 9: FINAL CLEANUP - SEGMENT 3]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = assign_remaining_households(hh_df, portfolio_stats, 3, 'RM')
    
    # ========== STEP 10: FINAL CLEANUP - SEGMENT 4 ==========
    print("\n[STEP 10: FINAL CLEANUP - SEGMENT 4]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = assign_remaining_households(hh_df, portfolio_stats, 4, 'RC')
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PORTFOLIO RECONSTRUCTION - COMPLETE")
    print("="*70)
    
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    
    print("\nFINAL PORTFOLIO SUMMARY:")
    print(f"\nRM IN MARKET portfolios:")
    for p in rm_in_market:
        s = portfolio_stats[p]
        status = "✓" if s['current_count'] >= s['min_required'] else "✗"
        print(f"  {status} {p}: {s['current_count']} Seg3 HHs (MIN: {s['min_required']}, MAX: {s['max_allowed']})")
    
    print(f"\nRM CENTRALIZED portfolios:")
    for p in rm_centralized:
        s = portfolio_stats[p]
        status = "✓" if s['current_count'] >= s['min_required'] else "✗"
        print(f"  {status} {p}: {s['current_count']} Seg3 HHs (MIN: {s['min_required']}, MAX: {s['max_allowed']})")
    
    print(f"\nRC CENTRALIZED portfolios:")
    for p in rc_centralized:
        s = portfolio_stats[p]
        status = "✓" if s['current_count'] >= s['min_required'] else "✗"
        print(f"  {status} {p}: {s['current_count']} Seg4 HHs (MIN: {s['min_required']}, MAX: {s['max_allowed']})")
    
    # Check for unassigned
    unassigned_seg3 = hh_df[(hh_df['NEW_SEGMENT'] == 3) & (hh_df['RULE'] == 'POOL') & (hh_df['CG_PORTFOLIO_CD'].isna())]
    unassigned_seg4 = hh_df[(hh_df['NEW_SEGMENT'] == 4) & (hh_df['RULE'] == 'POOL') & (hh_df['CG_PORTFOLIO_CD'].isna())]
    
    print(f"\nUNASSIGNED HOUSEHOLDS:")
    print(f"  Segment 3: {len(unassigned_seg3)}")
    print(f"  Segment 4: {len(unassigned_seg4)}")
    
    return hh_df, sbb_assignments_df, portfolio_stats


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Load your data
    # hh_df = pd.read_csv('hh_df.csv')
    # branch_data = pd.read_csv('branch_data.csv')
    # rbrm_data = pd.read_csv('rbrm_data.csv')
    # sbb_data = pd.read_csv('sbb_data.csv')
    # portfolio_centroids = pd.read_csv('portfolio_centroids.csv')
    
    # Run reconstruction
    # updated_hh_df, sbb_assignments, portfolio_stats = run_portfolio_reconstruction(
    #     hh_df, branch_data, rbrm_data, sbb_data, portfolio_centroids
    # )
    
    # Save results
    # updated_hh_df.to_csv('updated_hh_df.csv', index=False)
    # sbb_assignments.to_csv('sbb_assignments.csv', index=False)
    
    print("\nScript loaded. Ready to run with your data.")
