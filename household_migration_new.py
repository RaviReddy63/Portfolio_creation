import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import random

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


# ==================== UTILITY FUNCTIONS ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth in miles.
    """
    R = 3959  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def impute_missing_coordinates(hh_df):
    """
    Impute missing LAT_NUM and LON_NUM using BILLINGCITY, BILLINGSTATE, and PATR_AU_STR.
    Simple approach: Use median coordinates of similar locations.
    """
    hh_df = hh_df.copy()
    
    # Find rows with missing coordinates
    missing_mask = hh_df['LAT_NUM'].isna() | hh_df['LON_NUM'].isna()
    
    if missing_mask.sum() == 0:
        return hh_df
    
    # Impute based on BILLINGCITY and BILLINGSTATE
    for idx in hh_df[missing_mask].index:
        city = hh_df.loc[idx, 'BILLINGCITY']
        state = hh_df.loc[idx, 'BILLINGSTATE']
        au = hh_df.loc[idx, 'PATR_AU_STR']
        
        # Try to find similar records with coordinates
        similar = hh_df[
            (hh_df['BILLINGCITY'] == city) & 
            (hh_df['BILLINGSTATE'] == state) & 
            hh_df['LAT_NUM'].notna()
        ]
        
        if len(similar) > 0:
            hh_df.loc[idx, 'LAT_NUM'] = similar['LAT_NUM'].median()
            hh_df.loc[idx, 'LON_NUM'] = similar['LON_NUM'].median()
        else:
            # Try state level
            similar = hh_df[(hh_df['BILLINGSTATE'] == state) & hh_df['LAT_NUM'].notna()]
            if len(similar) > 0:
                hh_df.loc[idx, 'LAT_NUM'] = similar['LAT_NUM'].median()
                hh_df.loc[idx, 'LON_NUM'] = similar['LON_NUM'].median()
            else:
                # Use Eastern US random location as last resort
                hh_df.loc[idx, 'LAT_NUM'] = random.uniform(*EASTERN_US_LAT_RANGE)
                hh_df.loc[idx, 'LON_NUM'] = random.uniform(*EASTERN_US_LON_RANGE)
    
    return hh_df


def create_portfolio_location_map(rbrm_data, branch_data, portfolio_centroids):
    """
    Create a mapping of portfolio code to location (lat, lon).
    IN MARKET portfolios use branch locations, CENTRALIZED use centroids.
    """
    portfolio_locations = {}
    
    for _, row in rbrm_data.iterrows():
        portfolio_cd = row['CG_PORTFOLIO_CD']
        placement = row['PLACEMENT']
        
        if placement == 'IN MARKET':
            # Use branch location
            au = row['AU']
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
    Returns updated hh_df and a separate dataframe of SBB assignments.
    """
    hh_df = hh_df.copy()
    sbb_assignments = []
    
    # Filter Segment 2 with SBB/RETAIN rule
    sbb_retain_mask = (hh_df['NEW_SEGMENT'] == 2) & (hh_df['RULE'] == 'SBB/RETAIN')
    sbb_candidates = hh_df[sbb_retain_mask].copy()
    
    print(f"Processing {len(sbb_candidates)} Segment 2 SBB/RETAIN households...")
    
    for idx, hh in sbb_candidates.iterrows():
        assigned = False
        hh_lat = hh['LAT_NUM']
        hh_lon = hh['LON_NUM']
        hh_au = hh['PATR_AU_STR']
        
        # Check if SBB banker exists in same AU
        sbb_in_au = sbb_data[sbb_data['AU'] == hh_au]
        
        if len(sbb_in_au) > 0:
            # Assign to first SBB banker in AU
            sbb_banker = sbb_in_au.iloc[0]
            sbb_assignments.append({
                'BHH_SKEY': hh['BHH_SKEY'],
                'HH_ECN': hh['HH_ECN'],
                'SBB_BANKER': sbb_banker['FULL NAME'],
                'SBB_AU': sbb_banker['AU'],
                'ASSIGNMENT_TYPE': 'Same AU'
            })
            # Remove from HH_DF (unassign from portfolio)
            hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = None
            hh_df.loc[idx, 'BANKER_TYPE'] = 'SBB'
            assigned = True
        else:
            # Check within 10-mile radius
            min_distance = float('inf')
            closest_sbb = None
            
            for _, sbb in sbb_data.iterrows():
                sbb_au = sbb['AU']
                branch = branch_data[branch_data['AU'] == sbb_au]
                
                if len(branch) > 0:
                    branch_lat = branch.iloc[0]['BRANCH_LAT_NUM']
                    branch_lon = branch.iloc[0]['BRANCH_LON_NUM']
                    
                    distance = haversine_distance(hh_lat, hh_lon, branch_lat, branch_lon)
                    
                    if distance <= SBB_RADIUS and distance < min_distance:
                        min_distance = distance
                        closest_sbb = sbb
            
            if closest_sbb is not None:
                sbb_assignments.append({
                    'BHH_SKEY': hh['BHH_SKEY'],
                    'HH_ECN': hh['HH_ECN'],
                    'SBB_BANKER': closest_sbb['FULL NAME'],
                    'SBB_AU': closest_sbb['AU'],
                    'ASSIGNMENT_TYPE': f'Within {min_distance:.1f} miles'
                })
                # Remove from HH_DF (unassign from portfolio)
                hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = None
                hh_df.loc[idx, 'BANKER_TYPE'] = 'SBB'
                assigned = True
        
        # If not assigned, convert to RETAIN
        if not assigned:
            hh_df.loc[idx, 'RULE'] = 'RETAIN'
    
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
        
        portfolio_stats[portfolio_cd] = {
            'banker_type': banker_type,
            'placement': placement,
            'target_segment': target_segment,
            'current_count': segment_count,
            'total_count': len(portfolio_hhs),
            'min_required': min_required,
            'max_allowed': max_allowed,
            'deficit': max(0, min_required - segment_count),
            'excess': max(0, segment_count - max_allowed),
            'lat': location_info['lat'],
            'lon': location_info['lon']
        }
    
    return portfolio_stats


# ==================== STEP 3: FIND NEAREST HOUSEHOLDS ====================

def find_nearest_households(hh_df, portfolio_cd, portfolio_stats, radius, segment):
    """
    Find nearest POOL households of given segment within radius for a portfolio.
    Returns list of (hh_index, distance) tuples sorted by distance.
    """
    portfolio_info = portfolio_stats[portfolio_cd]
    portfolio_lat = portfolio_info['lat']
    portfolio_lon = portfolio_info['lon']
    
    # Get available POOL households of target segment
    available_hhs = hh_df[
        (hh_df['NEW_SEGMENT'] == segment) & 
        (hh_df['RULE'] == 'POOL') & 
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]
    
    candidates = []
    for idx, hh in available_hhs.iterrows():
        distance = haversine_distance(
            portfolio_lat, portfolio_lon,
            hh['LAT_NUM'], hh['LON_NUM']
        )
        if distance <= radius:
            candidates.append((idx, distance))
    
    # Sort by distance
    candidates.sort(key=lambda x: x[1])
    
    return candidates


# ==================== STEP 4: TRIM OVERSIZED PORTFOLIOS ====================

def trim_oversized_portfolios(hh_df, portfolio_stats):
    """
    For portfolios exceeding MAX, remove the farthest households and return them to POOL.
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
            
            # Calculate distances
            distances = []
            for idx, hh in portfolio_hhs.iterrows():
                distance = haversine_distance(
                    stats['lat'], stats['lon'],
                    hh['LAT_NUM'], hh['LON_NUM']
                )
                distances.append((idx, distance))
            
            # Sort by distance (farthest first)
            distances.sort(key=lambda x: x[1], reverse=True)
            
            # Remove excess households (farthest ones)
            num_to_remove = stats['excess']
            for i in range(min(num_to_remove, len(distances))):
                idx = distances[i][0]
                hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = None
                print(f"  Trimmed HH_ECN {hh_df.loc[idx, 'HH_ECN']} from {portfolio_cd} (distance: {distances[i][1]:.1f} miles)")
    
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
        
        # Assign households to undersized portfolios
        undersized = [p for p in in_market_portfolios if portfolio_stats[p]['deficit'] > 0]
        
        if not undersized:
            print(f"  No undersized portfolios. Breaking loop.")
            break
        
        print(f"  Found {len(undersized)} undersized portfolios")
        
        for portfolio_cd in undersized:
            deficit = portfolio_stats[portfolio_cd]['deficit']
            candidates = find_nearest_households(hh_df, portfolio_cd, portfolio_stats, radius, segment)
            
            assigned_count = 0
            for idx, distance in candidates:
                if assigned_count >= deficit:
                    break
                hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
                hh_df.loc[idx, 'BANKER_TYPE'] = banker_type
                assigned_count += 1
            
            print(f"    Portfolio {portfolio_cd}: Assigned {assigned_count} households (needed {deficit})")
        
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


# ==================== STEP 7 & 8: OPTIMIZE CENTRALIZED PORTFOLIOS ====================

def optimize_centralized_portfolios(hh_df, portfolio_stats, banker_type, segment):
    """
    Optimize CENTRALIZED portfolios using iterative radius expansion (200-1000 miles).
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
        
        # Assign households to undersized portfolios
        undersized = [p for p in centralized_portfolios if portfolio_stats[p]['deficit'] > 0]
        
        if not undersized:
            print(f"  No undersized portfolios. Breaking loop.")
            break
        
        print(f"  Found {len(undersized)} undersized portfolios")
        
        for portfolio_cd in undersized:
            deficit = portfolio_stats[portfolio_cd]['deficit']
            candidates = find_nearest_households(hh_df, portfolio_cd, portfolio_stats, radius, segment)
            
            assigned_count = 0
            for idx, distance in candidates:
                if assigned_count >= deficit:
                    break
                hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
                hh_df.loc[idx, 'BANKER_TYPE'] = banker_type
                assigned_count += 1
            
            print(f"    Portfolio {portfolio_cd}: Assigned {assigned_count} households (needed {deficit})")
        
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
        
        for portfolio_cd in undersized:
            deficit = portfolio_stats[portfolio_cd]['deficit']
            
            # Find all unassigned households of target segment
            unassigned = hh_df[
                (hh_df['NEW_SEGMENT'] == segment) & 
                (hh_df['RULE'] == 'POOL') & 
                (hh_df['CG_PORTFOLIO_CD'].isna())
            ]
            
            if len(unassigned) == 0:
                break
            
            # Calculate distances to all unassigned
            distances = []
            for idx, hh in unassigned.iterrows():
                distance = haversine_distance(
                    portfolio_stats[portfolio_cd]['lat'],
                    portfolio_stats[portfolio_cd]['lon'],
                    hh['LAT_NUM'], hh['LON_NUM']
                )
                distances.append((idx, distance))
            
            distances.sort(key=lambda x: x[1])
            
            assigned_count = 0
            for idx, distance in distances:
                if assigned_count >= deficit:
                    break
                hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
                hh_df.loc[idx, 'BANKER_TYPE'] = banker_type
                assigned_count += 1
            
            print(f"    Portfolio {portfolio_cd}: Assigned {assigned_count} households to reach MIN")
            
            # Recalculate stats
            portfolio_stats = calculate_portfolio_requirements(hh_df, 
                {p: {'lat': portfolio_stats[p]['lat'], 
                     'lon': portfolio_stats[p]['lon'],
                     'placement': portfolio_stats[p]['placement'],
                     'banker_type': portfolio_stats[p]['banker_type']} 
                 for p in portfolio_stats})
    
    # Phase B: Assign all remaining to nearest portfolio
    remaining = hh_df[
        (hh_df['NEW_SEGMENT'] == segment) & 
        (hh_df['RULE'] == 'POOL') & 
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]
    
    print(f"  Phase B: Assigning {len(remaining)} remaining households to nearest portfolio")
    
    for idx, hh in remaining.iterrows():
        hh_lat = hh['LAT_NUM']
        hh_lon = hh['LON_NUM']
        
        # Find nearest centralized portfolio
        min_distance = float('inf')
        nearest_portfolio = None
        
        for portfolio_cd in centralized_portfolios:
            distance = haversine_distance(
                hh_lat, hh_lon,
                portfolio_stats[portfolio_cd]['lat'],
                portfolio_stats[portfolio_cd]['lon']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_portfolio = portfolio_cd
        
        if nearest_portfolio:
            hh_df.loc[idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
            hh_df.loc[idx, 'BANKER_TYPE'] = banker_type
    
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
    print("Imputing missing coordinates...")
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
