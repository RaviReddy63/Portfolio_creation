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

# State name to code mapping
STATE_NAME_TO_CODE = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
    'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
    'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
    'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
    'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
    'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
    'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
    'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
    'DISTRICT OF COLUMBIA': 'DC', 'PUERTO RICO': 'PR', 'GUAM': 'GU', 'VIRGIN ISLANDS': 'VI'
}


# ==================== UTILITY FUNCTIONS ====================

def convert_state_to_code(state_name):
    """
    Convert full state name to 2-letter state code.
    Returns None if state name is not found.
    """
    if pd.isna(state_name):
        return None

    state_upper = str(state_name).strip().upper()
    return STATE_NAME_TO_CODE.get(state_upper, None)


def build_portfolio_coverage_map(rbrm_data):
    """
    Build a mapping of portfolio code to its allowed set of state codes
    from the COVERAGE column in rbrm_data.

    Handles both:
      - Multiple rows per portfolio (one state per row)
      - Single row with comma-separated states (e.g., 'LA, AR, OK, KS')

    Args:
        rbrm_data: DataFrame with columns CG_PORTFOLIO_CD and COVERAGE

    Returns:
        dict: { portfolio_cd -> set of state codes }
              e.g., { 'P1': {'LA', 'AR', 'OK', 'KS'}, ... }
    """
    portfolio_coverage = {}

    for portfolio_cd, group in rbrm_data.groupby('CG_PORTFOLIO_CD'):
        states = set()
        for val in group['COVERAGE'].dropna():
            # Handle comma-separated values within a single cell
            for state in str(val).split(','):
                state = state.strip().upper()
                if state:
                    states.add(state)
        portfolio_coverage[portfolio_cd] = states

    print(f"Built coverage map for {len(portfolio_coverage)} portfolios.")
    return portfolio_coverage


def check_portfolio_state_match(customer_state_code, portfolio_cd, portfolio_coverage_map):
    """
    Check if a customer's state code is in the portfolio's allowed coverage states.

    Args:
        customer_state_code : 2-letter state code (e.g., 'LA')
        portfolio_cd        : Portfolio code (e.g., 'P1')
        portfolio_coverage_map: dict returned by build_portfolio_coverage_map()

    Returns:
        True if the customer state is covered by the portfolio, False otherwise.
        If the portfolio has no coverage defined, returns True (no restriction).
    """
    if pd.isna(customer_state_code) or not customer_state_code:
        return False

    allowed_states = portfolio_coverage_map.get(portfolio_cd, None)

    # If no coverage defined for this portfolio, allow all states
    if allowed_states is None or len(allowed_states) == 0:
        return True

    return str(customer_state_code).strip().upper() in allowed_states


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
    Treats NULL, 0, 0.00, and any value equal to 0 as missing coordinates.
    """
    hh_df = hh_df.copy()

    missing_mask = (hh_df['LAT_NUM'].isna() | hh_df['LON_NUM'].isna() |
                   (hh_df['LAT_NUM'].abs() < 1e-9) | (hh_df['LON_NUM'].abs() < 1e-9))

    if missing_mask.sum() == 0:
        print("No missing coordinates to impute.")
        return hh_df

    print(f"Imputing {missing_mask.sum()} missing coordinates using KNN...")

    hh_df.loc[hh_df['LAT_NUM'].abs() < 1e-9, 'LAT_NUM'] = np.nan
    hh_df.loc[hh_df['LON_NUM'].abs() < 1e-9, 'LON_NUM'] = np.nan

    hh_df['location_group'] = hh_df['BILLINGCITY'].astype(str) + '_' + hh_df['BILLINGSTATE'].astype(str)

    for group in hh_df['location_group'].unique():
        group_mask = hh_df['location_group'] == group
        group_data = hh_df[group_mask].copy()

        group_missing = group_data['LAT_NUM'].isna() | group_data['LON_NUM'].isna()

        if group_missing.sum() > 0 and group_missing.sum() < len(group_data):
            coords = group_data[['LAT_NUM', 'LON_NUM']].values
            num_valid = len(group_data) - group_missing.sum()
            n_neighbors = 5 if num_valid >= 5 else num_valid
            imputer = KNNImputer(n_neighbors=n_neighbors)
            coords_imputed = imputer.fit_transform(coords)
            hh_df.loc[group_mask, 'LAT_NUM'] = coords_imputed[:, 0]
            hh_df.loc[group_mask, 'LON_NUM'] = coords_imputed[:, 1]

        elif group_missing.sum() == len(group_data):
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
                hh_df.loc[group_mask, 'LAT_NUM'] = random.uniform(*EASTERN_US_LAT_RANGE)
                hh_df.loc[group_mask, 'LON_NUM'] = random.uniform(*EASTERN_US_LON_RANGE)

    hh_df.drop('location_group', axis=1, inplace=True)

    print("Coordinate imputation complete.")
    return hh_df


def create_balltree(lat_lon_array):
    """
    Create a BallTree for efficient distance queries.
    Input: array of [lat, lon] in degrees
    Returns: BallTree object
    """
    lat_lon_array = np.array(lat_lon_array, dtype=np.float64)
    lat_lon_radians = np.radians(lat_lon_array)
    return BallTree(lat_lon_radians, metric='haversine')


def query_balltree(tree, query_points, radius_miles):
    """
    Query BallTree for points within radius.

    Args:
        tree        : BallTree object
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

    distances_miles = [d * EARTH_RADIUS_MILES for d in distances]

    return indices, distances_miles


def create_portfolio_location_map(rbrm_data, branch_data, portfolio_centroids):
    """
    Create a mapping of portfolio code to location (lat, lon).
    IN MARKET portfolios use branch locations, CENTRALIZED use centroids.
    Director field is retained for reference but no longer used for state matching.
    """
    portfolio_locations = {}

    branch_data = branch_data.copy()
    branch_data['AU'] = branch_data['AU'].astype(int)

    for _, row in rbrm_data.iterrows():
        portfolio_cd = row['CG_PORTFOLIO_CD']
        placement = row['PLACEMENT']

        if placement == 'IN MARKET':
            au = int(row['AU'])
            branch = branch_data[branch_data['AU'] == au]
            if len(branch) > 0:
                portfolio_locations[portfolio_cd] = {
                    'lat': branch.iloc[0]['BRANCH_LAT_NUM'],
                    'lon': branch.iloc[0]['BRANCH_LON_NUM'],
                    'placement': 'IN MARKET',
                    'banker_type': row['BANKER_TYPE'],
                }
        else:  # CENTRALIZED
            centroid = portfolio_centroids[portfolio_centroids['CG_PORTFOLIO_CD'] == portfolio_cd]
            if len(centroid) > 0 and pd.notna(centroid.iloc[0]['CENTROID_LAT_NUM']):
                portfolio_locations[portfolio_cd] = {
                    'lat': centroid.iloc[0]['CENTROID_LAT_NUM'],
                    'lon': centroid.iloc[0]['CENTROID_LON_NUM'],
                    'placement': 'CENTRALIZED',
                    'banker_type': row['BANKER_TYPE'],
                }
            else:
                portfolio_locations[portfolio_cd] = {
                    'lat': random.uniform(*EASTERN_US_LAT_RANGE),
                    'lon': random.uniform(*EASTERN_US_LON_RANGE),
                    'placement': 'CENTRALIZED',
                    'banker_type': row['BANKER_TYPE'],
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

    sbb_retain_mask = (hh_df['NEW_SEGMENT'] == 2) & (hh_df['RULE'] == 'SBB/RETAIN')
    sbb_candidates = hh_df[sbb_retain_mask].copy()

    print(f"Processing {len(sbb_candidates)} Segment 2 SBB/RETAIN households...")

    if len(sbb_candidates) == 0:
        return hh_df, pd.DataFrame()

    valid_coords_mask = sbb_candidates['LAT_NUM'].notna() & sbb_candidates['LON_NUM'].notna()
    sbb_candidates_valid = sbb_candidates[valid_coords_mask].copy()
    sbb_candidates_invalid = sbb_candidates[~valid_coords_mask].copy()

    if len(sbb_candidates_invalid) > 0:
        print(f"Warning: {len(sbb_candidates_invalid)} households have invalid coordinates, converting to RETAIN")
        for idx in sbb_candidates_invalid.index:
            hh_df.loc[idx, 'RULE'] = 'RETAIN'

    if len(sbb_candidates_valid) == 0:
        return hh_df, pd.DataFrame()

    branch_data = branch_data.copy()
    branch_data['AU'] = branch_data['AU'].astype(int)

    sbb_locations = []
    sbb_info = []

    for _, sbb in sbb_data.iterrows():
        sbb_au = int(sbb['AU'])
        branch = branch_data[branch_data['AU'] == sbb_au]

        if len(branch) > 0:
            branch_lat = branch.iloc[0]['BRANCH_LAT_NUM']
            branch_lon = branch.iloc[0]['BRANCH_LON_NUM']

            if pd.notna(branch_lat) and pd.notna(branch_lon):
                sbb_locations.append([branch_lat, branch_lon])
                sbb_info.append({
                    'full_name': sbb['FULL NAME'],
                    'au': sbb_au
                })

    if len(sbb_locations) == 0:
        print("No SBB bankers with valid branch locations found.")
        for idx in sbb_candidates_valid.index:
            hh_df.loc[idx, 'RULE'] = 'RETAIN'
        return hh_df, pd.DataFrame()

    sbb_tree = create_balltree(np.array(sbb_locations))
    hh_locations = sbb_candidates_valid[['LAT_NUM', 'LON_NUM']].values

    for idx, (hh_idx, hh) in enumerate(sbb_candidates_valid.iterrows()):
        assigned = False
        hh_au = int(hh['PATR_AU_STR']) if pd.notna(hh['PATR_AU_STR']) else 0

        sbb_in_au = [i for i, info in enumerate(sbb_info) if info['au'] == hh_au]

        if len(sbb_in_au) > 0:
            sbb_idx = sbb_in_au[0]
            sbb_banker = sbb_info[sbb_idx]
            sbb_assignments.append({
                'BHH_SKEY': hh['BHH_SKEY'],
                'HH_ECN': hh['HH_ECN'],
                'SBB_BANKER': sbb_banker['full_name'],
                'SBB_AU': sbb_banker['au'],
                'ASSIGNMENT_TYPE': 'Same AU'
            })
            hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = None
            hh_df.loc[hh_idx, 'BANKER_TYPE'] = 'SBB'
            assigned = True
        else:
            query_point = hh_locations[idx:idx+1]
            indices, distances = query_balltree(sbb_tree, query_point, SBB_RADIUS)

            if len(indices[0]) > 0:
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
                hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = None
                hh_df.loc[hh_idx, 'BANKER_TYPE'] = 'SBB'
                assigned = True

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
    Counts BOTH Segment 3 and Segment 4 for all portfolios.
    Returns a dictionary with portfolio statistics.
    """
    portfolio_stats = {}

    for portfolio_cd, location_info in portfolio_locations.items():
        banker_type = location_info['banker_type']
        placement = location_info['placement']

        portfolio_hhs = hh_df[hh_df['CG_PORTFOLIO_CD'] == portfolio_cd]
        segment_count = len(portfolio_hhs[portfolio_hhs['NEW_SEGMENT'].isin([3, 4])])

        if banker_type == 'RM':
            min_required = RM_MIN
            max_allowed = RM_MAX
        else:  # RC
            min_required = RC_MIN
            max_allowed = RC_MAX

        deficit_val = min_required - segment_count
        excess_val = segment_count - max_allowed

        portfolio_stats[portfolio_cd] = {
            'banker_type': banker_type,
            'placement': placement,
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


# ==================== STEP 4: TRIM OVERSIZED PORTFOLIOS ====================

def trim_oversized_portfolios(hh_df, portfolio_stats):
    """
    For portfolios exceeding MAX, remove the farthest households and return them to POOL.
    Trims from BOTH Segment 3 and Segment 4.
    Uses BallTree for efficient distance calculation.
    Returns updated hh_df.
    """
    hh_df = hh_df.copy()

    for portfolio_cd, stats in portfolio_stats.items():
        if stats['excess'] > 0:
            portfolio_hhs = hh_df[
                (hh_df['CG_PORTFOLIO_CD'] == portfolio_cd) &
                (hh_df['NEW_SEGMENT'].isin([3, 4])) &
                (hh_df['RULE'] == 'POOL')
            ]

            if len(portfolio_hhs) == 0:
                continue

            hh_locations = portfolio_hhs[['LAT_NUM', 'LON_NUM']].values
            portfolio_location = np.array([[stats['lat'], stats['lon']]])

            hh_tree = create_balltree(hh_locations)
            indices, distances = query_balltree(
                hh_tree,
                portfolio_location,
                radius_miles=10000
            )

            distances_list = []
            original_indices = portfolio_hhs.index.values
            for i, dist in zip(indices[0], distances[0]):
                distances_list.append((original_indices[i], dist))

            distances_list.sort(key=lambda x: x[1], reverse=True)

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


# ==================== STEP 6: OPTIMIZE PORTFOLIOS ====================

def optimize_in_market_portfolios(hh_df, portfolio_stats, portfolio_coverage_map, placement='IN MARKET'):
    """
    Optimize portfolios using iterative radius expansion.
    Assigns each household to nearest undersized portfolio within coverage states.
    Assigns BOTH Segment 3 and Segment 4 customers to BOTH RM and RC portfolios.
    Uses portfolio-level COVERAGE state matching instead of director-level matching.
    Uses portfolio_locations BallTree created once.
    """
    hh_df = hh_df.copy()

    portfolios = [
        p for p, s in portfolio_stats.items()
        if s['placement'] == placement
    ]

    print(f"\nOptimizing {len(portfolios)} {placement} portfolios (Segments 3 & 4, both RM & RC)...")

    if len(portfolios) == 0:
        print(f"  No {placement} portfolios found.")
        return hh_df

    # Create BallTree ONCE for all portfolios
    portfolio_locs_array = np.array([
        [portfolio_stats[p]['lat'], portfolio_stats[p]['lon']]
        for p in portfolios
    ])
    portfolio_tree = create_balltree(portfolio_locs_array)

    if placement == 'IN MARKET':
        start_radius = IN_MARKET_START_RADIUS
        max_radius = IN_MARKET_MAX_RADIUS
        increment = IN_MARKET_RADIUS_INCREMENT
    else:  # CENTRALIZED
        start_radius = CENTRALIZED_START_RADIUS
        max_radius = CENTRALIZED_MAX_RADIUS
        increment = CENTRALIZED_RADIUS_INCREMENT

    for radius in range(start_radius, max_radius + 1, increment):
        print(f"\n  Radius: {radius} miles")

        # Recalculate portfolio stats
        portfolio_stats = calculate_portfolio_requirements(hh_df,
            {p: {'lat': portfolio_stats[p]['lat'],
                 'lon': portfolio_stats[p]['lon'],
                 'placement': portfolio_stats[p]['placement'],
                 'banker_type': portfolio_stats[p]['banker_type']}
             for p in portfolio_stats})

        undersized_indices = [i for i, p in enumerate(portfolios) if portfolio_stats[p]['deficit'] > 0]

        if not undersized_indices:
            print(f"  No undersized portfolios. Breaking loop.")
            break

        print(f"  Found {len(undersized_indices)} undersized portfolios")

        # Get all unassigned households of BOTH Segment 3 and 4
        unassigned_hhs = hh_df[
            (hh_df['NEW_SEGMENT'].isin([3, 4])) &
            (hh_df['RULE'] == 'POOL') &
            (hh_df['CG_PORTFOLIO_CD'].isna())
        ].copy()

        if len(unassigned_hhs) == 0:
            print(f"  No unassigned households available.")
            break

        # Add state code column for portfolio coverage matching
        unassigned_hhs['STATE_CODE'] = unassigned_hhs['BILLINGSTATE'].apply(convert_state_to_code)

        portfolio_counts = {portfolios[i]: portfolio_stats[portfolios[i]]['current_count']
                           for i in undersized_indices}

        assigned_count = 0
        skipped_coverage_mismatch = 0

        hh_locations = unassigned_hhs[['LAT_NUM', 'LON_NUM']].values
        indices_list, distances_list = query_balltree(portfolio_tree, hh_locations, radius)

        for hh_idx, (indices, distances) in zip(unassigned_hhs.index, zip(indices_list, distances_list)):
            if len(indices) == 0:
                continue

            customer_state_code = unassigned_hhs.loc[hh_idx, 'STATE_CODE']

            assigned = False
            for i, dist in zip(indices, distances):
                if i not in undersized_indices:
                    continue

                portfolio_cd = portfolios[i]

                # Portfolio-level coverage check
                if not check_portfolio_state_match(customer_state_code, portfolio_cd, portfolio_coverage_map):
                    skipped_coverage_mismatch += 1
                    continue

                current_count = portfolio_counts.get(portfolio_cd, portfolio_stats[portfolio_cd]['current_count'])
                deficit = portfolio_stats[portfolio_cd]['min_required'] - current_count

                if deficit > 0:
                    hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
                    hh_df.loc[hh_idx, 'BANKER_TYPE'] = portfolio_stats[portfolio_cd]['banker_type']
                    assigned_count += 1

                    portfolio_counts[portfolio_cd] = current_count + 1

                    new_deficit = portfolio_stats[portfolio_cd]['min_required'] - portfolio_counts[portfolio_cd]
                    if new_deficit <= 0:
                        undersized_indices.remove(i)

                    assigned = True
                    break

            if not assigned:
                continue

            if len(undersized_indices) == 0:
                break

        print(f"  Assigned {assigned_count} households to nearest portfolios")
        print(f"  Skipped {skipped_coverage_mismatch} assignments due to portfolio coverage mismatch")

        # Trim oversized portfolios
        portfolio_stats = calculate_portfolio_requirements(hh_df,
            {p: {'lat': portfolio_stats[p]['lat'],
                 'lon': portfolio_stats[p]['lon'],
                 'placement': portfolio_stats[p]['placement'],
                 'banker_type': portfolio_stats[p]['banker_type']}
             for p in portfolio_stats})

        oversized = [p for p in portfolios if portfolio_stats[p]['excess'] > 0]
        if oversized:
            print(f"  Trimming {len(oversized)} oversized portfolios")
            hh_df = trim_oversized_portfolios(hh_df, portfolio_stats)

    return hh_df


# ==================== STEP 6.5: FILL PORTFOLIOS TO MAX ====================

def fill_portfolios_to_max(hh_df, portfolio_stats, portfolio_coverage_map, placement='IN MARKET', radius=20):
    """
    Fill portfolios up to MAX by assigning nearby households to nearest portfolio.
    Each household is assigned to the nearest portfolio that has capacity.
    Assigns BOTH Segment 3 and Segment 4 customers to BOTH RM and RC portfolios.
    Uses portfolio-level COVERAGE state matching instead of director-level matching.
    Uses portfolio_locations BallTree created once.
    """
    hh_df = hh_df.copy()

    portfolios = [
        p for p, s in portfolio_stats.items()
        if s['placement'] == placement
    ]

    print(f"\nFilling {len(portfolios)} {placement} portfolios to MAX (Segments 3 & 4, both RM & RC)...")
    print(f"  Using {radius} mile radius")

    if len(portfolios) == 0:
        print(f"  No {placement} portfolios found.")
        return hh_df

    # Recalculate portfolio stats
    portfolio_stats = calculate_portfolio_requirements(hh_df,
        {p: {'lat': portfolio_stats[p]['lat'],
             'lon': portfolio_stats[p]['lon'],
             'placement': portfolio_stats[p]['placement'],
             'banker_type': portfolio_stats[p]['banker_type']}
         for p in portfolio_stats})

    portfolios_with_capacity_indices = [
        i for i, p in enumerate(portfolios)
        if portfolio_stats[p]['current_count'] < portfolio_stats[p]['max_allowed']
    ]

    if not portfolios_with_capacity_indices:
        print("  All portfolios are at MAX capacity.")
        return hh_df

    unassigned_hhs = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ].copy()

    if len(unassigned_hhs) == 0:
        print("  No unassigned households available.")
        return hh_df

    # Add state code column for portfolio coverage matching
    unassigned_hhs['STATE_CODE'] = unassigned_hhs['BILLINGSTATE'].apply(convert_state_to_code)

    # Create BallTree ONCE for all portfolios
    portfolio_locs_array = np.array([
        [portfolio_stats[p]['lat'], portfolio_stats[p]['lon']]
        for p in portfolios
    ])
    portfolio_tree = create_balltree(portfolio_locs_array)

    portfolio_counts = {portfolios[i]: portfolio_stats[portfolios[i]]['current_count']
                       for i in portfolios_with_capacity_indices}

    assigned_count = 0
    skipped_coverage_mismatch = 0

    hh_locations = unassigned_hhs[['LAT_NUM', 'LON_NUM']].values
    indices_list, distances_list = query_balltree(portfolio_tree, hh_locations, radius)

    for hh_idx, (indices, distances) in zip(unassigned_hhs.index, zip(indices_list, distances_list)):
        if len(indices) == 0:
            continue

        customer_state_code = unassigned_hhs.loc[hh_idx, 'STATE_CODE']

        assigned = False
        for i, dist in zip(indices, distances):
            if i not in portfolios_with_capacity_indices:
                continue

            portfolio_cd = portfolios[i]

            # Portfolio-level coverage check
            if not check_portfolio_state_match(customer_state_code, portfolio_cd, portfolio_coverage_map):
                skipped_coverage_mismatch += 1
                continue

            current_count = portfolio_counts.get(portfolio_cd, portfolio_stats[portfolio_cd]['current_count'])
            max_allowed = portfolio_stats[portfolio_cd]['max_allowed']

            if current_count < max_allowed:
                hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
                hh_df.loc[hh_idx, 'BANKER_TYPE'] = portfolio_stats[portfolio_cd]['banker_type']
                assigned_count += 1

                portfolio_counts[portfolio_cd] = current_count + 1

                if portfolio_counts[portfolio_cd] >= max_allowed:
                    portfolios_with_capacity_indices.remove(i)

                assigned = True
                break

        if not assigned:
            continue

        if len(portfolios_with_capacity_indices) == 0:
            break

    print(f"  Assigned {assigned_count} additional households to nearest portfolios with capacity")
    print(f"  Skipped {skipped_coverage_mismatch} assignments due to portfolio coverage mismatch")

    return hh_df


# ==================== STEP 9 & 10: FINAL CLEANUP ====================

def assign_remaining_households(hh_df, portfolio_stats, portfolio_coverage_map, placement='CENTRALIZED'):
    """
    Final cleanup: Assign all remaining unassigned households.
    Handles BOTH Segment 3 and Segment 4 together for BOTH RM and RC portfolios.
    Uses portfolio-level COVERAGE state matching instead of director-level matching.

    Phase A: Fill ALL portfolios to MAX
    Phase B: Assign remaining to nearest portfolio below MAX
    Phase C: If still unassigned, increase MAX by 20 iteratively and assign

    Uses portfolio_locations BallTree created once.
    """
    hh_df = hh_df.copy()

    portfolios = [
        p for p, s in portfolio_stats.items()
        if s['placement'] == placement
    ]

    print(f"\nFinal cleanup for {placement} portfolios (Segments 3 & 4, both RM & RC)...")

    if len(portfolios) == 0:
        print(f"  No {placement} portfolios found.")
        return hh_df

    # Create BallTree ONCE for all portfolios
    portfolio_locs_array = np.array([
        [portfolio_stats[p]['lat'], portfolio_stats[p]['lon']]
        for p in portfolios
    ])
    portfolio_tree = create_balltree(portfolio_locs_array)

    # ---------- Phase A: Fill ALL portfolios to MAX ----------
    portfolio_stats = calculate_portfolio_requirements(hh_df,
        {p: {'lat': portfolio_stats[p]['lat'],
             'lon': portfolio_stats[p]['lon'],
             'placement': portfolio_stats[p]['placement'],
             'banker_type': portfolio_stats[p]['banker_type']}
         for p in portfolio_stats})

    portfolios_with_capacity_indices = [
        i for i, p in enumerate(portfolios)
        if portfolio_stats[p]['current_count'] < portfolio_stats[p]['max_allowed']
    ]

    if portfolios_with_capacity_indices:
        print(f"  Phase A: Filling {len(portfolios_with_capacity_indices)} portfolios to MAX")

        unassigned = hh_df[
            (hh_df['NEW_SEGMENT'].isin([3, 4])) &
            (hh_df['RULE'] == 'POOL') &
            (hh_df['CG_PORTFOLIO_CD'].isna())
        ].copy()

        if len(unassigned) > 0:
            unassigned['STATE_CODE'] = unassigned['BILLINGSTATE'].apply(convert_state_to_code)

            portfolio_counts = {portfolios[i]: portfolio_stats[portfolios[i]]['current_count']
                               for i in portfolios_with_capacity_indices}

            portfolios_with_capacity_set = set(portfolios_with_capacity_indices)

            assigned_count = 0
            skipped_coverage_mismatch = 0

            hh_locations = unassigned[['LAT_NUM', 'LON_NUM']].values
            indices_list, distances_list = query_balltree(portfolio_tree, hh_locations, radius_miles=10000)

            for hh_idx, (indices, distances) in zip(unassigned.index, zip(indices_list, distances_list)):
                if len(indices) == 0:
                    continue

                if len(portfolios_with_capacity_set) == 0:
                    break

                customer_state_code = unassigned.loc[hh_idx, 'STATE_CODE']

                assigned = False
                for i, dist in zip(indices, distances):
                    if i not in portfolios_with_capacity_set:
                        continue

                    portfolio_cd = portfolios[i]

                    # Portfolio-level coverage check
                    if not check_portfolio_state_match(customer_state_code, portfolio_cd, portfolio_coverage_map):
                        skipped_coverage_mismatch += 1
                        continue

                    current_count = portfolio_counts.get(portfolio_cd, portfolio_stats[portfolio_cd]['current_count'])
                    max_allowed = portfolio_stats[portfolio_cd]['max_allowed']

                    if current_count < max_allowed:
                        hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
                        hh_df.loc[hh_idx, 'BANKER_TYPE'] = portfolio_stats[portfolio_cd]['banker_type']
                        assigned_count += 1

                        portfolio_counts[portfolio_cd] = current_count + 1

                        if portfolio_counts[portfolio_cd] >= max_allowed:
                            portfolios_with_capacity_set.discard(i)

                        assigned = True
                        break

                if not assigned:
                    continue

            print(f"    Assigned {assigned_count} households to reach MAX")
            print(f"    Skipped {skipped_coverage_mismatch} assignments due to portfolio coverage mismatch")

    # ---------- Phase B: Assign remaining to nearest portfolio below MAX ----------
    portfolio_stats = calculate_portfolio_requirements(hh_df,
        {p: {'lat': portfolio_stats[p]['lat'],
             'lon': portfolio_stats[p]['lon'],
             'placement': portfolio_stats[p]['placement'],
             'banker_type': portfolio_stats[p]['banker_type']}
         for p in portfolio_stats})

    remaining = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ].copy()

    print(f"  Phase B: Assigning {len(remaining)} remaining households to nearest portfolio (without exceeding MAX)")

    if len(remaining) > 0 and len(portfolios) > 0:
        remaining['STATE_CODE'] = remaining['BILLINGSTATE'].apply(convert_state_to_code)

        portfolio_counts = {p: portfolio_stats[p]['current_count'] for p in portfolios}

        assigned_count = 0
        not_assigned_count = 0
        skipped_coverage_mismatch = 0

        hh_locations = remaining[['LAT_NUM', 'LON_NUM']].values
        indices_list, distances_list = query_balltree(portfolio_tree, hh_locations, radius_miles=10000)

        for hh_idx, (indices, distances) in zip(remaining.index, zip(indices_list, distances_list)):
            if len(indices) == 0:
                not_assigned_count += 1
                continue

            customer_state_code = remaining.loc[hh_idx, 'STATE_CODE']

            assigned = False
            for i, dist in zip(indices, distances):
                nearest_portfolio = portfolios[i]

                # Portfolio-level coverage check
                if not check_portfolio_state_match(customer_state_code, nearest_portfolio, portfolio_coverage_map):
                    skipped_coverage_mismatch += 1
                    continue

                current_count = portfolio_counts.get(nearest_portfolio, portfolio_stats[nearest_portfolio]['current_count'])
                max_allowed = portfolio_stats[nearest_portfolio]['max_allowed']

                if current_count < max_allowed:
                    hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
                    hh_df.loc[hh_idx, 'BANKER_TYPE'] = portfolio_stats[nearest_portfolio]['banker_type']
                    assigned_count += 1
                    portfolio_counts[nearest_portfolio] = current_count + 1
                    assigned = True
                    break

            if not assigned:
                not_assigned_count += 1

        print(f"    Assigned {assigned_count} households to nearest portfolio below MAX")
        print(f"    Skipped {skipped_coverage_mismatch} assignments due to portfolio coverage mismatch")
        if not_assigned_count > 0:
            print(f"    Could not assign {not_assigned_count} households (all nearest portfolios at MAX or coverage mismatch)")

    # ---------- Phase C: Iteratively increase MAX by 20 ----------
    remaining = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ].copy()

    if len(remaining) > 0:
        print(f"\n  Phase C: Iteratively increasing MAX by 20 to assign {len(remaining)} remaining households")

        remaining['STATE_CODE'] = remaining['BILLINGSTATE'].apply(convert_state_to_code)

        portfolio_stats = calculate_portfolio_requirements(hh_df,
            {p: {'lat': portfolio_stats[p]['lat'],
                 'lon': portfolio_stats[p]['lon'],
                 'placement': portfolio_stats[p]['placement'],
                 'banker_type': portfolio_stats[p]['banker_type']}
             for p in portfolio_stats})

        portfolio_counts = {p: portfolio_stats[p]['current_count'] for p in portfolios}

        original_max_rm = RM_MAX
        original_max_rc = RC_MAX

        iteration = 1
        max_iterations = 50

        while len(remaining) > 0 and iteration <= max_iterations:
            current_max_rm = original_max_rm + (iteration * 20)
            current_max_rc = original_max_rc + (iteration * 20)
            print(f"    Iteration {iteration}: Increasing MAX to RM:{current_max_rm}, RC:{current_max_rc}")

            hh_locations = remaining[['LAT_NUM', 'LON_NUM']].values
            indices_list, distances_list = query_balltree(portfolio_tree, hh_locations, radius_miles=10000)

            assigned_count = 0
            skipped_coverage_mismatch = 0

            for hh_idx, (indices, distances) in zip(remaining.index, zip(indices_list, distances_list)):
                if len(indices) == 0:
                    continue

                customer_state_code = remaining.loc[hh_idx, 'STATE_CODE']

                assigned = False
                for i, dist in zip(indices, distances):
                    nearest_portfolio = portfolios[i]
                    banker_type = portfolio_stats[nearest_portfolio]['banker_type']

                    # Portfolio-level coverage check
                    if not check_portfolio_state_match(customer_state_code, nearest_portfolio, portfolio_coverage_map):
                        skipped_coverage_mismatch += 1
                        continue

                    current_count = portfolio_counts.get(nearest_portfolio, portfolio_stats[nearest_portfolio]['current_count'])
                    current_max = current_max_rm if banker_type == 'RM' else current_max_rc

                    if current_count < current_max:
                        hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = nearest_portfolio
                        hh_df.loc[hh_idx, 'BANKER_TYPE'] = banker_type
                        assigned_count += 1
                        portfolio_counts[nearest_portfolio] = current_count + 1
                        assigned = True
                        break

                if not assigned:
                    continue

            print(f"      Assigned {assigned_count} households in this iteration")
            print(f"      Skipped {skipped_coverage_mismatch} assignments due to portfolio coverage mismatch")

            remaining = hh_df[
                (hh_df['NEW_SEGMENT'].isin([3, 4])) &
                (hh_df['RULE'] == 'POOL') &
                (hh_df['CG_PORTFOLIO_CD'].isna())
            ].copy()

            if len(remaining) > 0:
                remaining['STATE_CODE'] = remaining['BILLINGSTATE'].apply(convert_state_to_code)

            if assigned_count == 0 and len(remaining) > 0:
                print(f"      No households assigned in this iteration. Stopping.")
                break

            iteration += 1

        if len(remaining) > 0:
            print(f"    Warning: {len(remaining)} households still unassigned after {iteration-1} iterations")

    # Verify final unassigned count
    final_unassigned = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]

    print(f"  Final unassigned (Segments 3 & 4): {len(final_unassigned)}")

    return hh_df


# ==================== MAIN ORCHESTRATOR ====================

def run_portfolio_reconstruction(hh_df, branch_data, rbrm_data, sbb_data, portfolio_centroids):
    """
    Main function to orchestrate the entire portfolio reconstruction process.
    Assigns BOTH Segment 3 and Segment 4 to both RM and RC portfolios.
    Uses portfolio-level COVERAGE state matching (from rbrm_data COVERAGE column)
    instead of the previous director-level state matching.
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

    # Build portfolio coverage map from rbrm_data COVERAGE column
    print("Building portfolio coverage map from COVERAGE column...")
    portfolio_coverage_map = build_portfolio_coverage_map(rbrm_data)

    # ========== STEP 1: SBB ASSIGNMENT ==========
    print("\n[STEP 1: SBB/RETAIN ASSIGNMENT]")
    hh_df, sbb_assignments_df = assign_sbb_bankers(hh_df, sbb_data, branch_data)

    # ========== STEP 2: CALCULATE REQUIREMENTS ==========
    print("\n[STEP 2: CALCULATE PORTFOLIO REQUIREMENTS]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)

    rm_in_market    = [p for p, s in portfolio_stats.items() if s['banker_type'] == 'RM' and s['placement'] == 'IN MARKET']
    rm_centralized  = [p for p, s in portfolio_stats.items() if s['banker_type'] == 'RM' and s['placement'] == 'CENTRALIZED']
    rc_in_market    = [p for p, s in portfolio_stats.items() if s['banker_type'] == 'RC' and s['placement'] == 'IN MARKET']
    rc_centralized  = [p for p, s in portfolio_stats.items() if s['banker_type'] == 'RC' and s['placement'] == 'CENTRALIZED']

    print(f"RM IN MARKET portfolios:    {len(rm_in_market)}")
    print(f"RM CENTRALIZED portfolios:  {len(rm_centralized)}")
    print(f"RC IN MARKET portfolios:    {len(rc_in_market)}")
    print(f"RC CENTRALIZED portfolios:  {len(rc_centralized)}")

    # ========== STEP 3: IN MARKET OPTIMIZATION (RM & RC together) ==========
    print("\n[STEP 3: IN MARKET OPTIMIZATION (RM & RC)]")
    hh_df = optimize_in_market_portfolios(hh_df, portfolio_stats, portfolio_coverage_map, placement='IN MARKET')

    # ========== STEP 4: FILL IN MARKET TO MAX (RM & RC together) ==========
    print("\n[STEP 4: FILL IN MARKET TO MAX (RM & RC)]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = fill_portfolios_to_max(hh_df, portfolio_stats, portfolio_coverage_map, placement='IN MARKET', radius=20)

    # ========== STEP 5: CENTRALIZED OPTIMIZATION (RM & RC together) ==========
    print("\n[STEP 5: CENTRALIZED OPTIMIZATION (RM & RC)]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = optimize_in_market_portfolios(hh_df, portfolio_stats, portfolio_coverage_map, placement='CENTRALIZED')

    # ========== STEP 6: FINAL CLEANUP (RM & RC together) ==========
    print("\n[STEP 6: FINAL CLEANUP (RM & RC)]")
    portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
    hh_df = assign_remaining_households(hh_df, portfolio_stats, portfolio_coverage_map, placement='CENTRALIZED')

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
        print(f"  {status} {p}: {s['current_count']} HHs (MIN: {s['min_required']}, MAX: {s['max_allowed']})")

    print(f"\nRM CENTRALIZED portfolios:")
    for p in rm_centralized:
        s = portfolio_stats[p]
        status = "✓" if s['current_count'] >= s['min_required'] else "✗"
        print(f"  {status} {p}: {s['current_count']} HHs (MIN: {s['min_required']}, MAX: {s['max_allowed']})")

    print(f"\nRC IN MARKET portfolios:")
    for p in rc_in_market:
        s = portfolio_stats[p]
        status = "✓" if s['current_count'] >= s['min_required'] else "✗"
        print(f"  {status} {p}: {s['current_count']} HHs (MIN: {s['min_required']}, MAX: {s['max_allowed']})")

    print(f"\nRC CENTRALIZED portfolios:")
    for p in rc_centralized:
        s = portfolio_stats[p]
        status = "✓" if s['current_count'] >= s['min_required'] else "✗"
        print(f"  {status} {p}: {s['current_count']} HHs (MIN: {s['min_required']}, MAX: {s['max_allowed']})")

    unassigned_seg3_4 = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]

    print(f"\nUNASSIGNED HOUSEHOLDS:")
    print(f"  Segments 3 & 4: {len(unassigned_seg3_4)}")

    return hh_df, sbb_assignments_df, portfolio_stats


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Load your data
    # hh_df               = pd.read_csv('hh_df.csv')
    # branch_data         = pd.read_csv('branch_data.csv')
    # rbrm_data           = pd.read_csv('rbrm_data.csv')   # Must have COVERAGE column
    # sbb_data            = pd.read_csv('sbb_data.csv')
    # portfolio_centroids = pd.read_csv('portfolio_centroids.csv')

    # Run reconstruction
    # updated_hh_df, sbb_assignments, portfolio_stats = run_portfolio_reconstruction(
    #     hh_df, branch_data, rbrm_data, sbb_data, portfolio_centroids
    # )

    # Save results
    # updated_hh_df.to_csv('updated_hh_df.csv', index=False)
    # sbb_assignments.to_csv('sbb_assignments.csv', index=False)

    print("\nScript loaded. Ready to run with your data.")
