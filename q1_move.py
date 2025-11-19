"""
Customer-Banker Assignment System with BallTree Spatial Indexing
New Methodology: Distance-based assignment with balancing
Enhanced Version: Includes existing customers and portfolio size tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ==================== DISTANCE CALCULATION ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth in miles"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    miles = 3959 * c
    return miles


def impute_missing_coordinates_knn(df, k=5):
    """
    Use K-Nearest Neighbors to impute missing LAT_NUM and LON_NUM
    based on BILLINGCITY, BILLINGSTATE, and BILLINGSTREET similarity
    """
    print("\nImputing missing coordinates using KNN...")
    
    # Separate rows with and without coordinates
    has_coords = df[df['LAT_NUM'].notna() & df['LON_NUM'].notna()].copy()
    missing_coords = df[df['LAT_NUM'].isna() | df['LON_NUM'].isna()].copy()
    
    if len(missing_coords) == 0:
        print("✓ No missing coordinates to impute")
        return df
    
    print(f"Found {len(missing_coords)} customers with missing coordinates")
    print(f"Using {len(has_coords)} customers with valid coordinates for KNN")
    
    # Fill missing address fields with empty string
    for col in ['BILLINGCITY', 'BILLINGSTATE', 'BILLINGSTREET']:
        if col in has_coords.columns:
            has_coords[col] = has_coords[col].fillna('UNKNOWN')
            missing_coords[col] = missing_coords[col].fillna('UNKNOWN')
    
    # Encode BILLINGCITY and BILLINGSTATE
    city_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    
    # Fit encoders on combined data to handle all categories
    all_cities = pd.concat([has_coords['BILLINGCITY'], missing_coords['BILLINGCITY']]).unique()
    all_states = pd.concat([has_coords['BILLINGSTATE'], missing_coords['BILLINGSTATE']]).unique()
    
    city_encoder.fit(all_cities)
    state_encoder.fit(all_states)
    
    # Transform training data (customers with coordinates)
    has_coords['CITY_ENCODED'] = city_encoder.transform(has_coords['BILLINGCITY'])
    has_coords['STATE_ENCODED'] = state_encoder.transform(has_coords['BILLINGSTATE'])
    
    # Transform test data (customers without coordinates)
    missing_coords['CITY_ENCODED'] = city_encoder.transform(missing_coords['BILLINGCITY'])
    missing_coords['STATE_ENCODED'] = state_encoder.transform(missing_coords['BILLINGSTATE'])
    
    # Prepare features for KNN (using city and state)
    X_train = has_coords[['CITY_ENCODED', 'STATE_ENCODED']].values
    y_train_lat = has_coords['LAT_NUM'].values
    y_train_lon = has_coords['LON_NUM'].values
    
    X_test = missing_coords[['CITY_ENCODED', 'STATE_ENCODED']].values
    
    # Build KNN model
    knn = NearestNeighbors(n_neighbors=min(k, len(has_coords)), metric='euclidean')
    knn.fit(X_train)
    
    # Find K nearest neighbors for each missing coordinate
    distances, indices = knn.kneighbors(X_test)
    
    # Impute coordinates as median of K nearest neighbors
    imputed_lats = []
    imputed_lons = []
    
    for neighbor_indices in indices:
        neighbor_lats = y_train_lat[neighbor_indices]
        neighbor_lons = y_train_lon[neighbor_indices]
        
        # Use median to be robust to outliers
        imputed_lat = np.median(neighbor_lats)
        imputed_lon = np.median(neighbor_lons)
        
        imputed_lats.append(imputed_lat)
        imputed_lons.append(imputed_lon)
    
    # Update missing coordinates
    missing_coords['LAT_NUM'] = imputed_lats
    missing_coords['LON_NUM'] = imputed_lons
    missing_coords['COORDS_IMPUTED'] = True
    
    # Mark non-imputed rows
    has_coords['COORDS_IMPUTED'] = False
    
    # Combine back
    result = pd.concat([has_coords, missing_coords], ignore_index=True)
    
    # Drop encoding columns
    result = result.drop(columns=['CITY_ENCODED', 'STATE_ENCODED'], errors='ignore')
    
    print(f"✓ Imputed coordinates for {len(missing_coords)} customers using {k}-NN")
    print(f"  Method: Median of {k} nearest neighbors based on BILLINGCITY + BILLINGSTATE")
    
    return result


def build_balltree(df, lat_col, lon_col):
    """Build BallTree for efficient spatial queries"""
    coords_rad = np.radians(df[[lat_col, lon_col]].values)
    tree = BallTree(coords_rad, metric='haversine')
    return tree


# ==================== DATA PREPARATION ====================

def load_and_prepare_data(banker_file, req_custs_file, available_custs_file):
    """Load and prepare all input data files"""
    print("Loading data files...")
    
    banker_data = pd.read_csv(banker_file)
    req_custs = pd.read_csv(req_custs_file)
    available_custs = pd.read_csv(available_custs_file)
    
    # Filter out null coordinates from bankers
    print(f"\nBankers before filtering: {len(banker_data)}")
    banker_data = banker_data.dropna(subset=['BANKER_LAT_NUM', 'BANKER_LON_NUM'])
    print(f"Bankers after filtering: {len(banker_data)}")
    
    # Impute missing coordinates for customers using KNN
    print(f"\nCustomers loaded: {len(available_custs)}")
    customers_with_missing = available_custs['LAT_NUM'].isna().sum()
    print(f"Customers with missing coordinates: {customers_with_missing}")
    
    if customers_with_missing > 0:
        available_custs = impute_missing_coordinates_knn(available_custs, k=5)
    else:
        available_custs['COORDS_IMPUTED'] = False
    
    # Now filter out any remaining nulls (if imputation wasn't possible)
    print(f"\nCustomers before final filtering: {len(available_custs)}")
    available_custs = available_custs.dropna(subset=['LAT_NUM', 'LON_NUM'])
    print(f"Customers after final filtering: {len(available_custs)}")
    
    bankers_df = banker_data.merge(req_custs, on='PORT_CODE', how='inner')
    
    bankers_df['CURRENT_ASSIGNED'] = 0
    bankers_df['REMAINING_MIN'] = bankers_df['MIN_COUNT_REQ']
    bankers_df['REMAINING_MAX'] = bankers_df['MAX_COUNT_REQ']
    
    available_custs['IS_ASSIGNED'] = False
    available_custs['ASSIGNED_TO_PORT_CODE'] = None
    available_custs['ASSIGNED_BANKER_EID'] = None
    available_custs['DISTANCE_MILES'] = None
    available_custs['ASSIGNMENT_PHASE'] = None
    available_custs['EXCEPTION_FLAG'] = None
    
    print(f"\nLoaded {len(bankers_df)} bankers and {len(available_custs)} available customers")
    
    return bankers_df, available_custs, banker_data, available_custs.copy()


def separate_bankers_by_type(bankers_df):
    """Separate bankers into IN MARKET and CENTRALIZED"""
    in_market = bankers_df[bankers_df['ROLE_TYPE'] == 'IN MARKET'].copy()
    centralized = bankers_df[bankers_df['ROLE_TYPE'] == 'CENTRALIZED'].copy()
    
    print(f"IN MARKET bankers: {len(in_market)}")
    print(f"CENTRALIZED bankers: {len(centralized)}")
    
    return in_market, centralized


# ==================== STEP 1: BUILD DISTANCE MAPPING ====================

def build_customer_banker_mapping(customers_df, bankers_df, banker_tree, max_radius):
    """Build mapping of customers to bankers within radius"""
    print(f"Building customer-banker mapping within {max_radius} miles...")
    
    customer_banker_map = {}
    
    for idx, customer in customers_df.iterrows():
        cust_id = customer['HH_ECN']
        cust_lat = customer['LAT_NUM']
        cust_lon = customer['LON_NUM']
        
        # Find all bankers within radius
        customer_rad = np.radians([[cust_lat, cust_lon]])
        radius_rad = max_radius / 3959.0
        
        indices, distances = banker_tree.query_radius(customer_rad, r=radius_rad, 
                                                       return_distance=True, sort_results=True)
        
        if len(indices[0]) > 0:
            distances_miles = distances[0] * 3959.0
            banker_indices = indices[0]
            
            # Store as list of (banker_port_code, distance) tuples sorted by distance
            bankers_in_range = []
            for bidx, dist in zip(banker_indices, distances_miles):
                banker_port = bankers_df.iloc[bidx]['PORT_CODE']
                bankers_in_range.append((banker_port, dist))
            
            customer_banker_map[cust_id] = bankers_in_range
    
    print(f"✓ Mapped {len(customer_banker_map)} customers to bankers")
    
    return customer_banker_map


# ==================== STEP 2: ASSIGN CUSTOMERS TO NEAREST BANKER ====================

def assign_customers_to_nearest(customers_df, bankers_df, customer_banker_map, phase_name):
    """Assign each customer to their nearest banker"""
    print(f"\n{'='*60}")
    print(f"{phase_name}: Assign Customers to Nearest Banker")
    print(f"{'='*60}")
    
    assignments = []
    
    for idx, customer in customers_df[customers_df['IS_ASSIGNED'] == False].iterrows():
        cust_id = customer['HH_ECN']
        
        if cust_id not in customer_banker_map:
            continue
        
        # Get bankers sorted by distance
        bankers_in_range = customer_banker_map[cust_id]
        
        # Assign to nearest banker
        for banker_port, distance in bankers_in_range:
            banker_idx = bankers_df[bankers_df['PORT_CODE'] == banker_port].index[0]
            banker_eid = bankers_df.at[banker_idx, 'EID']
            
            # Assign
            customers_df.at[idx, 'IS_ASSIGNED'] = True
            customers_df.at[idx, 'ASSIGNED_TO_PORT_CODE'] = banker_port
            customers_df.at[idx, 'ASSIGNED_BANKER_EID'] = banker_eid
            customers_df.at[idx, 'DISTANCE_MILES'] = distance
            customers_df.at[idx, 'ASSIGNMENT_PHASE'] = phase_name
            
            bankers_df.at[banker_idx, 'CURRENT_ASSIGNED'] += 1
            
            assignments.append({
                'HH_ECN': cust_id,
                'PORT_CODE': banker_port,
                'DISTANCE': distance
            })
            
            break  # Only assign to one banker
    
    print(f"✓ Assigned {len(assignments)} customers")
    
    return bankers_df, customers_df, assignments


# ==================== STEP 3: REMOVE EXCESS CUSTOMERS ====================

def remove_excess_customers(customers_df, bankers_df, phase_name):
    """Remove farthest customers from over-capacity bankers"""
    print(f"\n{'='*60}")
    print(f"{phase_name}: Remove Excess Customers (Keep MIN)")
    print(f"{'='*60}")
    
    removed_count = 0
    
    for idx, banker in bankers_df.iterrows():
        port_code = banker['PORT_CODE']
        min_req = banker['MIN_COUNT_REQ']
        current_assigned = banker['CURRENT_ASSIGNED']
        
        if current_assigned <= min_req:
            continue
        
        # Get customers assigned to this banker
        assigned_custs = customers_df[
            (customers_df['ASSIGNED_TO_PORT_CODE'] == port_code) &
            (customers_df['IS_ASSIGNED'] == True)
        ].copy()
        
        # Sort by distance (farthest first)
        assigned_custs = assigned_custs.sort_values('DISTANCE_MILES', ascending=False)
        
        # Remove excess customers
        excess = current_assigned - min_req
        customers_to_remove = assigned_custs.head(excess)
        
        for cust_idx, cust_row in customers_to_remove.iterrows():
            customers_df.at[cust_idx, 'IS_ASSIGNED'] = False
            customers_df.at[cust_idx, 'ASSIGNED_TO_PORT_CODE'] = None
            customers_df.at[cust_idx, 'ASSIGNED_BANKER_EID'] = None
            customers_df.at[cust_idx, 'DISTANCE_MILES'] = None
            customers_df.at[cust_idx, 'ASSIGNMENT_PHASE'] = None
            
            removed_count += 1
        
        bankers_df.at[idx, 'CURRENT_ASSIGNED'] = min_req
    
    print(f"✓ Removed {removed_count} customers from over-capacity bankers")
    
    return bankers_df, customers_df


# ==================== STEP 4-6: FILL UNDERSIZED PORTFOLIOS ====================

def fill_undersized_portfolios(customers_df, bankers_df, max_radius, phase_name, target='MIN'):
    """Fill undersized portfolios with nearest unassigned customers"""
    print(f"\n{'='*60}")
    print(f"{phase_name}: Fill Undersized to {target} ({max_radius} miles)")
    print(f"{'='*60}")
    
    # Find undersized bankers
    if target == 'MIN':
        undersized = bankers_df[bankers_df['CURRENT_ASSIGNED'] < bankers_df['MIN_COUNT_REQ']].copy()
    else:  # MAX
        undersized = bankers_df[bankers_df['CURRENT_ASSIGNED'] < bankers_df['MAX_COUNT_REQ']].copy()
    
    if len(undersized) == 0:
        print(f"✓ No undersized bankers")
        return bankers_df, customers_df, []
    
    print(f"Found {len(undersized)} undersized bankers")
    
    assignments = []
    
    # Get unassigned customers
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    
    if len(unassigned_custs) == 0:
        print("✗ No unassigned customers available")
        return bankers_df, customers_df, []
    
    # Build tree for unassigned customers
    cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
    
    for idx, banker in undersized.iterrows():
        port_code = banker['PORT_CODE']
        banker_lat = banker['BANKER_LAT_NUM']
        banker_lon = banker['BANKER_LON_NUM']
        banker_eid = banker['EID']
        banker_name = banker['EMPLOYEE_NAME']
        
        if target == 'MIN':
            needed = banker['MIN_COUNT_REQ'] - banker['CURRENT_ASSIGNED']
        else:
            needed = banker['MAX_COUNT_REQ'] - banker['CURRENT_ASSIGNED']
        
        if needed <= 0:
            continue
        
        print(f"  Banker: {banker_name} (Port: {port_code}) - Needs: {needed}")
        
        # Find customers within radius
        banker_rad = np.radians([[banker_lat, banker_lon]])
        radius_rad = max_radius / 3959.0
        
        unassigned_custs_current = customers_df[customers_df['IS_ASSIGNED'] == False]
        if len(unassigned_custs_current) == 0:
            continue
        
        cust_tree = build_balltree(unassigned_custs_current, 'LAT_NUM', 'LON_NUM')
        
        indices, distances = cust_tree.query_radius(banker_rad, r=radius_rad, 
                                                     return_distance=True, sort_results=True)
        
        if len(indices[0]) == 0:
            print(f"    ✗ No customers within {max_radius} miles")
            continue
        
        distances_miles = distances[0] * 3959.0
        cust_indices = indices[0]
        
        assigned_count = 0
        for cidx, dist in zip(cust_indices, distances_miles):
            if assigned_count >= needed:
                break
            
            actual_idx = unassigned_custs_current.iloc[cidx].name
            cust_id = customers_df.at[actual_idx, 'HH_ECN']
            
            if customers_df.at[actual_idx, 'IS_ASSIGNED']:
                continue
            
            # Assign
            customers_df.at[actual_idx, 'IS_ASSIGNED'] = True
            customers_df.at[actual_idx, 'ASSIGNED_TO_PORT_CODE'] = port_code
            customers_df.at[actual_idx, 'ASSIGNED_BANKER_EID'] = banker_eid
            customers_df.at[actual_idx, 'DISTANCE_MILES'] = dist
            customers_df.at[actual_idx, 'ASSIGNMENT_PHASE'] = phase_name
            
            if max_radius > 40 and max_radius <= 60:
                customers_df.at[actual_idx, 'EXCEPTION_FLAG'] = f'EXPANDED_RADIUS_TO_{max_radius}MILES'
            elif max_radius > 200:
                customers_df.at[actual_idx, 'EXCEPTION_FLAG'] = f'EXPANDED_RADIUS_TO_{max_radius}MILES'
            
            bankers_df.at[idx, 'CURRENT_ASSIGNED'] += 1
            
            assignments.append({
                'HH_ECN': cust_id,
                'PORT_CODE': port_code,
                'DISTANCE': dist
            })
            
            assigned_count += 1
        
        print(f"    ✓ Assigned {assigned_count} customers")
    
    print(f"✓ Total assignments: {len(assignments)}")
    
    return bankers_df, customers_df, assignments


# ==================== STEP 5: ASSIGN TO NEAREST (NO EXCEED MAX) ====================

def assign_remaining_to_nearest_no_exceed_max(customers_df, bankers_df, customer_banker_map, phase_name):
    """Assign remaining unassigned customers to nearest banker without exceeding MAX"""
    print(f"\n{'='*60}")
    print(f"{phase_name}: Assign Remaining to Nearest (No Exceed MAX)")
    print(f"{'='*60}")
    
    assignments = []
    
    for idx, customer in customers_df[customers_df['IS_ASSIGNED'] == False].iterrows():
        cust_id = customer['HH_ECN']
        
        if cust_id not in customer_banker_map:
            continue
        
        bankers_in_range = customer_banker_map[cust_id]
        
        for banker_port, distance in bankers_in_range:
            banker_idx = bankers_df[bankers_df['PORT_CODE'] == banker_port].index[0]
            
            # Check if banker can take more (not at MAX)
            current = bankers_df.at[banker_idx, 'CURRENT_ASSIGNED']
            max_allowed = bankers_df.at[banker_idx, 'CURR_COUNT'] + bankers_df.at[banker_idx, 'MAX_COUNT_REQ']
            total = bankers_df.at[banker_idx, 'CURR_COUNT'] + current
            
            if total >= max_allowed:
                continue
            
            banker_eid = bankers_df.at[banker_idx, 'EID']
            
            # Assign
            customers_df.at[idx, 'IS_ASSIGNED'] = True
            customers_df.at[idx, 'ASSIGNED_TO_PORT_CODE'] = banker_port
            customers_df.at[idx, 'ASSIGNED_BANKER_EID'] = banker_eid
            customers_df.at[idx, 'DISTANCE_MILES'] = distance
            customers_df.at[idx, 'ASSIGNMENT_PHASE'] = phase_name
            
            bankers_df.at[banker_idx, 'CURRENT_ASSIGNED'] += 1
            
            assignments.append({
                'HH_ECN': cust_id,
                'PORT_CODE': banker_port,
                'DISTANCE': distance
            })
            
            break
    
    print(f"✓ Assigned {len(assignments)} customers")
    
    return bankers_df, customers_df, assignments


# ==================== OUTPUT GENERATION ====================

def generate_customer_assignment_file(customers_df, bankers_df_full, bankers_df_working, 
                                     output_path, existing_custs_file='EXISTING_CUSTS.csv'):
    """Generate the main customer assignment output file with existing and new customers"""
    print("\nGenerating customer assignment file...")
    
    # ==================== PROCESS NEW ASSIGNMENTS ====================
    assigned = customers_df[customers_df['IS_ASSIGNED'] == True].copy()
    
    # Columns from original banker data (without CURR_COUNT)
    banker_cols = ['PORT_CODE', 'EID', 'EMPLOYEE_NAME', 'AU', 'BRANCH_NAME', 
                   'ROLE_TYPE', 'BANKER_LAT_NUM', 'BANKER_LON_NUM', 
                   'MANAGER_NAME', 'DIRECTOR_NAME', 'COVERAGE', 
                   'BRANCH_LOCATION_CODE']
    
    # Merge with full banker info
    result = assigned.merge(
        bankers_df_full[banker_cols],
        left_on='ASSIGNED_TO_PORT_CODE',
        right_on='PORT_CODE',
        how='left'
    )
    
    # Merge CURR_COUNT from working dataframe
    result = result.merge(
        bankers_df_working[['PORT_CODE', 'CURR_COUNT']],
        left_on='ASSIGNED_TO_PORT_CODE',
        right_on='PORT_CODE',
        how='left',
        suffixes=('', '_curr')
    )
    result = result.drop(columns=['PORT_CODE_curr'], errors='ignore')
    
    result['PROXIMITY_LIMIT_MILES'] = result['ROLE_TYPE'].apply(
        lambda x: 40 if x == 'IN MARKET' else 200
    )
    result['IS_WITHIN_PROXIMITY'] = result['DISTANCE_MILES'] <= result['PROXIMITY_LIMIT_MILES']
    result['ASSIGNMENT_TIMESTAMP'] = datetime.now()
    result['CUSTOMER_TYPE'] = 'New'
    
    # ==================== LOAD AND PROCESS EXISTING CUSTOMERS ====================
    print("Loading existing customers...")
    try:
        existing_custs = pd.read_csv(existing_custs_file)
        print(f"✓ Loaded {len(existing_custs)} existing customers")
        
        # Filter out null coordinates
        existing_custs = existing_custs.dropna(subset=['LAT_NUM', 'LON_NUM'])
        print(f"✓ After filtering nulls: {len(existing_custs)} existing customers")
        
        # Rename CG_ECN to HH_ECN for consistency
        if 'CG_ECN' in existing_custs.columns:
            existing_custs = existing_custs.rename(columns={'CG_ECN': 'HH_ECN'})
        
        # Merge with banker information
        existing_result = existing_custs.merge(
            bankers_df_full[banker_cols],
            on='PORT_CODE',
            how='left'
        )
        
        # Merge CURR_COUNT from working dataframe
        existing_result = existing_result.merge(
            bankers_df_working[['PORT_CODE', 'CURR_COUNT']],
            on='PORT_CODE',
            how='left'
        )
        
        # Calculate distance for existing customers
        print("Calculating distances for existing customers...")
        existing_result['DISTANCE_MILES'] = existing_result.apply(
            lambda row: haversine_distance(
                row['LAT_NUM'], row['LON_NUM'],
                row['BANKER_LAT_NUM'], row['BANKER_LON_NUM']
            ) if pd.notna(row['BANKER_LAT_NUM']) and pd.notna(row['BANKER_LON_NUM']) else None,
            axis=1
        )
        
        # Add required columns for existing customers
        existing_result['PROXIMITY_LIMIT_MILES'] = existing_result['ROLE_TYPE'].apply(
            lambda x: 40 if x == 'IN MARKET' else 200
        )
        existing_result['IS_WITHIN_PROXIMITY'] = existing_result['DISTANCE_MILES'] <= existing_result['PROXIMITY_LIMIT_MILES']
        existing_result['ASSIGNMENT_TIMESTAMP'] = datetime.now()
        existing_result['ASSIGNMENT_PHASE'] = 'EXISTING_CUSTOMER'
        existing_result['EXCEPTION_FLAG'] = None
        existing_result['CUSTOMER_TYPE'] = 'Existing'
        existing_result['ASSIGNED_TO_PORT_CODE'] = existing_result['PORT_CODE']
        existing_result['COORDS_IMPUTED'] = False  # Existing customers have real coordinates
        
        # Fill missing columns with None/0 if they don't exist
        for col in ['NEW_SEGMENT', 'DEPOSIT_BAL', 'CG_GROSS_SALES', 'BANK_REVENUE',
                    'BILLINGCITY', 'BILLINGSTATE', 'BILLINGPOSTALCODE']:
            if col not in existing_result.columns:
                existing_result[col] = None
        
        print(f"✓ Processed {len(existing_result)} existing customers")
        
    except FileNotFoundError:
        print(f"⚠ Warning: {existing_custs_file} not found. Skipping existing customers.")
        existing_result = pd.DataFrame()
    except Exception as e:
        print(f"⚠ Warning: Error loading existing customers: {str(e)}")
        existing_result = pd.DataFrame()
    
    # ==================== CALCULATE size_reach AT BANKER LEVEL ====================
    print("Calculating size_reach for bankers...")
    
    # Calculate size_reach for each banker
    banker_size_reach = []
    for _, banker in bankers_df_working.iterrows():
        port_code = banker['PORT_CODE']
        current_assigned = banker['CURRENT_ASSIGNED']
        curr_count = banker['CURR_COUNT']
        min_req = banker['MIN_COUNT_REQ']
        
        total = current_assigned + curr_count
        target_min = curr_count + min_req
        
        # size_reach: 0 if undersized, 1 if reached minimum or above
        size_reach = 1 if total >= target_min else 0
        
        banker_size_reach.append({
            'PORT_CODE': port_code,
            'size_reach': size_reach
        })
    
    size_reach_df = pd.DataFrame(banker_size_reach)
    
    # Merge size_reach into new assignments
    result = result.merge(size_reach_df, left_on='ASSIGNED_TO_PORT_CODE', right_on='PORT_CODE', 
                          how='left', suffixes=('', '_size'))
    result = result.drop(columns=['PORT_CODE_size'], errors='ignore')
    
    # Merge size_reach into existing assignments
    if len(existing_result) > 0:
        existing_result = existing_result.merge(size_reach_df, on='PORT_CODE', how='left')
    
    # ==================== COMBINE NEW AND EXISTING ====================
    if len(existing_result) > 0:
        combined_result = pd.concat([result, existing_result], ignore_index=True)
        print(f"✓ Combined {len(result)} new + {len(existing_result)} existing = {len(combined_result)} total customers")
    else:
        combined_result = result
        print(f"✓ Total: {len(combined_result)} customers (new only)")
    
    # ==================== STANDARDIZE OUTPUT COLUMNS ====================
    output_cols = {
        'HH_ECN': 'HH_ECN',
        'ASSIGNED_TO_PORT_CODE': 'ASSIGNED_PORT_CODE',
        'EID': 'ASSIGNED_BANKER_EID',
        'EMPLOYEE_NAME': 'ASSIGNED_BANKER_NAME',
        'AU': 'ASSIGNED_BANKER_AU',
        'BRANCH_NAME': 'ASSIGNED_BANKER_BRANCH_NAME',
        'DISTANCE_MILES': 'DISTANCE_MILES',
        'LAT_NUM': 'CUSTOMER_LAT',
        'LON_NUM': 'CUSTOMER_LON',
        'BANKER_LAT_NUM': 'BANKER_LAT',
        'BANKER_LON_NUM': 'BANKER_LON',
        'ROLE_TYPE': 'BANKER_ROLE_TYPE',
        'PROXIMITY_LIMIT_MILES': 'PROXIMITY_LIMIT_MILES',
        'ASSIGNMENT_PHASE': 'ASSIGNMENT_PHASE',
        'IS_WITHIN_PROXIMITY': 'IS_WITHIN_PROXIMITY',
        'EXCEPTION_FLAG': 'EXCEPTION_FLAG',
        'COORDS_IMPUTED': 'COORDS_IMPUTED',
        'NEW_SEGMENT': 'NEW_SEGMENT',
        'DEPOSIT_BAL': 'DEPOSIT_BAL',
        'CG_GROSS_SALES': 'CG_GROSS_SALES',
        'BANK_REVENUE': 'BANK_REVENUE',
        'BILLINGCITY': 'BILLINGCITY',
        'BILLINGSTATE': 'BILLINGSTATE',
        'BILLINGPOSTALCODE': 'BILLINGPOSTALCODE',
        'MANAGER_NAME': 'BANKER_MANAGER_NAME',
        'DIRECTOR_NAME': 'BANKER_DIRECTOR_NAME',
        'COVERAGE': 'BANKER_COVERAGE',
        'BRANCH_LOCATION_CODE': 'BANKER_BRANCH_LOCATION_CODE',
        'ASSIGNMENT_TIMESTAMP': 'ASSIGNMENT_TIMESTAMP',
        'CUSTOMER_TYPE': 'CUSTOMER_TYPE',
        'size_reach': 'size_reach'
    }
    
    output_df = combined_result.rename(columns=output_cols)
    output_df = output_df[[col for col in output_cols.values() if col in output_df.columns]]
    
    output_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(output_df)} records)")
    
    return output_df


def generate_banker_summary_file(bankers_df_full, customers_df, output_path):
    """Generate banker summary statistics file"""
    print("\nGenerating banker summary file...")
    
    summary_data = []
    
    for _, banker in bankers_df_full.iterrows():
        port_code = banker['PORT_CODE']
        
        assigned_custs = customers_df[
            (customers_df['ASSIGNED_TO_PORT_CODE'] == port_code) & 
            (customers_df['IS_ASSIGNED'] == True)
        ]
        
        newly_assigned = len(assigned_custs)
        final_total = banker['CURR_COUNT'] + newly_assigned
        
        if newly_assigned > 0:
            distances = assigned_custs['DISTANCE_MILES'].dropna()
            avg_distance = distances.mean()
            min_distance = distances.min()
            max_distance = distances.max()
            median_distance = distances.median()
            std_distance = distances.std()
            
            total_deposits = assigned_custs['DEPOSIT_BAL'].sum()
            total_revenue = assigned_custs['BANK_REVENUE'].sum()
            total_sales = assigned_custs['CG_GROSS_SALES'].sum()
            
            num_cities = assigned_custs['BILLINGCITY'].nunique()
            num_states = assigned_custs['BILLINGSTATE'].nunique()
            num_zips = assigned_custs['BILLINGPOSTALCODE'].nunique()
        else:
            avg_distance = min_distance = max_distance = median_distance = std_distance = 0
            total_deposits = total_revenue = total_sales = 0
            num_cities = num_states = num_zips = 0
        
        summary_data.append({
            'PORT_CODE': port_code,
            'EID': banker['EID'],
            'EMPLOYEE_NAME': banker['EMPLOYEE_NAME'],
            'AU': banker['AU'],
            'BRANCH_NAME': banker['BRANCH_NAME'],
            'ROLE_TYPE': banker['ROLE_TYPE'],
            'MANAGER_NAME': banker['MANAGER_NAME'],
            'DIRECTOR_NAME': banker['DIRECTOR_NAME'],
            'ISACTIVE': banker['ISACTIVE'],
            'CURR_COUNT': banker['CURR_COUNT'],
            'MIN_COUNT_REQ': banker['MIN_COUNT_REQ'],
            'MAX_COUNT_REQ': banker['MAX_COUNT_REQ'],
            'TARGET_MIN_TOTAL': banker['CURR_COUNT'] + banker['MIN_COUNT_REQ'],
            'TARGET_MAX_TOTAL': banker['CURR_COUNT'] + banker['MAX_COUNT_REQ'],
            'NEWLY_ASSIGNED_COUNT': newly_assigned,
            'FINAL_TOTAL_COUNT': final_total,
            'MIN_MET': newly_assigned >= banker['MIN_COUNT_REQ'],
            'AT_CAPACITY': newly_assigned >= banker['MAX_COUNT_REQ'],
            'REMAINING_CAPACITY': banker['MAX_COUNT_REQ'] - newly_assigned,
            'AVG_DISTANCE_MILES': round(avg_distance, 2),
            'MIN_DISTANCE_MILES': round(min_distance, 2),
            'MAX_DISTANCE_MILES': round(max_distance, 2),
            'MEDIAN_DISTANCE_MILES': round(median_distance, 2),
            'STD_DISTANCE_MILES': round(std_distance, 2),
            'TOTAL_NEW_DEPOSITS': round(total_deposits, 2),
            'TOTAL_NEW_REVENUE': round(total_revenue, 2),
            'TOTAL_NEW_GROSS_SALES': round(total_sales, 2),
            'AVG_DEPOSIT_PER_CUSTOMER': round(total_deposits / newly_assigned, 2) if newly_assigned > 0 else 0,
            'AVG_REVENUE_PER_CUSTOMER': round(total_revenue / newly_assigned, 2) if newly_assigned > 0 else 0,
            'NUM_UNIQUE_CITIES': num_cities,
            'NUM_UNIQUE_STATES': num_states,
            'NUM_UNIQUE_ZIPCODES': num_zips
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(summary_df)} records)")
    
    return summary_df


def generate_unassigned_customers_file(customers_df, bankers_df_full, output_path):
    """Generate file with customers that were not assigned"""
    print("\nGenerating unassigned customers file...")
    
    unassigned = customers_df[customers_df['IS_ASSIGNED'] == False].copy()
    
    if len(unassigned) == 0:
        print("✓ All customers assigned successfully!")
        return None
    
    unassigned_data = []
    
    for _, customer in unassigned.iterrows():
        cust_lat = customer['LAT_NUM']
        cust_lon = customer['LON_NUM']
        
        banker_distances = []
        for _, banker in bankers_df_full.iterrows():
            distance = haversine_distance(
                cust_lat, cust_lon,
                banker['BANKER_LAT_NUM'], banker['BANKER_LON_NUM']
            )
            banker_distances.append({
                'PORT_CODE': banker['PORT_CODE'],
                'EMPLOYEE_NAME': banker['EMPLOYEE_NAME'],
                'ROLE_TYPE': banker['ROLE_TYPE'],
                'DISTANCE': distance
            })
        
        banker_distances.sort(key=lambda x: x['DISTANCE'])
        
        closest = banker_distances[0] if banker_distances else None
        second_closest = banker_distances[1] if len(banker_distances) > 1 else None
        
        unassigned_data.append({
            'HH_ECN': customer['HH_ECN'],
            'NEW_SEGMENT': customer['NEW_SEGMENT'],
            'BILLINGCITY': customer['BILLINGCITY'],
            'BILLINGSTATE': customer['BILLINGSTATE'],
            'BILLINGPOSTALCODE': customer['BILLINGPOSTALCODE'],
            'LAT_NUM': customer['LAT_NUM'],
            'LON_NUM': customer['LON_NUM'],
            'COORDS_IMPUTED': customer.get('COORDS_IMPUTED', False),
            'DEPOSIT_BAL': customer['DEPOSIT_BAL'],
            'CG_GROSS_SALES': customer['CG_GROSS_SALES'],
            'BANK_REVENUE': customer['BANK_REVENUE'],
            'CLOSEST_BANKER_PORT_CODE': closest['PORT_CODE'] if closest else None,
            'CLOSEST_BANKER_NAME': closest['EMPLOYEE_NAME'] if closest else None,
            'CLOSEST_BANKER_ROLE_TYPE': closest['ROLE_TYPE'] if closest else None,
            'DISTANCE_TO_CLOSEST_BANKER_MILES': round(closest['DISTANCE'], 2) if closest else None,
            'SECOND_CLOSEST_BANKER_PORT_CODE': second_closest['PORT_CODE'] if second_closest else None,
            'DISTANCE_TO_SECOND_CLOSEST_MILES': round(second_closest['DISTANCE'], 2) if second_closest else None
        })
    
    unassigned_df = pd.DataFrame(unassigned_data)
    unassigned_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(unassigned_df)} records)")
    
    return unassigned_df


def generate_overall_summary(bankers_df_full, customers_df, start_time, output_path):
    """Generate overall summary statistics"""
    print("\nGenerating overall summary...")
    
    assigned = customers_df[customers_df['IS_ASSIGNED'] == True]
    unassigned = customers_df[customers_df['IS_ASSIGNED'] == False]
    
    in_market_bankers = bankers_df_full[bankers_df_full['ROLE_TYPE'] == 'IN MARKET']
    centralized_bankers = bankers_df_full[bankers_df_full['ROLE_TYPE'] == 'CENTRALIZED']
    
    total_available = len(customers_df)
    total_assigned = len(assigned)
    total_unassigned = len(unassigned)
    
    if total_assigned > 0:
        overall_avg_dist = assigned['DISTANCE_MILES'].mean()
        min_dist = assigned['DISTANCE_MILES'].min()
        max_dist = assigned['DISTANCE_MILES'].max()
        total_deposits = assigned['DEPOSIT_BAL'].sum()
        total_revenue = assigned['BANK_REVENUE'].sum()
    else:
        overall_avg_dist = min_dist = max_dist = 0
        total_deposits = total_revenue = 0
    
    bankers_min_met = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] >= bankers_df_full['MIN_COUNT_REQ']])
    bankers_failed_min = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] < bankers_df_full['MIN_COUNT_REQ']])
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    summary_data = {
        'TOTAL_AVAILABLE_CUSTOMERS': total_available,
        'TOTAL_ASSIGNED_CUSTOMERS': total_assigned,
        'TOTAL_UNASSIGNED_CUSTOMERS': total_unassigned,
        'ASSIGNMENT_SUCCESS_RATE_PCT': round((total_assigned / total_available * 100), 2),
        'TOTAL_BANKERS': len(bankers_df_full),
        'IN_MARKET_BANKERS_COUNT': len(in_market_bankers),
        'CENTRALIZED_BANKERS_COUNT': len(centralized_bankers),
        'BANKERS_MET_MIN': bankers_min_met,
        'BANKERS_FAILED_MIN': bankers_failed_min,
        'OVERALL_AVG_DISTANCE_MILES': round(overall_avg_dist, 2),
        'MIN_ASSIGNMENT_DISTANCE_MILES': round(min_dist, 2),
        'MAX_ASSIGNMENT_DISTANCE_MILES': round(max_dist, 2),
        'TOTAL_ASSIGNED_DEPOSITS': round(total_deposits, 2),
        'TOTAL_ASSIGNED_REVENUE': round(total_revenue, 2),
        'ASSIGNMENT_RUN_DATE': datetime.now(),
        'TOTAL_EXECUTION_TIME_SECONDS': round(execution_time, 2)
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    return summary_df


# ==================== MAIN ORCHESTRATOR ====================

def run_customer_banker_assignment(banker_file, req_custs_file, available_custs_file, 
                                   existing_custs_file='EXISTING_CUSTS.csv',
                                   output_dir='output'):
    """Main function to run the entire customer-banker assignment process"""
    import os
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("CUSTOMER-BANKER ASSIGNMENT SYSTEM")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    bankers_df, customers_df, banker_data_orig, customers_orig = load_and_prepare_data(
        banker_file, req_custs_file, available_custs_file
    )
    
    in_market_bankers, centralized_bankers = separate_bankers_by_type(bankers_df)
    
    # ==================== IN MARKET ASSIGNMENTS ====================
    
    print("\n" + "="*80)
    print("IN MARKET ASSIGNMENTS")
    print("="*80)
    
    if len(in_market_bankers) > 0:
        # Step 1: Build distance mapping (40 miles)
        print("\nStep 1: Building distance mapping (40 miles)...")
        im_tree = build_balltree(in_market_bankers, 'BANKER_LAT_NUM', 'BANKER_LON_NUM')
        customer_banker_map = build_customer_banker_mapping(customers_df, in_market_bankers, im_tree, 40)
        
        # Step 2: Assign to nearest banker
        in_market_bankers, customers_df, step2 = assign_customers_to_nearest(
            customers_df, in_market_bankers, customer_banker_map, 'IM_STEP2_NEAREST_40MI'
        )
        
        # Step 3: Remove excess (keep MIN)
        in_market_bankers, customers_df = remove_excess_customers(
            customers_df, in_market_bankers, 'IM_STEP3_REMOVE_EXCESS'
        )
        
        # Step 4: Fill undersized (40 miles)
        in_market_bankers, customers_df, step4 = fill_undersized_portfolios(
            customers_df, in_market_bankers, 40, 'IM_STEP4_FILL_40MI', 'MIN'
        )
        
        # Step 5: Assign remaining to nearest (no exceed MAX)
        in_market_bankers, customers_df, step5 = assign_remaining_to_nearest_no_exceed_max(
            customers_df, in_market_bankers, customer_banker_map, 'IM_STEP5_REMAINING'
        )
        
        # Step 6: Fill undersized (60 miles)
        in_market_bankers, customers_df, step6 = fill_undersized_portfolios(
            customers_df, in_market_bankers, 60, 'IM_STEP6_FILL_60MI', 'MIN'
        )
    else:
        print("\n⚠ No IN MARKET bankers found, skipping IN MARKET assignments")
        step2 = step4 = step5 = step6 = []
    
    # ==================== CENTRALIZED ASSIGNMENTS ====================
    
    print("\n" + "="*80)
    print("CENTRALIZED ASSIGNMENTS")
    print("="*80)
    
    if len(centralized_bankers) > 0:
        # Step 7: Build mapping and assign to nearest (200 miles, MIN only)
        print("\nStep 7: Assigning to nearest CENTRALIZED banker (200 miles, MIN only)...")
        cent_tree = build_balltree(centralized_bankers, 'BANKER_LAT_NUM', 'BANKER_LON_NUM')
        
        unassigned = customers_df[customers_df['IS_ASSIGNED'] == False]
        step7 = []
        
        for idx, customer in unassigned.iterrows():
            cust_lat = customer['LAT_NUM']
            cust_lon = customer['LON_NUM']
            cust_id = customer['HH_ECN']
            
            customer_rad = np.radians([[cust_lat, cust_lon]])
            radius_rad = 200 / 3959.0
            
            indices, distances = cent_tree.query_radius(customer_rad, r=radius_rad, 
                                                         return_distance=True, sort_results=True)
            
            if len(indices[0]) == 0:
                continue
            
            distances_miles = distances[0] * 3959.0
            banker_indices = indices[0]
            
            for bidx, dist in zip(banker_indices, distances_miles):
                banker_idx = centralized_bankers.iloc[bidx].name
                port_code = centralized_bankers.at[banker_idx, 'PORT_CODE']
                
                # Check if banker still needs customers (below MIN)
                current = centralized_bankers.at[banker_idx, 'CURRENT_ASSIGNED']
                min_req = centralized_bankers.at[banker_idx, 'MIN_COUNT_REQ']
                
                if current >= min_req:
                    continue
                
                banker_eid = centralized_bankers.at[banker_idx, 'EID']
                
                customers_df.at[idx, 'IS_ASSIGNED'] = True
                customers_df.at[idx, 'ASSIGNED_TO_PORT_CODE'] = port_code
                customers_df.at[idx, 'ASSIGNED_BANKER_EID'] = banker_eid
                customers_df.at[idx, 'DISTANCE_MILES'] = dist
                customers_df.at[idx, 'ASSIGNMENT_PHASE'] = 'CENT_STEP7_NEAREST_200MI'
                
                centralized_bankers.at[banker_idx, 'CURRENT_ASSIGNED'] += 1
                
                step7.append({
                    'HH_ECN': cust_id,
                    'PORT_CODE': port_code,
                    'DISTANCE': dist
                })
                
                break
        
        print(f"✓ Assigned {len(step7)} customers")
        
        # Step 8: Fill undersized (400 miles, MIN)
        centralized_bankers, customers_df, step8 = fill_undersized_portfolios(
            customers_df, centralized_bankers, 400, 'CENT_STEP8_FILL_400MI', 'MIN'
        )
        
        # Step 9: Fill undersized (600 miles, MIN)
        centralized_bankers, customers_df, step9 = fill_undersized_portfolios(
            customers_df, centralized_bankers, 600, 'CENT_STEP9_FILL_600MI', 'MIN'
        )
        
        # Step 10: Fill to MAX (200 miles)
        centralized_bankers, customers_df, step10 = fill_undersized_portfolios(
            customers_df, centralized_bankers, 200, 'CENT_STEP10_FILL_MAX_200MI', 'MAX'
        )
        
        # Step 11: Assign remaining unassigned to nearest centralized (no distance limit, up to MAX)
        print(f"\n{'='*60}")
        print("CENT_STEP11: Assign Remaining to Nearest CENTRALIZED (No Limit, Up to MAX)")
        print(f"{'='*60}")
        
        unassigned = customers_df[customers_df['IS_ASSIGNED'] == False]
        step11 = []
        
        if len(unassigned) > 0:
            print(f"Found {len(unassigned)} unassigned customers")
            
            for idx, customer in unassigned.iterrows():
                cust_lat = customer['LAT_NUM']
                cust_lon = customer['LON_NUM']
                cust_id = customer['HH_ECN']
                
                # Find nearest centralized banker regardless of distance
                min_distance = float('inf')
                nearest_banker_idx = None
                nearest_port_code = None
                
                for banker_idx, banker in centralized_bankers.iterrows():
                    # Check if banker can take more (not at MAX)
                    current = banker['CURRENT_ASSIGNED']
                    curr_count = banker['CURR_COUNT']
                    max_allowed = banker['MAX_COUNT_REQ']
                    total = curr_count + current
                    target_max = curr_count + max_allowed
                    
                    if total >= target_max:
                        continue
                    
                    # Calculate distance
                    banker_lat = banker['BANKER_LAT_NUM']
                    banker_lon = banker['BANKER_LON_NUM']
                    distance = haversine_distance(cust_lat, cust_lon, banker_lat, banker_lon)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_banker_idx = banker_idx
                        nearest_port_code = banker['PORT_CODE']
                
                # Assign to nearest banker if found
                if nearest_banker_idx is not None:
                    banker_eid = centralized_bankers.at[nearest_banker_idx, 'EID']
                    
                    customers_df.at[idx, 'IS_ASSIGNED'] = True
                    customers_df.at[idx, 'ASSIGNED_TO_PORT_CODE'] = nearest_port_code
                    customers_df.at[idx, 'ASSIGNED_BANKER_EID'] = banker_eid
                    customers_df.at[idx, 'DISTANCE_MILES'] = min_distance
                    customers_df.at[idx, 'ASSIGNMENT_PHASE'] = 'CENT_STEP11_NEAREST_NO_LIMIT'
                    
                    if min_distance > 600:
                        customers_df.at[idx, 'EXCEPTION_FLAG'] = f'DISTANCE_EXCEEDS_600MI_{int(min_distance)}MI'
                    
                    centralized_bankers.at[nearest_banker_idx, 'CURRENT_ASSIGNED'] += 1
                    
                    step11.append({
                        'HH_ECN': cust_id,
                        'PORT_CODE': nearest_port_code,
                        'DISTANCE': min_distance
                    })
            
            print(f"✓ Assigned {len(step11)} customers to nearest centralized bankers")
        else:
            print("✓ No unassigned customers remaining")
    else:
        print("\n⚠ No CENTRALIZED bankers found, skipping CENTRALIZED assignments")
        step7 = step8 = step9 = step10 = step11 = []
    
    # ==================== GENERATE OUTPUTS ====================
    
    all_bankers = pd.concat([in_market_bankers, centralized_bankers], ignore_index=True)
    
    print("\n" + "="*60)
    print("Generating Output Files")
    print("="*60)
    
    output_files = {}
    
    output_files['customer_assignment'] = os.path.join(output_dir, 'customer_assignment.csv')
    generate_customer_assignment_file(customers_df, banker_data_orig, all_bankers,
                                     output_files['customer_assignment'], existing_custs_file)
    
    output_files['banker_summary'] = os.path.join(output_dir, 'banker_summary.csv')
    generate_banker_summary_file(all_bankers, customers_df, output_files['banker_summary'])
    
    output_files['unassigned_customers'] = os.path.join(output_dir, 'unassigned_customers.csv')
    generate_unassigned_customers_file(customers_df, banker_data_orig, output_files['unassigned_customers'])
    
    output_files['overall_summary'] = os.path.join(output_dir, 'overall_summary.csv')
    generate_overall_summary(all_bankers, customers_df, start_time, output_files['overall_summary'])
    
    print("\n" + "="*80)
    print("ASSIGNMENT COMPLETE")
    print("="*80)
    
    assigned_count = len(customers_df[customers_df['IS_ASSIGNED'] == True])
    unassigned_count = len(customers_df[customers_df['IS_ASSIGNED'] == False])
    
    print(f"\n✓ Total Customers Assigned: {assigned_count}")
    print(f"✓ Total Customers Unassigned: {unassigned_count}")
    print(f"✓ Assignment Rate: {(assigned_count/len(customers_df)*100):.2f}%")
    
    print(f"\n--- IN MARKET Assignments ---")
    print(f"✓ Step 2 - Nearest (40 mi): {len(step2)}")
    print(f"✓ Step 4 - Fill MIN (40 mi): {len(step4)}")
    print(f"✓ Step 5 - Remaining (no exceed MAX): {len(step5)}")
    print(f"✓ Step 6 - Fill MIN (60 mi): {len(step6)}")
    print(f"  Total IN MARKET: {len(step2) + len(step4) + len(step5) + len(step6)}")
    
    print(f"\n--- CENTRALIZED Assignments ---")
    print(f"✓ Step 7 - Nearest (200 mi, MIN only): {len(step7)}")
    print(f"✓ Step 8 - Fill MIN (400 mi): {len(step8)}")
    print(f"✓ Step 9 - Fill MIN (600 mi): {len(step9)}")
    print(f"✓ Step 10 - Fill MAX (200 mi): {len(step10)}")
    print(f"✓ Step 11 - Nearest (no limit, up to MAX): {len(step11)}")
    print(f"  Total CENTRALIZED: {len(step7) + len(step8) + len(step9) + len(step10) + len(step11)}")
    
    print(f"\n✓ Execution Time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    print(f"\nOutput files saved to: {output_dir}/")
    
    return output_files


if __name__ == "__main__":
    BANKER_FILE = 'banker_data.csv'
    REQ_CUSTS_FILE = 'req_custs.csv'
    AVAILABLE_CUSTS_FILE = 'available_custs.csv'
    EXISTING_CUSTS_FILE = 'EXISTING_CUSTS.csv'
    
    output_files = run_customer_banker_assignment(
        banker_file=BANKER_FILE,
        req_custs_file=REQ_CUSTS_FILE,
        available_custs_file=AVAILABLE_CUSTS_FILE,
        existing_custs_file=EXISTING_CUSTS_FILE,
        output_dir='output'
    )
    
    print("\n" + "="*80)
    print("Output Files Generated:")
    print("="*80)
    for name, path in output_files.items():
        print(f"  • {name}: {path}")
