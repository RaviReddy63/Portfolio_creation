"""
Household Portfolio Migration System
=====================================
Migrates from Client Group-based portfolios to Household-based portfolios
with banker retention logic and SBB assignments.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ==================== UTILITY FUNCTIONS ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth in miles"""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    miles = 3959 * c
    return miles


def impute_missing_coordinates_knn(df, lat_col='LAT_NUM', lon_col='LON_NUM', 
                                    city_col='BILLINGCITY', state_col='BILLINGSTATE',
                                    k=5, entity_name='records'):
    """
    Use K-Nearest Neighbors to impute missing LAT_NUM and LON_NUM
    based on city and state similarity
    
    Parameters:
    -----------
    df : DataFrame - Input data with coordinates and address fields
    lat_col : str - Name of latitude column
    lon_col : str - Name of longitude column
    city_col : str - Name of city column (for KNN features)
    state_col : str - Name of state column (for KNN features)
    k : int - Number of neighbors to use
    entity_name : str - Name for logging purposes
    
    Returns:
    --------
    DataFrame with imputed coordinates and COORDS_IMPUTED flag
    """
    print(f"\n  Imputing missing coordinates for {entity_name} using KNN...")
    
    df = df.copy()
    
    # Check if required columns exist
    if city_col not in df.columns or state_col not in df.columns:
        print(f"    ⚠ Missing {city_col} or {state_col} columns, skipping imputation")
        df['COORDS_IMPUTED'] = False
        return df
    
    # Separate rows with and without coordinates
    has_coords = df[df[lat_col].notna() & df[lon_col].notna()].copy()
    missing_coords = df[df[lat_col].isna() | df[lon_col].isna()].copy()
    
    if len(missing_coords) == 0:
        print(f"    ✓ No missing coordinates to impute")
        df['COORDS_IMPUTED'] = False
        return df
    
    if len(has_coords) == 0:
        print(f"    ⚠ No records with valid coordinates for KNN training")
        df['COORDS_IMPUTED'] = False
        return df
    
    print(f"    Found {len(missing_coords)} {entity_name} with missing coordinates")
    print(f"    Using {len(has_coords)} {entity_name} with valid coordinates for KNN")
    
    # Fill missing address fields with 'UNKNOWN'
    for col in [city_col, state_col]:
        has_coords[col] = has_coords[col].fillna('UNKNOWN').astype(str)
        missing_coords[col] = missing_coords[col].fillna('UNKNOWN').astype(str)
    
    # Encode city and state
    city_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    
    # Fit encoders on combined data to handle all categories
    all_cities = pd.concat([has_coords[city_col], missing_coords[city_col]]).unique()
    all_states = pd.concat([has_coords[state_col], missing_coords[state_col]]).unique()
    
    city_encoder.fit(all_cities)
    state_encoder.fit(all_states)
    
    # Transform training data (records with coordinates)
    has_coords['CITY_ENCODED'] = city_encoder.transform(has_coords[city_col])
    has_coords['STATE_ENCODED'] = state_encoder.transform(has_coords[state_col])
    
    # Transform test data (records without coordinates)
    missing_coords['CITY_ENCODED'] = city_encoder.transform(missing_coords[city_col])
    missing_coords['STATE_ENCODED'] = state_encoder.transform(missing_coords[state_col])
    
    # Prepare features for KNN
    X_train = has_coords[['CITY_ENCODED', 'STATE_ENCODED']].values
    y_train_lat = has_coords[lat_col].values
    y_train_lon = has_coords[lon_col].values
    
    X_test = missing_coords[['CITY_ENCODED', 'STATE_ENCODED']].values
    
    # Build KNN model
    n_neighbors = min(k, len(has_coords))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
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
    missing_coords[lat_col] = imputed_lats
    missing_coords[lon_col] = imputed_lons
    missing_coords['COORDS_IMPUTED'] = True
    
    # Mark non-imputed rows
    has_coords['COORDS_IMPUTED'] = False
    
    # Drop encoding columns
    has_coords = has_coords.drop(columns=['CITY_ENCODED', 'STATE_ENCODED'], errors='ignore')
    missing_coords = missing_coords.drop(columns=['CITY_ENCODED', 'STATE_ENCODED'], errors='ignore')
    
    # Combine back
    result = pd.concat([has_coords, missing_coords], ignore_index=True)
    
    print(f"    ✓ Imputed coordinates for {len(missing_coords)} {entity_name} using {n_neighbors}-NN")
    print(f"      Method: Median of {n_neighbors} nearest neighbors based on {city_col} + {state_col}")
    
    return result


def build_balltree(df, lat_col, lon_col):
    """Build BallTree for efficient spatial queries"""
    valid_df = df.dropna(subset=[lat_col, lon_col])
    if len(valid_df) == 0:
        return None, valid_df
    coords_rad = np.radians(valid_df[[lat_col, lon_col]].values)
    tree = BallTree(coords_rad, metric='haversine')
    return tree, valid_df


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def print_step(step_num, description):
    """Print a formatted step header"""
    print(f"\n{'─'*60}")
    print(f" STEP {step_num}: {description}")
    print(f"{'─'*60}")


# ==================== DATA LOADING ====================

def load_all_data(data_dir=''):
    """Load all required data files"""
    print_section("LOADING DATA FILES")
    
    prefix = f"{data_dir}/" if data_dir else ""
    
    # Load all tables
    print("Loading clientgroup_data.csv...")
    clientgroup_data = pd.read_csv(f"{prefix}clientgroup_data.csv")
    print(f"  ✓ Loaded {len(clientgroup_data)} client groups")
    
    print("Loading relatedclient_data.csv...")
    relatedclient_data = pd.read_csv(f"{prefix}relatedclient_data.csv")
    print(f"  ✓ Loaded {len(relatedclient_data)} related client records")
    
    print("Loading banker_data.csv...")
    banker_data = pd.read_csv(f"{prefix}banker_data.csv")
    print(f"  ✓ Loaded {len(banker_data)} bankers")
    
    print("Loading sbb_data.csv...")
    sbb_data = pd.read_csv(f"{prefix}sbb_data.csv")
    print(f"  ✓ Loaded {len(sbb_data)} SBB bankers")
    
    print("Loading branch_data.csv...")
    branch_data = pd.read_csv(f"{prefix}branch_data.csv")
    print(f"  ✓ Loaded {len(branch_data)} branches")
    
    print("Loading household_data.csv...")
    household_data = pd.read_csv(f"{prefix}household_data.csv")
    print(f"  ✓ Loaded {len(household_data)} household records")
    
    print("Loading households_to_retain.csv...")
    try:
        households_to_retain = pd.read_csv(f"{prefix}households_to_retain.csv")
        print(f"  ✓ Loaded {len(households_to_retain)} households to retain")
    except FileNotFoundError:
        print("  ⚠ households_to_retain.csv not found, creating empty dataframe")
        households_to_retain = pd.DataFrame(columns=['HH_ECN', 'NEW_SEGMENT'])
    
    return {
        'clientgroup': clientgroup_data,
        'relatedclient': relatedclient_data,
        'banker': banker_data,
        'sbb': sbb_data,
        'branch': branch_data,
        'household': household_data,
        'hh_to_retain': households_to_retain
    }


# ==================== DATA PREPARATION ====================

def prepare_data(data):
    """Prepare and enrich data for processing"""
    print_section("PREPARING DATA")
    
    # Rename MASTER_CUST_ID to HH_ECN for consistency
    data['household'] = data['household'].rename(columns={'MASTER_CUST_ID': 'HH_ECN'})
    
    # ==================== COORDINATE IMPUTATION ====================
    
    # Check missing coordinates before imputation
    hh_missing_before = data['household'][data['household']['LAT_NUM'].isna() | data['household']['LON_NUM'].isna()]
    cg_missing_before = data['clientgroup'][data['clientgroup']['LAT_NUM'].isna() | data['clientgroup']['LON_NUM'].isna()]
    
    print(f"\n  Missing coordinates before imputation:")
    print(f"    Households: {len(hh_missing_before)} / {len(data['household'])}")
    print(f"    Client Groups: {len(cg_missing_before)} / {len(data['clientgroup'])}")
    
    # Impute client group coordinates FIRST (using KNN with address fields)
    # This way clientgroup coords can be used for household imputation
    data['clientgroup'] = impute_missing_coordinates_knn(
        data['clientgroup'],
        lat_col='LAT_NUM',
        lon_col='LON_NUM',
        city_col='BILLINGCITY',
        state_col='BILLINGSTATE',
        k=5,
        entity_name='client groups'
    )
    
    # Impute household coordinates using KNN with city/state
    data['household'] = impute_missing_coordinates_knn(
        data['household'],
        lat_col='LAT_NUM',
        lon_col='LON_NUM',
        city_col='BILLINGCITY',
        state_col='BILLINGSTATE',
        k=5,
        entity_name='households'
    )
    
    # Check missing coordinates after imputation
    hh_missing_after = data['household'][data['household']['LAT_NUM'].isna() | data['household']['LON_NUM'].isna()]
    cg_missing_after = data['clientgroup'][data['clientgroup']['LAT_NUM'].isna() | data['clientgroup']['LON_NUM'].isna()]
    
    print(f"\n  Missing coordinates after imputation:")
    print(f"    Households: {len(hh_missing_after)} / {len(data['household'])}")
    print(f"    Client Groups: {len(cg_missing_after)} / {len(data['clientgroup'])}")
    
    # ==================== BANKER SEPARATION ====================
    
    # Separate RM and RC bankers
    data['rm_bankers'] = data['banker'][data['banker']['BANKER_TYPE'] == 'RM'].copy()
    data['rc_bankers'] = data['banker'][data['banker']['BANKER_TYPE'] == 'RC'].copy()
    
    print(f"\n  Banker counts:")
    print(f"    RM Bankers: {len(data['rm_bankers'])}")
    print(f"    RC Bankers: {len(data['rc_bankers'])}")
    
    # ==================== HOUSEHOLD PRIMARY RECORDS ====================
    
    # Get unique households with their primary info
    # Aggregate to get one record per HH_ECN
    hh_primary = data['household'].groupby('HH_ECN').agg({
        'NEW_SEGMENT': 'first',
        'LAT_NUM': 'first',
        'LON_NUM': 'first',
        'PATR_AU_STR': 'first'
    }).reset_index()
    
    # Add COORDS_IMPUTED flag if available
    if 'COORDS_IMPUTED' in data['household'].columns:
        hh_imputed = data['household'].groupby('HH_ECN')['COORDS_IMPUTED'].any().reset_index()
        hh_primary = hh_primary.merge(hh_imputed, on='HH_ECN', how='left')
    else:
        hh_primary['COORDS_IMPUTED'] = False
    
    data['hh_primary'] = hh_primary
    
    print(f"\n  Unique Households: {len(hh_primary)}")
    print(f"    With imputed coordinates: {hh_primary['COORDS_IMPUTED'].sum()}")
    print(f"  Segment Distribution:")
    print(f"    Segment 1: {len(hh_primary[hh_primary['NEW_SEGMENT'] == 1])}")
    print(f"    Segment 2: {len(hh_primary[hh_primary['NEW_SEGMENT'] == 2])}")
    print(f"    Segment 3: {len(hh_primary[hh_primary['NEW_SEGMENT'] == 3])}")
    print(f"    Segment 4: {len(hh_primary[hh_primary['NEW_SEGMENT'] == 4])}")
    
    # ==================== SBB WITH COORDINATES ====================
    
    # Merge SBB with branch data for coordinates
    data['sbb'] = data['sbb'].merge(
        data['branch'][['AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']],
        on='AU',
        how='left'
    )
    print(f"\n  SBB Bankers with coordinates: {data['sbb']['BRANCH_LAT_NUM'].notna().sum()} / {len(data['sbb'])}")
    
    # ==================== MAPPINGS ====================
    
    # Build CG_ECN to Portfolio mapping
    cg_to_portfolio = data['clientgroup'][['CG_ECN', 'CG_PORTFOLIO_CD']].drop_duplicates()
    data['cg_to_portfolio'] = cg_to_portfolio
    
    # Build Portfolio to Banker mapping
    portfolio_to_banker = data['banker'][['PORT_CODE', 'EID', 'EMPLOYEE_NAME', 'BANKER_TYPE', 'AU']].copy()
    data['portfolio_to_banker'] = portfolio_to_banker
    
    # Build ECN to CG_ECN mapping (from relatedclient)
    ecn_to_cg = data['relatedclient'][['CG_ECN', 'RC_ECN']].drop_duplicates()
    data['ecn_to_cg'] = ecn_to_cg
    
    # Build HH_ECN to ECN mapping
    hh_to_ecn = data['household'][['HH_ECN', 'ECN']].drop_duplicates()
    data['hh_to_ecn'] = hh_to_ecn
    
    print(f"\n  Mappings built:")
    print(f"    CG to Portfolio: {len(cg_to_portfolio)}")
    print(f"    Portfolio to Banker: {len(portfolio_to_banker)}")
    print(f"    ECN to CG: {len(ecn_to_cg)}")
    print(f"    HH to ECN: {len(hh_to_ecn)}")
    
    return data


# ==================== STEP 1: DEDUPLICATE CLIENT GROUPS ====================

def deduplicate_client_groups(data):
    """
    Rule 1: Map all client groups to just one portfolio
    - CG_ECN should not be part of other client groups as an ECN (except itself)
    """
    print_step("1", "Deduplicate Client Groups")
    
    cg_data = data['clientgroup'].copy()
    rc_data = data['relatedclient'].copy()
    
    # Find CG_ECNs that appear as RC_ECN in other client groups
    cg_ecns = set(cg_data['CG_ECN'].unique())
    
    # Get cases where CG_ECN appears as related client in another CG
    cross_references = rc_data[
        (rc_data['RC_ECN'].isin(cg_ecns)) & 
        (rc_data['CG_ECN'] != rc_data['RC_ECN'])
    ].copy()
    
    print(f"  CG_ECNs appearing as RC_ECN in other groups: {len(cross_references)}")
    
    # For each CG_ECN in multiple portfolios, keep only one (first occurrence)
    cg_portfolio_count = cg_data.groupby('CG_ECN')['CG_PORTFOLIO_CD'].nunique()
    multi_portfolio_cgs = cg_portfolio_count[cg_portfolio_count > 1].index.tolist()
    
    print(f"  CG_ECNs in multiple portfolios: {len(multi_portfolio_cgs)}")
    
    # Create canonical CG to Portfolio mapping (keep first)
    canonical_cg_portfolio = cg_data.groupby('CG_ECN').first().reset_index()
    data['canonical_cg_portfolio'] = canonical_cg_portfolio
    
    # Create canonical ECN to CG mapping (keep first)
    canonical_ecn_cg = rc_data.groupby('RC_ECN').first().reset_index()
    data['canonical_ecn_cg'] = canonical_ecn_cg
    
    print(f"  ✓ Created canonical mappings")
    print(f"    Unique CG_ECNs: {len(canonical_cg_portfolio)}")
    print(f"    Unique RC_ECNs: {len(canonical_ecn_cg)}")
    
    return data


# ==================== MAPPING BUILDERS ====================

def build_ecn_to_banker_mapping(data):
    """Build mapping from ECN to current banker"""
    print("\n  Building ECN → Banker mapping...")
    
    # CG_ECN → Portfolio → Banker
    cg_to_portfolio = data['canonical_cg_portfolio'][['CG_ECN', 'CG_PORTFOLIO_CD']]
    portfolio_to_banker = data['portfolio_to_banker']
    
    # Map CG_ECN to banker
    cg_to_banker = cg_to_portfolio.merge(
        portfolio_to_banker,
        left_on='CG_PORTFOLIO_CD',
        right_on='PORT_CODE',
        how='left'
    )
    
    # RC_ECN → CG_ECN → Banker
    ecn_to_cg = data['canonical_ecn_cg'][['RC_ECN', 'CG_ECN']]
    ecn_to_banker = ecn_to_cg.merge(
        cg_to_banker[['CG_ECN', 'PORT_CODE', 'EID', 'EMPLOYEE_NAME', 'BANKER_TYPE']],
        on='CG_ECN',
        how='left'
    )
    
    data['cg_to_banker'] = cg_to_banker
    data['ecn_to_banker'] = ecn_to_banker
    
    print(f"    CG_ECN with bankers: {cg_to_banker['EID'].notna().sum()}")
    print(f"    RC_ECN with bankers: {ecn_to_banker['EID'].notna().sum()}")
    
    return data


def build_hh_to_current_banker_mapping(data):
    """Build mapping from HH_ECN to current banker(s) based on ECN overlap"""
    print("\n  Building HH_ECN → Current Banker mapping...")
    
    hh_to_ecn = data['hh_to_ecn']
    ecn_to_banker = data['ecn_to_banker']
    
    # Join HH ECNs to their current bankers
    hh_ecn_banker = hh_to_ecn.merge(
        ecn_to_banker[['RC_ECN', 'CG_ECN', 'PORT_CODE', 'EID', 'EMPLOYEE_NAME', 'BANKER_TYPE']],
        left_on='ECN',
        right_on='RC_ECN',
        how='left'
    )
    
    # Group by HH_ECN to find all associated bankers
    hh_banker_counts = hh_ecn_banker.groupby(['HH_ECN', 'PORT_CODE', 'EID', 'EMPLOYEE_NAME', 'BANKER_TYPE']).size().reset_index(name='ECN_COUNT')
    
    data['hh_ecn_banker'] = hh_ecn_banker
    data['hh_banker_counts'] = hh_banker_counts
    
    hh_with_bankers = hh_ecn_banker[hh_ecn_banker['EID'].notna()]['HH_ECN'].nunique()
    print(f"    HH_ECNs with current banker mapping: {hh_with_bankers}")
    
    return data


def resolve_hh_to_single_banker(data, hh_ecn):
    """Resolve HH to single banker by majority ECN count"""
    hh_banker_counts = data['hh_banker_counts']
    
    hh_bankers = hh_banker_counts[hh_banker_counts['HH_ECN'] == hh_ecn].copy()
    
    if len(hh_bankers) == 0 or hh_bankers['EID'].isna().all():
        return None
    
    hh_bankers = hh_bankers[hh_bankers['EID'].notna()]
    
    if len(hh_bankers) == 0:
        return None
    
    # Get banker with most ECNs
    best_banker = hh_bankers.loc[hh_bankers['ECN_COUNT'].idxmax()]
    
    return {
        'PORT_CODE': best_banker['PORT_CODE'],
        'EID': best_banker['EID'],
        'EMPLOYEE_NAME': best_banker['EMPLOYEE_NAME'],
        'BANKER_TYPE': best_banker['BANKER_TYPE']
    }


# ==================== ASSIGNMENT TRACKING ====================

def initialize_assignment_tracking(data):
    """Initialize tracking dataframe for HH assignments"""
    print("\n  Initializing assignment tracking...")
    
    hh_primary = data['hh_primary'].copy()
    
    hh_primary['IS_ASSIGNED'] = False
    hh_primary['ASSIGNED_PORT_CODE'] = None
    hh_primary['ASSIGNED_EID'] = None
    hh_primary['ASSIGNED_BANKER_NAME'] = None
    hh_primary['ASSIGNED_BANKER_TYPE'] = None  # RM, RC, SBB
    hh_primary['ASSIGNMENT_STEP'] = None
    hh_primary['ASSIGNMENT_REASON'] = None
    
    # Ensure COORDS_IMPUTED column exists
    if 'COORDS_IMPUTED' not in hh_primary.columns:
        hh_primary['COORDS_IMPUTED'] = False
    
    data['hh_assignments'] = hh_primary
    
    imputed_count = hh_primary['COORDS_IMPUTED'].sum()
    print(f"    Tracking {len(hh_primary)} households")
    print(f"    With imputed coordinates: {imputed_count}")
    print(f"    With original coordinates: {len(hh_primary) - imputed_count}")
    
    return data


def assign_household(data, hh_ecn, port_code, eid, banker_name, banker_type, step, reason):
    """Assign a household to a banker"""
    idx = data['hh_assignments'][data['hh_assignments']['HH_ECN'] == hh_ecn].index
    
    if len(idx) == 0:
        return False
    
    idx = idx[0]
    
    if data['hh_assignments'].at[idx, 'IS_ASSIGNED']:
        return False  # Already assigned
    
    data['hh_assignments'].at[idx, 'IS_ASSIGNED'] = True
    data['hh_assignments'].at[idx, 'ASSIGNED_PORT_CODE'] = port_code
    data['hh_assignments'].at[idx, 'ASSIGNED_EID'] = eid
    data['hh_assignments'].at[idx, 'ASSIGNED_BANKER_NAME'] = banker_name
    data['hh_assignments'].at[idx, 'ASSIGNED_BANKER_TYPE'] = banker_type
    data['hh_assignments'].at[idx, 'ASSIGNMENT_STEP'] = step
    data['hh_assignments'].at[idx, 'ASSIGNMENT_REASON'] = reason
    
    return True


def get_unassigned_hh(data, segment=None):
    """Get unassigned households, optionally filtered by segment"""
    hh = data['hh_assignments']
    unassigned = hh[hh['IS_ASSIGNED'] == False]
    
    if segment is not None:
        unassigned = unassigned[unassigned['NEW_SEGMENT'] == segment]
    
    return unassigned


# ==================== STEP 5: SEGMENT 4 → RC RETENTION ====================

def step5_segment4_rc_retention(data):
    """
    Step 5: Get all segment 4 HH ECNs mapped to current RC bankers' CG ECNs
    Retain with same RC banker
    """
    print_step("5", "Segment 4 HH → RC Banker Retention")
    
    # Get segment 4 households
    seg4_hh = data['hh_assignments'][data['hh_assignments']['NEW_SEGMENT'] == 4]['HH_ECN'].tolist()
    print(f"  Segment 4 Households: {len(seg4_hh)}")
    
    # Get RC banker portfolios
    rc_portfolios = data['rc_bankers']['PORT_CODE'].tolist()
    
    # Find HHs that have ECNs in RC portfolios
    hh_ecn_banker = data['hh_ecn_banker']
    seg4_in_rc = hh_ecn_banker[
        (hh_ecn_banker['HH_ECN'].isin(seg4_hh)) &
        (hh_ecn_banker['BANKER_TYPE'] == 'RC')
    ]['HH_ECN'].unique()
    
    print(f"  Segment 4 HHs with RC banker connection: {len(seg4_in_rc)}")
    
    assigned_count = 0
    for hh_ecn in seg4_in_rc:
        banker_info = resolve_hh_to_single_banker(data, hh_ecn)
        
        if banker_info and banker_info['BANKER_TYPE'] == 'RC':
            success = assign_household(
                data, hh_ecn,
                banker_info['PORT_CODE'],
                banker_info['EID'],
                banker_info['EMPLOYEE_NAME'],
                'RC',
                'STEP_5',
                'SEG4_RC_RETENTION'
            )
            if success:
                assigned_count += 1
    
    print(f"  ✓ Assigned {assigned_count} Segment 4 HHs to RC bankers")
    
    return data, {'step5_assigned': assigned_count}


# ==================== STEP 6: SEGMENT 3 → RM RETENTION ====================

def step6_segment3_rm_retention(data):
    """
    Step 6: All segment 3 CG ECNs and related ECNs → retain with same RM
    Resolve conflicts by majority ECN count
    """
    print_step("6", "Segment 3 CG/ECN → RM Banker Retention")
    
    # Get segment 3 households
    seg3_hh = get_unassigned_hh(data, segment=3)['HH_ECN'].tolist()
    print(f"  Unassigned Segment 3 Households: {len(seg3_hh)}")
    
    # Find HHs that have ECNs in RM portfolios
    hh_ecn_banker = data['hh_ecn_banker']
    seg3_in_rm = hh_ecn_banker[
        (hh_ecn_banker['HH_ECN'].isin(seg3_hh)) &
        (hh_ecn_banker['BANKER_TYPE'] == 'RM')
    ]['HH_ECN'].unique()
    
    print(f"  Segment 3 HHs with RM banker connection: {len(seg3_in_rm)}")
    
    assigned_count = 0
    for hh_ecn in seg3_in_rm:
        banker_info = resolve_hh_to_single_banker(data, hh_ecn)
        
        if banker_info and banker_info['BANKER_TYPE'] == 'RM':
            success = assign_household(
                data, hh_ecn,
                banker_info['PORT_CODE'],
                banker_info['EID'],
                banker_info['EMPLOYEE_NAME'],
                'RM',
                'STEP_6',
                'SEG3_RM_RETENTION'
            )
            if success:
                assigned_count += 1
    
    print(f"  ✓ Assigned {assigned_count} Segment 3 HHs to RM bankers")
    
    return data, {'step6_assigned': assigned_count}


# ==================== STEP 7: HOUSEHOLDS TO RETAIN ====================

def step7_households_to_retain(data):
    """
    Step 7: HHs from households_to_retain file → retain with existing banker
    """
    print_step("7", "Households to Retain (from file)")
    
    hh_to_retain = data['hh_to_retain']['HH_ECN'].tolist()
    print(f"  Households in retain file: {len(hh_to_retain)}")
    
    # Filter to unassigned only
    unassigned = get_unassigned_hh(data)
    hh_to_retain_unassigned = [hh for hh in hh_to_retain if hh in unassigned['HH_ECN'].values]
    print(f"  Still unassigned: {len(hh_to_retain_unassigned)}")
    
    assigned_count = 0
    for hh_ecn in hh_to_retain_unassigned:
        banker_info = resolve_hh_to_single_banker(data, hh_ecn)
        
        if banker_info:
            success = assign_household(
                data, hh_ecn,
                banker_info['PORT_CODE'],
                banker_info['EID'],
                banker_info['EMPLOYEE_NAME'],
                banker_info['BANKER_TYPE'],
                'STEP_7',
                'RETAIN_FILE'
            )
            if success:
                assigned_count += 1
    
    print(f"  ✓ Assigned {assigned_count} HHs from retain file")
    
    return data, {'step7_assigned': assigned_count}


# ==================== STEP 8: SEGMENT 2 → SBB (EXISTING BOOKS) ====================

def step8_segment2_sbb_existing(data):
    """
    Step 8: Unassigned segment 2 HHs in existing books
    - If patronage AU matches SBB AU → assign to SBB
    - Else if within 10 miles of SBB → assign to SBB
    """
    print_step("8", "Segment 2 → SBB (Existing Books)")
    
    sbb_mappings = []
    
    # Get unassigned segment 2 HHs that are in existing books
    seg2_hh = get_unassigned_hh(data, segment=2)
    
    # Filter to those in existing books (have ECNs in current portfolios)
    hh_ecn_banker = data['hh_ecn_banker']
    seg2_in_books = hh_ecn_banker[
        (hh_ecn_banker['HH_ECN'].isin(seg2_hh['HH_ECN'])) &
        (hh_ecn_banker['EID'].notna())
    ]['HH_ECN'].unique()
    
    print(f"  Unassigned Segment 2 HHs: {len(seg2_hh)}")
    print(f"  Segment 2 HHs in existing books: {len(seg2_in_books)}")
    
    # Get SBB AUs
    sbb_data = data['sbb']
    sbb_aus = set(sbb_data['AU'].dropna().unique())
    
    assigned_au_count = 0
    assigned_proximity_count = 0
    
    # Build SBB BallTree for proximity search
    sbb_tree, sbb_valid = build_balltree(sbb_data, 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM')
    
    for hh_ecn in seg2_in_books:
        hh_row = data['hh_assignments'][data['hh_assignments']['HH_ECN'] == hh_ecn].iloc[0]
        
        if hh_row['IS_ASSIGNED']:
            continue
        
        patr_au = hh_row['PATR_AU_STR']
        hh_lat = hh_row['LAT_NUM']
        hh_lon = hh_row['LON_NUM']
        
        assigned = False
        
        # Check 1: Patronage AU matches SBB AU
        if patr_au in sbb_aus:
            sbb_match = sbb_data[sbb_data['AU'] == patr_au].iloc[0]
            
            success = assign_household(
                data, hh_ecn,
                patr_au,  # Using AU as port code for SBB
                patr_au,  # Using AU as EID for SBB
                sbb_match['Full Name'],
                'SBB',
                'STEP_8',
                'SBB_AU_MATCH'
            )
            
            if success:
                assigned_au_count += 1
                assigned = True
                
                # Add all ECNs to SBB mapping
                hh_ecns = data['hh_to_ecn'][data['hh_to_ecn']['HH_ECN'] == hh_ecn]['ECN'].tolist()
                for ecn in hh_ecns:
                    sbb_mappings.append({
                        'ECN': ecn,
                        'HH_ECN': hh_ecn,
                        'AU': patr_au,
                        'Full Name': sbb_match['Full Name'],
                        'ASSIGNMENT_REASON': 'AU_MATCH'
                    })
        
        # Check 2: Within 10 miles of SBB
        if not assigned and sbb_tree is not None and pd.notna(hh_lat) and pd.notna(hh_lon):
            hh_rad = np.radians([[hh_lat, hh_lon]])
            radius_rad = 10 / 3959.0  # 10 miles
            
            indices, distances = sbb_tree.query_radius(hh_rad, r=radius_rad, 
                                                        return_distance=True, sort_results=True)
            
            if len(indices[0]) > 0:
                nearest_idx = indices[0][0]
                nearest_sbb = sbb_valid.iloc[nearest_idx]
                
                success = assign_household(
                    data, hh_ecn,
                    nearest_sbb['AU'],
                    nearest_sbb['AU'],
                    nearest_sbb['Full Name'],
                    'SBB',
                    'STEP_8',
                    'SBB_PROXIMITY_10MI'
                )
                
                if success:
                    assigned_proximity_count += 1
                    
                    hh_ecns = data['hh_to_ecn'][data['hh_to_ecn']['HH_ECN'] == hh_ecn]['ECN'].tolist()
                    for ecn in hh_ecns:
                        sbb_mappings.append({
                            'ECN': ecn,
                            'HH_ECN': hh_ecn,
                            'AU': nearest_sbb['AU'],
                            'Full Name': nearest_sbb['Full Name'],
                            'ASSIGNMENT_REASON': 'PROXIMITY_10MI'
                        })
    
    print(f"  ✓ Assigned by AU match: {assigned_au_count}")
    print(f"  ✓ Assigned by 10mi proximity: {assigned_proximity_count}")
    
    data['sbb_mappings'] = sbb_mappings
    
    return data, {
        'step8_au_match': assigned_au_count,
        'step8_proximity': assigned_proximity_count
    }


# ==================== STEP 9: REMAINING SEGMENT 2 → RM/RC ====================

def step9_segment2_rm_rc_retention(data):
    """
    Step 9: Remaining segment 2 HHs → stay with existing RM/RC bankers
    """
    print_step("9", "Remaining Segment 2 → RM/RC Retention")
    
    seg2_hh = get_unassigned_hh(data, segment=2)
    print(f"  Unassigned Segment 2 HHs: {len(seg2_hh)}")
    
    assigned_count = 0
    
    for _, hh_row in seg2_hh.iterrows():
        hh_ecn = hh_row['HH_ECN']
        banker_info = resolve_hh_to_single_banker(data, hh_ecn)
        
        if banker_info:
            success = assign_household(
                data, hh_ecn,
                banker_info['PORT_CODE'],
                banker_info['EID'],
                banker_info['EMPLOYEE_NAME'],
                banker_info['BANKER_TYPE'],
                'STEP_9',
                'SEG2_RMRC_RETENTION'
            )
            if success:
                assigned_count += 1
    
    print(f"  ✓ Assigned {assigned_count} Segment 2 HHs to RM/RC bankers")
    
    return data, {'step9_assigned': assigned_count}


# ==================== STEP 10: VERIFY SEGMENT 2 PLACEMENT ====================

def step10_verify_segment2_placement(data):
    """
    Step 10: Verify all segment 2 CG ECNs and related ECNs are placed
    For unplaced: check SBB AU match, then 10mi proximity
    """
    print_step("10", "Verify Segment 2 Placement & Assign to SBB")
    
    sbb_mappings = data.get('sbb_mappings', [])
    
    # Get all segment 2 CG_ECNs and their related ECNs
    cg_data = data['clientgroup']
    rc_data = data['relatedclient']
    
    # Get segment 2 from old segmentation (CS_NEW_NS = 2)
    # Note: Need to map old segments to new - assuming CS_NEW_NS maps approximately
    seg2_cg_ecns = cg_data[cg_data['CS_NEW_NS'] == 2]['CG_ECN'].unique()
    seg2_rc_ecns = rc_data[rc_data['CS_NEW_NS'] == 2]['RC_ECN'].unique()
    
    all_seg2_ecns = set(seg2_cg_ecns) | set(seg2_rc_ecns)
    print(f"  Total Segment 2 ECNs (old seg): {len(all_seg2_ecns)}")
    
    # Find HHs containing these ECNs
    hh_to_ecn = data['hh_to_ecn']
    hh_with_seg2_ecns = hh_to_ecn[hh_to_ecn['ECN'].isin(all_seg2_ecns)]['HH_ECN'].unique()
    
    # Filter to unassigned
    unassigned_hh = get_unassigned_hh(data)
    unassigned_with_seg2 = [hh for hh in hh_with_seg2_ecns if hh in unassigned_hh['HH_ECN'].values]
    
    print(f"  HHs with Segment 2 ECNs: {len(hh_with_seg2_ecns)}")
    print(f"  Still unassigned: {len(unassigned_with_seg2)}")
    
    sbb_data = data['sbb']
    sbb_aus = set(sbb_data['AU'].dropna().unique())
    sbb_tree, sbb_valid = build_balltree(sbb_data, 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM')
    
    assigned_au_count = 0
    assigned_proximity_count = 0
    assigned_rmrc_count = 0
    
    for hh_ecn in unassigned_with_seg2:
        hh_row = data['hh_assignments'][data['hh_assignments']['HH_ECN'] == hh_ecn].iloc[0]
        
        if hh_row['IS_ASSIGNED']:
            continue
        
        patr_au = hh_row['PATR_AU_STR']
        hh_lat = hh_row['LAT_NUM']
        hh_lon = hh_row['LON_NUM']
        
        assigned = False
        
        # Check 1: Patronage AU matches SBB AU
        if patr_au in sbb_aus:
            sbb_match = sbb_data[sbb_data['AU'] == patr_au].iloc[0]
            
            success = assign_household(
                data, hh_ecn,
                patr_au,
                patr_au,
                sbb_match['Full Name'],
                'SBB',
                'STEP_10',
                'SBB_AU_MATCH_VERIFY'
            )
            
            if success:
                assigned_au_count += 1
                assigned = True
                
                hh_ecns = data['hh_to_ecn'][data['hh_to_ecn']['HH_ECN'] == hh_ecn]['ECN'].tolist()
                for ecn in hh_ecns:
                    sbb_mappings.append({
                        'ECN': ecn,
                        'HH_ECN': hh_ecn,
                        'AU': patr_au,
                        'Full Name': sbb_match['Full Name'],
                        'ASSIGNMENT_REASON': 'AU_MATCH_STEP10'
                    })
        
        # Check 2: Within 10 miles of SBB
        if not assigned and sbb_tree is not None and pd.notna(hh_lat) and pd.notna(hh_lon):
            hh_rad = np.radians([[hh_lat, hh_lon]])
            radius_rad = 10 / 3959.0
            
            indices, distances = sbb_tree.query_radius(hh_rad, r=radius_rad, 
                                                        return_distance=True, sort_results=True)
            
            if len(indices[0]) > 0:
                nearest_idx = indices[0][0]
                nearest_sbb = sbb_valid.iloc[nearest_idx]
                
                success = assign_household(
                    data, hh_ecn,
                    nearest_sbb['AU'],
                    nearest_sbb['AU'],
                    nearest_sbb['Full Name'],
                    'SBB',
                    'STEP_10',
                    'SBB_PROXIMITY_10MI_VERIFY'
                )
                
                if success:
                    assigned_proximity_count += 1
                    assigned = True
                    
                    hh_ecns = data['hh_to_ecn'][data['hh_to_ecn']['HH_ECN'] == hh_ecn]['ECN'].tolist()
                    for ecn in hh_ecns:
                        sbb_mappings.append({
                            'ECN': ecn,
                            'HH_ECN': hh_ecn,
                            'AU': nearest_sbb['AU'],
                            'Full Name': nearest_sbb['Full Name'],
                            'ASSIGNMENT_REASON': 'PROXIMITY_10MI_STEP10'
                        })
        
        # Check 3: Retain with RM/RC
        if not assigned:
            banker_info = resolve_hh_to_single_banker(data, hh_ecn)
            
            if banker_info:
                success = assign_household(
                    data, hh_ecn,
                    banker_info['PORT_CODE'],
                    banker_info['EID'],
                    banker_info['EMPLOYEE_NAME'],
                    banker_info['BANKER_TYPE'],
                    'STEP_10',
                    'SEG2_ECN_RMRC_FALLBACK'
                )
                if success:
                    assigned_rmrc_count += 1
    
    print(f"  ✓ Assigned by AU match: {assigned_au_count}")
    print(f"  ✓ Assigned by 10mi proximity: {assigned_proximity_count}")
    print(f"  ✓ Assigned to RM/RC (fallback): {assigned_rmrc_count}")
    
    data['sbb_mappings'] = sbb_mappings
    
    return data, {
        'step10_au_match': assigned_au_count,
        'step10_proximity': assigned_proximity_count,
        'step10_rmrc': assigned_rmrc_count
    }


# ==================== STEP 11: SPATIAL ASSIGNMENT (SEG 3 → RM, SEG 4 → RC) ====================

def step11_spatial_assignment(data):
    """
    Step 11: Remaining segment 3 → RMs, segment 4 → RCs
    Using BallTree spatial assignment with min/max targets
    """
    print_step("11", "Spatial Assignment (Seg 3 → RM, Seg 4 → RC)")
    
    # Calculate current retention counts per banker
    assigned_hh = data['hh_assignments'][data['hh_assignments']['IS_ASSIGNED'] == True]
    
    # RM bankers - segment 3
    rm_retention_counts = assigned_hh[
        (assigned_hh['ASSIGNED_BANKER_TYPE'] == 'RM') & 
        (assigned_hh['NEW_SEGMENT'] == 3)
    ].groupby('ASSIGNED_PORT_CODE').size().to_dict()
    
    # RC bankers - segment 4
    rc_retention_counts = assigned_hh[
        (assigned_hh['ASSIGNED_BANKER_TYPE'] == 'RC') & 
        (assigned_hh['NEW_SEGMENT'] == 4)
    ].groupby('ASSIGNED_PORT_CODE').size().to_dict()
    
    # Prepare RM bankers for assignment
    rm_bankers = data['rm_bankers'].copy()
    rm_bankers = rm_bankers.merge(
        data['branch'][['AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']],
        on='AU',
        how='left'
    )
    rm_bankers['RETAINED_COUNT'] = rm_bankers['PORT_CODE'].map(rm_retention_counts).fillna(0).astype(int)
    rm_bankers['MIN_NEEDED'] = 270 - rm_bankers['RETAINED_COUNT']
    rm_bankers['MAX_NEEDED'] = 350 - rm_bankers['RETAINED_COUNT']
    rm_bankers['CURRENT_ASSIGNED'] = 0
    
    # Prepare RC bankers for assignment
    rc_bankers = data['rc_bankers'].copy()
    # RC bankers are CENTRALIZED - they don't have AU, need different location logic
    # For now, use a default location or skip coordinates
    rc_bankers['RETAINED_COUNT'] = rc_bankers['PORT_CODE'].map(rc_retention_counts).fillna(0).astype(int)
    rc_bankers['MIN_NEEDED'] = 220 - rc_bankers['RETAINED_COUNT']
    rc_bankers['MAX_NEEDED'] = 270 - rc_bankers['RETAINED_COUNT']
    rc_bankers['CURRENT_ASSIGNED'] = 0
    
    print(f"\n  RM Bankers: {len(rm_bankers)}")
    print(f"    Total MIN needed: {rm_bankers['MIN_NEEDED'].clip(lower=0).sum()}")
    print(f"    Total MAX capacity: {rm_bankers['MAX_NEEDED'].clip(lower=0).sum()}")
    
    print(f"\n  RC Bankers: {len(rc_bankers)}")
    print(f"    Total MIN needed: {rc_bankers['MIN_NEEDED'].clip(lower=0).sum()}")
    print(f"    Total MAX capacity: {rc_bankers['MAX_NEEDED'].clip(lower=0).sum()}")
    
    # Get unassigned households
    seg3_unassigned = get_unassigned_hh(data, segment=3)
    seg4_unassigned = get_unassigned_hh(data, segment=4)
    
    print(f"\n  Unassigned Segment 3: {len(seg3_unassigned)}")
    print(f"  Unassigned Segment 4: {len(seg4_unassigned)}")
    
    # Assign Segment 3 to RMs
    seg3_assigned = assign_spatial_with_balltree(
        data, seg3_unassigned, rm_bankers, 
        'RM', 'STEP_11', 'SPATIAL_SEG3_RM',
        in_market_radius=40, expanded_radius=60
    )
    
    # Assign Segment 4 to RCs (CENTRALIZED - larger radius)
    seg4_assigned = assign_spatial_with_balltree(
        data, seg4_unassigned, rc_bankers,
        'RC', 'STEP_11', 'SPATIAL_SEG4_RC',
        in_market_radius=200, expanded_radius=600, is_centralized=True
    )
    
    print(f"\n  ✓ Segment 3 → RM assignments: {seg3_assigned}")
    print(f"  ✓ Segment 4 → RC assignments: {seg4_assigned}")
    
    return data, {
        'step11_seg3_rm': seg3_assigned,
        'step11_seg4_rc': seg4_assigned
    }


def assign_spatial_with_balltree(data, unassigned_hh, bankers, banker_type, step, reason,
                                  in_market_radius=40, expanded_radius=60, is_centralized=False):
    """Assign households to bankers using BallTree spatial indexing"""
    
    if len(unassigned_hh) == 0:
        return 0
    
    # Get banker coordinates
    if is_centralized:
        # For centralized, use HH coordinates to find nearest banker
        # Since RC bankers don't have AU/branch, we'll assign based on capacity
        return assign_centralized_by_capacity(data, unassigned_hh, bankers, banker_type, step, reason)
    
    # Build BallTree for bankers
    bankers_valid = bankers.dropna(subset=['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']).copy()
    
    if len(bankers_valid) == 0:
        print(f"    ⚠ No bankers with valid coordinates")
        return 0
    
    banker_tree = BallTree(
        np.radians(bankers_valid[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values),
        metric='haversine'
    )
    
    assigned_count = 0
    
    # Phase 1: Initial radius
    for _, hh_row in unassigned_hh.iterrows():
        hh_ecn = hh_row['HH_ECN']
        hh_lat = hh_row['LAT_NUM']
        hh_lon = hh_row['LON_NUM']
        
        if data['hh_assignments'][data['hh_assignments']['HH_ECN'] == hh_ecn].iloc[0]['IS_ASSIGNED']:
            continue
        
        if pd.isna(hh_lat) or pd.isna(hh_lon):
            continue
        
        hh_rad = np.radians([[hh_lat, hh_lon]])
        radius_rad = in_market_radius / 3959.0
        
        indices, distances = banker_tree.query_radius(hh_rad, r=radius_rad,
                                                       return_distance=True, sort_results=True)
        
        for idx, dist in zip(indices[0], distances[0] * 3959.0):
            banker = bankers_valid.iloc[idx]
            banker_idx = bankers[bankers['PORT_CODE'] == banker['PORT_CODE']].index[0]
            
            current = bankers.at[banker_idx, 'CURRENT_ASSIGNED']
            max_needed = bankers.at[banker_idx, 'MAX_NEEDED']
            
            if current >= max_needed:
                continue
            
            success = assign_household(
                data, hh_ecn,
                banker['PORT_CODE'],
                banker['EID'],
                banker['EMPLOYEE_NAME'],
                banker_type,
                step,
                reason
            )
            
            if success:
                bankers.at[banker_idx, 'CURRENT_ASSIGNED'] += 1
                assigned_count += 1
                break
    
    # Phase 2: Expanded radius for undersized bankers
    undersized = bankers[bankers['CURRENT_ASSIGNED'] < bankers['MIN_NEEDED']]
    
    if len(undersized) > 0:
        print(f"    Filling {len(undersized)} undersized bankers (expanded to {expanded_radius}mi)")
        
        for idx, banker in undersized.iterrows():
            if pd.isna(banker.get('BRANCH_LAT_NUM')) or pd.isna(banker.get('BRANCH_LON_NUM')):
                continue
            
            needed = banker['MIN_NEEDED'] - banker['CURRENT_ASSIGNED']
            if needed <= 0:
                continue
            
            # Get remaining unassigned
            still_unassigned = get_unassigned_hh(data, segment=unassigned_hh['NEW_SEGMENT'].iloc[0] if len(unassigned_hh) > 0 else None)
            still_unassigned = still_unassigned.dropna(subset=['LAT_NUM', 'LON_NUM'])
            
            if len(still_unassigned) == 0:
                break
            
            # Build tree for unassigned HHs
            hh_tree = BallTree(
                np.radians(still_unassigned[['LAT_NUM', 'LON_NUM']].values),
                metric='haversine'
            )
            
            banker_rad = np.radians([[banker['BRANCH_LAT_NUM'], banker['BRANCH_LON_NUM']]])
            radius_rad = expanded_radius / 3959.0
            
            indices, distances = hh_tree.query_radius(banker_rad, r=radius_rad,
                                                       return_distance=True, sort_results=True)
            
            assigned_to_banker = 0
            for hh_idx, dist in zip(indices[0], distances[0] * 3959.0):
                if assigned_to_banker >= needed:
                    break
                
                hh_row = still_unassigned.iloc[hh_idx]
                hh_ecn = hh_row['HH_ECN']
                
                if data['hh_assignments'][data['hh_assignments']['HH_ECN'] == hh_ecn].iloc[0]['IS_ASSIGNED']:
                    continue
                
                success = assign_household(
                    data, hh_ecn,
                    banker['PORT_CODE'],
                    banker['EID'],
                    banker['EMPLOYEE_NAME'],
                    banker_type,
                    step,
                    f"{reason}_EXPANDED_{expanded_radius}MI"
                )
                
                if success:
                    bankers.at[idx, 'CURRENT_ASSIGNED'] += 1
                    assigned_count += 1
                    assigned_to_banker += 1
    
    return assigned_count


def assign_centralized_by_capacity(data, unassigned_hh, bankers, banker_type, step, reason):
    """Assign to centralized bankers by capacity (round-robin to fill minimums)"""
    
    assigned_count = 0
    
    # Sort bankers by how much they need
    bankers = bankers.copy()
    bankers = bankers.sort_values('MIN_NEEDED', ascending=False)
    
    hh_list = unassigned_hh['HH_ECN'].tolist()
    
    for hh_ecn in hh_list:
        if data['hh_assignments'][data['hh_assignments']['HH_ECN'] == hh_ecn].iloc[0]['IS_ASSIGNED']:
            continue
        
        # Find banker with most remaining capacity under MIN
        for idx, banker in bankers.iterrows():
            current = banker['CURRENT_ASSIGNED']
            min_needed = banker['MIN_NEEDED']
            
            if current >= min_needed:
                continue
            
            success = assign_household(
                data, hh_ecn,
                banker['PORT_CODE'],
                banker['EID'],
                banker['EMPLOYEE_NAME'],
                banker_type,
                step,
                reason
            )
            
            if success:
                bankers.at[idx, 'CURRENT_ASSIGNED'] += 1
                assigned_count += 1
                break
        else:
            # All at MIN, now fill to MAX
            for idx, banker in bankers.iterrows():
                current = banker['CURRENT_ASSIGNED']
                max_needed = banker['MAX_NEEDED']
                
                if current >= max_needed:
                    continue
                
                success = assign_household(
                    data, hh_ecn,
                    banker['PORT_CODE'],
                    banker['EID'],
                    banker['EMPLOYEE_NAME'],
                    banker_type,
                    step,
                    f"{reason}_FILL_MAX"
                )
                
                if success:
                    bankers.at[idx, 'CURRENT_ASSIGNED'] += 1
                    assigned_count += 1
                    break
    
    return assigned_count


# ==================== STEP 12: GENERATE METRICS ====================

def step12_generate_metrics(data, metrics):
    """Generate comprehensive metrics report"""
    print_step("12", "Generate Metrics")
    
    hh_assignments = data['hh_assignments']
    
    total_hh = len(hh_assignments)
    assigned_hh = hh_assignments[hh_assignments['IS_ASSIGNED'] == True]
    unassigned_hh = hh_assignments[hh_assignments['IS_ASSIGNED'] == False]
    
    print(f"\n  OVERALL METRICS:")
    print(f"  {'─'*40}")
    print(f"  Total Households: {total_hh}")
    print(f"  Assigned: {len(assigned_hh)} ({len(assigned_hh)/total_hh*100:.1f}%)")
    print(f"  Unassigned: {len(unassigned_hh)} ({len(unassigned_hh)/total_hh*100:.1f}%)")
    
    # Coordinate imputation metrics
    if 'COORDS_IMPUTED' in hh_assignments.columns:
        imputed_count = hh_assignments['COORDS_IMPUTED'].sum()
        print(f"\n  COORDINATE IMPUTATION:")
        print(f"  {'─'*40}")
        print(f"  Households with imputed coordinates: {imputed_count} ({imputed_count/total_hh*100:.1f}%)")
        print(f"  Households with original coordinates: {total_hh - imputed_count}")
    
    print(f"\n  BY SEGMENT:")
    print(f"  {'─'*40}")
    for seg in [1, 2, 3, 4]:
        seg_total = len(hh_assignments[hh_assignments['NEW_SEGMENT'] == seg])
        seg_assigned = len(assigned_hh[assigned_hh['NEW_SEGMENT'] == seg])
        pct = seg_assigned/seg_total*100 if seg_total > 0 else 0
        print(f"  Segment {seg}: {seg_assigned}/{seg_total} assigned ({pct:.1f}%)")
    
    print(f"\n  BY BANKER TYPE:")
    print(f"  {'─'*40}")
    for bt in ['RM', 'RC', 'SBB']:
        bt_count = len(assigned_hh[assigned_hh['ASSIGNED_BANKER_TYPE'] == bt])
        print(f"  {bt}: {bt_count}")
    
    print(f"\n  BY ASSIGNMENT STEP:")
    print(f"  {'─'*40}")
    step_counts = assigned_hh.groupby('ASSIGNMENT_STEP').size().sort_index()
    for step, count in step_counts.items():
        print(f"  {step}: {count}")
    
    print(f"\n  SBB MAPPINGS:")
    print(f"  {'─'*40}")
    sbb_mappings = data.get('sbb_mappings', [])
    print(f"  Total ECNs mapped to SBB: {len(sbb_mappings)}")
    if len(sbb_mappings) > 0:
        sbb_df = pd.DataFrame(sbb_mappings)
        print(f"  Unique HHs: {sbb_df['HH_ECN'].nunique()}")
        print(f"  Unique SBB AUs: {sbb_df['AU'].nunique()}")
    
    # Compile all metrics
    final_metrics = {
        'total_households': total_hh,
        'assigned_households': len(assigned_hh),
        'unassigned_households': len(unassigned_hh),
        'assignment_rate': len(assigned_hh)/total_hh*100,
        'coords_imputed_count': hh_assignments['COORDS_IMPUTED'].sum() if 'COORDS_IMPUTED' in hh_assignments.columns else 0,
        **metrics
    }
    
    # Add segment-wise metrics
    for seg in [1, 2, 3, 4]:
        seg_total = len(hh_assignments[hh_assignments['NEW_SEGMENT'] == seg])
        seg_assigned = len(assigned_hh[assigned_hh['NEW_SEGMENT'] == seg])
        final_metrics[f'segment_{seg}_total'] = seg_total
        final_metrics[f'segment_{seg}_assigned'] = seg_assigned
    
    # Add banker type metrics
    for bt in ['RM', 'RC', 'SBB']:
        bt_count = len(assigned_hh[assigned_hh['ASSIGNED_BANKER_TYPE'] == bt])
        final_metrics[f'assigned_to_{bt.lower()}'] = bt_count
    
    return final_metrics


# ==================== OUTPUT GENERATION ====================

def generate_outputs(data, metrics, output_dir='output'):
    """Generate all output files with columns similar to original customer_assignment.csv"""
    print_section("GENERATING OUTPUT FILES")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Main HH Assignment File (with enriched columns like original)
    # EXCLUDE SBB assignments - they go to separate file
    hh_assignments = data['hh_assignments'].copy()
    
    # Get assigned households (excluding SBB)
    assigned_hh = hh_assignments[
        (hh_assignments['IS_ASSIGNED'] == True) & 
        (hh_assignments['ASSIGNED_BANKER_TYPE'] != 'SBB')
    ].copy()
    
    # Merge with banker data for RM/RC bankers
    banker_data = data['banker'].copy()
    branch_data = data['branch'].copy()
    sbb_data = data['sbb'].copy()
    
    # Add branch coordinates to banker data
    banker_with_coords = banker_data.merge(
        branch_data[['AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM', 'NAME', 'CITY', 'POSTALCODE']],
        on='AU',
        how='left'
    )
    banker_with_coords = banker_with_coords.rename(columns={
        'NAME': 'BRANCH_NAME',
        'BRANCH_LAT_NUM': 'BANKER_LAT',
        'BRANCH_LON_NUM': 'BANKER_LON'
    })
    
    # Merge assigned HHs with banker info
    assigned_enriched = assigned_hh.merge(
        banker_with_coords[['PORT_CODE', 'EID', 'EMPLOYEE_NAME', 'AU', 'BRANCH_NAME',
                            'ROLE_TYPE', 'BANKER_LAT', 'BANKER_LON', 
                            'MANAGER_NAME', 'DIRECTOR_NAME', 'COVERAGE', 'BANKER_TYPE']],
        left_on='ASSIGNED_PORT_CODE',
        right_on='PORT_CODE',
        how='left'
    )
    
    # For SBB bankers, merge separately
    sbb_assigned = assigned_enriched[assigned_enriched['ASSIGNED_BANKER_TYPE'] == 'SBB'].copy()
    non_sbb_assigned = assigned_enriched[assigned_enriched['ASSIGNED_BANKER_TYPE'] != 'SBB'].copy()
    
    if len(sbb_assigned) > 0:
        # Get SBB details
        sbb_details = sbb_data[['AU', 'Full Name', 'Manager', 'Director', 'State', 
                                 'BRANCH NAME', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].copy()
        sbb_details = sbb_details.rename(columns={
            'Full Name': 'EMPLOYEE_NAME_SBB',
            'Manager': 'MANAGER_NAME_SBB',
            'Director': 'DIRECTOR_NAME_SBB',
            'BRANCH NAME': 'BRANCH_NAME_SBB',
            'BRANCH_LAT_NUM': 'BANKER_LAT_SBB',
            'BRANCH_LON_NUM': 'BANKER_LON_SBB'
        })
        
        sbb_assigned = sbb_assigned.merge(
            sbb_details,
            left_on='ASSIGNED_PORT_CODE',
            right_on='AU',
            how='left',
            suffixes=('', '_sbb')
        )
        
        # Fill in banker columns from SBB data
        sbb_assigned['EMPLOYEE_NAME'] = sbb_assigned['EMPLOYEE_NAME'].fillna(sbb_assigned['EMPLOYEE_NAME_SBB'])
        sbb_assigned['MANAGER_NAME'] = sbb_assigned['MANAGER_NAME'].fillna(sbb_assigned['MANAGER_NAME_SBB'])
        sbb_assigned['DIRECTOR_NAME'] = sbb_assigned['DIRECTOR_NAME'].fillna(sbb_assigned['DIRECTOR_NAME_SBB'])
        sbb_assigned['BRANCH_NAME'] = sbb_assigned['BRANCH_NAME'].fillna(sbb_assigned['BRANCH_NAME_SBB'])
        sbb_assigned['BANKER_LAT'] = sbb_assigned['BANKER_LAT'].fillna(sbb_assigned['BANKER_LAT_SBB'])
        sbb_assigned['BANKER_LON'] = sbb_assigned['BANKER_LON'].fillna(sbb_assigned['BANKER_LON_SBB'])
        sbb_assigned['AU'] = sbb_assigned['ASSIGNED_PORT_CODE']
        sbb_assigned['ROLE_TYPE'] = 'SBB'
        
        # Drop temporary columns
        sbb_cols_to_drop = [col for col in sbb_assigned.columns if col.endswith('_SBB') or col.endswith('_sbb')]
        sbb_assigned = sbb_assigned.drop(columns=sbb_cols_to_drop, errors='ignore')
        
        # Combine back
        assigned_enriched = pd.concat([non_sbb_assigned, sbb_assigned], ignore_index=True)
    
    # Calculate distance
    assigned_enriched['DISTANCE_MILES'] = assigned_enriched.apply(
        lambda row: haversine_distance(
            row['LAT_NUM'], row['LON_NUM'],
            row['BANKER_LAT'], row['BANKER_LON']
        ) if pd.notna(row['BANKER_LAT']) and pd.notna(row['BANKER_LON']) else None,
        axis=1
    )
    
    # Add proximity limit based on role type
    def get_proximity_limit(row):
        if row['ASSIGNED_BANKER_TYPE'] == 'SBB':
            return 10
        elif row.get('ROLE_TYPE') == 'IN MARKET' or row['ASSIGNED_BANKER_TYPE'] == 'RM':
            return 40
        else:  # CENTRALIZED / RC
            return 200
    
    assigned_enriched['PROXIMITY_LIMIT_MILES'] = assigned_enriched.apply(get_proximity_limit, axis=1)
    assigned_enriched['IS_WITHIN_PROXIMITY'] = assigned_enriched['DISTANCE_MILES'] <= assigned_enriched['PROXIMITY_LIMIT_MILES']
    
    # Add exception flag for expanded radius
    def get_exception_flag(row):
        if pd.isna(row['DISTANCE_MILES']):
            return None
        if row['ASSIGNED_BANKER_TYPE'] == 'RM' and row['DISTANCE_MILES'] > 40:
            return f"EXPANDED_RADIUS_{int(row['DISTANCE_MILES'])}MI"
        elif row['ASSIGNED_BANKER_TYPE'] == 'RC' and row['DISTANCE_MILES'] > 200:
            return f"EXPANDED_RADIUS_{int(row['DISTANCE_MILES'])}MI"
        elif row['ASSIGNED_BANKER_TYPE'] == 'SBB' and row['DISTANCE_MILES'] > 10:
            return f"EXPANDED_RADIUS_{int(row['DISTANCE_MILES'])}MI"
        return None
    
    assigned_enriched['EXCEPTION_FLAG'] = assigned_enriched.apply(get_exception_flag, axis=1)
    
    # Add timestamp
    assigned_enriched['ASSIGNMENT_TIMESTAMP'] = datetime.now()
    
    # Calculate size_reach (1 if banker met minimum, 0 otherwise)
    # This requires knowing retained counts per banker - simplified version
    assigned_enriched['size_reach'] = 1  # Default to 1, can be enhanced
    
    # Rename columns to match original format
    output_df = assigned_enriched.rename(columns={
        'LAT_NUM': 'CUSTOMER_LAT',
        'LON_NUM': 'CUSTOMER_LON',
        'ASSIGNED_PORT_CODE': 'ASSIGNED_PORT_CODE',
        'ASSIGNED_EID': 'ASSIGNED_BANKER_EID',
        'ASSIGNED_BANKER_NAME': 'ASSIGNED_BANKER_NAME',
        'AU': 'ASSIGNED_BANKER_AU',
        'BRANCH_NAME': 'ASSIGNED_BANKER_BRANCH_NAME',
        'BANKER_LAT': 'BANKER_LAT',
        'BANKER_LON': 'BANKER_LON',
        'ROLE_TYPE': 'BANKER_ROLE_TYPE',
        'MANAGER_NAME': 'BANKER_MANAGER_NAME',
        'DIRECTOR_NAME': 'BANKER_DIRECTOR_NAME',
        'COVERAGE': 'BANKER_COVERAGE',
        'ASSIGNMENT_STEP': 'ASSIGNMENT_PHASE'
    })
    
    # Select and order columns
    output_cols = [
        'HH_ECN',
        'ASSIGNED_PORT_CODE',
        'ASSIGNED_BANKER_EID',
        'ASSIGNED_BANKER_NAME',
        'ASSIGNED_BANKER_AU',
        'ASSIGNED_BANKER_BRANCH_NAME',
        'ASSIGNED_BANKER_TYPE',
        'DISTANCE_MILES',
        'CUSTOMER_LAT',
        'CUSTOMER_LON',
        'BANKER_LAT',
        'BANKER_LON',
        'BANKER_ROLE_TYPE',
        'PROXIMITY_LIMIT_MILES',
        'ASSIGNMENT_PHASE',
        'ASSIGNMENT_REASON',
        'IS_WITHIN_PROXIMITY',
        'EXCEPTION_FLAG',
        'COORDS_IMPUTED',
        'NEW_SEGMENT',
        'PATR_AU_STR',
        'BANKER_MANAGER_NAME',
        'BANKER_DIRECTOR_NAME',
        'BANKER_COVERAGE',
        'ASSIGNMENT_TIMESTAMP',
        'size_reach'
    ]
    
    # Keep only existing columns
    output_cols = [col for col in output_cols if col in output_df.columns]
    output_df = output_df[output_cols]
    
    output_df.to_csv(f"{output_dir}/hh_assignments.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/hh_assignments.csv ({len(output_df)} records)")
    
    # 2. SBB Mapping File
    sbb_mappings = data.get('sbb_mappings', [])
    if len(sbb_mappings) > 0:
        sbb_df = pd.DataFrame(sbb_mappings)
        sbb_output = sbb_df[['ECN', 'AU', 'Full Name']].copy()
        sbb_output.to_csv(f"{output_dir}/SBB_MAPPING.csv", index=False)
        print(f"  ✓ Saved: {output_dir}/SBB_MAPPING.csv ({len(sbb_output)} records)")
    else:
        print(f"  ⚠ No SBB mappings to save")
    
    # 3. Unassigned Households (excluding those assigned to SBB)
    unassigned = hh_assignments[
        (hh_assignments['IS_ASSIGNED'] == False) | 
        (hh_assignments['ASSIGNED_BANKER_TYPE'] == 'SBB')
    ].copy()
    unassigned_only = hh_assignments[hh_assignments['IS_ASSIGNED'] == False].copy()
    if len(unassigned_only) > 0:
        unassigned_only = unassigned_only.rename(columns={
            'LAT_NUM': 'CUSTOMER_LAT',
            'LON_NUM': 'CUSTOMER_LON'
        })
        unassigned_only.to_csv(f"{output_dir}/unassigned_households.csv", index=False)
        print(f"  ✓ Saved: {output_dir}/unassigned_households.csv ({len(unassigned_only)} records)")
    
    # 4. ECN-level Assignment File (with all ECNs for each HH)
    ecn_assignments = data['hh_to_ecn'].merge(
        hh_assignments[['HH_ECN', 'ASSIGNED_PORT_CODE', 'ASSIGNED_EID', 
                        'ASSIGNED_BANKER_NAME', 'ASSIGNED_BANKER_TYPE',
                        'ASSIGNMENT_STEP', 'ASSIGNMENT_REASON', 'IS_ASSIGNED']],
        on='HH_ECN',
        how='left'
    )
    ecn_assignments.to_csv(f"{output_dir}/ecn_assignments.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/ecn_assignments.csv ({len(ecn_assignments)} records)")
    
    # 5. Banker Summary
    banker_summary = output_df.groupby(['ASSIGNED_PORT_CODE', 'ASSIGNED_BANKER_EID', 
                                        'ASSIGNED_BANKER_NAME', 'ASSIGNED_BANKER_TYPE']).agg({
        'HH_ECN': 'count',
        'NEW_SEGMENT': lambda x: dict(Counter(x)),
        'DISTANCE_MILES': ['mean', 'min', 'max']
    }).reset_index()
    banker_summary.columns = ['PORT_CODE', 'EID', 'BANKER_NAME', 'BANKER_TYPE', 
                              'HH_COUNT', 'SEGMENT_BREAKDOWN', 
                              'AVG_DISTANCE_MILES', 'MIN_DISTANCE_MILES', 'MAX_DISTANCE_MILES']
    banker_summary.to_csv(f"{output_dir}/banker_summary.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/banker_summary.csv ({len(banker_summary)} records)")
    
    # 6. Metrics Summary
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/metrics_summary.csv")
    
    return {
        'hh_assignments': f"{output_dir}/hh_assignments.csv",
        'sbb_mapping': f"{output_dir}/SBB_MAPPING.csv",
        'ecn_assignments': f"{output_dir}/ecn_assignments.csv",
        'banker_summary': f"{output_dir}/banker_summary.csv",
        'metrics_summary': f"{output_dir}/metrics_summary.csv"
    }


# ==================== MAIN ORCHESTRATOR ====================

def run_household_migration(data_dir='', output_dir='output'):
    """Main function to run the complete household migration process"""
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print(" HOUSEHOLD PORTFOLIO MIGRATION SYSTEM")
    print(" " + "="*78)
    print(f" Started: {start_time}")
    print("="*80)
    
    # Load data
    data = load_all_data(data_dir)
    
    # Prepare data
    data = prepare_data(data)
    
    # Step 1: Deduplicate client groups
    data = deduplicate_client_groups(data)
    
    # Build mappings
    data = build_ecn_to_banker_mapping(data)
    data = build_hh_to_current_banker_mapping(data)
    
    # Initialize assignment tracking
    data = initialize_assignment_tracking(data)
    
    # Collect metrics
    all_metrics = {}
    
    # Step 5: Segment 4 → RC retention
    data, m5 = step5_segment4_rc_retention(data)
    all_metrics.update(m5)
    
    # Step 6: Segment 3 → RM retention
    data, m6 = step6_segment3_rm_retention(data)
    all_metrics.update(m6)
    
    # Step 7: Households to retain
    data, m7 = step7_households_to_retain(data)
    all_metrics.update(m7)
    
    # Step 8: Segment 2 → SBB (existing books)
    data, m8 = step8_segment2_sbb_existing(data)
    all_metrics.update(m8)
    
    # Step 9: Remaining Segment 2 → RM/RC
    data, m9 = step9_segment2_rm_rc_retention(data)
    all_metrics.update(m9)
    
    # Step 10: Verify Segment 2 placement
    data, m10 = step10_verify_segment2_placement(data)
    all_metrics.update(m10)
    
    # Step 11: Spatial assignment
    data, m11 = step11_spatial_assignment(data)
    all_metrics.update(m11)
    
    # Step 12: Generate metrics
    final_metrics = step12_generate_metrics(data, all_metrics)
    
    # Generate outputs
    output_files = generate_outputs(data, final_metrics, output_dir)
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_section("MIGRATION COMPLETE")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Output directory: {output_dir}/")
    print(f"\n  Output files:")
    for name, path in output_files.items():
        print(f"    • {name}: {path}")
    
    return data, final_metrics, output_files


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    data, metrics, outputs = run_household_migration(
        data_dir='',  # Set to your data directory
        output_dir='output'
    )
