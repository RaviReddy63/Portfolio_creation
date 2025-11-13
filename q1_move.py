"""
Customer-Banker Assignment System
Assigns available customers to banker portfolios based on proximity and capacity constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')


# ==================== DISTANCE CALCULATION ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on earth in miles
    
    Args:
        lat1, lon1: Latitude and longitude of point 1
        lat2, lon2: Latitude and longitude of point 2
    
    Returns:
        Distance in miles
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in miles
    miles = 3959 * c
    return miles


def calculate_distance_matrix(bankers_df, customers_df, role_type, proximity_limit):
    """
    Calculate eligible customers for each banker based on proximity
    
    Args:
        bankers_df: DataFrame with banker information
        customers_df: DataFrame with customer information
        role_type: 'IN MARKET' or 'CENTRALIZED'
        proximity_limit: Maximum distance in miles (20 for IN MARKET, 200 for CENTRALIZED)
    
    Returns:
        Dictionary: {PORT_CODE: [(customer_id, distance), ...]} sorted by distance
    """
    eligible_customers = {}
    
    for _, banker in bankers_df.iterrows():
        port_code = banker['PORT_CODE']
        banker_lat = banker['BANKER_LAT_NUM']
        banker_lon = banker['BANKER_LON_NUM']
        
        customer_distances = []
        
        for _, customer in customers_df.iterrows():
            cust_lat = customer['LAT_NUM']
            cust_lon = customer['LON_NUM']
            
            # Calculate distance
            distance = haversine_distance(banker_lat, banker_lon, cust_lat, cust_lon)
            
            # Check if within proximity limit
            if distance <= proximity_limit:
                customer_distances.append((customer['HH_ECN'], distance))
        
        # Sort by distance (closest first)
        customer_distances.sort(key=lambda x: x[1])
        eligible_customers[port_code] = customer_distances
    
    return eligible_customers


# ==================== DATA PREPARATION ====================

def load_and_prepare_data(banker_file, req_custs_file, available_custs_file):
    """
    Load and prepare all input data files
    
    Returns:
        Tuple: (bankers_df, customers_df, original_banker_data, original_customers)
    """
    print("Loading data files...")
    
    # Load data
    banker_data = pd.read_csv(banker_file)
    req_custs = pd.read_csv(req_custs_file)
    available_custs = pd.read_csv(available_custs_file)
    
    # Merge banker data with requirements
    bankers_df = banker_data.merge(req_custs, on='PORT_CODE', how='inner')
    
    # Add tracking columns to bankers
    bankers_df['CURRENT_ASSIGNED'] = 0
    bankers_df['REMAINING_MIN'] = bankers_df['MIN_COUNT_REQ']
    bankers_df['REMAINING_MAX'] = bankers_df['MAX_COUNT_REQ']
    
    # Add tracking columns to customers
    available_custs['IS_ASSIGNED'] = False
    available_custs['ASSIGNED_TO_PORT_CODE'] = None
    available_custs['ASSIGNED_BANKER_EID'] = None
    available_custs['DISTANCE_MILES'] = None
    available_custs['ASSIGNMENT_PHASE'] = None
    available_custs['EXCEPTION_FLAG'] = None
    
    print(f"Loaded {len(bankers_df)} bankers and {len(available_custs)} available customers")
    
    return bankers_df, available_custs, banker_data, available_custs.copy()


def separate_bankers_by_type(bankers_df):
    """
    Separate bankers into IN MARKET and CENTRALIZED
    
    Returns:
        Tuple: (in_market_df, centralized_df)
    """
    in_market = bankers_df[bankers_df['ROLE_TYPE'] == 'IN MARKET'].copy()
    centralized = bankers_df[bankers_df['ROLE_TYPE'] == 'CENTRALIZED'].copy()
    
    print(f"IN MARKET bankers: {len(in_market)}")
    print(f"CENTRALIZED bankers: {len(centralized)}")
    
    return in_market, centralized


def prioritize_bankers(bankers_df, eligible_customers_dict):
    """
    Sort bankers by priority (most urgent needs first)
    
    Args:
        bankers_df: Banker DataFrame
        eligible_customers_dict: Dictionary of eligible customers per banker
    
    Returns:
        Sorted DataFrame
    """
    # Add number of eligible customers
    bankers_df['NUM_ELIGIBLE'] = bankers_df['PORT_CODE'].apply(
        lambda x: len(eligible_customers_dict.get(x, []))
    )
    
    # Sort by REMAINING_MIN (descending) then NUM_ELIGIBLE (ascending)
    sorted_bankers = bankers_df.sort_values(
        by=['REMAINING_MIN', 'NUM_ELIGIBLE'],
        ascending=[False, True]
    ).copy()
    
    return sorted_bankers


# ==================== ASSIGNMENT LOGIC ====================

def assign_customers_to_banker(banker_row, customers_df, eligible_customers, 
                               phase, min_or_max='MIN', proximity_limit=20):
    """
    Assign customers to a single banker
    
    Args:
        banker_row: Series containing banker information
        customers_df: DataFrame of all customers
        eligible_customers: List of (customer_id, distance) tuples for this banker
        phase: Assignment phase name (e.g., 'IN_MARKET_MIN')
        min_or_max: 'MIN' to fill minimum, 'MAX' to fill up to maximum
        proximity_limit: Base proximity limit for this banker type
    
    Returns:
        Tuple: (assignments_list, updated_customers_df, exceptions_list)
    """
    port_code = banker_row['PORT_CODE']
    banker_eid = banker_row['EID']
    
    assignments = []
    exceptions = []
    
    # Determine target count
    if min_or_max == 'MIN':
        target_count = int(banker_row['REMAINING_MIN'])
    else:
        target_count = int(banker_row['REMAINING_MAX'])
    
    if target_count <= 0:
        return assignments, customers_df, exceptions
    
    # Get unassigned eligible customers
    unassigned_eligible = []
    for cust_id, distance in eligible_customers:
        if not customers_df.loc[customers_df['HH_ECN'] == cust_id, 'IS_ASSIGNED'].values[0]:
            unassigned_eligible.append((cust_id, distance))
    
    assigned_count = 0
    
    # First pass: Assign customers within proximity
    for cust_id, distance in unassigned_eligible:
        if assigned_count >= target_count:
            break
        
        # Assign customer
        idx = customers_df[customers_df['HH_ECN'] == cust_id].index[0]
        customers_df.at[idx, 'IS_ASSIGNED'] = True
        customers_df.at[idx, 'ASSIGNED_TO_PORT_CODE'] = port_code
        customers_df.at[idx, 'ASSIGNED_BANKER_EID'] = banker_eid
        customers_df.at[idx, 'DISTANCE_MILES'] = distance
        customers_df.at[idx, 'ASSIGNMENT_PHASE'] = phase
        
        # Flag if this is an expanded radius assignment (e.g., IN_MARKET_40MILE or CENTRALIZED_400MILE)
        if '40MILE' in phase and distance > 20:
            customers_df.at[idx, 'EXCEPTION_FLAG'] = f'EXPANDED_RADIUS_20_TO_40_MILES'
        elif '400MILE' in phase and distance > 200:
            customers_df.at[idx, 'EXCEPTION_FLAG'] = f'EXPANDED_RADIUS_200_TO_400_MILES'
        
        assignments.append({
            'HH_ECN': cust_id,
            'PORT_CODE': port_code,
            'DISTANCE': distance,
            'PHASE': phase
        })
        
        assigned_count += 1
    
    # If minimum not met and this is MIN phase, try expanding radius
    if min_or_max == 'MIN' and assigned_count < target_count:
        exception_msg = f"Could not meet minimum. Assigned {assigned_count} of {target_count} required."
        exceptions.append({
            'PORT_CODE': port_code,
            'BANKER_EID': banker_eid,
            'EXCEPTION_TYPE': 'MINIMUM_NOT_MET',
            'ASSIGNED_COUNT': assigned_count,
            'REQUIRED_COUNT': target_count,
            'MESSAGE': exception_msg
        })
    
    return assignments, customers_df, exceptions


def process_banker_assignments(bankers_df, customers_df, eligible_customers_dict, 
                               phase_name, banker_type, proximity_limit):
    """
    Process assignments for a group of bankers (MIN then MAX phases)
    
    Args:
        bankers_df: DataFrame of bankers to process
        customers_df: DataFrame of customers
        eligible_customers_dict: Eligible customers per banker
        phase_name: Base phase name (e.g., 'IN_MARKET')
        banker_type: 'IN MARKET' or 'CENTRALIZED'
        proximity_limit: Distance limit for this banker type
    
    Returns:
        Tuple: (updated_bankers_df, updated_customers_df, all_assignments, all_exceptions)
    """
    all_assignments = []
    all_exceptions = []
    
    print(f"\n{'='*60}")
    print(f"Processing {phase_name} Bankers - MINIMUM Requirements")
    print(f"{'='*60}")
    
    # Prioritize bankers
    sorted_bankers = prioritize_bankers(bankers_df, eligible_customers_dict)
    
    # PHASE 1: Meet minimum requirements
    for idx, banker in sorted_bankers.iterrows():
        port_code = banker['PORT_CODE']
        banker_name = banker['EMPLOYEE_NAME']
        
        if banker['REMAINING_MIN'] <= 0:
            continue
        
        eligible_custs = eligible_customers_dict.get(port_code, [])
        
        print(f"  Banker: {banker_name} (Port: {port_code}) - Need: {banker['REMAINING_MIN']}, Eligible: {len(eligible_custs)}")
        
        assignments, customers_df, exceptions = assign_customers_to_banker(
            banker, customers_df, eligible_custs, 
            f"{phase_name}_MIN", 'MIN', proximity_limit
        )
        
        # Update banker's tracking
        assigned_count = len(assignments)
        bankers_df.at[idx, 'CURRENT_ASSIGNED'] += assigned_count
        bankers_df.at[idx, 'REMAINING_MIN'] = max(0, banker['REMAINING_MIN'] - assigned_count)
        bankers_df.at[idx, 'REMAINING_MAX'] = max(0, banker['REMAINING_MAX'] - assigned_count)
        
        all_assignments.extend(assignments)
        all_exceptions.extend(exceptions)
        
        print(f"    ✓ Assigned: {assigned_count}, Remaining MIN: {bankers_df.at[idx, 'REMAINING_MIN']}")
    
    print(f"\n{'='*60}")
    print(f"Processing {phase_name} Bankers - MAXIMUM Capacity")
    print(f"{'='*60}")
    
    # PHASE 2: Fill up to maximum
    for idx, banker in sorted_bankers.iterrows():
        port_code = banker['PORT_CODE']
        banker_name = banker['EMPLOYEE_NAME']
        
        if bankers_df.at[idx, 'REMAINING_MAX'] <= 0:
            continue
        
        eligible_custs = eligible_customers_dict.get(port_code, [])
        
        print(f"  Banker: {banker_name} (Port: {port_code}) - Can add: {bankers_df.at[idx, 'REMAINING_MAX']}")
        
        assignments, customers_df, exceptions = assign_customers_to_banker(
            bankers_df.loc[idx], customers_df, eligible_custs, 
            f"{phase_name}_MAX", 'MAX', proximity_limit
        )
        
        # Update banker's tracking
        assigned_count = len(assignments)
        bankers_df.at[idx, 'CURRENT_ASSIGNED'] += assigned_count
        bankers_df.at[idx, 'REMAINING_MAX'] = max(0, bankers_df.at[idx, 'REMAINING_MAX'] - assigned_count)
        
        all_assignments.extend(assignments)
        
        print(f"    ✓ Assigned: {assigned_count}, Remaining MAX: {bankers_df.at[idx, 'REMAINING_MAX']}")
    
    return bankers_df, customers_df, all_assignments, all_exceptions


# ==================== REASSIGNMENT LOGIC ====================

def reassign_from_overcapacity_bankers(struggling_bankers_df, all_bankers_df, customers_df, 
                                       max_distance, min_portfolio_size=280):
    """
    Reassign customers from over-capacity bankers to struggling bankers
    Only takes from 20-mile IN MARKET assignments
    Source banker must maintain >= min_portfolio_size total customers
    
    Args:
        struggling_bankers_df: Bankers who still haven't met minimum
        all_bankers_df: All IN MARKET bankers
        customers_df: Customer DataFrame
        max_distance: Maximum distance in miles for reassignment
        min_portfolio_size: Minimum total portfolio size (default 280)
    
    Returns:
        Tuple: (updated_bankers_df, updated_customers_df, reassignments_list)
    """
    reassignments = []
    
    print(f"\n{'='*60}")
    print(f"Reassigning from Over-Capacity Bankers (within {max_distance} miles)")
    print(f"{'='*60}")
    
    if len(struggling_bankers_df) == 0:
        print("✓ No struggling bankers - all met minimum requirements")
        return all_bankers_df, customers_df, reassignments
    
    # Get customers from 20-mile IN MARKET assignments only
    reassignable_customers = customers_df[
        (customers_df['IS_ASSIGNED'] == True) & 
        (customers_df['ASSIGNMENT_PHASE'].str.contains('IN_MARKET_MIN|IN_MARKET_MAX', na=False)) &
        (~customers_df['ASSIGNMENT_PHASE'].str.contains('40MILE', na=False))
    ].copy()
    
    print(f"Found {len(reassignable_customers)} customers in 20-mile IN MARKET assignments")
    print(f"Attempting to help {len(struggling_bankers_df)} struggling bankers\n")
    
    for idx, struggling_banker in struggling_bankers_df.iterrows():
        port_code = struggling_banker['PORT_CODE']
        banker_name = struggling_banker['EMPLOYEE_NAME']
        needs = int(struggling_banker['REMAINING_MIN'])
        
        if needs <= 0:
            continue
        
        print(f"Banker: {banker_name} (Port: {port_code}) - Needs: {needs} more customers")
        
        # Calculate distance from struggling banker to all reassignable customers
        banker_lat = struggling_banker['BANKER_LAT_NUM']
        banker_lon = struggling_banker['BANKER_LON_NUM']
        
        candidate_customers = []
        
        for _, customer in reassignable_customers.iterrows():
            if not customer['IS_ASSIGNED']:  # Skip if already reassigned
                continue
                
            source_port = customer['ASSIGNED_TO_PORT_CODE']
            
            # Don't take from self
            if source_port == port_code:
                continue
            
            # Get source banker info
            source_banker = all_bankers_df[all_bankers_df['PORT_CODE'] == source_port]
            if len(source_banker) == 0:
                continue
            
            source_banker = source_banker.iloc[0]
            source_total = source_banker['CURR_COUNT'] + source_banker['CURRENT_ASSIGNED']
            
            # Check if source banker can spare this customer (stay >= 280)
            if source_total - 1 < min_portfolio_size:
                continue
            
            # Calculate distance to struggling banker
            distance = haversine_distance(
                banker_lat, banker_lon,
                customer['LAT_NUM'], customer['LON_NUM']
            )
            
            # Check if within max_distance
            if distance > max_distance:
                continue
            
            candidate_customers.append({
                'HH_ECN': customer['HH_ECN'],
                'SOURCE_PORT': source_port,
                'SOURCE_BANKER_NAME': source_banker['EMPLOYEE_NAME'],
                'SOURCE_TOTAL': source_total,
                'DISTANCE': distance
            })
        
        # Sort by distance (closest first)
        candidate_customers.sort(key=lambda x: x['DISTANCE'])
        
        print(f"  Found {len(candidate_customers)} candidate customers for reassignment")
        
        reassigned_count = 0
        
        for candidate in candidate_customers:
            if reassigned_count >= needs:
                break
            
            cust_id = candidate['HH_ECN']
            source_port = candidate['SOURCE_PORT']
            distance = candidate['DISTANCE']
            
            # Double-check source banker can still spare (in case multiple taken)
            source_banker_idx = all_bankers_df[all_bankers_df['PORT_CODE'] == source_port].index[0]
            source_banker = all_bankers_df.loc[source_banker_idx]
            source_total = source_banker['CURR_COUNT'] + source_banker['CURRENT_ASSIGNED']
            
            if source_total - 1 < min_portfolio_size:
                continue
            
            # Perform reassignment
            cust_idx = customers_df[customers_df['HH_ECN'] == cust_id].index[0]
            
            # Update customer assignment
            old_phase = customers_df.at[cust_idx, 'ASSIGNMENT_PHASE']
            customers_df.at[cust_idx, 'ASSIGNED_TO_PORT_CODE'] = port_code
            customers_df.at[cust_idx, 'ASSIGNED_BANKER_EID'] = struggling_banker['EID']
            customers_df.at[cust_idx, 'DISTANCE_MILES'] = distance
            customers_df.at[cust_idx, 'ASSIGNMENT_PHASE'] = f'IN_MARKET_REASSIGNED_{max_distance}MILE'
            customers_df.at[cust_idx, 'EXCEPTION_FLAG'] = f'REASSIGNED_FROM_{source_port}_WITHIN_{max_distance}MILES'
            
            # Update source banker (decrement)
            all_bankers_df.at[source_banker_idx, 'CURRENT_ASSIGNED'] -= 1
            all_bankers_df.at[source_banker_idx, 'REMAINING_MAX'] += 1
            
            # Update struggling banker (increment)
            struggling_idx = all_bankers_df[all_bankers_df['PORT_CODE'] == port_code].index[0]
            all_bankers_df.at[struggling_idx, 'CURRENT_ASSIGNED'] += 1
            all_bankers_df.at[struggling_idx, 'REMAINING_MIN'] = max(0, all_bankers_df.at[struggling_idx, 'REMAINING_MIN'] - 1)
            all_bankers_df.at[struggling_idx, 'REMAINING_MAX'] = max(0, all_bankers_df.at[struggling_idx, 'REMAINING_MAX'] - 1)
            
            reassignments.append({
                'HH_ECN': cust_id,
                'FROM_PORT': source_port,
                'TO_PORT': port_code,
                'DISTANCE': distance,
                'FROM_BANKER': candidate['SOURCE_BANKER_NAME'],
                'TO_BANKER': banker_name
            })
            
            reassigned_count += 1
        
        if reassigned_count > 0:
            print(f"  ✓ Reassigned {reassigned_count} customers to {banker_name}")
            remaining = int(all_bankers_df[all_bankers_df['PORT_CODE'] == port_code]['REMAINING_MIN'].values[0])
            if remaining > 0:
                print(f"    ⚠ Still needs {remaining} more customers")
            else:
                print(f"    ✓ Minimum requirement now met!")
        else:
            print(f"  ✗ Could not find any reassignable customers")
    
    return all_bankers_df, customers_df, reassignments


# ==================== OUTPUT GENERATION ====================

def generate_customer_assignment_file(customers_df, bankers_df_full, output_path):
    """
    Generate the main customer assignment output file
    """
    print("\nGenerating customer assignment file...")
    
    # Get only assigned customers
    assigned = customers_df[customers_df['IS_ASSIGNED'] == True].copy()
    
    # Merge with banker details
    banker_cols = ['PORT_CODE', 'EID', 'EMPLOYEE_NAME', 'AU', 'BRANCH_NAME', 
                   'ROLE_TYPE', 'BANKER_LAT_NUM', 'BANKER_LON_NUM', 
                   'MANAGER_NAME', 'DIRECTOR_NAME', 'COVERAGE', 
                   'BRANCH_LOCATION_CODE']
    
    result = assigned.merge(
        bankers_df_full[banker_cols],
        left_on='ASSIGNED_TO_PORT_CODE',
        right_on='PORT_CODE',
        how='left'
    )
    
    # Calculate proximity compliance
    result['PROXIMITY_LIMIT_MILES'] = result['ROLE_TYPE'].apply(
        lambda x: 20 if x == 'IN MARKET' else 200
    )
    result['IS_WITHIN_PROXIMITY'] = result['DISTANCE_MILES'] <= result['PROXIMITY_LIMIT_MILES']
    result['ASSIGNMENT_TIMESTAMP'] = datetime.now()
    
    # Select and rename columns for output
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
        'ASSIGNMENT_TIMESTAMP': 'ASSIGNMENT_TIMESTAMP'
    }
    
    output_df = result.rename(columns=output_cols)
    output_df = output_df[[col for col in output_cols.values() if col in output_df.columns]]
    
    output_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(output_df)} records)")
    
    return output_df


def generate_banker_summary_file(bankers_df_full, customers_df, output_path):
    """
    Generate banker summary statistics file
    """
    print("\nGenerating banker summary file...")
    
    summary_data = []
    
    for _, banker in bankers_df_full.iterrows():
        port_code = banker['PORT_CODE']
        
        # Get assigned customers for this banker
        assigned_custs = customers_df[
            (customers_df['ASSIGNED_TO_PORT_CODE'] == port_code) & 
            (customers_df['IS_ASSIGNED'] == True)
        ]
        
        newly_assigned = len(assigned_custs)
        final_total = banker['CURR_COUNT'] + newly_assigned
        
        # Distance statistics
        if newly_assigned > 0:
            distances = assigned_custs['DISTANCE_MILES'].dropna()
            avg_distance = distances.mean()
            min_distance = distances.min()
            max_distance = distances.max()
            median_distance = distances.median()
            std_distance = distances.std()
            
            # Business metrics
            total_deposits = assigned_custs['DEPOSIT_BAL'].sum()
            total_revenue = assigned_custs['BANK_REVENUE'].sum()
            total_sales = assigned_custs['CG_GROSS_SALES'].sum()
            
            # Geographic diversity
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
            'NUM_UNIQUE_ZIPCODES': num_zips,
            'HAS_EXCEPTIONS': False,  # Will be updated if exceptions exist
            'EXCEPTION_COUNT': 0,
            'EXCEPTION_DETAILS': None
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(summary_df)} records)")
    
    return summary_df


def generate_unassigned_customers_file(customers_df, bankers_df_full, output_path):
    """
    Generate file with customers that were not assigned
    """
    print("\nGenerating unassigned customers file...")
    
    unassigned = customers_df[customers_df['IS_ASSIGNED'] == False].copy()
    
    if len(unassigned) == 0:
        print("✓ All customers assigned successfully!")
        return None
    
    # Find closest banker for each unassigned customer
    unassigned_data = []
    
    for _, customer in unassigned.iterrows():
        cust_lat = customer['LAT_NUM']
        cust_lon = customer['LON_NUM']
        
        # Calculate distance to all bankers
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
        
        # Sort by distance
        banker_distances.sort(key=lambda x: x['DISTANCE'])
        
        closest = banker_distances[0] if banker_distances else None
        second_closest = banker_distances[1] if len(banker_distances) > 1 else None
        
        # Determine reason for non-assignment
        reason = "UNKNOWN"
        if closest:
            if closest['ROLE_TYPE'] == 'IN MARKET' and closest['DISTANCE'] > 20:
                reason = "NO_BANKER_WITHIN_PROXIMITY"
            elif closest['ROLE_TYPE'] == 'CENTRALIZED' and closest['DISTANCE'] > 200:
                reason = "NO_BANKER_WITHIN_PROXIMITY"
            else:
                reason = "ALL_ELIGIBLE_BANKERS_AT_MAX_CAPACITY"
        
        unassigned_data.append({
            'HH_ECN': customer['HH_ECN'],
            'NEW_SEGMENT': customer['NEW_SEGMENT'],
            'BILLINGCITY': customer['BILLINGCITY'],
            'BILLINGSTATE': customer['BILLINGSTATE'],
            'BILLINGPOSTALCODE': customer['BILLINGPOSTALCODE'],
            'LAT_NUM': customer['LAT_NUM'],
            'LON_NUM': customer['LON_NUM'],
            'DEPOSIT_BAL': customer['DEPOSIT_BAL'],
            'CG_GROSS_SALES': customer['CG_GROSS_SALES'],
            'BANK_REVENUE': customer['BANK_REVENUE'],
            'REASON': reason,
            'CLOSEST_BANKER_PORT_CODE': closest['PORT_CODE'] if closest else None,
            'CLOSEST_BANKER_NAME': closest['EMPLOYEE_NAME'] if closest else None,
            'CLOSEST_BANKER_ROLE_TYPE': closest['ROLE_TYPE'] if closest else None,
            'DISTANCE_TO_CLOSEST_BANKER_MILES': round(closest['DISTANCE'], 2) if closest else None,
            'SECOND_CLOSEST_BANKER_PORT_CODE': second_closest['PORT_CODE'] if second_closest else None,
            'DISTANCE_TO_SECOND_CLOSEST_MILES': round(second_closest['DISTANCE'], 2) if second_closest else None,
            'NUM_BANKERS_WITHIN_50_MILES': sum(1 for b in banker_distances if b['DISTANCE'] <= 50),
            'NUM_BANKERS_WITHIN_100_MILES': sum(1 for b in banker_distances if b['DISTANCE'] <= 100)
        })
    
    unassigned_df = pd.DataFrame(unassigned_data)
    unassigned_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(unassigned_df)} records)")
    
    return unassigned_df


def generate_exception_report(all_exceptions, output_path):
    """
    Generate exception report file
    """
    if not all_exceptions:
        print("\n✓ No exceptions to report")
        return None
    
    print("\nGenerating exception report...")
    
    exceptions_df = pd.DataFrame(all_exceptions)
    exceptions_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(exceptions_df)} records)")
    
    return exceptions_df


def generate_overall_summary(bankers_df_full, customers_df, start_time, output_path):
    """
    Generate overall summary statistics
    """
    print("\nGenerating overall summary...")
    
    assigned = customers_df[customers_df['IS_ASSIGNED'] == True]
    unassigned = customers_df[customers_df['IS_ASSIGNED'] == False]
    
    in_market_bankers = bankers_df_full[bankers_df_full['ROLE_TYPE'] == 'IN MARKET']
    centralized_bankers = bankers_df_full[bankers_df_full['ROLE_TYPE'] == 'CENTRALIZED']
    
    # Calculate metrics
    total_available = len(customers_df)
    total_assigned = len(assigned)
    total_unassigned = len(unassigned)
    
    # Distance metrics
    if total_assigned > 0:
        overall_avg_dist = assigned['DISTANCE_MILES'].mean()
        
        in_market_assigned = assigned[assigned['ASSIGNMENT_PHASE'].str.contains('IN_MARKET', na=False)]
        centralized_assigned = assigned[assigned['ASSIGNMENT_PHASE'].str.contains('CENTRALIZED', na=False)]
        
        im_avg_dist = in_market_assigned['DISTANCE_MILES'].mean() if len(in_market_assigned) > 0 else 0
        cent_avg_dist = centralized_assigned['DISTANCE_MILES'].mean() if len(centralized_assigned) > 0 else 0
        
        min_dist = assigned['DISTANCE_MILES'].min()
        max_dist = assigned['DISTANCE_MILES'].max()
        
        total_deposits = assigned['DEPOSIT_BAL'].sum()
        total_revenue = assigned['BANK_REVENUE'].sum()
    else:
        overall_avg_dist = im_avg_dist = cent_avg_dist = 0
        min_dist = max_dist = 0
        total_deposits = total_revenue = 0
    
    # Banker statistics
    bankers_min_met = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] >= bankers_df_full['MIN_COUNT_REQ']])
    bankers_failed_min = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] < bankers_df_full['MIN_COUNT_REQ']])
    bankers_at_max = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] >= bankers_df_full['MAX_COUNT_REQ']])
    
    # Portfolio statistics
    portfolio_sizes = (bankers_df_full['CURR_COUNT'] + bankers_df_full['CURRENT_ASSIGNED']).values
    avg_portfolio = portfolio_sizes.mean()
    min_portfolio = portfolio_sizes.min()
    max_portfolio = portfolio_sizes.max()
    
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
        'BANKERS_AT_MAX': bankers_at_max,
        'OVERALL_AVG_DISTANCE_MILES': round(overall_avg_dist, 2),
        'IN_MARKET_AVG_DISTANCE_MILES': round(im_avg_dist, 2),
        'CENTRALIZED_AVG_DISTANCE_MILES': round(cent_avg_dist, 2),
        'MIN_ASSIGNMENT_DISTANCE_MILES': round(min_dist, 2),
        'MAX_ASSIGNMENT_DISTANCE_MILES': round(max_dist, 2),
        'TOTAL_ASSIGNED_DEPOSITS': round(total_deposits, 2),
        'TOTAL_ASSIGNED_REVENUE': round(total_revenue, 2),
        'AVG_PORTFOLIO_SIZE': round(avg_portfolio, 2),
        'MIN_PORTFOLIO_SIZE': int(min_portfolio),
        'MAX_PORTFOLIO_SIZE': int(max_portfolio),
        'ASSIGNMENT_RUN_DATE': datetime.now(),
        'TOTAL_EXECUTION_TIME_SECONDS': round(execution_time, 2)
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    return summary_df


# ==================== MAIN ORCHESTRATOR ====================

def run_customer_banker_assignment(banker_file, req_custs_file, available_custs_file, 
                                   output_dir='output'):
    """
    Main function to run the entire customer-banker assignment process
    
    Args:
        banker_file: Path to banker_data.csv
        req_custs_file: Path to req_custs.csv
        available_custs_file: Path to available_custs.csv
        output_dir: Directory to save output files
    
    Returns:
        Dictionary with paths to all output files
    """
    import os
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("CUSTOMER-BANKER ASSIGNMENT SYSTEM")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and prepare data
    bankers_df, customers_df, banker_data_orig, customers_orig = load_and_prepare_data(
        banker_file, req_custs_file, available_custs_file
    )
    
    # Step 2: Separate bankers by type
    in_market_bankers, centralized_bankers = separate_bankers_by_type(bankers_df)
    
    # Step 3: Calculate distance matrices
    print("\n" + "="*60)
    print("Calculating Distance Matrices")
    print("="*60)
    
    print("Calculating eligible customers for IN MARKET bankers (20 mile radius)...")
    in_market_eligible = calculate_distance_matrix(
        in_market_bankers, customers_df, 'IN MARKET', 20
    )
    
    print("Calculating eligible customers for CENTRALIZED bankers (200 mile radius)...")
    centralized_eligible = calculate_distance_matrix(
        centralized_bankers, customers_df, 'CENTRALIZED', 200
    )
    
    # Step 4: Process IN MARKET assignments (20 miles)
    in_market_bankers, customers_df, im_assignments, im_exceptions = process_banker_assignments(
        in_market_bankers, customers_df, in_market_eligible,
        'IN_MARKET', 'IN MARKET', 20
    )
    
    # Step 4.5: Retry IN MARKET bankers with 40 miles if they didn't meet minimum
    print("\n" + "="*60)
    print("Checking for IN MARKET Bankers Below Minimum")
    print("="*60)
    
    # Identify bankers who didn't meet minimum
    bankers_below_min = in_market_bankers[in_market_bankers['REMAINING_MIN'] > 0].copy()
    
    if len(bankers_below_min) > 0:
        print(f"Found {len(bankers_below_min)} IN MARKET bankers below minimum.")
        print("Retrying with expanded 40-mile radius...")
        
        # Recalculate eligible customers with 40-mile radius for these bankers
        in_market_40mile_eligible = calculate_distance_matrix(
            bankers_below_min, customers_df, 'IN MARKET', 40
        )
        
        # Process these bankers again with 40-mile radius
        in_market_bankers_retry, customers_df, im_40_assignments, im_40_exceptions = process_banker_assignments(
            bankers_below_min, customers_df, in_market_40mile_eligible,
            'IN_MARKET_40MILE', 'IN MARKET', 40
        )
        
        # Update the main in_market_bankers dataframe with retry results
        for idx, retry_banker in in_market_bankers_retry.iterrows():
            port_code = retry_banker['PORT_CODE']
            main_idx = in_market_bankers[in_market_bankers['PORT_CODE'] == port_code].index[0]
            in_market_bankers.at[main_idx, 'CURRENT_ASSIGNED'] = retry_banker['CURRENT_ASSIGNED']
            in_market_bankers.at[main_idx, 'REMAINING_MIN'] = retry_banker['REMAINING_MIN']
            in_market_bankers.at[main_idx, 'REMAINING_MAX'] = retry_banker['REMAINING_MAX']
        
        # Combine assignments and exceptions
        im_assignments.extend(im_40_assignments)
        im_exceptions.extend(im_40_exceptions)
        
        print(f"✓ Additional {len(im_40_assignments)} assignments made with 40-mile radius")
    else:
        print("✓ All IN MARKET bankers met their minimum requirements within 20 miles")
    
    # Step 4.6: Reassign from over-capacity bankers (20 miles) if still below minimum
    print("\n" + "="*60)
    print("Step 3: Reassignment from Over-Capacity Bankers (20 miles)")
    print("="*60)
    
    bankers_still_struggling = in_market_bankers[in_market_bankers['REMAINING_MIN'] > 0].copy()
    
    if len(bankers_still_struggling) > 0:
        print(f"Found {len(bankers_still_struggling)} IN MARKET bankers still below minimum.")
        print("Attempting reassignment from over-capacity bankers within 20 miles...\n")
        
        in_market_bankers, customers_df, reassignments_20 = reassign_from_overcapacity_bankers(
            bankers_still_struggling, 
            in_market_bankers, 
            customers_df,
            max_distance=20,
            min_portfolio_size=280
        )
        
        print(f"\n✓ Completed {len(reassignments_20)} reassignments (20 miles)")
    else:
        reassignments_20 = []
        print("✓ All IN MARKET bankers have met their minimum requirements")
    
    # Step 4.7: Reassign from over-capacity bankers (40 miles) if STILL below minimum
    print("\n" + "="*60)
    print("Step 4: Reassignment from Over-Capacity Bankers (40 miles)")
    print("="*60)
    
    bankers_still_struggling_40 = in_market_bankers[in_market_bankers['REMAINING_MIN'] > 0].copy()
    
    if len(bankers_still_struggling_40) > 0:
        print(f"Found {len(bankers_still_struggling_40)} IN MARKET bankers STILL below minimum.")
        print("Attempting reassignment from over-capacity bankers within 40 miles...\n")
        
        in_market_bankers, customers_df, reassignments_40 = reassign_from_overcapacity_bankers(
            bankers_still_struggling_40, 
            in_market_bankers, 
            customers_df,
            max_distance=40,
            min_portfolio_size=280
        )
        
        print(f"\n✓ Completed {len(reassignments_40)} reassignments (40 miles)")
    else:
        reassignments_40 = []
        print("✓ All IN MARKET bankers have met their minimum requirements")
    
    # Combine all reassignments
    all_reassignments = reassignments_20 + reassignments_40
    
    # Add reassignments to the assignments list for tracking
    for r in all_reassignments:
        im_assignments.append({
            'HH_ECN': r['HH_ECN'],
            'PORT_CODE': r['TO_PORT'],
            'DISTANCE': r['DISTANCE'],
            'PHASE': 'IN_MARKET_REASSIGNED'
        })
    
    # Step 5: Process CENTRALIZED assignments (200 miles)
    centralized_bankers, customers_df, cent_assignments, cent_exceptions = process_banker_assignments(
        centralized_bankers, customers_df, centralized_eligible,
        'CENTRALIZED', 'CENTRALIZED', 200
    )
    
    # Step 5.5: Retry CENTRALIZED bankers with 400 miles if they didn't meet minimum
    print("\n" + "="*60)
    print("Checking for CENTRALIZED Bankers Below Minimum")
    print("="*60)
    
    # Identify CENTRALIZED bankers who didn't meet minimum
    cent_bankers_below_min = centralized_bankers[centralized_bankers['REMAINING_MIN'] > 0].copy()
    
    if len(cent_bankers_below_min) > 0:
        print(f"Found {len(cent_bankers_below_min)} CENTRALIZED bankers below minimum.")
        print("Retrying with expanded 400-mile radius...")
        
        # Recalculate eligible customers with 400-mile radius for these bankers
        centralized_400mile_eligible = calculate_distance_matrix(
            cent_bankers_below_min, customers_df, 'CENTRALIZED', 400
        )
        
        # Process these bankers again with 400-mile radius
        centralized_bankers_retry, customers_df, cent_400_assignments, cent_400_exceptions = process_banker_assignments(
            cent_bankers_below_min, customers_df, centralized_400mile_eligible,
            'CENTRALIZED_400MILE', 'CENTRALIZED', 400
        )
        
        # Update the main centralized_bankers dataframe with retry results
        for idx, retry_banker in centralized_bankers_retry.iterrows():
            port_code = retry_banker['PORT_CODE']
            main_idx = centralized_bankers[centralized_bankers['PORT_CODE'] == port_code].index[0]
            centralized_bankers.at[main_idx, 'CURRENT_ASSIGNED'] = retry_banker['CURRENT_ASSIGNED']
            centralized_bankers.at[main_idx, 'REMAINING_MIN'] = retry_banker['REMAINING_MIN']
            centralized_bankers.at[main_idx, 'REMAINING_MAX'] = retry_banker['REMAINING_MAX']
        
        # Combine assignments and exceptions
        cent_assignments.extend(cent_400_assignments)
        cent_exceptions.extend(cent_400_exceptions)
        
        print(f"✓ Additional {len(cent_400_assignments)} assignments made with 400-mile radius")
    else:
        print("✓ All CENTRALIZED bankers met their minimum requirements within 200 miles")
    
    # Step 6: Combine results
    all_bankers = pd.concat([in_market_bankers, centralized_bankers], ignore_index=True)
    all_exceptions = im_exceptions + cent_exceptions
    
    # Step 7: Generate output files
    print("\n" + "="*60)
    print("Generating Output Files")
    print("="*60)
    
    output_files = {}
    
    # File 1: Customer Assignment
    output_files['customer_assignment'] = os.path.join(output_dir, 'customer_assignment.csv')
    generate_customer_assignment_file(customers_df, banker_data_orig, output_files['customer_assignment'])
    
    # File 2: Banker Summary
    output_files['banker_summary'] = os.path.join(output_dir, 'banker_summary.csv')
    generate_banker_summary_file(all_bankers, customers_df, output_files['banker_summary'])
    
    # File 3: Unassigned Customers
    output_files['unassigned_customers'] = os.path.join(output_dir, 'unassigned_customers.csv')
    generate_unassigned_customers_file(customers_df, banker_data_orig, output_files['unassigned_customers'])
    
    # File 4: Exception Report
    if all_exceptions:
        output_files['exception_report'] = os.path.join(output_dir, 'exception_report.csv')
        generate_exception_report(all_exceptions, output_files['exception_report'])
    
    # File 5: Overall Summary
    output_files['overall_summary'] = os.path.join(output_dir, 'overall_summary.csv')
    generate_overall_summary(all_bankers, customers_df, start_time, output_files['overall_summary'])
    
    # Print final summary
    print("\n" + "="*80)
    print("ASSIGNMENT COMPLETE")
    print("="*80)
    
    assigned_count = len(customers_df[customers_df['IS_ASSIGNED'] == True])
    unassigned_count = len(customers_df[customers_df['IS_ASSIGNED'] == False])
    
    # Count assignments by phase
    im_20_count = len([a for a in im_assignments if '40MILE' not in a['PHASE'] and 'REASSIGNED' not in a['PHASE']])
    im_40_count = len([a for a in im_assignments if '40MILE' in a['PHASE']])
    im_reassigned_20_count = len(reassignments_20)
    im_reassigned_40_count = len(reassignments_40)
    
    cent_200_count = len([a for a in cent_assignments if '400MILE' not in a['PHASE']])
    cent_400_count = len([a for a in cent_assignments if '400MILE' in a['PHASE']])
    
    print(f"\n✓ Total Customers Assigned: {assigned_count}")
    print(f"✓ Total Customers Unassigned: {unassigned_count}")
    print(f"✓ Assignment Rate: {(assigned_count/len(customers_df)*100):.2f}%")
    
    print(f"\n--- IN MARKET Assignments ---")
    print(f"✓ Step 1 - Standard (20 miles): {im_20_count}")
    print(f"✓ Step 2 - Expanded (40 miles): {im_40_count}")
    print(f"✓ Step 3 - Reassigned (20 miles): {im_reassigned_20_count}")
    print(f"✓ Step 4 - Reassigned (40 miles): {im_reassigned_40_count}")
    print(f"  Total IN MARKET: {im_20_count + im_40_count + im_reassigned_20_count + im_reassigned_40_count}")
    
    print(f"\n--- CENTRALIZED Assignments ---")
    print(f"✓ Step 1 - Standard (200 miles): {cent_200_count}")
    print(f"✓ Step 2 - Expanded (400 miles): {cent_400_count}")
    print(f"  Total CENTRALIZED: {len(cent_assignments)}")
    
    print(f"\n✓ Exceptions: {len(all_exceptions)}")
    print(f"\n✓ Execution Time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    
    print(f"\nOutput files saved to: {output_dir}/")
    
    return output_files


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    """
    Example usage of the customer-banker assignment system
    """
    
    # Define your input file paths
    BANKER_FILE = 'banker_data.csv'
    REQ_CUSTS_FILE = 'req_custs.csv'
    AVAILABLE_CUSTS_FILE = 'available_custs.csv'
    
    # Run the assignment
    output_files = run_customer_banker_assignment(
        banker_file=BANKER_FILE,
        req_custs_file=REQ_CUSTS_FILE,
        available_custs_file=AVAILABLE_CUSTS_FILE,
        output_dir='output'
    )
    
    print("\n" + "="*80)
    print("Output Files Generated:")
    print("="*80)
    for name, path in output_files.items():
        print(f"  • {name}: {path}")
