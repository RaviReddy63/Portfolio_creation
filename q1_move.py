"""
Customer-Banker Assignment System with BallTree Spatial Indexing
Assigns available customers to banker portfolios based on proximity and capacity constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree
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


def build_balltree(df, lat_col, lon_col):
    """Build BallTree for efficient spatial queries"""
    coords_rad = np.radians(df[[lat_col, lon_col]].values)
    tree = BallTree(coords_rad, metric='haversine')
    return tree


def find_nearest_banker_within_radius(customer_lat, customer_lon, banker_tree, 
                                      bankers_df, max_radius_miles):
    """Find nearest banker within radius for a customer using BallTree"""
    customer_rad = np.radians([[customer_lat, customer_lon]])
    radius_rad = max_radius_miles / 3959.0
    
    indices, distances = banker_tree.query_radius(customer_rad, r=radius_rad, 
                                                   return_distance=True, sort_results=True)
    
    if len(indices[0]) == 0:
        return None, None
    
    distances_miles = distances[0] * 3959.0
    banker_indices = indices[0]
    
    return banker_indices, distances_miles


def find_customers_within_radius(banker_lat, banker_lon, customer_tree, 
                                 customers_df, max_radius_miles):
    """Find customers within radius for a banker using BallTree"""
    banker_rad = np.radians([[banker_lat, banker_lon]])
    radius_rad = max_radius_miles / 3959.0
    
    indices, distances = customer_tree.query_radius(banker_rad, r=radius_rad, 
                                                     return_distance=True, sort_results=True)
    
    if len(indices[0]) == 0:
        return [], []
    
    distances_miles = distances[0] * 3959.0
    customer_indices = indices[0]
    
    return customer_indices, distances_miles


# ==================== DATA PREPARATION ====================
def load_and_prepare_data(banker_file, req_custs_file, available_custs_file):
    """Load and prepare all input data files"""
    print("Loading data files...")
    
    banker_data = pd.read_csv(banker_file)
    req_custs = pd.read_csv(req_custs_file)
    available_custs = pd.read_csv(available_custs_file)
    
    # ===== FILTER OUT NULL COORDINATES =====
    print(f"Bankers before filtering: {len(banker_data)}")
    banker_data = banker_data.dropna(subset=['BANKER_LAT_NUM', 'BANKER_LON_NUM'])
    print(f"Bankers after filtering: {len(banker_data)}")
    
    print(f"Customers before filtering: {len(available_custs)}")
    available_custs = available_custs.dropna(subset=['LAT_NUM', 'LON_NUM'])
    print(f"Customers after filtering: {len(available_custs)}")
    # ========================================
    
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
    
    print(f"Loaded {len(bankers_df)} bankers and {len(available_custs)} available customers")
    
    return bankers_df, available_custs, banker_data, available_custs.copy()

def separate_bankers_by_type(bankers_df):
    """Separate bankers into IN MARKET and CENTRALIZED"""
    in_market = bankers_df[bankers_df['ROLE_TYPE'] == 'IN MARKET'].copy()
    centralized = bankers_df[bankers_df['ROLE_TYPE'] == 'CENTRALIZED'].copy()
    
    print(f"IN MARKET bankers: {len(in_market)}")
    print(f"CENTRALIZED bankers: {len(centralized)}")
    
    return in_market, centralized


# ==================== STEP 1 & 5: CUSTOMER TO NEAREST BANKER ====================

def assign_customers_to_nearest_banker(customers_df, bankers_df, banker_tree, 
                                      max_radius_miles, phase_name):
    """For each customer, find nearest banker and assign (up to MIN only)"""
    print(f"\n{'='*60}")
    print(f"{phase_name}: Assign Customers to Nearest Banker ({max_radius_miles} miles)")
    print(f"{'='*60}")
    
    assignments = []
    bankers_df = bankers_df.copy()
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False].copy()
    
    for idx, customer in unassigned_custs.iterrows():
        cust_lat = customer['LAT_NUM']
        cust_lon = customer['LON_NUM']
        cust_id = customer['HH_ECN']
        
        banker_indices, distances = find_nearest_banker_within_radius(
            cust_lat, cust_lon, banker_tree, bankers_df, max_radius_miles
        )
        
        if banker_indices is None:
            continue
        
        assigned = False
        for bidx, distance in zip(banker_indices, distances):
            banker_row = bankers_df.iloc[bidx]
            port_code = banker_row['PORT_CODE']
            
            banker_main_idx = bankers_df[bankers_df['PORT_CODE'] == port_code].index[0]
            current_count = bankers_df.at[banker_main_idx, 'CURR_COUNT'] + bankers_df.at[banker_main_idx, 'CURRENT_ASSIGNED']
            min_req = bankers_df.at[banker_main_idx, 'CURR_COUNT'] + bankers_df.at[banker_main_idx, 'MIN_COUNT_REQ']
            
            if current_count >= min_req:
                continue
            
            customers_df.at[idx, 'IS_ASSIGNED'] = True
            customers_df.at[idx, 'ASSIGNED_TO_PORT_CODE'] = port_code
            customers_df.at[idx, 'ASSIGNED_BANKER_EID'] = banker_row['EID']
            customers_df.at[idx, 'DISTANCE_MILES'] = distance
            customers_df.at[idx, 'ASSIGNMENT_PHASE'] = phase_name
            
            bankers_df.at[banker_main_idx, 'CURRENT_ASSIGNED'] += 1
            bankers_df.at[banker_main_idx, 'REMAINING_MIN'] = max(0, bankers_df.at[banker_main_idx, 'REMAINING_MIN'] - 1)
            bankers_df.at[banker_main_idx, 'REMAINING_MAX'] = max(0, bankers_df.at[banker_main_idx, 'REMAINING_MAX'] - 1)
            
            assignments.append({
                'HH_ECN': cust_id,
                'PORT_CODE': port_code,
                'DISTANCE': distance,
                'PHASE': phase_name
            })
            
            assigned = True
            break
        
    print(f"✓ Assigned {len(assignments)} customers to nearest bankers")
    
    return bankers_df, customers_df, assignments


# ==================== STEP 2-4, 6-8: FILL PORTFOLIOS ====================

def fill_banker_portfolios(bankers_df, customers_df, customer_tree, max_radius_miles, 
                          phase_name, target_type='MIN'):
    """Fill banker portfolios to MIN or MAX by finding nearest unassigned customers"""
    print(f"\n{'='*60}")
    print(f"{phase_name}: Fill to {target_type} ({max_radius_miles} miles)")
    print(f"{'='*60}")
    
    assignments = []
    bankers_df = bankers_df.copy()
    
    if target_type == 'MIN':
        bankers_to_fill = bankers_df[bankers_df['REMAINING_MIN'] > 0].copy()
    else:
        bankers_to_fill = bankers_df[bankers_df['REMAINING_MAX'] > 0].copy()
    
    if len(bankers_to_fill) == 0:
        print(f"✓ All bankers already at {target_type}")
        return bankers_df, customers_df, assignments
    
    print(f"Found {len(bankers_to_fill)} bankers below {target_type}")
    
    if target_type == 'MIN':
        bankers_to_fill = bankers_to_fill.sort_values('REMAINING_MIN', ascending=False)
    else:
        bankers_to_fill = bankers_to_fill.sort_values('REMAINING_MAX', ascending=False)
    
    for _, banker in bankers_to_fill.iterrows():
        port_code = banker['PORT_CODE']
        banker_lat = banker['BANKER_LAT_NUM']
        banker_lon = banker['BANKER_LON_NUM']
        banker_eid = banker['EID']
        banker_name = banker['EMPLOYEE_NAME']
        
        if target_type == 'MIN':
            needed = int(banker['REMAINING_MIN'])
        else:
            needed = int(banker['REMAINING_MAX'])
        
        if needed <= 0:
            continue
        
        print(f"  Banker: {banker_name} (Port: {port_code}) - Needs: {needed}")
        
        unassigned_indices = customers_df[customers_df['IS_ASSIGNED'] == False].index.tolist()
        
        if len(unassigned_indices) == 0:
            print(f"    ✗ No unassigned customers available")
            continue
        
        unassigned_custs = customers_df.loc[unassigned_indices]
        if len(unassigned_custs) == 0:
            continue
            
        temp_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        
        cust_indices, distances = find_customers_within_radius(
            banker_lat, banker_lon, temp_tree, unassigned_custs, max_radius_miles
        )
        
        if len(cust_indices) == 0:
            print(f"    ✗ No customers found within {max_radius_miles} miles")
            continue
        
        assigned_count = 0
        for cidx, distance in zip(cust_indices, distances):
            if assigned_count >= needed:
                break
            
            actual_idx = unassigned_custs.iloc[cidx].name
            cust_id = customers_df.at[actual_idx, 'HH_ECN']
            
            if customers_df.at[actual_idx, 'IS_ASSIGNED']:
                continue
            
            customers_df.at[actual_idx, 'IS_ASSIGNED'] = True
            customers_df.at[actual_idx, 'ASSIGNED_TO_PORT_CODE'] = port_code
            customers_df.at[actual_idx, 'ASSIGNED_BANKER_EID'] = banker_eid
            customers_df.at[actual_idx, 'DISTANCE_MILES'] = distance
            customers_df.at[actual_idx, 'ASSIGNMENT_PHASE'] = phase_name
            
            if '40MILE' in phase_name and distance > 20:
                customers_df.at[actual_idx, 'EXCEPTION_FLAG'] = 'EXPANDED_RADIUS_20_TO_40_MILES'
            elif '400MILE' in phase_name and distance > 200:
                customers_df.at[actual_idx, 'EXCEPTION_FLAG'] = 'EXPANDED_RADIUS_200_TO_400_MILES'
            
            banker_main_idx = bankers_df[bankers_df['PORT_CODE'] == port_code].index[0]
            bankers_df.at[banker_main_idx, 'CURRENT_ASSIGNED'] += 1
            bankers_df.at[banker_main_idx, 'REMAINING_MIN'] = max(0, bankers_df.at[banker_main_idx, 'REMAINING_MIN'] - 1)
            bankers_df.at[banker_main_idx, 'REMAINING_MAX'] = max(0, bankers_df.at[banker_main_idx, 'REMAINING_MAX'] - 1)
            
            assignments.append({
                'HH_ECN': cust_id,
                'PORT_CODE': port_code,
                'DISTANCE': distance,
                'PHASE': phase_name
            })
            
            assigned_count += 1
        
        print(f"    ✓ Assigned {assigned_count} customers")
    
    print(f"✓ Total assignments in this step: {len(assignments)}")
    
    return bankers_df, customers_df, assignments


# ==================== OUTPUT GENERATION ====================

def generate_customer_assignment_file(customers_df, bankers_df_full, output_path):
    """Generate the main customer assignment output file"""
    print("\nGenerating customer assignment file...")
    
    assigned = customers_df[customers_df['IS_ASSIGNED'] == True].copy()
    
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
    
    result['PROXIMITY_LIMIT_MILES'] = result['ROLE_TYPE'].apply(
        lambda x: 20 if x == 'IN MARKET' else 200
    )
    result['IS_WITHIN_PROXIMITY'] = result['DISTANCE_MILES'] <= result['PROXIMITY_LIMIT_MILES']
    result['ASSIGNMENT_TIMESTAMP'] = datetime.now()
    
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
            'NUM_UNIQUE_ZIPCODES': num_zips,
            'HAS_EXCEPTIONS': False,
            'EXCEPTION_COUNT': 0,
            'EXCEPTION_DETAILS': None
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


def generate_overall_summary(bankers_df_full, customers_df, start_time, output_path, 
                             im_step1, im_step2, im_step3, im_step4, 
                             cent_step5, cent_step6, cent_step7, cent_step8a, cent_step8b):
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
        
        in_market_assigned = assigned[assigned['ASSIGNMENT_PHASE'].str.contains('IM_', na=False)]
        centralized_assigned = assigned[assigned['ASSIGNMENT_PHASE'].str.contains('CENT_', na=False)]
        
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
    
    bankers_min_met = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] >= bankers_df_full['MIN_COUNT_REQ']])
    bankers_failed_min = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] < bankers_df_full['MIN_COUNT_REQ']])
    bankers_at_max = len(bankers_df_full[bankers_df_full['CURRENT_ASSIGNED'] >= bankers_df_full['MAX_COUNT_REQ']])
    
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
    """Main function to run the entire customer-banker assignment process"""
    import os
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("CUSTOMER-BANKER ASSIGNMENT SYSTEM (BallTree Optimized)")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    bankers_df, customers_df, banker_data_orig, customers_orig = load_and_prepare_data(
        banker_file, req_custs_file, available_custs_file
    )
    
    in_market_bankers, centralized_bankers = separate_bankers_by_type(bankers_df)
    
    print("\nBuilding spatial index for IN MARKET bankers...")
    im_tree = build_balltree(in_market_bankers, 'BANKER_LAT_NUM', 'BANKER_LON_NUM')
    
    print("\n" + "="*80)
    print("IN MARKET ASSIGNMENTS")
    print("="*80)
    
    in_market_bankers, customers_df, im_step1 = assign_customers_to_nearest_banker(
        customers_df, in_market_bankers, im_tree, 20, 'IM_STEP1_NEAREST_20MI'
    )
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    if len(unassigned_custs) > 0:
        cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        in_market_bankers, customers_df, im_step2 = fill_banker_portfolios(
            in_market_bankers, customers_df, cust_tree, 20, 'IM_STEP2_FILL_MIN_20MI', 'MIN'
        )
    else:
        im_step2 = []
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    if len(unassigned_custs) > 0:
        cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        in_market_bankers, customers_df, im_step3 = fill_banker_portfolios(
            in_market_bankers, customers_df, cust_tree, 40, 'IM_STEP3_FILL_MIN_40MILE', 'MIN'
        )
    else:
        im_step3 = []
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    if len(unassigned_custs) > 0:
        cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        in_market_bankers, customers_df, im_step4 = fill_banker_portfolios(
            in_market_bankers, customers_df, cust_tree, 20, 'IM_STEP4_FILL_MAX_20MI', 'MAX'
        )
    else:
        im_step4 = []
    
    print("\n" + "="*80)
    print("CENTRALIZED ASSIGNMENTS")
    print("="*80)
    
    print("\nBuilding spatial index for CENTRALIZED bankers...")
    cent_tree = build_balltree(centralized_bankers, 'BANKER_LAT_NUM', 'BANKER_LON_NUM')
    
    centralized_bankers, customers_df, cent_step5 = assign_customers_to_nearest_banker(
        customers_df, centralized_bankers, cent_tree, 200, 'CENT_STEP5_NEAREST_200MI'
    )
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    if len(unassigned_custs) > 0:
        cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        centralized_bankers, customers_df, cent_step6 = fill_banker_portfolios(
            centralized_bankers, customers_df, cust_tree, 200, 'CENT_STEP6_FILL_MIN_200MI', 'MIN'
        )
    else:
        cent_step6 = []
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    if len(unassigned_custs) > 0:
        cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        centralized_bankers, customers_df, cent_step7 = fill_banker_portfolios(
            centralized_bankers, customers_df, cust_tree, 400, 'CENT_STEP7_FILL_MIN_400MILE', 'MIN'
        )
    else:
        cent_step7 = []
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    if len(unassigned_custs) > 0:
        cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        centralized_bankers, customers_df, cent_step8a = fill_banker_portfolios(
            centralized_bankers, customers_df, cust_tree, 200, 'CENT_STEP8A_FILL_MAX_200MI', 'MAX'
        )
    else:
        cent_step8a = []
    
    unassigned_custs = customers_df[customers_df['IS_ASSIGNED'] == False]
    if len(unassigned_custs) > 0:
        cust_tree = build_balltree(unassigned_custs, 'LAT_NUM', 'LON_NUM')
        centralized_bankers, customers_df, cent_step8b = fill_banker_portfolios(
            centralized_bankers, customers_df, cust_tree, 400, 'CENT_STEP8B_FILL_MAX_400MILE', 'MAX'
        )
    else:
        cent_step8b = []
    
    all_bankers = pd.concat([in_market_bankers, centralized_bankers], ignore_index=True)
    
    print("\n" + "="*60)
    print("Generating Output Files")
    print("="*60)
    
    output_files = {}
    
    output_files['customer_assignment'] = os.path.join(output_dir, 'customer_assignment.csv')
    generate_customer_assignment_file(customers_df, banker_data_orig, output_files['customer_assignment'])
    
    output_files['banker_summary'] = os.path.join(output_dir, 'banker_summary.csv')
    generate_banker_summary_file(all_bankers, customers_df, output_files['banker_summary'])
    
    output_files['unassigned_customers'] = os.path.join(output_dir, 'unassigned_customers.csv')
    generate_unassigned_customers_file(customers_df, banker_data_orig, output_files['unassigned_customers'])
    
    output_files['overall_summary'] = os.path.join(output_dir, 'overall_summary.csv')
    generate_overall_summary(all_bankers, customers_df, start_time, output_files['overall_summary'],
                            im_step1, im_step2, im_step3, im_step4,
                            cent_step5, cent_step6, cent_step7, cent_step8a, cent_step8b)
    
    print("\n" + "="*80)
    print("ASSIGNMENT COMPLETE")
    print("="*80)
    
    assigned_count = len(customers_df[customers_df['IS_ASSIGNED'] == True])
    unassigned_count = len(customers_df[customers_df['IS_ASSIGNED'] == False])
    
    print(f"\n✓ Total Customers Assigned: {assigned_count}")
    print(f"✓ Total Customers Unassigned: {unassigned_count}")
    print(f"✓ Assignment Rate: {(assigned_count/len(customers_df)*100):.2f}%")
    
    print(f"\n--- IN MARKET Assignments ---")
    print(f"✓ Step 1 - Nearest banker (20 mi, up to MIN): {len(im_step1)}")
    print(f"✓ Step 2 - Fill MIN (20 mi): {len(im_step2)}")
    print(f"✓ Step 3 - Fill MIN (40 mi): {len(im_step3)}")
    print(f"✓ Step 4 - Fill MAX (20 mi): {len(im_step4)}")
    print(f"  Total IN MARKET: {len(im_step1) + len(im_step2) + len(im_step3) + len(im_step4)}")
    
    print(f"\n--- CENTRALIZED Assignments ---")
    print(f"✓ Step 5 - Nearest banker (200 mi, up to MIN): {len(cent_step5)}")
    print(f"✓ Step 6 - Fill MIN (200 mi): {len(cent_step6)}")
    print(f"✓ Step 7 - Fill MIN (400 mi): {len(cent_step7)}")
    print(f"✓ Step 8a - Fill MAX (200 mi): {len(cent_step8a)}")
    print(f"✓ Step 8b - Fill MAX (400 mi): {len(cent_step8b)}")
    print(f"  Total CENTRALIZED: {len(cent_step5) + len(cent_step6) + len(cent_step7) + len(cent_step8a) + len(cent_step8b)}")
    
    print(f"\n✓ Execution Time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    print(f"\nOutput files saved to: {output_dir}/")
    
    return output_files


if __name__ == "__main__":
    BANKER_FILE = 'banker_data.csv'
    REQ_CUSTS_FILE = 'req_custs.csv'
    AVAILABLE_CUSTS_FILE = 'available_custs.csv'
    
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
