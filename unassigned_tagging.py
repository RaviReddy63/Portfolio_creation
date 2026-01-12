import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

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

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two points"""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def convert_state_name_to_code(state_name):
    """Convert full state name to state code"""
    if pd.isna(state_name):
        return None
    
    state_name_clean = str(state_name).strip().upper()
    
    # If already a 2-letter code, return as is
    if len(state_name_clean) == 2:
        return state_name_clean
    
    # Otherwise, look up in mapping
    return STATE_NAME_TO_CODE.get(state_name_clean, None)

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
    
    # Combine both - include COVERAGE column
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
    
    if len(bankers) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
    
    # Expand COVERAGE column (comma-separated states) into multiple rows
    bankers['COVERAGE'] = bankers['COVERAGE'].fillna('')
    bankers_expanded = []
    
    for _, banker in bankers.iterrows():
        coverage_str = str(banker['COVERAGE']).strip()
        if coverage_str:
            states = [s.strip().upper() for s in coverage_str.split(',') if s.strip()]
            for state in states:
                banker_row = banker.to_dict()
                banker_row['STATE'] = state
                bankers_expanded.append(banker_row)
    
    if len(bankers_expanded) == 0:
        print(f"  Warning: No {banker_type} bankers with valid COVERAGE found")
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
    
    bankers_by_state = pd.DataFrame(bankers_expanded)
    
    assignments = []
    unmatched_states = set()
    
    for _, customer in customers_no_coords.iterrows():
        # Convert full state name to state code
        customer_state_code = convert_state_name_to_code(customer['BILLINGSTATE'])
        
        if not customer_state_code:
            continue
        
        # Find bankers covering this state code
        matching_bankers = bankers_by_state[bankers_by_state['STATE'] == customer_state_code]
        
        if len(matching_bankers) > 0:
            # Assign to first matching banker (or you can implement load balancing)
            banker = matching_bankers.iloc[0]
            assignments.append({
                'CG_ECN': customer['CG_ECN'],
                'HH_ECN': customer['HH_ECN'],
                'PORT_CODE': banker['PORT_CODE'],
                'PROM_SEG_RAW': customer['PROM_SEG_RAW'],
                'BANKER_TYPE': banker_type
            })
        else:
            unmatched_states.add(customer_state_code)
    
    if unmatched_states:
        print(f"  Warning: No {banker_type} coverage for state codes: {sorted(unmatched_states)}")
    
    return pd.DataFrame(assignments)

def assign_customers(unassigned_pop, banker_locations, prom_seg, banker_type, 
                     max_radius, in_market_radius=None, centralized_radius=600):
    """Assign customers to nearest bankers within radius"""
    
    # Filter customers by segment
    customers = unassigned_pop[unassigned_pop['PROM_SEG_RAW'] == prom_seg].copy()
    
    if len(customers) == 0:
        empty_df = pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
        return empty_df, pd.DataFrame()
    
    # Separate customers with and without coordinates
    customers_with_coords = customers.dropna(subset=['ECN_LAT', 'ECN_LON']).copy()
    customers_no_coords = customers[customers['ECN_LAT'].isna() | customers['ECN_LON'].isna()].copy()
    
    bankers = banker_locations[banker_locations['BANKER_TYPE'] == banker_type].copy()
    
    assignments = []
    
    # Process customers WITH coordinates
    if len(customers_with_coords) > 0 and len(bankers) > 0:
        
        # For RM (PROM_SEG_RAW=3), try IN MARKET first
        if in_market_radius is not None:
            in_market_bankers = bankers[bankers['ROLE_TYPE'] == 'IN MARKET'].copy()
            
            if len(in_market_bankers) > 0:
                # Build BallTree for IN MARKET bankers
                banker_coords = in_market_bankers[['LAT', 'LON']].values.astype(float)
                banker_coords_rad = np.radians(banker_coords)
                tree = BallTree(banker_coords_rad, metric='haversine')
                
                customer_coords = customers_with_coords[['ECN_LAT', 'ECN_LON']].values.astype(float)
                customer_coords_rad = np.radians(customer_coords)
                distances, indices = tree.query(customer_coords_rad, k=1)
                distances_miles = distances.flatten() * 3959
                
                # Assign customers within IN MARKET radius
                mask = distances_miles <= in_market_radius
                assigned_indices = np.where(mask)[0]
                
                for idx in assigned_indices:
                    banker_idx = indices[idx][0]
                    assignments.append({
                        'CG_ECN': customers_with_coords.iloc[idx]['CG_ECN'],
                        'HH_ECN': customers_with_coords.iloc[idx]['HH_ECN'],
                        'PORT_CODE': in_market_bankers.iloc[banker_idx]['PORT_CODE'],
                        'PROM_SEG_RAW': prom_seg,
                        'BANKER_TYPE': banker_type
                    })
                
                # Remove assigned customers
                customers_with_coords = customers_with_coords.iloc[~mask].copy()
        
        # Assign remaining customers to CENTRALIZED or all bankers
        if len(customers_with_coords) > 0 and len(bankers) > 0:
            # For RM after IN MARKET attempt, use only CENTRALIZED bankers
            if in_market_radius is not None:
                bankers_for_assignment = bankers[bankers['ROLE_TYPE'] == 'CENTRALIZED'].copy()
                radius_to_use = centralized_radius
            else:
                # For RC, use all bankers
                bankers_for_assignment = bankers.copy()
                radius_to_use = max_radius
            
            if len(bankers_for_assignment) > 0:
                banker_coords = bankers_for_assignment[['LAT', 'LON']].values.astype(float)
                banker_coords_rad = np.radians(banker_coords)
                tree = BallTree(banker_coords_rad, metric='haversine')
                
                customer_coords = customers_with_coords[['ECN_LAT', 'ECN_LON']].values.astype(float)
                customer_coords_rad = np.radians(customer_coords)
                distances, indices = tree.query(customer_coords_rad, k=1)
                distances_miles = distances.flatten() * 3959
                
                # Assign customers within radius
                mask = distances_miles <= radius_to_use
                assigned_indices = np.where(mask)[0]
                
                for idx in assigned_indices:
                    banker_idx = indices[idx][0]
                    assignments.append({
                        'CG_ECN': customers_with_coords.iloc[idx]['CG_ECN'],
                        'HH_ECN': customers_with_coords.iloc[idx]['HH_ECN'],
                        'PORT_CODE': bankers_for_assignment.iloc[banker_idx]['PORT_CODE'],
                        'PROM_SEG_RAW': prom_seg,
                        'BANKER_TYPE': banker_type
                    })
    
    assignments_df = pd.DataFrame(assignments) if assignments else pd.DataFrame(
        columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']
    )
    
    return assignments_df, customers_no_coords

def main(df1, df2, df3, df4):
    """Main function to assign all customers"""
    
    print("="*60)
    print("CUSTOMER-BANKER ASSIGNMENT PROCESS")
    print("="*60)
    
    # Get banker locations
    banker_locations = get_banker_locations(df2, df3, df4)
    
    print(f"\nBanker Summary:")
    print(f"  Total bankers with valid locations: {len(banker_locations)}")
    print(f"  RC bankers: {len(banker_locations[banker_locations['BANKER_TYPE']=='RC'])}")
    print(f"  RM bankers: {len(banker_locations[banker_locations['BANKER_TYPE']=='RM'])}")
    print(f"  IN MARKET: {len(banker_locations[banker_locations['ROLE_TYPE']=='IN MARKET'])}")
    print(f"  CENTRALIZED: {len(banker_locations[banker_locations['ROLE_TYPE']=='CENTRALIZED'])}")
    
    # Customer summary
    total_customers = len(df1)
    seg4_customers = len(df1[df1['PROM_SEG_RAW'] == 4])
    seg3_customers = len(df1[df1['PROM_SEG_RAW'] == 3])
    
    print(f"\nCustomer Summary:")
    print(f"  Total unassigned customers: {total_customers}")
    print(f"  Segment 4 (RC): {seg4_customers}")
    print(f"  Segment 3 (RM): {seg3_customers}")
    
    # Step 1: Assign PROM_SEG_RAW=4 to RC bankers (400 miles max)
    print(f"\n{'='*60}")
    print("STEP 1: Assigning Segment 4 customers to RC bankers")
    print(f"{'='*60}")
    
    rc_assignments, rc_no_coords = assign_customers(
        df1, banker_locations, 
        prom_seg=4, 
        banker_type='RC', 
        max_radius=400
    )
    print(f"  RC assignments (geographic): {len(rc_assignments)}")
    print(f"  RC customers without coords: {len(rc_no_coords)}")
    
    # Step 1b: Assign RC customers without coordinates by state
    rc_state_assignments = assign_by_state(rc_no_coords, df2, 'RC')
    print(f"  RC assignments (by state): {len(rc_state_assignments)}")
    
    # Step 2: Assign PROM_SEG_RAW=3 to RM bankers (40 miles IN MARKET, then 600 miles CENTRALIZED)
    print(f"\n{'='*60}")
    print("STEP 2: Assigning Segment 3 customers to RM bankers")
    print(f"{'='*60}")
    
    rm_assignments, rm_no_coords = assign_customers(
        df1, banker_locations, 
        prom_seg=3, 
        banker_type='RM', 
        max_radius=400,
        in_market_radius=40,
        centralized_radius=600
    )
    print(f"  RM assignments (geographic): {len(rm_assignments)}")
    print(f"  RM customers without coords: {len(rm_no_coords)}")
    
    # Step 2b: Assign RM customers without coordinates by state
    rm_state_assignments = assign_by_state(rm_no_coords, df2, 'RM')
    print(f"  RM assignments (by state): {len(rm_state_assignments)}")
    
    # Combine all assignments
    all_assignments = [rc_assignments, rc_state_assignments, rm_assignments, rm_state_assignments]
    final_assignments = pd.concat([df for df in all_assignments if len(df) > 0], ignore_index=True)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL ASSIGNMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total customers assigned: {len(final_assignments)}")
    print(f"  Total customers unassigned: {total_customers - len(final_assignments)}")
    print(f"  Assignment rate: {len(final_assignments)/total_customers*100:.2f}%")
    
    print(f"\nBreakdown by Banker Type:")
    if len(final_assignments) > 0:
        banker_type_counts = final_assignments['BANKER_TYPE'].value_counts()
        for banker_type, count in banker_type_counts.items():
            print(f"  {banker_type}: {count}")
    
    print(f"\nBreakdown by Segment:")
    if len(final_assignments) > 0:
        seg_counts = final_assignments['PROM_SEG_RAW'].value_counts()
        for seg, count in seg_counts.items():
            print(f"  Segment {seg}: {count}")
    
    return final_assignments[['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']]

# Run the assignment
if __name__ == "__main__":
    # Assuming df1, df2, df3, df4 are already loaded
    result = main(df1, df2, df3, df4)
    df1['PROM_SEG_RAW'] = df1.groupby('CG_ECN')['HH_SEG'].transform('max')
    
    # Display sample results
    print(f"\n{'='*60}")
    print("SAMPLE ASSIGNMENTS (First 10 rows)")
    print(f"{'='*60}")
    print(result.head(10).to_string(index=False))
    
    # Save to CSV
    output_file = 'customer_banker_assignments.csv'
    result.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
