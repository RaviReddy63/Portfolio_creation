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

def assign_by_state(customers_no_coords, active_bankers):
    """Assign customers without coordinates based on state coverage - ANY banker type"""
    
    if len(customers_no_coords) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']), pd.DataFrame()
    
    # Use ALL bankers (no filtering by BANKER_TYPE)
    bankers = active_bankers.copy()
    
    if len(bankers) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']), customers_no_coords
    
    # Expand COVERAGE column (/ separated states) into multiple rows
    bankers['COVERAGE'] = bankers['COVERAGE'].fillna('')
    bankers_expanded = []
    
    for _, banker in bankers.iterrows():
        coverage_str = str(banker['COVERAGE']).strip()
        if coverage_str:
            states = [s.strip().upper() for s in coverage_str.split('/') if s.strip()]
            for state in states:
                banker_row = banker.to_dict()
                banker_row['STATE'] = state
                bankers_expanded.append(banker_row)
    
    if len(bankers_expanded) == 0:
        print(f"  Warning: No bankers with valid COVERAGE found")
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']), customers_no_coords
    
    bankers_by_state = pd.DataFrame(bankers_expanded)
    
    assignments = []
    unmatched_customers = []
    unmatched_states = set()
    
    for _, customer in customers_no_coords.iterrows():
        # Convert full state name to state code
        customer_state_code = convert_state_name_to_code(customer['BILLINGSTATE'])
        
        if not customer_state_code:
            unmatched_customers.append(customer)
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
                'BANKER_TYPE': banker['BANKER_TYPE']
            })
        else:
            unmatched_states.add(customer_state_code)
            unmatched_customers.append(customer)
    
    if unmatched_states:
        print(f"  Warning: No banker coverage for state codes: {sorted(unmatched_states)}")
    
    unmatched_df = pd.DataFrame(unmatched_customers) if unmatched_customers else pd.DataFrame()
    
    return pd.DataFrame(assignments), unmatched_df

def assign_customers(unassigned_pop, banker_locations, in_market_radius=40, centralized_radius=600):
    """Assign customers to nearest bankers within radius - ANY banker type"""
    
    customers = unassigned_pop.copy()
    
    if len(customers) == 0:
        empty_df = pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
        return empty_df, pd.DataFrame()
    
    # Separate customers with and without coordinates
    customers_with_coords = customers.dropna(subset=['ECN_LAT', 'ECN_LON']).copy()
    customers_no_coords = customers[customers['ECN_LAT'].isna() | customers['ECN_LON'].isna()].copy()
    
    # Use ALL bankers (no filtering)
    bankers = banker_locations.copy()
    
    assignments = []
    unassigned_with_coords = []
    
    # Process customers WITH coordinates
    if len(customers_with_coords) > 0 and len(bankers) > 0:
        
        # Step 1: Try IN MARKET bankers first (40 miles)
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
                    'PROM_SEG_RAW': customers_with_coords.iloc[idx]['PROM_SEG_RAW'],
                    'BANKER_TYPE': in_market_bankers.iloc[banker_idx]['BANKER_TYPE']
                })
            
            # Remove assigned customers
            customers_with_coords = customers_with_coords.iloc[~mask].copy()
        
        # Step 2: Assign remaining customers to CENTRALIZED bankers (600 miles)
        if len(customers_with_coords) > 0:
            centralized_bankers = bankers[bankers['ROLE_TYPE'] == 'CENTRALIZED'].copy()
            
            if len(centralized_bankers) > 0:
                banker_coords = centralized_bankers[['LAT', 'LON']].values.astype(float)
                banker_coords_rad = np.radians(banker_coords)
                tree = BallTree(banker_coords_rad, metric='haversine')
                
                customer_coords = customers_with_coords[['ECN_LAT', 'ECN_LON']].values.astype(float)
                customer_coords_rad = np.radians(customer_coords)
                distances, indices = tree.query(customer_coords_rad, k=1)
                distances_miles = distances.flatten() * 3959
                
                # Assign customers within CENTRALIZED radius
                mask = distances_miles <= centralized_radius
                assigned_indices = np.where(mask)[0]
                
                for idx in assigned_indices:
                    banker_idx = indices[idx][0]
                    assignments.append({
                        'CG_ECN': customers_with_coords.iloc[idx]['CG_ECN'],
                        'HH_ECN': customers_with_coords.iloc[idx]['HH_ECN'],
                        'PORT_CODE': centralized_bankers.iloc[banker_idx]['PORT_CODE'],
                        'PROM_SEG_RAW': customers_with_coords.iloc[idx]['PROM_SEG_RAW'],
                        'BANKER_TYPE': centralized_bankers.iloc[banker_idx]['BANKER_TYPE']
                    })
                
                # Track unassigned customers beyond radius
                unassigned_mask = ~mask
                for idx in np.where(unassigned_mask)[0]:
                    unassigned_with_coords.append(customers_with_coords.iloc[idx])
    
    assignments_df = pd.DataFrame(assignments) if assignments else pd.DataFrame(
        columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']
    )
    
    # Combine unassigned with coords and no coords
    all_unassigned = pd.concat([
        pd.DataFrame(unassigned_with_coords) if unassigned_with_coords else pd.DataFrame(),
        customers_no_coords
    ], ignore_index=True)
    
    return assignments_df, all_unassigned

def assign_randomly(customers_unassigned, active_bankers):
    """Randomly assign remaining unassigned customers - ANY banker type"""
    
    if len(customers_unassigned) == 0:
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
    
    # Use ALL bankers (no filtering)
    bankers = active_bankers.copy()
    
    if len(bankers) == 0:
        print(f"  Warning: No bankers available for random assignment")
        return pd.DataFrame(columns=['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE'])
    
    # Get unique PORT_CODEs with their BANKER_TYPE
    portfolio_banker_map = bankers[['PORT_CODE', 'BANKER_TYPE']].drop_duplicates()
    
    assignments = []
    
    for _, customer in customers_unassigned.iterrows():
        # Randomly select a portfolio
        random_portfolio_row = portfolio_banker_map.sample(n=1).iloc[0]
        
        assignments.append({
            'CG_ECN': customer['CG_ECN'],
            'HH_ECN': customer['HH_ECN'],
            'PORT_CODE': random_portfolio_row['PORT_CODE'],
            'PROM_SEG_RAW': customer['PROM_SEG_RAW'],
            'BANKER_TYPE': random_portfolio_row['BANKER_TYPE']
        })
    
    return pd.DataFrame(assignments)

def main(df1, df2, df3, df4, df5):
    """Main function to assign all customers"""
    
    print("="*60)
    print("CUSTOMER-BANKER ASSIGNMENT PROCESS")
    print("="*60)
    
    # Transform PROM_SEG_RAW to max HH_SEG per CG_ECN
    df1['PROM_SEG_RAW'] = df1.groupby('CG_ECN')['HH_SEG'].transform('max')
    
    # Filter df5 to only include CG_ECNs that exist in df1
    df5_filtered = df5[df5['CG_ECN'].isin(df1['CG_ECN'])].copy()
    
    print(f"\nBackfill Portfolio Summary:")
    print(f"  Total CG_ECNs in backfill file: {len(df5)}")
    print(f"  CG_ECNs in backfill that exist in df1: {len(df5_filtered)}")
    print(f"  CG_ECNs excluded (not in df1): {len(df5) - len(df5_filtered)}")
    
    # Separate customers into backfill and needs-assignment
    backfill_cg_ecns = set(df5_filtered['CG_ECN'].unique())
    df1_backfill = df1[df1['CG_ECN'].isin(backfill_cg_ecns)].copy()
    df1_to_assign = df1[~df1['CG_ECN'].isin(backfill_cg_ecns)].copy()
    
    print(f"\nCustomer Split:")
    print(f"  Customers with backfill portfolios: {len(df1_backfill)}")
    print(f"  Customers needing assignment: {len(df1_to_assign)}")
    
    # Create backfill assignments by merging df1_backfill with df5_filtered
    backfill_assignments = df1_backfill[['CG_ECN', 'HH_ECN', 'PROM_SEG_RAW']].merge(
        df5_filtered[['CG_ECN', 'PORT_CODE']], 
        on='CG_ECN', 
        how='left'
    )
    
    # Get BANKER_TYPE from df2 based on PORT_CODE
    backfill_assignments = backfill_assignments.merge(
        df2[['PORT_CODE', 'BANKER_TYPE']].drop_duplicates(),
        on='PORT_CODE',
        how='left'
    )
    
    print(f"  Backfill assignments created: {len(backfill_assignments)}")
    
    # Get banker locations
    banker_locations = get_banker_locations(df2, df3, df4)
    
    print(f"\nBanker Summary:")
    print(f"  Total bankers with valid locations: {len(banker_locations)}")
    print(f"  RC bankers: {len(banker_locations[banker_locations['BANKER_TYPE']=='RC'])}")
    print(f"  RM bankers: {len(banker_locations[banker_locations['BANKER_TYPE']=='RM'])}")
    print(f"  IN MARKET: {len(banker_locations[banker_locations['ROLE_TYPE']=='IN MARKET'])}")
    print(f"  CENTRALIZED: {len(banker_locations[banker_locations['ROLE_TYPE']=='CENTRALIZED'])}")
    
    # Customer summary for assignment
    total_customers_to_assign = len(df1_to_assign)
    seg4_customers = len(df1_to_assign[df1_to_assign['PROM_SEG_RAW'] == 4])
    seg3_customers = len(df1_to_assign[df1_to_assign['PROM_SEG_RAW'] == 3])
    
    print(f"\nCustomer Assignment Summary:")
    print(f"  Total customers needing assignment: {total_customers_to_assign}")
    print(f"  Segment 4: {seg4_customers}")
    print(f"  Segment 3: {seg3_customers}")
    
    # Step 1: Assign ALL customers geographically (40 miles IN MARKET, 600 miles CENTRALIZED)
    print(f"\n{'='*60}")
    print("STEP 1: Assigning customers geographically to ANY banker")
    print(f"{'='*60}")
    
    geographic_assignments, unassigned_after_geo = assign_customers(
        df1_to_assign, banker_locations, 
        in_market_radius=40,
        centralized_radius=600
    )
    print(f"  Geographic assignments: {len(geographic_assignments)}")
    print(f"  Customers remaining: {len(unassigned_after_geo)}")
    
    # Step 2: Assign remaining customers by state
    print(f"\n{'='*60}")
    print("STEP 2: Assigning remaining customers by state")
    print(f"{'='*60}")
    
    state_assignments, still_unassigned = assign_by_state(unassigned_after_geo, df2)
    print(f"  State-based assignments: {len(state_assignments)}")
    print(f"  Customers still unassigned: {len(still_unassigned)}")
    
    # Step 3: Randomly assign remaining customers
    print(f"\n{'='*60}")
    print("STEP 3: Randomly assigning remaining customers")
    print(f"{'='*60}")
    
    random_assignments = assign_randomly(still_unassigned, df2)
    print(f"  Random assignments: {len(random_assignments)}")
    
    # Combine all assignments (backfill + new assignments)
    all_assignments = [
        backfill_assignments[['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE']],
        geographic_assignments,
        state_assignments,
        random_assignments
    ]
    final_assignments = pd.concat([df for df in all_assignments if len(df) > 0], ignore_index=True)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL ASSIGNMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total customers assigned: {len(final_assignments)}")
    print(f"  - From backfill: {len(backfill_assignments)}")
    print(f"  - From assignment logic: {len(final_assignments) - len(backfill_assignments)}")
    print(f"  Total customers unassigned: {len(df1) - len(final_assignments)}")
    print(f"  Assignment rate: {len(final_assignments)/len(df1)*100:.2f}%")
    
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
    
    print(f"\nCross-tabulation (Segment vs Banker Type):")
    if len(final_assignments) > 0:
        crosstab = pd.crosstab(final_assignments['PROM_SEG_RAW'], 
                               final_assignments['BANKER_TYPE'], 
                               margins=True)
        print(crosstab)

    # Merge with df1 to get ECN_LAT and ECN_LON
    final_assignments = final_assignments.merge(
        df1[['CG_ECN', 'HH_ECN', 'ECN_LAT', 'ECN_LON']].drop_duplicates(),
        on=['CG_ECN', 'HH_ECN'],
        how='left'
    )
    
    # Create banker output file with locations
    banker_output = df2.copy()
    
    # Merge IN MARKET bankers with branch locations
    banker_output = banker_output.merge(
        df3[['AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
        on='AU', 
        how='left'
    )
    
    # Merge all bankers with portfolio centroids
    banker_output = banker_output.merge(
        df4[['PORT_CODE', 'PORT_CENTROID_LAT', 'PORT_CENTROID_LON']], 
        on='PORT_CODE', 
        how='left'
    )
    
    # Set LAT and LON based on ROLE_TYPE
    banker_output['BANKER_LAT'] = banker_output.apply(
        lambda row: row['BRANCH_LAT_NUM'] if row['ROLE_TYPE'] == 'IN MARKET' else row['PORT_CENTROID_LAT'],
        axis=1
    )
    banker_output['BANKER_LON'] = banker_output.apply(
        lambda row: row['BRANCH_LON_NUM'] if row['ROLE_TYPE'] == 'IN MARKET' else row['PORT_CENTROID_LON'],
        axis=1
    )
    
    # Drop temporary columns
    banker_output = banker_output.drop(columns=['BRANCH_LAT_NUM', 'BRANCH_LON_NUM', 'PORT_CENTROID_LAT', 'PORT_CENTROID_LON'])
    
    # Save banker output
    banker_output.to_csv('active_bankers_with_locations.csv', index=False)
    print(f"\n✓ Banker locations saved to: active_bankers_with_locations.csv")
    
    return final_assignments[['CG_ECN', 'HH_ECN', 'PORT_CODE', 'PROM_SEG_RAW', 'BANKER_TYPE', 'ECN_LAT', 'ECN_LON']]

# Run the assignment
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Assuming df1, df2, df3, df4, df5 are already loaded
    result = main(df1, df2, df3, df4, df5)
    
    # Display sample results
    print(f"\n{'='*60}")
    print("SAMPLE ASSIGNMENTS (First 10 rows)")
    print(f"{'='*60}")
    print(result.head(10).to_string(index=False))
    
    # Save to CSV
    output_file = 'customer_banker_assignments.csv'
    result.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
