import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from utils import haversine_distance

def enhanced_customer_au_assignment_with_two_inmarket_iterations(
    customer_data,
    branch_data,
    min_portfolio_size=200,
    max_portfolio_size_inmarket=240,
    max_portfolio_size_proximity=250,
    max_portfolio_size_centralized=240,
    inmarket_radius_miles=20,
    centralized_radius_miles=100
):
    """
    Enhanced customer-to-AU assignment with configurable portfolio sizes and radii
    
    Parameters:
    -----------
    customer_data : pd.DataFrame
        Customer data with LAT_NUM, LON_NUM columns
    branch_data : pd.DataFrame  
        Branch data with AU, LAT, LON columns
    min_portfolio_size : int
        Minimum customers per portfolio (default: 200)
    max_portfolio_size_inmarket : int
        Maximum size for in-market portfolios (default: 240)
    max_portfolio_size_proximity : int
        Maximum size for proximity portfolios (default: 250)
    max_portfolio_size_centralized : int
        Maximum size for centralized portfolios (default: 240)
    inmarket_radius_miles : float
        Radius for in-market assignments in miles (default: 20)
        Second iteration automatically uses 2x this value
    centralized_radius_miles : float
        Radius for centralized assignments in miles (default: 100)
    
    Returns:
    --------
    pd.DataFrame
        Customer data with ASSIGNED_AU, ASSIGNMENT_TYPE, DISTANCE_TO_AU columns
    """
    
    # Initialize result DataFrame
    results = customer_data.copy()
    results['ASSIGNED_AU'] = None
    results['ASSIGNMENT_TYPE'] = None
    results['DISTANCE_TO_AU'] = None
    
    # Preserve ECN column (rename if needed for consistency)
    if 'CG_ECN' in results.columns and 'ECN' not in results.columns:
        results['ECN'] = results['CG_ECN']
    elif 'ECN' not in results.columns:
        st.error("No customer ID column (ECN or CG_ECN) found in data")
        return results
    
    unassigned_mask = results['ASSIGNED_AU'].isna()
    
    # Calculate second iteration radius (double the first)
    inmarket_radius_miles_second = inmarket_radius_miles * 2
    
    # ============================================================
    # PHASE 1: IN-MARKET ASSIGNMENTS (FIRST ITERATION)
    # ============================================================
    print(f"\n=== Phase 1: In-Market Assignment (First Iteration - {inmarket_radius_miles} miles) ===")
    
    for _, branch in branch_data.iterrows():
        au = branch['AU']
        branch_lat = branch['LAT']
        branch_lon = branch['LON']
        
        # Find unassigned customers within radius
        unassigned = results[unassigned_mask].copy()
        
        if len(unassigned) == 0:
            break
        
        unassigned['distance'] = unassigned.apply(
            lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], branch_lat, branch_lon),
            axis=1
        )
        
        nearby = unassigned[unassigned['distance'] <= inmarket_radius_miles].sort_values('distance')
        
        if len(nearby) >= min_portfolio_size:
            assigned = nearby.head(max_portfolio_size_inmarket)
            results.loc[assigned.index, 'ASSIGNED_AU'] = au
            results.loc[assigned.index, 'ASSIGNMENT_TYPE'] = 'In-Market (First)'
            results.loc[assigned.index, 'DISTANCE_TO_AU'] = assigned['distance'].values
            unassigned_mask = results['ASSIGNED_AU'].isna()
            
            print(f"AU {au}: Assigned {len(assigned)} customers (In-Market First)")
    
    # ============================================================
    # PHASE 2: IN-MARKET ASSIGNMENTS (SECOND ITERATION - DOUBLE RADIUS)
    # ============================================================
    print(f"\n=== Phase 2: In-Market Assignment (Second Iteration - {inmarket_radius_miles_second} miles) ===")
    
    for _, branch in branch_data.iterrows():
        au = branch['AU']
        branch_lat = branch['LAT']
        branch_lon = branch['LON']
        
        # Skip if AU already has a portfolio
        if au in results[results['ASSIGNED_AU'].notna()]['ASSIGNED_AU'].values:
            continue
        
        unassigned = results[unassigned_mask].copy()
        
        if len(unassigned) == 0:
            break
        
        unassigned['distance'] = unassigned.apply(
            lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], branch_lat, branch_lon),
            axis=1
        )
        
        nearby = unassigned[unassigned['distance'] <= inmarket_radius_miles_second].sort_values('distance')
        
        if len(nearby) >= min_portfolio_size:
            assigned = nearby.head(max_portfolio_size_inmarket)
            results.loc[assigned.index, 'ASSIGNED_AU'] = au
            results.loc[assigned.index, 'ASSIGNMENT_TYPE'] = 'In-Market (Second)'
            results.loc[assigned.index, 'DISTANCE_TO_AU'] = assigned['distance'].values
            unassigned_mask = results['ASSIGNED_AU'].isna()
            
            print(f"AU {au}: Assigned {len(assigned)} customers (In-Market Second)")
    
    # ============================================================
    # PHASE 3: PROXIMITY CLUSTERING
    # ============================================================
    print("\n=== Phase 3: Proximity Clustering ===")
    
    unassigned = results[unassigned_mask].copy()
    
    if len(unassigned) > 0:
        # Determine number of clusters
        num_clusters = max(1, len(unassigned) // max_portfolio_size_proximity)
        
        if num_clusters > 0:
            # Perform KMeans clustering
            coords = unassigned[['LAT_NUM', 'LON_NUM']].values
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            unassigned['cluster'] = kmeans.fit_predict(coords)
            
            # Assign clusters to nearest AUs
            for cluster_id in range(num_clusters):
                cluster_customers = unassigned[unassigned['cluster'] == cluster_id]
                
                if len(cluster_customers) >= min_portfolio_size:
                    # Find centroid
                    centroid_lat = cluster_customers['LAT_NUM'].mean()
                    centroid_lon = cluster_customers['LON_NUM'].mean()
                    
                    # Find nearest available AU
                    available_aus = branch_data[~branch_data['AU'].isin(results[results['ASSIGNED_AU'].notna()]['ASSIGNED_AU'].values)]
                    
                    if len(available_aus) > 0:
                        available_aus['distance_to_centroid'] = available_aus.apply(
                            lambda row: haversine_distance(centroid_lat, centroid_lon, row['LAT'], row['LON']),
                            axis=1
                        )
                        nearest_au = available_aus.sort_values('distance_to_centroid').iloc[0]['AU']
                        
                        # Calculate distances and assign
                        nearest_branch = branch_data[branch_data['AU'] == nearest_au].iloc[0]
                        cluster_customers['distance'] = cluster_customers.apply(
                            lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], nearest_branch['LAT'], nearest_branch['LON']),
                            axis=1
                        )
                        
                        assigned = cluster_customers.head(max_portfolio_size_proximity)
                        results.loc[assigned.index, 'ASSIGNED_AU'] = nearest_au
                        results.loc[assigned.index, 'ASSIGNMENT_TYPE'] = 'Proximity'
                        results.loc[assigned.index, 'DISTANCE_TO_AU'] = assigned['distance'].values
                        unassigned_mask = results['ASSIGNED_AU'].isna()
                        
                        print(f"AU {nearest_au}: Assigned {len(assigned)} customers (Proximity)")
    
    # ============================================================
    # PHASE 4: CENTRALIZED ASSIGNMENTS
    # ============================================================
    print("\n=== Phase 4: Centralized Assignments ===")
    
    unassigned = results[unassigned_mask].copy()
    
    if len(unassigned) > 0:
        available_aus = branch_data[~branch_data['AU'].isin(results[results['ASSIGNED_AU'].notna()]['ASSIGNED_AU'].values)]
        
        for _, branch in available_aus.iterrows():
            au = branch['AU']
            branch_lat = branch['LAT']
            branch_lon = branch['LON']
            
            unassigned = results[unassigned_mask].copy()
            
            if len(unassigned) == 0:
                break
            
            unassigned['distance'] = unassigned.apply(
                lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], branch_lat, branch_lon),
                axis=1
            )
            
            nearby = unassigned[unassigned['distance'] <= centralized_radius_miles].sort_values('distance')
            
            if len(nearby) >= min_portfolio_size:
                assigned = nearby.head(max_portfolio_size_centralized)
                results.loc[assigned.index, 'ASSIGNED_AU'] = au
                results.loc[assigned.index, 'ASSIGNMENT_TYPE'] = 'Centralized'
                results.loc[assigned.index, 'DISTANCE_TO_AU'] = assigned['distance'].values
                unassigned_mask = results['ASSIGNED_AU'].isna()
                
                print(f"AU {au}: Assigned {len(assigned)} customers (Centralized)")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n=== Assignment Summary ===")
    print(f"Total customers: {len(results):,}")
    print(f"Assigned: {(~results['ASSIGNED_AU'].isna()).sum():,}")
    print(f"Unassigned: {results['ASSIGNED_AU'].isna().sum():,}")
    
    if (~results['ASSIGNED_AU'].isna()).sum() > 0:
        print("\nBy Assignment Type:")
        print(results[results['ASSIGNED_AU'].notna()]['ASSIGNMENT_TYPE'].value_counts())
    
    # Add TYPE column for compatibility (copy from ASSIGNMENT_TYPE)
    results['TYPE'] = results['ASSIGNMENT_TYPE']
    
    return results
