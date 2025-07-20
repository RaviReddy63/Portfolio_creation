import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    R = 3959  # Earth's radius in miles
    
    # Convert to numpy arrays if not already and ensure proper dtype
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical errors
    
    distance = R * c
    
    # Handle scalar case
    if np.isscalar(distance):
        return float(distance)
    return distance

def compute_distance_matrix(coords1, coords2):
    """Compute distance matrix between two sets of coordinates"""
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)
    
    lat1 = coords1[:, 0][:, np.newaxis]
    lon1 = coords1[:, 1][:, np.newaxis]
    lat2 = coords2[:, 0][np.newaxis, :]
    lon2 = coords2[:, 1][np.newaxis, :]
    
    return haversine_distance_vectorized(lat1, lon1, lat2, lon2)

def calculate_cluster_radius_vectorized(coords):
    """Vectorized calculation of cluster radius"""
    coords = np.asarray(coords, dtype=np.float64)
    
    if len(coords) <= 1:
        return 0.0
    
    centroid = coords.mean(axis=0)
    distances = haversine_distance_vectorized(
        coords[:, 0], coords[:, 1], 
        centroid[0], centroid[1]
    )
    return float(np.max(distances))

def find_candidates_spatial(customer_coords, seed_coord, max_radius, ball_tree=None):
    """Use spatial indexing to find candidates within radius"""
    customer_coords = np.asarray(customer_coords, dtype=np.float64)
    seed_coord = np.asarray(seed_coord, dtype=np.float64)
    
    if ball_tree is None:
        coords_rad = np.radians(customer_coords)
        ball_tree = BallTree(coords_rad, metric='haversine')
    
    seed_rad = np.radians(seed_coord).reshape(1, -1)
    radius_rad = max_radius / 3959
    
    indices = ball_tree.query_radius(seed_rad, r=radius_rad)[0]
    
    if len(indices) > 0:
        candidate_coords = customer_coords[indices]
        distances = haversine_distance_vectorized(
            candidate_coords[:, 0], candidate_coords[:, 1],
            seed_coord[0], seed_coord[1]
        )
        
        sorted_indices = np.argsort(distances)
        return indices[sorted_indices], distances[sorted_indices]
    
    return np.array([]), np.array([])

def constrained_clustering_optimized(customer_df, min_size=200, max_size=225, max_radius=20):
    """Optimized clustering with vectorized operations and spatial indexing"""
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    
    coords_rad = np.radians(coords)
    ball_tree = BallTree(coords_rad, metric='haversine')
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    unassigned_count = np.count_nonzero(unassigned_mask)
    while unassigned_count >= min_size:
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        candidate_indices, distances = find_candidates_spatial(
            coords[unassigned_mask], seed_coord, max_radius, None
        )
        
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[candidate_indices]
        
        if len(candidate_original_indices) == 0:
            unassigned_mask[seed_idx] = False
            unassigned_count = np.count_nonzero(unassigned_mask)
            continue
        
        current_cluster = [seed_idx]
        current_coords = [coords[seed_idx]]
        
        for i, (candidate_idx, distance) in enumerate(zip(candidate_original_indices, distances)):
            if candidate_idx == seed_idx:
                continue
                
            if len(current_cluster) >= max_size:
                break
            
            test_coords = np.array(current_coords + [coords[candidate_idx]], dtype=np.float64)
            test_radius = calculate_cluster_radius_vectorized(test_coords)
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append(coords[candidate_idx])
        
        if len(current_cluster) < min_size:
            unassigned_mask[seed_idx] = False
        elif len(current_cluster) <= max_size:
            cluster_coords = np.array(current_coords, dtype=np.float64)
            cluster_radius = calculate_cluster_radius_vectorized(cluster_coords)
            
            for idx in current_cluster:
                customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                unassigned_mask[idx] = False
            
            final_clusters.append({
                'cluster_id': cluster_id,
                'size': len(current_cluster),
                'radius': cluster_radius,
                'centroid_lat': np.mean(cluster_coords[:, 0]),
                'centroid_lon': np.mean(cluster_coords[:, 1])
            })
            
            cluster_id += 1
        else:
            coords_array = np.array(current_coords, dtype=np.float64)
            
            if len(current_cluster) > 0 and min_size > 0:
                if len(current_cluster) // min_size < 3:
                    n_splits = np.maximum(1, len(current_cluster) // min_size)
                else:
                    n_splits = 3
            else:
                n_splits = 1
            
            if n_splits > 1:
                kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                subcluster_labels = kmeans.fit_predict(coords_array)
                
                for sub_id in range(n_splits):
                    sub_mask = subcluster_labels == sub_id
                    sub_indices = [current_cluster[i] for i in range(len(current_cluster)) if sub_mask[i]]
                    
                    if len(sub_indices) >= min_size and len(sub_indices) <= max_size:
                        sub_coords = coords_array[sub_mask]
                        sub_radius = calculate_cluster_radius_vectorized(sub_coords)
                        
                        if sub_radius <= max_radius:
                            for idx in sub_indices:
                                customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                                unassigned_mask[idx] = False
                            
                            final_clusters.append({
                                'cluster_id': cluster_id,
                                'size': len(sub_indices),
                                'radius': sub_radius,
                                'centroid_lat': np.mean(sub_coords[:, 0]),
                                'centroid_lon': np.mean(sub_coords[:, 1])
                            })
                            cluster_id += 1
            
            for idx in current_cluster:
                unassigned_mask[idx] = False
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    return customers_clean, pd.DataFrame(final_clusters)

def constrained_clustering_with_radius(customer_df, min_size=200, max_size=240, max_radius=100):
    """Clustering with both size constraints AND radius constraint for centralized portfolios"""
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    coords_rad = np.radians(coords)
    ball_tree = BallTree(coords_rad, metric='haversine')
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    unassigned_count = np.count_nonzero(unassigned_mask)
    while unassigned_count >= min_size:
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        candidate_indices, distances = find_candidates_spatial(
            coords[unassigned_mask], seed_coord, max_radius, None
        )
        
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[candidate_indices]
        
        if len(candidate_original_indices) == 0:
            unassigned_mask[seed_idx] = False
            unassigned_count = np.count_nonzero(unassigned_mask)
            continue
        
        current_cluster = [seed_idx]
        current_coords = [coords[seed_idx]]
        
        for i, (candidate_idx, distance) in enumerate(zip(candidate_original_indices, distances)):
            if candidate_idx == seed_idx:
                continue
                
            if len(current_cluster) >= max_size:
                break
            
            test_coords = np.array(current_coords + [coords[candidate_idx]], dtype=np.float64)
            test_radius = calculate_cluster_radius_vectorized(test_coords)
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append(coords[candidate_idx])
        
        if len(current_cluster) < min_size:
            unassigned_mask[seed_idx] = False
        elif len(current_cluster) <= max_size:
            cluster_coords = np.array(current_coords, dtype=np.float64)
            cluster_radius = calculate_cluster_radius_vectorized(cluster_coords)
            
            if cluster_radius <= max_radius:
                for idx in current_cluster:
                    customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                    unassigned_mask[idx] = False
                
                final_clusters.append({
                    'cluster_id': cluster_id,
                    'size': len(current_cluster),
                    'radius': cluster_radius,
                    'centroid_lat': np.mean(cluster_coords[:, 0]),
                    'centroid_lon': np.mean(cluster_coords[:, 1])
                })
                
                cluster_id += 1
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    return customers_clean, pd.DataFrame(final_clusters)

def assign_clusters_to_branches_vectorized(cluster_info, branch_df):
    """Vectorized cluster to branch assignment"""
    if len(cluster_info) == 0:
        return pd.DataFrame()
    
    cluster_coords = cluster_info[['centroid_lat', 'centroid_lon']].values.astype(np.float64)
    branch_coords = branch_df[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values.astype(np.float64)
    
    distance_matrix = compute_distance_matrix(cluster_coords, branch_coords)
    
    nearest_branch_indices = np.argmin(distance_matrix, axis=1)
    min_distances = np.min(distance_matrix, axis=1)
    
    cluster_assignments = []
    for i, cluster in cluster_info.iterrows():
        branch_idx = nearest_branch_indices[i]
        assigned_branch = branch_df.iloc[branch_idx]['AU']
        distance = min_distances[i]
        
        cluster_assignments.append({
            'cluster_id': cluster['cluster_id'],
            'assigned_branch': assigned_branch,
            'cluster_to_branch_distance': distance
        })
    
    return pd.DataFrame(cluster_assignments)

def greedy_assign_customers_to_branches(clustered_customers, cluster_assignments, branch_df, max_distance=20, max_customers_per_branch=225):
    """Fast greedy assignment of customers to branches"""
    
    identified_branches = cluster_assignments['assigned_branch'].unique()
    identified_branch_coords = branch_df[branch_df['AU'].isin(identified_branches)].copy()
    
    customers_to_assign = clustered_customers[clustered_customers['cluster'] != -1].copy()
    
    if len(customers_to_assign) == 0 or len(identified_branch_coords) == 0:
        return {}, list(customers_to_assign.index)
    
    customer_coords = customers_to_assign[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values.astype(np.float64)
    
    distance_matrix = compute_distance_matrix(customer_coords, branch_coords)
    distance_matrix[distance_matrix > max_distance] = np.inf
    
    customer_indices = list(customers_to_assign.index)
    branch_aus = list(identified_branch_coords['AU'])
    
    branch_capacity = {branch_au: max_customers_per_branch for branch_au in branch_aus}
    customer_assignments = {branch_au: [] for branch_au in branch_aus}
    unassigned_customers = []
    
    assignment_candidates = []
    for i, customer_idx in enumerate(customer_indices):
        for j, branch_au in enumerate(branch_aus):
            distance = distance_matrix[i, j]
            if distance < np.inf:
                assignment_candidates.append((i, customer_idx, j, branch_au, distance))
    
    assignment_candidates.sort(key=lambda x: x[4])
    
    assigned_customers = set()
    
    for customer_i, customer_idx, branch_j, branch_au, distance in assignment_candidates:
        if customer_idx in assigned_customers:
            continue
        
        if branch_capacity[branch_au] <= 0:
            continue
        
        customer_assignments[branch_au].append({
            'customer_idx': customer_idx,
            'distance': distance
        })
        
        assigned_customers.add(customer_idx)
        branch_capacity[branch_au] -= 1
    
    for customer_idx in customer_indices:
        if customer_idx not in assigned_customers:
            unassigned_customers.append(customer_idx)
    
    customer_assignments = {k: v for k, v in customer_assignments.items() if v}
    
    for branch_au in customer_assignments:
        customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return customer_assignments, unassigned_customers

def enhanced_customer_au_assignment_with_streamlit_progress(customer_df, branch_df):
    """Enhanced main function with Streamlit progress indicators"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"Starting enhanced assignment with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Step 1: Create first INMARKET clusters (20-mile radius)
    progress_bar.progress(10)
    status_text.text("Step 1: Creating first INMARKET clusters (20-mile radius)...")
    
    clustered_customers, cluster_info = constrained_clustering_optimized(
        customer_df, min_size=200, max_size=225, max_radius=20
    )
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        status_text.text(f"Created {len(cluster_info)} first INMARKET clusters")
        
        # Step 2: Assign clusters to branches
        progress_bar.progress(20)
        status_text.text("Step 2: Assigning first INMARKET clusters to branches...")
        cluster_assignments = assign_clusters_to_branches_vectorized(cluster_info, branch_df)
        
        # Step 3: Use greedy assignment for customer-AU assignment
        progress_bar.progress(30)
        status_text.text("Step 3: Using greedy assignment for first INMARKET customer-AU assignment...")
        customer_assignments, unassigned = greedy_assign_customers_to_branches(
            clustered_customers, cluster_assignments, branch_df
        )
        
        # Create first INMARKET results
        for branch_au, customers in customer_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                
                inmarket_results.append({
                    'CG_ECN': customer_data.get('CG_ECN', customer_data.get('ECN', '')),
                    'BILLINGCITY': customer_data.get('BILLINGCITY', ''),
                    'BILLINGSTATE': customer_data.get('BILLINGSTATE', ''),
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': distance_value,
                    'TYPE': 'INMARKET'
                })
        
        unassigned_customer_indices.extend(unassigned)
    
    # Add customers that were never assigned to any cluster
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    status_text.text(f"Total unassigned customers after first INMARKET: {len(unassigned_customer_indices)}")
    
    # Step 4: Create second INMARKET clusters (40-mile radius)
    progress_bar.progress(50)
    status_text.text("Step 4: Creating second INMARKET clusters (40-mile radius)...")
    
    second_inmarket_results = []
    unassigned_after_second_inmarket = unassigned_customer_indices.copy()
    
    if unassigned_customer_indices:
        remaining_customers_df = customer_df.loc[unassigned_customer_indices]
        
        clustered_customers_2, cluster_info_2 = constrained_clustering_optimized(
            remaining_customers_df, min_size=200, max_size=225, max_radius=40
        )
        
        if len(cluster_info_2) > 0:
            status_text.text(f"Created {len(cluster_info_2)} second INMARKET clusters")
            
            cluster_assignments_2 = assign_clusters_to_branches_vectorized(cluster_info_2, branch_df)
            customer_assignments_2, unassigned_2 = greedy_assign_customers_to_branches(
                clustered_customers_2, cluster_assignments_2, branch_df
            )
            
            for branch_au, customers in customer_assignments_2.items():
                for customer in customers:
                    customer_idx = customer['customer_idx']
                    customer_data = remaining_customers_df.loc[customer_idx]
                    
                    distance_value = customer.get('distance', 0)
                    
                    second_inmarket_results.append({
                        'CG_ECN': customer_data.get('CG_ECN', customer_data.get('ECN', '')),
                        'BILLINGCITY': customer_data.get('BILLINGCITY', ''),
                        'BILLINGSTATE': customer_data.get('BILLINGSTATE', ''),
                        'LAT_NUM': customer_data['LAT_NUM'],
                        'LON_NUM': customer_data['LON_NUM'],
                        'ASSIGNED_AU': branch_au,
                        'DISTANCE_TO_AU': distance_value,
                        'TYPE': 'INMARKET'
                    })
            
            never_assigned_2 = clustered_customers_2[clustered_customers_2['cluster'] == -1].index.tolist()
            unassigned_after_second_inmarket = list(set(unassigned_2 + never_assigned_2))
        else:
            unassigned_after_second_inmarket = unassigned_customer_indices.copy()
    
    # Step 5: Create CENTRALIZED clusters
    progress_bar.progress(70)
    status_text.text("Step 5: Creating CENTRALIZED clusters...")
    
    centralized_results = []
    final_unassigned = []
    
    if unassigned_after_second_inmarket:
        remaining_unassigned_df = customer_df.loc[unassigned_after_second_inmarket]
        
        clustered_centralized, centralized_cluster_info = constrained_clustering_with_radius(
            remaining_unassigned_df, min_size=200, max_size=240, max_radius=100
        )
        
        if len(centralized_cluster_info) > 0:
            cluster_assignments_cent = assign_clusters_to_branches_vectorized(
                centralized_cluster_info, branch_df
            )
            
            for _, assignment in cluster_assignments_cent.iterrows():
                cluster_id = assignment['cluster_id']
                assigned_branch = assignment['assigned_branch']
                
                cluster_customers = clustered_centralized[
                    clustered_centralized['cluster'] == cluster_id
                ]
                
                branch_coords = branch_df[
                    branch_df['AU'] == assigned_branch
                ][['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].iloc[0]
                
                for idx, customer in cluster_customers.iterrows():
                    distance = haversine_distance_vectorized(
                        customer['LAT_NUM'], customer['LON_NUM'],
                        branch_coords['BRANCH_LAT_NUM'], branch_coords['BRANCH_LON_NUM']
                    )
                    
                    original_customer = remaining_unassigned_df.loc[idx]
                    
                    centralized_results.append({
                        'CG_ECN': original_customer.get('CG_ECN', original_customer.get('ECN', '')),
                        'BILLINGCITY': original_customer.get('BILLINGCITY', ''),
                        'BILLINGSTATE': original_customer.get('BILLINGSTATE', ''),
                        'LAT_NUM': original_customer['LAT_NUM'],
                        'LON_NUM': original_customer['LON_NUM'],
                        'ASSIGNED_AU': assigned_branch,
                        'DISTANCE_TO_AU': distance,
                        'TYPE': 'CENTRALIZED',
                        'CLUSTER_ID': cluster_id
                    })
            
            unassigned_centralized = clustered_centralized[
                clustered_centralized['cluster'] == -1
            ]
            final_unassigned = list(unassigned_centralized.index)
        else:
            final_unassigned = list(remaining_unassigned_df.index)
    
    # Combine all results
    progress_bar.progress(90)
    status_text.text("Finalizing results...")
    
    all_results = inmarket_results + second_inmarket_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    progress_bar.progress(100)
    status_text.text("Portfolio mapping completed!")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return result_df, final_unassigned

def convert_mapping_results_to_portfolio_format(result_df, branch_df):
    """Convert mapping results to portfolio assignment format"""
    if len(result_df) == 0:
        return {}, {}
    
    portfolios_created = {}
    portfolio_summaries = {}
    
    # Group by assigned AU
    for au_id in result_df['ASSIGNED_AU'].unique():
        au_customers = result_df[result_df['ASSIGNED_AU'] == au_id].copy()
        
        # Add required columns for compatibility
        au_customers['AU'] = au_id
        au_customers['PORT_CODE'] = 'MAPPED_' + str(au_id)
        au_customers['TYPE'] = au_customers['TYPE']  # Keep INMARKET or CENTRALIZED
        au_customers['Distance'] = au_customers['DISTANCE_TO_AU']
        
        # Get branch coordinates
        branch_info = branch_df[branch_df['AU'] == au_id]
        if not branch_info.empty:
            au_customers['BRANCH_LAT_NUM'] = branch_info.iloc[0]['BRANCH_LAT_NUM']
            au_customers['BRANCH_LON_NUM'] = branch_info.iloc[0]['BRANCH_LON_NUM']
        
        # Add dummy financial data if not present
        if 'BANK_REVENUE' not in au_customers.columns:
            au_customers['BANK_REVENUE'] = np.random.uniform(5000, 50000, len(au_customers))
        if 'DEPOSIT_BAL' not in au_customers.columns:
            au_customers['DEPOSIT_BAL'] = np.random.uniform(100000, 1000000, len(au_customers))
        
        portfolios_created[au_id] = au_customers
        
        # Create portfolio summary
        inmarket_count = len(au_customers[au_customers['TYPE'] == 'INMARKET'])
        centralized_count = len(au_customers[au_customers['TYPE'] == 'CENTRALIZED'])
        
        summary_items = []
        
        if inmarket_count > 0:
            summary_items.append({
                'Include': True,
                'Portfolio ID': f'INMARKET_{au_id}',
                'Portfolio Type': 'INMARKET',
                'Total Customers': inmarket_count,
                'Available for this portfolio': inmarket_count,
                'Select': inmarket_count
            })
        
        if centralized_count > 0:
            summary_items.append({
                'Include': True,
                'Portfolio ID': f'CENTRALIZED_{au_id}',
                'Portfolio Type': 'CENTRALIZED', 
                'Total Customers': centralized_count,
                'Available for this portfolio': centralized_count,
                'Select': centralized_count
            })
        
        portfolio_summaries[au_id] = summary_items
    
    return portfolios_created, portfolio_summaries
