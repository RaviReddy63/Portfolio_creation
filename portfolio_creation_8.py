import pandas as pd
import numpy as np
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
    radius_rad = max_radius / 3959  # Convert miles to radians
    
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
            else:
                for idx in current_cluster:
                    unassigned_mask[idx] = False
        else:
            for idx in current_cluster:
                unassigned_mask[idx] = False
        
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
    
    unassigned_customers = [idx for idx in customer_indices if idx not in assigned_customers]
    customer_assignments = {k: v for k, v in customer_assignments.items() if v}
    
    for branch_au in customer_assignments:
        customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return customer_assignments, unassigned_customers

def assign_proximity_customers_to_existing_portfolios(unassigned_customers_df, customer_assignments, branch_df, proximity_threshold=20, max_portfolio_size=250):
    """Check if unassigned customers are within proximity of identified AUs"""
    if len(unassigned_customers_df) == 0 or not customer_assignments:
        return [], list(unassigned_customers_df.index), customer_assignments
    
    identified_aus = list(customer_assignments.keys())
    identified_branch_coords = branch_df[branch_df['AU'].isin(identified_aus)].copy()
    
    current_portfolio_sizes = {}
    for branch_au, customers in customer_assignments.items():
        current_portfolio_sizes[branch_au] = len(customers)
    
    unassigned_coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values.astype(np.float64)
    
    distance_matrix = compute_distance_matrix(unassigned_coords, branch_coords)
    
    proximity_results = []
    remaining_unassigned = []
    updated_customer_assignments = customer_assignments.copy()
    
    for i, (customer_idx, customer_data) in enumerate(unassigned_customers_df.iterrows()):
        assigned = False
        
        customer_distances = distance_matrix[i, :]
        within_proximity = customer_distances <= proximity_threshold
        
        if np.any(within_proximity):
            proximity_aus = []
            for j, is_within in enumerate(within_proximity):
                if is_within:
                    branch_au = identified_branch_coords.iloc[j]['AU']
                    distance = customer_distances[j]
                    current_size = current_portfolio_sizes[branch_au]
                    
                    if current_size < max_portfolio_size:
                        proximity_aus.append((branch_au, distance, current_size))
            
            if proximity_aus:
                proximity_aus.sort(key=lambda x: x[1])
                
                for branch_au, distance, current_size in proximity_aus:
                    if current_portfolio_sizes[branch_au] < max_portfolio_size:
                        updated_customer_assignments[branch_au].append({
                            'customer_idx': customer_idx,
                            'distance': distance
                        })
                        
                        current_portfolio_sizes[branch_au] += 1
                        
                        proximity_results.append({
                            'ECN': customer_data['CG_ECN'],
                            'BILLINGCITY': customer_data['BILLINGCITY'],
                            'BILLINGSTATE': customer_data['BILLINGSTATE'],
                            'LAT_NUM': customer_data['LAT_NUM'],
                            'LON_NUM': customer_data['LON_NUM'],
                            'ASSIGNED_AU': branch_au,
                            'DISTANCE_TO_AU': distance,
                            'TYPE': 'INMARKET'
                        })
                        
                        assigned = True
                        break
        
        if not assigned:
            remaining_unassigned.append(customer_idx)
    
    for branch_au in updated_customer_assignments:
        updated_customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return proximity_results, remaining_unassigned, updated_customer_assignments

def create_centralized_clusters_with_radius_and_assign(unassigned_customers_df, branch_df, min_size=200, max_size=240, max_radius=100):
    """Create centralized clusters WITH radius constraint and assign to branches"""
    if len(unassigned_customers_df) == 0:
        return [], []
    
    clustered_centralized, centralized_cluster_info = constrained_clustering_with_radius(
        unassigned_customers_df, min_size=min_size, max_size=max_size, max_radius=max_radius
    )
    
    centralized_results = []
    final_unassigned = []
    
    if len(centralized_cluster_info) > 0:
        cluster_assignments = assign_clusters_to_branches_vectorized(
            centralized_cluster_info, branch_df
        )
        
        for _, assignment in cluster_assignments.iterrows():
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
                
                original_customer = unassigned_customers_df.loc[idx]
                
                centralized_results.append({
                    'customer_idx': idx,
                    'ECN': original_customer['CG_ECN'],
                    'BILLINGCITY': original_customer['BILLINGCITY'],
                    'BILLINGSTATE': original_customer['BILLINGSTATE'],
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
        final_unassigned = list(unassigned_customers_df.index)
    
    return centralized_results, final_unassigned

def enhanced_customer_au_assignment_with_two_inmarket_iterations(customer_df, branch_df, 
                                                               min_size=200, max_inmarket_size=225, 
                                                               max_centralized_size=240, max_proximity_size=250,
                                                               inmarket_radius_first=20, inmarket_radius_second=40,
                                                               centralized_radius=100):
    """Enhanced main function with two INMARKET iterations and centralized portfolios - NOW WITH CONFIGURABLE SIZES AND RADIUS
    
    Args:
        customer_df: DataFrame with customer data
        branch_df: DataFrame with branch/AU data
        min_size: Minimum portfolio size (default 200)
        max_inmarket_size: Maximum size for in-market portfolios (default 225)
        max_centralized_size: Maximum size for centralized portfolios (default 240)
        max_proximity_size: Maximum size after proximity assignment (default 250)
        inmarket_radius_first: Radius for first in-market iteration in miles (default 20)
        inmarket_radius_second: Radius for second in-market iteration in miles (default 40)
        centralized_radius: Radius for centralized portfolios in miles (default 100)
    """
    
    # Step 1: Create first INMARKET clusters - USE inmarket_radius_first PARAMETER
    clustered_customers, cluster_info = constrained_clustering_optimized(
        customer_df, min_size=min_size, max_size=max_inmarket_size, max_radius=inmarket_radius_first
    )
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        cluster_assignments = assign_clusters_to_branches_vectorized(cluster_info, branch_df)
        
        # USE max_inmarket_size and inmarket_radius_first PARAMETERS
        customer_assignments, unassigned = greedy_assign_customers_to_branches(
            clustered_customers, cluster_assignments, branch_df, max_distance=inmarket_radius_first, max_customers_per_branch=max_inmarket_size
        )
        
        for branch_au, customers in customer_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                
                inmarket_results.append({
                    'ECN': customer_data['CG_ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': distance_value,
                    'TYPE': 'INMARKET'
                })
        
        unassigned_customer_indices.extend(unassigned)
    
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    # Step 2: Check proximity of unassigned customers to identified AUs - USE max_proximity_size PARAMETER
    proximity_results = []
    unassigned_after_proximity = unassigned_customer_indices.copy()
    
    if unassigned_customer_indices and inmarket_results:
        customer_assignments = {}
        for result in inmarket_results:
            au = result['ASSIGNED_AU']
            if au not in customer_assignments:
                customer_assignments[au] = []
            
            customer_idx = customer_df[
                (customer_df['CG_ECN'] == result['ECN']) &
                (customer_df['LAT_NUM'] == result['LAT_NUM']) &
                (customer_df['LON_NUM'] == result['LON_NUM'])
            ].index[0]
            
            customer_assignments[au].append({
                'customer_idx': customer_idx,
                'distance': result['DISTANCE_TO_AU']
            })
        
        unassigned_customers_df = customer_df.loc[unassigned_customer_indices]
        
        # USE max_proximity_size and inmarket_radius_first PARAMETERS
        proximity_results, unassigned_after_proximity, updated_customer_assignments = assign_proximity_customers_to_existing_portfolios(
            unassigned_customers_df, customer_assignments, branch_df, 
            proximity_threshold=inmarket_radius_first, max_portfolio_size=max_proximity_size
        )
    
    # Step 3: Create second INMARKET clusters - USE max_inmarket_size and inmarket_radius_second PARAMETERS
    second_inmarket_results = []
    unassigned_after_second_inmarket = unassigned_after_proximity.copy()
    
    if unassigned_after_proximity:
        remaining_customers_df = customer_df.loc[unassigned_after_proximity]
        
        # USE max_inmarket_size and inmarket_radius_second PARAMETERS
        clustered_customers_2, cluster_info_2 = constrained_clustering_optimized(
            remaining_customers_df, min_size=min_size, max_size=max_inmarket_size, max_radius=inmarket_radius_second
        )
        
        if len(cluster_info_2) > 0:
            cluster_assignments_2 = assign_clusters_to_branches_vectorized(cluster_info_2, branch_df)
            
            # USE max_inmarket_size and inmarket_radius_second PARAMETERS
            customer_assignments_2, unassigned_2 = greedy_assign_customers_to_branches(
                clustered_customers_2, cluster_assignments_2, branch_df, max_distance=inmarket_radius_second, max_customers_per_branch=max_inmarket_size
            )
            
            for branch_au, customers in customer_assignments_2.items():
                for customer in customers:
                    customer_idx = customer['customer_idx']
                    customer_data = remaining_customers_df.loc[customer_idx]
                    
                    distance_value = customer.get('distance', 0)
                    
                    second_inmarket_results.append({
                        'ECN': customer_data['CG_ECN'],
                        'BILLINGCITY': customer_data['BILLINGCITY'],
                        'BILLINGSTATE': customer_data['BILLINGSTATE'],
                        'LAT_NUM': customer_data['LAT_NUM'],
                        'LON_NUM': customer_data['LON_NUM'],
                        'ASSIGNED_AU': branch_au,
                        'DISTANCE_TO_AU': distance_value,
                        'TYPE': 'INMARKET'
                    })
            
            never_assigned_2 = clustered_customers_2[clustered_customers_2['cluster'] == -1].index.tolist()
            unassigned_after_second_inmarket = list(set(unassigned_2 + never_assigned_2))
        else:
            unassigned_after_second_inmarket = unassigned_after_proximity.copy()
    
    # Step 4: Create CENTRALIZED clusters - USE max_centralized_size and centralized_radius PARAMETERS
    centralized_results = []
    final_unassigned = []
    
    if unassigned_after_second_inmarket:
        remaining_unassigned_df = customer_df.loc[unassigned_after_second_inmarket]
        
        # USE min_size, max_centralized_size, and centralized_radius PARAMETERS
        centralized_results, final_unassigned = create_centralized_clusters_with_radius_and_assign(
            remaining_unassigned_df, branch_df, min_size=min_size, max_size=max_centralized_size, max_radius=centralized_radius
        )
    
    # Combine all results
    all_results = inmarket_results + proximity_results + second_inmarket_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    return result_df
