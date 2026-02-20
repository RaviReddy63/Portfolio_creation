import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import builtins

# ==================== CONFIGURATION ====================
NEW_PORTFOLIO_MIN = 280
NEW_PORTFOLIO_MAX = 330
IN_MARKET_MAX_RADIUS = 20       # miles
EARTH_RADIUS_MILES = 3959

# ==================== UTILITY ====================

def get_director_for_state(state_code):
    """Return director name for a given state code."""
    for director, states in DIRECTOR_STATES.items():
        if state_code in states:
            return director
    return None


def compute_centroid(lats, lons):
    """Compute geographic centroid of a set of lat/lon points."""
    return np.mean(lats), np.mean(lons)


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS_MILES * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ==================== STEP 1: IN MARKET CLUSTERING ====================

def build_in_market_clusters(unassigned_df, max_radius=IN_MARKET_MAX_RADIUS):
    """
    Cluster unassigned customers into IN MARKET portfolios.
    - Cluster radius: max 20 miles (uses smallest radius that fits 280-330 customers)
    - Cluster size: 280-330 customers
    - Respects director-state boundaries (no cross-director clusters)
    
    Returns list of cluster dicts with keys:
        customers (list of indices), centroid_lat, centroid_lon, director, state_codes
    """
    clusters = []
    unassigned_df = unassigned_df.copy()
    
    # Add state code if not present
    if 'STATE_CODE' not in unassigned_df.columns:
        unassigned_df['STATE_CODE'] = unassigned_df['BILLINGSTATE'].apply(convert_state_to_code)

    # Work with valid coordinates only
    valid_mask = unassigned_df['LAT_NUM'].notna() & unassigned_df['LON_NUM'].notna()
    working_df = unassigned_df[valid_mask].copy()
    assigned_indices = set()

    print(f"  Starting IN MARKET clustering on {len(working_df)} customers...")

    # Build BallTree once
    coords = working_df[['LAT_NUM', 'LON_NUM']].values
    tree = BallTree(np.radians(coords), metric='haversine')

    for idx, row in working_df.iterrows():
        if idx in assigned_indices:
            continue

        customer_state = row['STATE_CODE']
        if pd.isna(customer_state):
            continue

        director = get_director_for_state(customer_state)
        if director is None:
            continue

        # Get allowed states for this director
        allowed_states = set(DIRECTOR_STATES.get(director, []))

        query_point = np.radians([[row['LAT_NUM'], row['LON_NUM']]])

        # Try increasing radii to find minimum radius with 280-330 customers
        best_cluster = None

        for radius_miles in range(5, max_radius + 1, 5):
            radius_rad = radius_miles / EARTH_RADIUS_MILES
            neighbor_indices = tree.query_radius(query_point, r=radius_rad)[0]

            # Get actual df indices
            candidate_df_indices = working_df.iloc[neighbor_indices].index.tolist()

            # Filter: not already assigned + same director region
            eligible = [
                i for i in candidate_df_indices
                if i not in assigned_indices
                and working_df.loc[i, 'STATE_CODE'] in allowed_states
            ]

            if len(eligible) >= NEW_PORTFOLIO_MIN:
                # Cap at MAX
                cluster_indices = eligible[:NEW_PORTFOLIO_MAX]
                cluster_rows = working_df.loc[cluster_indices]
                centroid_lat, centroid_lon = compute_centroid(
                    cluster_rows['LAT_NUM'].values,
                    cluster_rows['LON_NUM'].values
                )
                best_cluster = {
                    'customers': cluster_indices,
                    'centroid_lat': centroid_lat,
                    'centroid_lon': centroid_lon,
                    'director': director,
                    'placement': 'IN MARKET',
                    'radius_used': radius_miles
                }
                break  # Use smallest radius that works

        if best_cluster:
            clusters.append(best_cluster)
            assigned_indices.update(best_cluster['customers'])
            print(f"    IN MARKET cluster formed: {len(best_cluster['customers'])} customers, "
                  f"radius={best_cluster['radius_used']}mi, director={best_cluster['director']}")

    print(f"  Formed {len(clusters)} IN MARKET clusters")
    print(f"  Customers assigned: {sum(len(c['customers']) for c in clusters)}")
    print(f"  Customers remaining: {len(working_df) - len(assigned_indices)}")

    return clusters, assigned_indices


# ==================== STEP 2: ASSIGN AU TO IN MARKET CLUSTERS ====================

def assign_au_to_clusters(clusters, branch_data, used_aus=None):
    """
    For each IN MARKET cluster, find the nearest AU (branch) to the centroid.
    Each AU can only be used once across ALL new portfolios.
    Assigns portfolio code as 'P{AU}'.
    
    Returns updated clusters with 'portfolio_cd' and 'au' fields.
    """
    if used_aus is None:
        used_aus = set()

    branch_data = branch_data.copy()
    branch_data['AU'] = branch_data['AU'].astype(int)

    # Valid branches only
    valid_branches = branch_data[
        branch_data['BRANCH_LAT_NUM'].notna() &
        branch_data['BRANCH_LON_NUM'].notna()
    ].copy()

    branch_coords = valid_branches[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    branch_tree = BallTree(np.radians(branch_coords), metric='haversine')

    assigned_clusters = []

    for cluster in clusters:
        centroid = np.radians([[cluster['centroid_lat'], cluster['centroid_lon']]])

        # Query nearest branches (get top N to handle already-used AUs)
        k = builtins.min(len(valid_branches), 20)
        distances, indices = branch_tree.query(centroid, k=k)

        assigned_au = None
        for i in indices[0]:
            candidate_au = int(valid_branches.iloc[i]['AU'])
            if candidate_au not in used_aus:
                assigned_au = candidate_au
                used_aus.add(candidate_au)
                break

        if assigned_au is not None:
            cluster['au'] = assigned_au
            cluster['portfolio_cd'] = f'P{assigned_au}'
            assigned_clusters.append(cluster)
            print(f"    Cluster → Portfolio {cluster['portfolio_cd']} (AU={assigned_au}, "
                  f"director={cluster['director']})")
        else:
            print(f"    Warning: No available AU found for cluster at "
                  f"({cluster['centroid_lat']:.2f}, {cluster['centroid_lon']:.2f})")
            cluster['au'] = None
            cluster['portfolio_cd'] = None
            assigned_clusters.append(cluster)

    return assigned_clusters, used_aus


# ==================== STEP 3: CENTRALIZED CLUSTERING ====================

def build_centralized_clusters(unassigned_df, assigned_in_market_indices):
    """
    Cluster remaining unassigned customers (those not placed in IN MARKET clusters).
    No radius constraint. Cluster size: 280-330.
    Respects director-state boundaries.
    Portfolio codes: PC1, PC2, ...
    
    Returns list of cluster dicts.
    """
    clusters = []

    # Exclude already assigned customers
    remaining_df = unassigned_df[
        ~unassigned_df.index.isin(assigned_in_market_indices)
    ].copy()

    if 'STATE_CODE' not in remaining_df.columns:
        remaining_df['STATE_CODE'] = remaining_df['BILLINGSTATE'].apply(convert_state_to_code)

    valid_mask = remaining_df['LAT_NUM'].notna() & remaining_df['LON_NUM'].notna()
    working_df = remaining_df[valid_mask].copy()

    print(f"\n  Starting CENTRALIZED clustering on {len(working_df)} remaining customers...")

    assigned_indices = set()
    pc_counter = 1

    # Group by director region first to respect boundaries
    for director, allowed_states in DIRECTOR_STATES.items():
        director_df = working_df[
            working_df['STATE_CODE'].isin(allowed_states) &
            ~working_df.index.isin(assigned_indices)
        ].copy()

        if len(director_df) < NEW_PORTFOLIO_MIN:
            print(f"    {director}: Only {len(director_df)} customers — insufficient for a cluster")
            continue

        print(f"    Processing {director}: {len(director_df)} customers")

        # Build BallTree for this director's customers
        coords = director_df[['LAT_NUM', 'LON_NUM']].values
        tree = BallTree(np.radians(coords), metric='haversine')

        dir_assigned = set()

        for idx, row in director_df.iterrows():
            if idx in dir_assigned:
                continue

            # Get all unassigned customers for this director
            remaining_in_director = [
                i for i in director_df.index
                if i not in dir_assigned
            ]

            if len(remaining_in_director) < NEW_PORTFOLIO_MIN:
                break

            # Find nearest NEW_PORTFOLIO_MAX customers to this seed point
            query_point = np.radians([[row['LAT_NUM'], row['LON_NUM']]])
            k = builtins.min(NEW_PORTFOLIO_MAX, len(remaining_in_director))
            distances, indices = tree.query(query_point, k=k)

            candidate_df_indices = director_df.iloc[indices[0]].index.tolist()
            eligible = [i for i in candidate_df_indices if i not in dir_assigned]

            if len(eligible) < NEW_PORTFOLIO_MIN:
                continue

            cluster_indices = eligible[:NEW_PORTFOLIO_MAX]
            cluster_rows = director_df.loc[cluster_indices]
            centroid_lat, centroid_lon = compute_centroid(
                cluster_rows['LAT_NUM'].values,
                cluster_rows['LON_NUM'].values
            )

            portfolio_cd = f'PC{pc_counter}'
            clusters.append({
                'customers': cluster_indices,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'director': director,
                'placement': 'CENTRALIZED',
                'portfolio_cd': portfolio_cd,
                'au': None
            })

            dir_assigned.update(cluster_indices)
            assigned_indices.update(cluster_indices)
            pc_counter += 1

            print(f"      CENTRALIZED cluster {portfolio_cd}: {len(cluster_indices)} customers, "
                  f"director={director}")

    print(f"\n  Formed {len(clusters)} CENTRALIZED clusters")
    print(f"  Customers assigned: {sum(len(c['customers']) for c in clusters)}")

    return clusters, assigned_indices


# ==================== STEP 4: UPDATE HH_DF AND PORTFOLIO STRUCTURES ====================

def update_hh_and_portfolio_structures(hh_df, rbrm_data, portfolio_stats,
                                        in_market_clusters, centralized_clusters):
    """
    Assign portfolio codes back to hh_df and update rbrm_data + portfolio_stats
    with new IN MARKET and CENTRALIZED portfolios.
    """
    hh_df = hh_df.copy()
    rbrm_data = rbrm_data.copy()

    all_clusters = in_market_clusters + centralized_clusters
    new_rbrm_rows = []

    for cluster in all_clusters:
        portfolio_cd = cluster.get('portfolio_cd')
        if portfolio_cd is None:
            continue

        placement = cluster['placement']
        director = cluster['director']
        centroid_lat = cluster['centroid_lat']
        centroid_lon = cluster['centroid_lon']
        au = cluster.get('au', None)

        # Assign portfolio code to households in hh_df
        for hh_idx in cluster['customers']:
            hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
            hh_df.loc[hh_idx, 'BANKER_TYPE'] = 'RM'  # New hires default to RM

        # Add to portfolio_stats
        portfolio_stats[portfolio_cd] = {
            'banker_type': 'RM',
            'placement': placement,
            'director': director,
            'current_count': len(cluster['customers']),
            'total_count': len(cluster['customers']),
            'min_required': NEW_PORTFOLIO_MIN,
            'max_allowed': NEW_PORTFOLIO_MAX,
            'deficit': 0,
            'excess': 0,
            'lat': centroid_lat,
            'lon': centroid_lon
        }

        # Add to rbrm_data
        new_rbrm_rows.append({
            'CG_PORTFOLIO_CD': portfolio_cd,
            'PLACEMENT': placement,
            'BANKER_TYPE': 'RM',
            'DIRECTOR': director,
            'AU': au if au else np.nan,
            'CENTROID_LAT_NUM': centroid_lat,
            'CENTROID_LON_NUM': centroid_lon,
            'IS_NEW_PORTFOLIO': True
        })

    if new_rbrm_rows:
        new_rbrm_df = pd.DataFrame(new_rbrm_rows)
        rbrm_data = pd.concat([rbrm_data, new_rbrm_df], ignore_index=True)

    print(f"\n  Added {len(new_rbrm_rows)} new portfolios to rbrm_data")
    print(f"  Total portfolios now: {len(portfolio_stats)}")

    return hh_df, rbrm_data, portfolio_stats


# ==================== MAIN ORCHESTRATOR ====================

def create_new_portfolios(hh_df, branch_data, rbrm_data, portfolio_stats, used_aus=None):
    """
    Main function to create new portfolios for unassigned customers.
    
    Steps:
        1. Cluster unassigned into IN MARKET portfolios (max 20 mile radius, 280-330 size)
        2. Assign nearest available AU to each IN MARKET cluster → portfolio code P{AU}
        3. Cluster remaining into CENTRALIZED portfolios → portfolio code PC1, PC2, ...
        4. Update hh_df, rbrm_data, portfolio_stats
    
    Returns:
        hh_df, rbrm_data, portfolio_stats
    """
    print("=" * 70)
    print("NEW PORTFOLIO CREATION - START")
    print("=" * 70)

    # Get unassigned Segment 3 & 4 POOL customers
    unassigned_df = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ].copy()

    print(f"\nTotal unassigned customers: {len(unassigned_df)}")

    if len(unassigned_df) == 0:
        print("No unassigned customers found. Exiting.")
        return hh_df, rbrm_data, portfolio_stats

    # Add state code
    unassigned_df['STATE_CODE'] = unassigned_df['BILLINGSTATE'].apply(convert_state_to_code)

    # Collect existing AUs to avoid conflicts
    if used_aus is None:
        existing_aus = set(
            pd.to_numeric(rbrm_data['AU'], errors='coerce')
            .dropna()
            .astype(int)
            .tolist()
        )
    else:
        existing_aus = used_aus

    # ---- STEP 1: IN MARKET CLUSTERING ----
    print("\n[STEP 1: IN MARKET CLUSTERING]")
    in_market_clusters, in_market_assigned = build_in_market_clusters(
        unassigned_df, max_radius=IN_MARKET_MAX_RADIUS
    )

    # ---- STEP 2: ASSIGN AU TO IN MARKET CLUSTERS ----
    print("\n[STEP 2: ASSIGN AU TO IN MARKET CLUSTERS]")
    in_market_clusters, existing_aus = assign_au_to_clusters(
        in_market_clusters, branch_data, used_aus=existing_aus
    )

    # ---- STEP 3: CENTRALIZED CLUSTERING ----
    print("\n[STEP 3: CENTRALIZED CLUSTERING]")
    centralized_clusters, centralized_assigned = build_centralized_clusters(
        unassigned_df, in_market_assigned
    )

    # ---- STEP 4: UPDATE STRUCTURES ----
    print("\n[STEP 4: UPDATING HH_DF AND PORTFOLIO STRUCTURES]")
    hh_df, rbrm_data, portfolio_stats = update_hh_and_portfolio_structures(
        hh_df, rbrm_data, portfolio_stats,
        in_market_clusters, centralized_clusters
    )

    # ---- FINAL SUMMARY ----
    total_new = len(in_market_clusters) + len(centralized_clusters)
    total_assigned = sum(len(c['customers']) for c in in_market_clusters + centralized_clusters)
    still_unassigned = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]

    print("\n" + "=" * 70)
    print("NEW PORTFOLIO CREATION - COMPLETE")
    print("=" * 70)
    print(f"  New IN MARKET portfolios created : {len(in_market_clusters)}")
    print(f"  New CENTRALIZED portfolios created: {len(centralized_clusters)}")
    print(f"  Total new portfolios             : {total_new}")
    print(f"  Customers assigned               : {total_assigned}")
    print(f"  Still unassigned                 : {len(still_unassigned)}")

    return hh_df, rbrm_data, portfolio_stats


# ==================== USAGE ====================

# After running run_portfolio_reconstruction(), call this:
#
# updated_hh_df, updated_rbrm_data, updated_portfolio_stats = create_new_portfolios(
#     hh_df=updated_hh_df,
#     branch_data=branch_data,
#     rbrm_data=rbrm_data,
#     portfolio_stats=portfolio_stats
# )
