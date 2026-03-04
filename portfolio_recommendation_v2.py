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
    - Each cluster contains customers from ONE state only (no cross-state mixing)

    Returns list of cluster dicts with keys:
        customers (list of indices), centroid_lat, centroid_lon, state_code, placement, radius_used
    """
    clusters = []
    unassigned_df = unassigned_df.copy()

    if 'STATE_CODE' not in unassigned_df.columns:
        unassigned_df['STATE_CODE'] = unassigned_df['BILLINGSTATE'].apply(convert_state_to_code)

    valid_mask = unassigned_df['LAT_NUM'].notna() & unassigned_df['LON_NUM'].notna()
    working_df = unassigned_df[valid_mask].copy()

    print(f"  Starting IN MARKET clustering on {len(working_df)} customers...")

    all_assigned_indices = set()

    # ---- Group by STATE first ----
    for state_code, state_df in working_df.groupby('STATE_CODE'):

        if pd.isna(state_code) or not state_code:
            continue

        print(f"\n    State: {state_code} — {len(state_df)} customers")

        # Build BallTree once per state
        coords = state_df[['LAT_NUM', 'LON_NUM']].values
        tree = BallTree(np.radians(coords), metric='haversine')

        state_assigned = set()

        for idx, row in state_df.iterrows():
            if idx in state_assigned:
                continue

            query_point = np.radians([[row['LAT_NUM'], row['LON_NUM']]])

            best_cluster = None

            for radius_miles in range(5, max_radius + 1, 5):
                radius_rad = radius_miles / EARTH_RADIUS_MILES
                neighbor_indices = tree.query_radius(query_point, r=radius_rad)[0]

                # Map back to original df indices
                candidate_df_indices = state_df.iloc[neighbor_indices].index.tolist()

                # Filter: not already assigned, same state (guaranteed by groupby)
                eligible = [
                    i for i in candidate_df_indices
                    if i not in state_assigned
                ]

                if len(eligible) >= NEW_PORTFOLIO_MIN:
                    cluster_indices = eligible[:NEW_PORTFOLIO_MAX]
                    cluster_rows = state_df.loc[cluster_indices]
                    centroid_lat, centroid_lon = compute_centroid(
                        cluster_rows['LAT_NUM'].values,
                        cluster_rows['LON_NUM'].values
                    )
                    best_cluster = {
                        'customers': cluster_indices,
                        'centroid_lat': centroid_lat,
                        'centroid_lon': centroid_lon,
                        'state_code': state_code,
                        'placement': 'IN MARKET',
                        'radius_used': radius_miles
                    }
                    break  # Use smallest radius that works

            if best_cluster:
                clusters.append(best_cluster)
                state_assigned.update(best_cluster['customers'])
                all_assigned_indices.update(best_cluster['customers'])
                print(f"      IN MARKET cluster formed: {len(best_cluster['customers'])} customers, "
                      f"radius={best_cluster['radius_used']}mi, state={state_code}")

    print(f"\n  Formed {len(clusters)} IN MARKET clusters")
    print(f"  Customers assigned: {sum(len(c['customers']) for c in clusters)}")
    print(f"  Customers remaining: {len(working_df) - len(all_assigned_indices)}")

    return clusters, all_assigned_indices


# ==================== STEP 2: ASSIGN AU TO IN MARKET CLUSTERS ====================

def assign_au_to_clusters(clusters, branch_data, used_aus=None):
    """
    For each IN MARKET cluster, find the nearest AU (branch) to the centroid.
    Each AU can only be used once across ALL new portfolios.
    Assigns portfolio code as 'P{AU}'.

    Returns updated clusters with 'portfolio_cd' and 'au' fields.
    """
    if used_aus is None:
        used_aus = set(
            pd.to_numeric(branch_data['AU'], errors='coerce')
            .dropna()
            .astype(int)
            .tolist()
        )

    branch_data = branch_data.copy()
    branch_data['AU'] = pd.to_numeric(branch_data['AU'], errors='coerce')
    branch_data['BRANCH_LAT_NUM'] = pd.to_numeric(branch_data['BRANCH_LAT_NUM'], errors='coerce')
    branch_data['BRANCH_LON_NUM'] = pd.to_numeric(branch_data['BRANCH_LON_NUM'], errors='coerce')
    branch_data = branch_data.dropna(subset=['AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM'])
    branch_data['AU'] = branch_data['AU'].astype(int)

    valid_branches = branch_data.copy()
    branch_coords = valid_branches[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    branch_tree = BallTree(np.radians(branch_coords), metric='haversine')

    assigned_clusters = []

    for cluster in clusters:
        centroid = np.radians([[cluster['centroid_lat'], cluster['centroid_lon']]])

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
                  f"state={cluster['state_code']})")
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
    Cluster remaining unassigned customers into CENTRALIZED portfolios.
    No radius constraint. Cluster size: 280-330.
    Each cluster contains customers from ONE state only (no cross-state mixing).
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

    all_assigned_indices = set()
    pc_counter = 1

    # ---- Group by STATE first ----
    for state_code, state_df in working_df.groupby('STATE_CODE'):

        if pd.isna(state_code) or not state_code:
            continue

        # Skip already assigned
        state_df = state_df[~state_df.index.isin(all_assigned_indices)].copy()

        if len(state_df) < NEW_PORTFOLIO_MIN:
            print(f"    State {state_code}: Only {len(state_df)} customers — insufficient for a cluster, skipping")
            continue

        print(f"\n    State: {state_code} — {len(state_df)} customers")

        # Build BallTree for this state's customers
        coords = state_df[['LAT_NUM', 'LON_NUM']].values
        tree = BallTree(np.radians(coords), metric='haversine')

        state_assigned = set()

        for idx, row in state_df.iterrows():
            if idx in state_assigned:
                continue

            # Check enough remaining in state
            remaining_in_state = [
                i for i in state_df.index
                if i not in state_assigned
            ]

            if len(remaining_in_state) < NEW_PORTFOLIO_MIN:
                break

            # Find nearest NEW_PORTFOLIO_MAX customers to this seed point
            query_point = np.radians([[row['LAT_NUM'], row['LON_NUM']]])
            k = builtins.min(NEW_PORTFOLIO_MAX, len(remaining_in_state))
            distances, indices = tree.query(query_point, k=k)

            candidate_df_indices = state_df.iloc[indices[0]].index.tolist()
            eligible = [i for i in candidate_df_indices if i not in state_assigned]

            if len(eligible) < NEW_PORTFOLIO_MIN:
                continue

            cluster_indices = eligible[:NEW_PORTFOLIO_MAX]
            cluster_rows = state_df.loc[cluster_indices]
            centroid_lat, centroid_lon = compute_centroid(
                cluster_rows['LAT_NUM'].values,
                cluster_rows['LON_NUM'].values
            )

            portfolio_cd = f'PC{pc_counter}'
            clusters.append({
                'customers': cluster_indices,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'state_code': state_code,
                'placement': 'CENTRALIZED',
                'portfolio_cd': portfolio_cd,
                'au': None
            })

            state_assigned.update(cluster_indices)
            all_assigned_indices.update(cluster_indices)
            pc_counter += 1

            print(f"      CENTRALIZED cluster {portfolio_cd}: {len(cluster_indices)} customers, "
                  f"state={state_code}")

    print(f"\n  Formed {len(clusters)} CENTRALIZED clusters")
    print(f"  Customers assigned: {sum(len(c['customers']) for c in clusters)}")

    return clusters, all_assigned_indices


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
        state_code = cluster['state_code']
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
            'state_code': state_code,
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
            'STATE_CODE': state_code,
            'COVERAGE': state_code,           # Single state coverage
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
        1. Cluster unassigned into IN MARKET portfolios
           - Max 20 mile radius, 280-330 size
           - Customers from ONE state only per cluster
        2. Assign nearest available AU to each IN MARKET cluster → portfolio code P{AU}
        3. Cluster remaining into CENTRALIZED portfolios → portfolio code PC1, PC2, ...
           - Customers from ONE state only per cluster
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
    print("\n[STEP 1: IN MARKET CLUSTERING (single-state per cluster)]")
    in_market_clusters, in_market_assigned = build_in_market_clusters(
        unassigned_df, max_radius=IN_MARKET_MAX_RADIUS
    )

    # ---- STEP 2: ASSIGN AU TO IN MARKET CLUSTERS ----
    print("\n[STEP 2: ASSIGN AU TO IN MARKET CLUSTERS]")
    in_market_clusters, existing_aus = assign_au_to_clusters(
        in_market_clusters, branch_data, used_aus=existing_aus
    )

    # ---- STEP 3: CENTRALIZED CLUSTERING ----
    print("\n[STEP 3: CENTRALIZED CLUSTERING (single-state per cluster)]")
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
    print(f"  New IN MARKET portfolios created  : {len(in_market_clusters)}")
    print(f"  New CENTRALIZED portfolios created : {len(centralized_clusters)}")
    print(f"  Total new portfolios               : {total_new}")
    print(f"  Customers assigned                 : {total_assigned}")
    print(f"  Still unassigned                   : {len(still_unassigned)}")

    # Per-state summary
    print("\n  Per-state breakdown:")
    all_clusters = in_market_clusters + centralized_clusters
    state_summary = {}
    for c in all_clusters:
        s = c['state_code']
        state_summary.setdefault(s, {'portfolios': 0, 'customers': 0})
        state_summary[s]['portfolios'] += 1
        state_summary[s]['customers'] += len(c['customers'])
    for state, info in sorted(state_summary.items()):
        print(f"    {state}: {info['portfolios']} portfolios, {info['customers']} customers")

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
