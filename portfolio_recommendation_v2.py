import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import builtins

# ==================== CONFIGURATION ====================
NEW_PORTFOLIO_MIN  = 280
NEW_PORTFOLIO_MAX  = 330
IN_MARKET_MAX_RADIUS = 20       # miles
EARTH_RADIUS_MILES = 3959

# Director to States mapping
DIRECTOR_STATES = {
    'JOHN KLEINER':   ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'HI', 'OR'],
    'SEAN APPENRODT': ['IL', 'IN', 'IA', 'KS', 'KY', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
    'MOJGAN MADADI':  ['CA'],
    'MEHNOOSH ASKARI': ['CT', 'DE', 'DC', 'ME', 'MD', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT',
                        'AL', 'AR', 'FL', 'GA', 'LA', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV', 'WA']
}

# Build reverse lookup: state -> director
STATE_TO_DIRECTOR = {}
for _director, _states in DIRECTOR_STATES.items():
    for _state in _states:
        STATE_TO_DIRECTOR[_state] = _director


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


def convert_state_to_code(state_name):
    """Convert full state name to 2-letter code. If already 2-letter, return as-is."""
    STATE_NAME_TO_CODE = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
        'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
        'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC',
        'PUERTO RICO': 'PR', 'GUAM': 'GU', 'VIRGIN ISLANDS': 'VI'
    }
    if pd.isna(state_name):
        return None
    state_upper = str(state_name).strip().upper()
    if len(state_upper) == 2:
        return state_upper
    return STATE_NAME_TO_CODE.get(state_upper, None)


# ==================== STEP 1: IN MARKET CLUSTERING ====================

def build_in_market_clusters(unassigned_df, max_radius=IN_MARKET_MAX_RADIUS):
    """
    Cluster unassigned customers into IN MARKET portfolios.
    - Cluster radius: max 20 miles (uses smallest radius that fits 280-330 customers)
    - Cluster size: 280-330 customers
    - Each cluster contains customers from ONE state only (no cross-state mixing)

    Returns list of cluster dicts with keys:
        customers, centroid_lat, centroid_lon, state_code, coverage,
        placement, radius_used
    """
    clusters = []
    unassigned_df = unassigned_df.copy()

    if 'STATE_CODE' not in unassigned_df.columns:
        unassigned_df['STATE_CODE'] = unassigned_df['BILLINGSTATE'].apply(convert_state_to_code)

    valid_mask = unassigned_df['LAT_NUM'].notna() & unassigned_df['LON_NUM'].notna()
    working_df = unassigned_df[valid_mask].copy()

    print(f"  Starting IN MARKET clustering on {len(working_df)} customers...")

    all_assigned_indices = set()

    for state_code, state_df in working_df.groupby('STATE_CODE'):

        if pd.isna(state_code) or not state_code:
            continue

        print(f"\n    State: {state_code} — {len(state_df)} customers")

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
                candidate_df_indices = state_df.iloc[neighbor_indices].index.tolist()
                eligible = [i for i in candidate_df_indices if i not in state_assigned]

                if len(eligible) >= NEW_PORTFOLIO_MIN:
                    cluster_indices = eligible[:NEW_PORTFOLIO_MAX]
                    cluster_rows = state_df.loc[cluster_indices]
                    centroid_lat, centroid_lon = compute_centroid(
                        cluster_rows['LAT_NUM'].values,
                        cluster_rows['LON_NUM'].values
                    )
                    best_cluster = {
                        'customers':    cluster_indices,
                        'centroid_lat': centroid_lat,
                        'centroid_lon': centroid_lon,
                        'state_code':   state_code,
                        'coverage':     {state_code},       # single-state coverage set
                        'placement':    'IN MARKET',
                        'radius_used':  radius_miles
                    }
                    break

            if best_cluster:
                clusters.append(best_cluster)
                state_assigned.update(best_cluster['customers'])
                all_assigned_indices.update(best_cluster['customers'])
                print(f"      IN MARKET cluster formed: {len(best_cluster['customers'])} customers, "
                      f"radius={best_cluster['radius_used']}mi, state={state_code}")

    total_in_market_assigned = 0
    for c in clusters:
        total_in_market_assigned += len(c['customers'])
    print(f"\n  Formed {len(clusters)} IN MARKET clusters")
    print(f"  Customers assigned: {total_in_market_assigned}")
    print(f"  Customers remaining: {len(working_df) - len(all_assigned_indices)}")

    return clusters, all_assigned_indices


# ==================== STEP 2: ASSIGN AU TO IN MARKET CLUSTERS ====================

def assign_au_to_clusters(clusters, branch_data, used_aus=None):
    """
    For each IN MARKET cluster, find the nearest AU (branch) to the centroid.
    Each AU can only be used once across ALL new portfolios.
    Assigns portfolio code as 'P{AU}'.
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
            cluster['au']           = assigned_au
            cluster['portfolio_cd'] = f'P{assigned_au}'
            assigned_clusters.append(cluster)
            print(f"    Cluster → Portfolio {cluster['portfolio_cd']} (AU={assigned_au}, "
                  f"state={cluster['state_code']})")
        else:
            print(f"    Warning: No available AU found for cluster at "
                  f"({cluster['centroid_lat']:.2f}, {cluster['centroid_lon']:.2f})")
            cluster['au']           = None
            cluster['portfolio_cd'] = None
            assigned_clusters.append(cluster)

    return assigned_clusters, used_aus


# ==================== STEP 3: CENTRALIZED CLUSTERING ====================

def build_centralized_clusters(unassigned_df, assigned_in_market_indices):
    """
    Cluster remaining unassigned customers into CENTRALIZED portfolios.
    No radius constraint. Cluster size: 280-330.
    Each cluster contains customers from ONE state only.
    Portfolio codes: PC1, PC2, ...
    """
    clusters = []

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

    for state_code, state_df in working_df.groupby('STATE_CODE'):

        if pd.isna(state_code) or not state_code:
            continue

        state_df = state_df[~state_df.index.isin(all_assigned_indices)].copy()

        if len(state_df) < NEW_PORTFOLIO_MIN:
            print(f"    State {state_code}: Only {len(state_df)} customers — "
                  f"insufficient for a cluster, skipping")
            continue

        print(f"\n    State: {state_code} — {len(state_df)} customers")

        coords = state_df[['LAT_NUM', 'LON_NUM']].values
        tree = BallTree(np.radians(coords), metric='haversine')

        state_assigned = set()

        for idx, row in state_df.iterrows():
            if idx in state_assigned:
                continue

            remaining_in_state = [i for i in state_df.index if i not in state_assigned]

            if len(remaining_in_state) < NEW_PORTFOLIO_MIN:
                break

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
                'customers':    cluster_indices,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'state_code':   state_code,
                'coverage':     {state_code},       # single-state coverage set
                'placement':    'CENTRALIZED',
                'portfolio_cd': portfolio_cd,
                'au':           None
            })

            state_assigned.update(cluster_indices)
            all_assigned_indices.update(cluster_indices)
            pc_counter += 1

            print(f"      CENTRALIZED cluster {portfolio_cd}: {len(cluster_indices)} customers, "
                  f"state={state_code}")

    total_centralized_assigned = 0
    for c in clusters:
        total_centralized_assigned += len(c['customers'])
    print(f"\n  Formed {len(clusters)} CENTRALIZED clusters")
    print(f"  Customers assigned: {total_centralized_assigned}")

    return clusters, all_assigned_indices


# ==================== STEP 4: UPDATE HH_DF AND PORTFOLIO STRUCTURES ====================

def update_hh_and_portfolio_structures(hh_df, rbrm_data, portfolio_stats,
                                        in_market_clusters, centralized_clusters):
    """
    Assign portfolio codes back to hh_df and update rbrm_data + portfolio_stats
    with new IN MARKET and CENTRALIZED portfolios.
    """
    hh_df     = hh_df.copy()
    rbrm_data = rbrm_data.copy()

    all_clusters  = in_market_clusters + centralized_clusters
    new_rbrm_rows = []

    for cluster in all_clusters:
        portfolio_cd = cluster.get('portfolio_cd')
        if portfolio_cd is None:
            continue

        placement    = cluster['placement']
        state_code   = cluster['state_code']
        centroid_lat = cluster['centroid_lat']
        centroid_lon = cluster['centroid_lon']
        au           = cluster.get('au', None)

        # Coverage string — may be multi-state for combined portfolios
        coverage_val = cluster.get('coverage', {state_code})
        if isinstance(coverage_val, set):
            coverage_str = ','.join(sorted(coverage_val))
        else:
            coverage_str = str(coverage_val)

        # Director from coverage states
        director = None
        for state in coverage_str.split(','):
            director = STATE_TO_DIRECTOR.get(state.strip(), None)
            if director:
                break

        # Assign portfolio code to households
        for hh_idx in cluster['customers']:
            hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
            hh_df.loc[hh_idx, 'BANKER_TYPE']     = 'RM'

        # Update portfolio_stats
        portfolio_stats[portfolio_cd] = {
            'banker_type':   'RM',
            'placement':     placement,
            'state_code':    state_code,
            'director':      director,
            'coverage':      coverage_str,
            'current_count': len(cluster['customers']),
            'total_count':   len(cluster['customers']),
            'min_required':  NEW_PORTFOLIO_MIN,
            'max_allowed':   NEW_PORTFOLIO_MAX,
            'deficit':       0,
            'excess':        0,
            'lat':           centroid_lat,
            'lon':           centroid_lon
        }

        # Add to rbrm_data
        new_rbrm_rows.append({
            'CG_PORTFOLIO_CD': portfolio_cd,
            'PLACEMENT':       placement,
            'BANKER_TYPE':     'RM',
            'STATE_CODE':      state_code,
            'COVERAGE':        coverage_str,
            'DIRECTOR':        director,
            'AU':              au if au else np.nan,
            'CENTROID_LAT_NUM': centroid_lat,
            'CENTROID_LON_NUM': centroid_lon,
            'IS_NEW_PORTFOLIO': True
        })

    if new_rbrm_rows:
        new_rbrm_df = pd.DataFrame(new_rbrm_rows)
        rbrm_data   = pd.concat([rbrm_data, new_rbrm_df], ignore_index=True)

    print(f"\n  Added {len(new_rbrm_rows)} new portfolios to rbrm_data")
    print(f"  Total portfolios now: {len(portfolio_stats)}")

    return hh_df, rbrm_data, portfolio_stats


# ==================== STEP 5: BUILD DIRECTOR COVERAGE MAP ====================

def build_new_portfolio_director_coverage(in_market_clusters, centralized_clusters,
                                           branch_data):
    """
    Build a DataFrame mapping new portfolios to their director, coverage states,
    AU, and branch/centroid coordinates.

    Args:
        in_market_clusters   : list of cluster dicts from build_in_market_clusters()
        centralized_clusters : list of cluster dicts from build_centralized_clusters()
                               (includes combined-state portfolios if applicable)
        branch_data          : DataFrame with AU, BRANCH_LAT_NUM, BRANCH_LON_NUM

    Returns:
        DataFrame with columns:
            CG_PORTFOLIO_CD : new portfolio code
            DIRECTOR        : director name
            COVERAGE        : comma-separated state codes (e.g. 'TX' or 'TX,OK')
            AU              : AU number for IN MARKET, null for CENTRALIZED
            BRANCH_LAT_NUM  : branch lat for IN MARKET, customer centroid lat for CENTRALIZED
            BRANCH_LON_NUM  : branch lon for IN MARKET, customer centroid lon for CENTRALIZED
    """
    # Build AU -> branch coordinates lookup
    branch_lookup = (
        branch_data.copy()
        .assign(AU=lambda df: pd.to_numeric(df['AU'], errors='coerce'))
        .assign(BRANCH_LAT_NUM=lambda df: pd.to_numeric(df['BRANCH_LAT_NUM'], errors='coerce'))
        .assign(BRANCH_LON_NUM=lambda df: pd.to_numeric(df['BRANCH_LON_NUM'], errors='coerce'))
        .dropna(subset=['AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM'])
        .drop_duplicates(subset=['AU'])
        .set_index('AU')[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']]
        .to_dict(orient='index')
    )

    rows = []

    for cluster in in_market_clusters + centralized_clusters:
        portfolio_cd = cluster.get('portfolio_cd')
        if portfolio_cd is None:
            continue

        placement = cluster.get('placement', 'CENTRALIZED')

        # Resolve coverage
        coverage_val = cluster.get('coverage', None)
        if coverage_val:
            if isinstance(coverage_val, set):
                coverage_states = sorted(coverage_val)
            else:
                coverage_states = sorted([s.strip() for s in str(coverage_val).split(',')])
        else:
            state_code = cluster.get('state_code', None)
            coverage_states = [state_code] if state_code else []

        coverage_str = ','.join(coverage_states)

        # Director from first matching state
        director = None
        for state in coverage_states:
            director = STATE_TO_DIRECTOR.get(state, None)
            if director:
                break

        # AU and coordinates
        if placement == 'IN MARKET':
            au = cluster.get('au', None)
            if au and int(au) in branch_lookup:
                branch_lat = branch_lookup[int(au)]['BRANCH_LAT_NUM']
                branch_lon = branch_lookup[int(au)]['BRANCH_LON_NUM']
            else:
                # Fallback to centroid if branch not found
                branch_lat = cluster.get('centroid_lat', np.nan)
                branch_lon = cluster.get('centroid_lon', np.nan)
        else:
            # CENTRALIZED — AU is null, use customer centroid
            au         = np.nan
            branch_lat = cluster.get('centroid_lat', np.nan)
            branch_lon = cluster.get('centroid_lon', np.nan)

        rows.append({
            'CG_PORTFOLIO_CD': portfolio_cd,
            'DIRECTOR':        director,
            'COVERAGE':        coverage_str,
            'AU':              au,
            'BRANCH_LAT_NUM':  branch_lat,
            'BRANCH_LON_NUM':  branch_lon,
        })

    portfolio_director_coverage_df = pd.DataFrame(
        rows,
        columns=['CG_PORTFOLIO_CD', 'DIRECTOR', 'COVERAGE',
                 'AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
    )

    print(f"New portfolio director-coverage map: {len(portfolio_director_coverage_df)} portfolios")
    return portfolio_director_coverage_df


# ==================== MAIN ORCHESTRATOR ====================

def create_new_portfolios(hh_df, branch_data, rbrm_data, portfolio_stats, used_aus=None):
    """
    Main function to create new portfolios for unassigned customers.

    Steps:
        1. Cluster unassigned into IN MARKET portfolios
        2. Assign nearest available AU to each IN MARKET cluster
        3. Cluster remaining into CENTRALIZED portfolios
        4. Update hh_df, rbrm_data, portfolio_stats
        5. Build director-coverage mapping DataFrame

    Returns:
        hh_df, rbrm_data, portfolio_stats, portfolio_director_coverage_df
    """
    print("=" * 70)
    print("NEW PORTFOLIO CREATION - START")
    print("=" * 70)

    unassigned_df = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ].copy()

    print(f"\nTotal unassigned customers: {len(unassigned_df)}")

    if len(unassigned_df) == 0:
        print("No unassigned customers found. Exiting.")
        empty_df = pd.DataFrame(columns=['CG_PORTFOLIO_CD', 'DIRECTOR', 'COVERAGE',
                                          'AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM'])
        return hh_df, rbrm_data, portfolio_stats, empty_df

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

    # ---- STEP 5: BUILD DIRECTOR COVERAGE MAP ----
    print("\n[STEP 5: BUILDING DIRECTOR-COVERAGE MAP]")
    portfolio_director_coverage_df = build_new_portfolio_director_coverage(
        in_market_clusters, centralized_clusters, branch_data
    )

    # ---- FINAL SUMMARY ----
    total_new      = len(in_market_clusters) + len(centralized_clusters)
    total_assigned = 0
    for c in in_market_clusters + centralized_clusters:
        total_assigned += len(c['customers'])
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

    print("\n  Per-state breakdown:")
    state_summary = {}
    for c in in_market_clusters + centralized_clusters:
        s = c['state_code']
        state_summary.setdefault(s, {'portfolios': 0, 'customers': 0})
        state_summary[s]['portfolios'] += 1
        state_summary[s]['customers']  += len(c['customers'])
    for state, info in sorted(state_summary.items()):
        print(f"    {state}: {info['portfolios']} portfolios, {info['customers']} customers")

    print("\n  Director-Coverage summary:")
    print(portfolio_director_coverage_df.to_string(index=False))

    return hh_df, rbrm_data, portfolio_stats, portfolio_director_coverage_df


# ==================== USAGE ====================

# hh_df, rbrm_data, portfolio_stats, portfolio_director_coverage_df = create_new_portfolios(
#     hh_df=hh_df,
#     branch_data=branch_data,
#     rbrm_data=rbrm_data,
#     portfolio_stats=portfolio_stats
# )
#
# portfolio_director_coverage_df.head()
