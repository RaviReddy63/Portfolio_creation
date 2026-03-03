import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import builtins

# ==================== CONFIGURATION ====================
NEW_PORTFOLIO_MIN = 280
NEW_PORTFOLIO_MAX = 330
EARTH_RADIUS_MILES = 3959

# ==================== US STATE ADJACENCY MAP ====================
STATE_ADJACENCY = {
    'AL': {'FL', 'GA', 'MS', 'TN'},
    'AK': set(),
    'AZ': {'CA', 'CO', 'NM', 'NV', 'UT'},
    'AR': {'LA', 'MO', 'MS', 'OK', 'TN', 'TX'},
    'CA': {'AZ', 'NV', 'OR'},
    'CO': {'AZ', 'KS', 'NE', 'NM', 'OK', 'UT', 'WY'},
    'CT': {'MA', 'NY', 'RI'},
    'DE': {'MD', 'NJ', 'PA'},
    'FL': {'AL', 'GA'},
    'GA': {'AL', 'FL', 'NC', 'SC', 'TN'},
    'HI': set(),
    'ID': {'MT', 'NV', 'OR', 'UT', 'WA', 'WY'},
    'IL': {'IN', 'IA', 'KY', 'MO', 'WI'},
    'IN': {'IL', 'KY', 'MI', 'OH'},
    'IA': {'IL', 'MN', 'MO', 'NE', 'SD', 'WI'},
    'KS': {'CO', 'MO', 'NE', 'OK'},
    'KY': {'IL', 'IN', 'MO', 'OH', 'TN', 'VA', 'WV'},
    'LA': {'AR', 'MS', 'TX'},
    'ME': {'NH'},
    'MD': {'DE', 'PA', 'VA', 'WV'},
    'MA': {'CT', 'NH', 'NY', 'RI', 'VT'},
    'MI': {'IN', 'OH', 'WI'},
    'MN': {'IA', 'ND', 'SD', 'WI'},
    'MS': {'AL', 'AR', 'LA', 'TN'},
    'MO': {'AR', 'IL', 'IA', 'KS', 'KY', 'NE', 'OK', 'TN'},
    'MT': {'ID', 'ND', 'SD', 'WY'},
    'NE': {'CO', 'IA', 'KS', 'MO', 'SD', 'WY'},
    'NV': {'AZ', 'CA', 'ID', 'OR', 'UT'},
    'NH': {'MA', 'ME', 'VT'},
    'NJ': {'DE', 'NY', 'PA'},
    'NM': {'AZ', 'CO', 'OK', 'TX', 'UT'},
    'NY': {'CT', 'MA', 'NJ', 'PA', 'VT'},
    'NC': {'GA', 'SC', 'TN', 'VA'},
    'ND': {'MN', 'MT', 'SD'},
    'OH': {'IN', 'KY', 'MI', 'PA', 'WV'},
    'OK': {'AR', 'CO', 'KS', 'MO', 'NM', 'TX'},
    'OR': {'CA', 'ID', 'NV', 'WA'},
    'PA': {'DE', 'MD', 'NJ', 'NY', 'OH', 'WV'},
    'RI': {'CT', 'MA'},
    'SC': {'GA', 'NC'},
    'SD': {'IA', 'MN', 'MT', 'ND', 'NE', 'WY'},
    'TN': {'AL', 'AR', 'GA', 'KY', 'MO', 'MS', 'NC', 'VA'},
    'TX': {'AR', 'LA', 'NM', 'OK'},
    'UT': {'AZ', 'CO', 'ID', 'NV', 'NM', 'WY'},
    'VT': {'MA', 'NH', 'NY'},
    'VA': {'KY', 'MD', 'NC', 'TN', 'WV'},
    'WA': {'ID', 'OR'},
    'WV': {'KY', 'MD', 'OH', 'PA', 'VA'},
    'WI': {'IL', 'IA', 'MI', 'MN'},
    'WY': {'CO', 'ID', 'MT', 'NE', 'SD', 'UT'},
    'DC': {'MD', 'VA'},
    'PR': set(),
    'GU': set(),
    'VI': set(),
}

# Director to States mapping
DIRECTOR_STATES = {
    'JOHN KLEINER': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'HI', 'OR'],
    'SEAN APPENRODT': ['IL', 'IN', 'IA', 'KS', 'KY', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
    'MOJGAN MADADI': ['CA'],
    'MEHNOOSH ASKARI': ['CT', 'DE', 'DC', 'ME', 'MD', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT',
                        'AL', 'AR', 'FL', 'GA', 'LA', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV', 'WA']
}

# Build reverse lookup: state -> director
STATE_TO_DIRECTOR = {}
for director, states in DIRECTOR_STATES.items():
    for state in states:
        STATE_TO_DIRECTOR[state] = director


# ==================== UTILITY ====================

def compute_centroid(lats, lons):
    """Compute geographic centroid of a set of lat/lon points."""
    return np.mean(lats), np.mean(lons)


def get_director_for_state(state_code):
    """Return director name for a given state code."""
    if pd.isna(state_code) or not state_code:
        return None
    return STATE_TO_DIRECTOR.get(state_code.strip().upper(), None)


def are_neighboring_states(state_a, state_b):
    """Check if two states share a border."""
    return state_b in STATE_ADJACENCY.get(state_a, set())


def are_same_director(state_a, state_b):
    """Check if two states belong to the same director."""
    dir_a = get_director_for_state(state_a)
    dir_b = get_director_for_state(state_b)
    return dir_a is not None and dir_a == dir_b


# ==================== MAIN FUNCTION ====================

def combine_neighboring_state_leftovers(hh_df, pc_counter_start=1):
    """
    After main clustering is done, collect leftover unassigned customers
    per state and try to combine neighboring states (within the same director)
    to form new CENTRALIZED portfolios.

    Rules:
        - Only states WITH leftover customers can be combined (no bridge states)
        - Maximum 2 states per combined portfolio
        - Both states must be neighbors AND under the same director
        - Combined pool creates as many 280-330 sized portfolios as possible
        - Centroids computed from actual customer locations
        - Portfolio codes: PC{n} continuing from pc_counter_start

    Args:
        hh_df            : Main household dataframe
        pc_counter_start : Starting counter for PC portfolio codes

    Returns:
        hh_df            : Updated with new portfolio assignments
        new_portfolios   : List of dicts describing each new combined portfolio
    """
    print("=" * 70)
    print("NEIGHBORING STATE COMBINATION - START")
    print("=" * 70)

    hh_df = hh_df.copy()

    # ---- Collect leftover customers per state ----
    leftover_df = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ].copy()

    if 'STATE_CODE' not in leftover_df.columns:
        leftover_df['STATE_CODE'] = leftover_df['BILLINGSTATE'].apply(
            lambda x: x.strip().upper() if pd.notna(x) else None
        )

    print(f"\nTotal leftover unassigned customers: {len(leftover_df)}")

    if len(leftover_df) == 0:
        print("No leftover customers. Exiting.")
        return hh_df, []

    # ---- Build per-state leftover pools ----
    state_pools = {}
    for state_code, state_df in leftover_df.groupby('STATE_CODE'):
        if pd.isna(state_code) or not state_code:
            continue
        state_pools[state_code] = list(state_df.index)

    print(f"\nStates with leftover customers:")
    for state, indices in sorted(state_pools.items()):
        director = get_director_for_state(state)
        print(f"  {state} ({director}): {len(indices)} customers")

    # ---- Find valid neighboring state pairs ----
    # Greedy pairing — each state used in at most one pair
    states_with_leftovers_sorted = sorted(state_pools.keys())
    paired_states = set()
    valid_pairs = []

    for state_a in states_with_leftovers_sorted:
        if state_a in paired_states:
            continue

        best_partner = None
        best_combined = 0

        for state_b in states_with_leftovers_sorted:
            if state_b == state_a:
                continue
            if state_b in paired_states:
                continue
            if not are_neighboring_states(state_a, state_b):
                continue
            if not are_same_director(state_a, state_b):
                continue

            combined_count = len(state_pools[state_a]) + len(state_pools[state_b])

            # Only pair if combined count meets minimum
            if combined_count >= NEW_PORTFOLIO_MIN:
                if combined_count > best_combined:
                    best_combined = combined_count
                    best_partner = state_b

        if best_partner is not None:
            valid_pairs.append((state_a, best_partner))
            paired_states.add(state_a)
            paired_states.add(best_partner)
            print(f"\n  Pairing: {state_a} + {best_partner} "
                  f"({len(state_pools[state_a])} + {len(state_pools[best_partner])} "
                  f"= {best_combined} customers)")

    if len(valid_pairs) == 0:
        print("\nNo valid neighboring state pairs found with sufficient customers.")
        return hh_df, []

    print(f"\nFound {len(valid_pairs)} valid state pairs")

    # ---- Create portfolios from each valid pair ----
    new_portfolios = []
    pc_counter = pc_counter_start

    for state_a, state_b in valid_pairs:
        director = get_director_for_state(state_a)
        combined_states = f"{state_a}+{state_b}"

        combined_indices = state_pools[state_a] + state_pools[state_b]
        combined_df = hh_df.loc[combined_indices].copy()
        valid_coords = combined_df['LAT_NUM'].notna() & combined_df['LON_NUM'].notna()
        combined_df = combined_df[valid_coords].copy()

        if len(combined_df) < NEW_PORTFOLIO_MIN:
            print(f"\n  Skipping {combined_states}: Only {len(combined_df)} customers "
                  f"with valid coordinates")
            continue

        print(f"\n  Processing pair {combined_states} | Director: {director} "
              f"| {len(combined_df)} customers")

        # Build BallTree on combined pool
        coords = combined_df[['LAT_NUM', 'LON_NUM']].values
        tree = BallTree(np.radians(coords), metric='haversine')

        pool_assigned = set()
        portfolios_from_pair = 0

        for idx, row in combined_df.iterrows():
            if idx in pool_assigned:
                continue

            remaining = [i for i in combined_df.index if i not in pool_assigned]
            if len(remaining) < NEW_PORTFOLIO_MIN:
                break

            query_point = np.radians([[row['LAT_NUM'], row['LON_NUM']]])
            k = builtins.min(NEW_PORTFOLIO_MAX, len(remaining))
            distances, indices = tree.query(query_point, k=k)

            candidate_indices = combined_df.iloc[indices[0]].index.tolist()
            eligible = [i for i in candidate_indices if i not in pool_assigned]

            if len(eligible) < NEW_PORTFOLIO_MIN:
                continue

            cluster_indices = eligible[:NEW_PORTFOLIO_MAX]
            cluster_rows = combined_df.loc[cluster_indices]

            centroid_lat, centroid_lon = compute_centroid(
                cluster_rows['LAT_NUM'].values,
                cluster_rows['LON_NUM'].values
            )

            portfolio_cd = f'PC{pc_counter}'

            # Assign in hh_df
            for hh_idx in cluster_indices:
                hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
                hh_df.loc[hh_idx, 'BANKER_TYPE'] = 'RM'

            cluster_count = len(cluster_indices)

            new_portfolios.append({
                'portfolio_cd': portfolio_cd,
                'placement': 'CENTRALIZED',
                'state_a': state_a,
                'state_b': state_b,
                'combined_states': combined_states,
                'director': director,
                'coverage': f'{state_a},{state_b}',
                'customer_count': cluster_count,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'au': None,
                'customers': cluster_indices
            })

            pool_assigned.update(cluster_indices)
            pc_counter += 1
            portfolios_from_pair += 1

            print(f"    Portfolio {portfolio_cd}: {cluster_count} customers "
                  f"from {combined_states}, director={director}")

        print(f"  Created {portfolios_from_pair} portfolios from {combined_states}")

    # ---- Final summary ----
    total_assigned = 0
    for p in new_portfolios:
        total_assigned += p['customer_count']

    still_unassigned = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ]

    print("\n" + "=" * 70)
    print("NEIGHBORING STATE COMBINATION - COMPLETE")
    print("=" * 70)
    print(f"  State pairs processed          : {len(valid_pairs)}")
    print(f"  New combined portfolios created: {len(new_portfolios)}")
    print(f"  Customers assigned             : {total_assigned}")
    print(f"  Still unassigned               : {len(still_unassigned)}")

    if new_portfolios:
        print("\n  New portfolios summary:")
        for p in new_portfolios:
            print(f"    {p['portfolio_cd']}: {p['combined_states']} | "
                  f"{p['customer_count']} customers | director={p['director']}")

    return hh_df, new_portfolios


# ==================== HELPER: UPDATE RBRM AND PORTFOLIO STATS ====================

def update_structures_with_combined_portfolios(rbrm_data, portfolio_stats, new_portfolios):
    """
    Update rbrm_data and portfolio_stats with newly created combined portfolios.
    Call this after combine_neighboring_state_leftovers().
    """
    rbrm_data = rbrm_data.copy()
    new_rbrm_rows = []

    for p in new_portfolios:
        portfolio_cd = p['portfolio_cd']

        portfolio_stats[portfolio_cd] = {
            'banker_type': 'RM',
            'placement': 'CENTRALIZED',
            'state_code': p['combined_states'],
            'director': p['director'],
            'coverage': p['coverage'],
            'current_count': p['customer_count'],
            'total_count': p['customer_count'],
            'min_required': NEW_PORTFOLIO_MIN,
            'max_allowed': NEW_PORTFOLIO_MAX,
            'deficit': 0,
            'excess': 0,
            'lat': p['centroid_lat'],
            'lon': p['centroid_lon']
        }

        new_rbrm_rows.append({
            'CG_PORTFOLIO_CD': portfolio_cd,
            'PLACEMENT': 'CENTRALIZED',
            'BANKER_TYPE': 'RM',
            'DIRECTOR': p['director'],
            'STATE_CODE': p['combined_states'],
            'COVERAGE': p['coverage'],       # e.g. 'TX,OK'
            'AU': np.nan,                    # No AU for CENTRALIZED
            'CENTROID_LAT_NUM': p['centroid_lat'],
            'CENTROID_LON_NUM': p['centroid_lon'],
            'IS_NEW_PORTFOLIO': True,
            'IS_COMBINED_STATES': True
        })

    if new_rbrm_rows:
        new_rbrm_df = pd.DataFrame(new_rbrm_rows)
        rbrm_data = pd.concat([rbrm_data, new_rbrm_df], ignore_index=True)

    print(f"Updated rbrm_data and portfolio_stats with {len(new_portfolios)} combined portfolios")

    return rbrm_data, portfolio_stats


# ==================== USAGE ====================

# Run after create_new_portfolios():
#
# # Determine starting PC counter to avoid conflicts with existing PC portfolios
# existing_pc = [p for p in portfolio_stats if p.startswith('PC')]
# pc_start = max([int(p[2:]) for p in existing_pc], default=0) + 1
#
# updated_hh_df, new_combined_portfolios = combine_neighboring_state_leftovers(
#     hh_df=updated_hh_df,
#     pc_counter_start=pc_start
# )
#
# updated_rbrm_data, updated_portfolio_stats = update_structures_with_combined_portfolios(
#     rbrm_data=updated_rbrm_data,
#     portfolio_stats=updated_portfolio_stats,
#     new_portfolios=new_combined_portfolios
# )
