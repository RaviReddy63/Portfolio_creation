import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import builtins

# ==================== CONFIGURATION ====================
CENTRALIZED_NEIGHBOR_RADIUS = 2000   # miles
EARTH_RADIUS_MILES = 3959

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

# US State Adjacency Map
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


# ==================== UTILITY ====================

def get_director_for_state(state_code):
    """Return director name for a given state code."""
    if pd.isna(state_code) or not state_code:
        return None
    return STATE_TO_DIRECTOR.get(state_code.strip().upper(), None)


def get_coverage_states(portfolio_cd, portfolio_coverage_map):
    """
    Return the set of states covered by a portfolio.
    Handles both single-state and combined (e.g. 'TX,OK') coverage.
    """
    return portfolio_coverage_map.get(portfolio_cd, set())


def get_neighboring_states_same_director(coverage_states, director):
    """
    Given a set of coverage states for a portfolio, return all neighboring states
    that belong to the same director (excluding states already in coverage).

    Args:
        coverage_states : set of state codes already covered by the portfolio
        director        : director name for the portfolio

    Returns:
        set of neighboring state codes within the same director's territory
    """
    director_states = set(DIRECTOR_STATES.get(director, []))
    neighbors = set()

    for state in coverage_states:
        state_neighbors = STATE_ADJACENCY.get(state.strip().upper(), set())
        for neighbor in state_neighbors:
            # Must be same director, not already in coverage
            if neighbor in director_states and neighbor not in coverage_states:
                neighbors.add(neighbor)

    return neighbors


# ==================== MAIN FUNCTION ====================

def fill_undersized_centralized_from_neighbors(
    hh_df,
    portfolio_stats,
    portfolio_coverage_map,
    rbrm_data
):
    """
    For CENTRALIZED portfolios still below MIN, attempt to fill them to MIN
    by pulling unassigned customers from a single neighboring state within
    2000 miles of the portfolio centroid. Respects director boundaries.

    Steps per undersized portfolio:
        1. Get portfolio's coverage states and director
        2. Find all neighboring states within the same director
        3. For each neighbor, count unassigned customers within 2000 miles
        4. Pick the single best neighbor (most customers available)
        5. Assign enough customers from that neighbor to reach MIN
        6. Update COVERAGE in portfolio_coverage_map and add UPDATED_COVERAGE
           column in rbrm_data

    Args:
        hh_df                 : Household dataframe
        portfolio_stats       : Portfolio stats dict
        portfolio_coverage_map: Dict of portfolio_cd -> set of state codes
        rbrm_data             : rbrm_data DataFrame

    Returns:
        hh_df, portfolio_stats, portfolio_coverage_map, rbrm_data
    """
    print("=" * 70)
    print("FILL UNDERSIZED CENTRALIZED FROM NEIGHBORING STATES - START")
    print("=" * 70)

    hh_df = hh_df.copy()
    rbrm_data = rbrm_data.copy()

    # Add UPDATED_COVERAGE column if not present
    if 'UPDATED_COVERAGE' not in rbrm_data.columns:
        rbrm_data['UPDATED_COVERAGE'] = np.nan

    # Get all unassigned customers (Seg 3 & 4, POOL, no portfolio)
    unassigned_df = hh_df[
        (hh_df['NEW_SEGMENT'].isin([3, 4])) &
        (hh_df['RULE'] == 'POOL') &
        (hh_df['CG_PORTFOLIO_CD'].isna())
    ].copy()

    if 'STATE_CODE' not in unassigned_df.columns:
        unassigned_df['STATE_CODE'] = unassigned_df['BILLINGSTATE'].apply(
            lambda x: x.strip().upper() if pd.notna(x) else None
        )

    print(f"\nTotal unassigned customers available: {len(unassigned_df)}")

    # Get undersized CENTRALIZED portfolios
    undersized_centralized = [
        p for p, s in portfolio_stats.items()
        if s['placement'] == 'CENTRALIZED' and s['deficit'] > 0
    ]

    print(f"Undersized CENTRALIZED portfolios   : {len(undersized_centralized)}")

    if len(undersized_centralized) == 0:
        print("No undersized CENTRALIZED portfolios. Exiting.")
        return hh_df, portfolio_stats, portfolio_coverage_map, rbrm_data

    if len(unassigned_df) == 0:
        print("No unassigned customers available. Exiting.")
        return hh_df, portfolio_stats, portfolio_coverage_map, rbrm_data

    # Track which unassigned customers have been used in this step
    # so we don't double-assign across portfolios
    used_in_this_step = set()

    total_assigned = 0
    portfolios_filled = 0
    portfolios_not_filled = 0

    for portfolio_cd in undersized_centralized:
        stats = portfolio_stats[portfolio_cd]
        deficit = stats['deficit']
        p_lat = stats['lat']
        p_lon = stats['lon']

        # Get portfolio coverage states and director
        coverage_states = get_coverage_states(portfolio_cd, portfolio_coverage_map)
        if not coverage_states:
            print(f"\n  {portfolio_cd}: No coverage states defined, skipping")
            portfolios_not_filled += 1
            continue

        # Infer director from coverage states
        director = None
        for state in coverage_states:
            director = get_director_for_state(state)
            if director:
                break

        if not director:
            print(f"\n  {portfolio_cd}: No director found for coverage states {coverage_states}, skipping")
            portfolios_not_filled += 1
            continue

        print(f"\n  Portfolio: {portfolio_cd} | Coverage: {coverage_states} | "
              f"Director: {director} | Deficit: {deficit}")

        # Get neighboring states within same director
        neighbor_states = get_neighboring_states_same_director(coverage_states, director)

        if not neighbor_states:
            print(f"    No valid neighboring states found within director boundary")
            portfolios_not_filled += 1
            continue

        print(f"    Candidate neighboring states: {neighbor_states}")

        # Build pool of unassigned customers not yet used in this step
        available_df = unassigned_df[
            ~unassigned_df.index.isin(used_in_this_step)
        ].copy()

        if len(available_df) == 0:
            print(f"    No available unassigned customers remaining")
            portfolios_not_filled += 1
            continue

        # For each neighboring state, find unassigned customers within 2000 miles
        # of portfolio centroid
        best_neighbor_state = None
        best_neighbor_indices = []

        portfolio_point = np.array([[p_lat, p_lon]])

        for neighbor_state in sorted(neighbor_states):
            # Filter unassigned customers from this neighbor state
            state_candidates = available_df[
                available_df['STATE_CODE'] == neighbor_state
            ].copy()

            valid_coords = (
                state_candidates['LAT_NUM'].notna() &
                state_candidates['LON_NUM'].notna()
            )
            state_candidates = state_candidates[valid_coords].copy()

            if len(state_candidates) == 0:
                continue

            # Build BallTree for this state's candidates
            coords = state_candidates[['LAT_NUM', 'LON_NUM']].values
            tree = BallTree(np.radians(coords), metric='haversine')

            # Query within 2000 miles of portfolio centroid
            radius_rad = CENTRALIZED_NEIGHBOR_RADIUS / EARTH_RADIUS_MILES
            query_point = np.radians(portfolio_point)
            neighbor_indices_in_tree = tree.query_radius(query_point, r=radius_rad)[0]

            if len(neighbor_indices_in_tree) == 0:
                continue

            # Map back to hh_df indices
            candidate_hh_indices = state_candidates.iloc[neighbor_indices_in_tree].index.tolist()

            print(f"    Neighbor {neighbor_state}: {len(candidate_hh_indices)} unassigned "
                  f"customers within {CENTRALIZED_NEIGHBOR_RADIUS} miles")

            # Pick the best neighbor — most customers available
            if len(candidate_hh_indices) > len(best_neighbor_indices):
                best_neighbor_indices = candidate_hh_indices
                best_neighbor_state = neighbor_state

        if best_neighbor_state is None or len(best_neighbor_indices) == 0:
            print(f"    No unassigned customers found in any neighboring state within "
                  f"{CENTRALIZED_NEIGHBOR_RADIUS} miles")
            portfolios_not_filled += 1
            continue

        print(f"    Selected neighbor: {best_neighbor_state} "
              f"({len(best_neighbor_indices)} candidates available)")

        # Take only as many as needed to reach MIN
        num_to_assign = builtins.min(deficit, len(best_neighbor_indices))
        assign_indices = best_neighbor_indices[:num_to_assign]

        # Assign customers to portfolio
        for hh_idx in assign_indices:
            hh_df.loc[hh_idx, 'CG_PORTFOLIO_CD'] = portfolio_cd
            hh_df.loc[hh_idx, 'BANKER_TYPE'] = stats['banker_type']

        used_in_this_step.update(assign_indices)
        total_assigned += num_to_assign

        # Update portfolio coverage map
        updated_coverage = coverage_states | {best_neighbor_state}
        portfolio_coverage_map[portfolio_cd] = updated_coverage

        # Build updated coverage string e.g. 'TX,OK,AR'
        updated_coverage_str = ','.join(sorted(updated_coverage))

        # Update UPDATED_COVERAGE in rbrm_data for this portfolio
        rbrm_mask = rbrm_data['CG_PORTFOLIO_CD'] == portfolio_cd
        rbrm_data.loc[rbrm_mask, 'UPDATED_COVERAGE'] = updated_coverage_str

        # Update portfolio stats
        new_count = stats['current_count'] + num_to_assign
        new_deficit = builtins.max(0, stats['min_required'] - new_count)
        new_excess = builtins.max(0, new_count - stats['max_allowed'])

        portfolio_stats[portfolio_cd]['current_count'] = new_count
        portfolio_stats[portfolio_cd]['deficit'] = new_deficit
        portfolio_stats[portfolio_cd]['excess'] = new_excess

        if new_deficit == 0:
            portfolios_filled += 1
            print(f"    Assigned {num_to_assign} customers from {best_neighbor_state} "
                  f"→ Portfolio now at {new_count} (MIN met)")
        else:
            portfolios_not_filled += 1
            print(f"    Assigned {num_to_assign} customers from {best_neighbor_state} "
                  f"→ Portfolio at {new_count}, still {new_deficit} short of MIN")

        print(f"    Updated coverage: {updated_coverage_str}")

    # ---- Final summary ----
    still_undersized = [
        p for p, s in portfolio_stats.items()
        if s['placement'] == 'CENTRALIZED' and s['deficit'] > 0
    ]

    print("\n" + "=" * 70)
    print("FILL UNDERSIZED CENTRALIZED FROM NEIGHBORING STATES - COMPLETE")
    print("=" * 70)
    print(f"  Portfolios fully filled to MIN : {portfolios_filled}")
    print(f"  Portfolios still below MIN     : {portfolios_not_filled}")
    print(f"  Total customers assigned       : {total_assigned}")
    print(f"  Still undersized (CENTRALIZED) : {len(still_undersized)}")

    if still_undersized:
        print(f"\n  Still undersized portfolios:")
        for p in still_undersized:
            s = portfolio_stats[p]
            print(f"    {p}: {s['current_count']} customers "
                  f"(MIN: {s['min_required']}, deficit: {s['deficit']})")

    return hh_df, portfolio_stats, portfolio_coverage_map, rbrm_data


# ==================== INTEGRATION IN MAIN ORCHESTRATOR ====================
#
# Add this step in run_portfolio_reconstruction() BEFORE fill_portfolios_to_max
# for CENTRALIZED portfolios:
#
# # ========== STEP 5: CENTRALIZED OPTIMIZATION (RM & RC together) ==========
# print("\n[STEP 5: CENTRALIZED OPTIMIZATION (RM & RC)]")
# portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
# hh_df = optimize_in_market_portfolios(hh_df, portfolio_stats, portfolio_coverage_map, placement='CENTRALIZED')
#
# # ========== STEP 5.5: FILL UNDERSIZED CENTRALIZED FROM NEIGHBORING STATES ==========
# print("\n[STEP 5.5: FILL UNDERSIZED CENTRALIZED FROM NEIGHBORING STATES]")
# portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
# hh_df, portfolio_stats, portfolio_coverage_map, rbrm_data = fill_undersized_centralized_from_neighbors(
#     hh_df=hh_df,
#     portfolio_stats=portfolio_stats,
#     portfolio_coverage_map=portfolio_coverage_map,
#     rbrm_data=rbrm_data
# )
#
# # ========== STEP 6: FINAL CLEANUP (RM & RC together) ==========
# print("\n[STEP 6: FINAL CLEANUP (RM & RC)]")
# portfolio_stats = calculate_portfolio_requirements(hh_df, portfolio_locations)
# hh_df = assign_remaining_households(hh_df, portfolio_stats, portfolio_coverage_map, placement='CENTRALIZED')
