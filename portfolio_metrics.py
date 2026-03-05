import pandas as pd


def generate_portfolio_metrics(current_portfolio_data, new_portfolio_data):
    """
    Generate a consolidated metrics summary DataFrame with one row per portfolio.

    Metrics:
        CG_PORTFOLIO_CD              : Portfolio code
        CG_COUNT                     : Total CG_ECNs in current portfolio
        CG_RC_COUNT                  : Total RC_ECNs in current portfolio
        HH_COUNT                     : Total HH_ECNs in future portfolio
        HH_RC_COUNT                  : Total RC_ECNs in future portfolio
        PERCENTAGE_CG_ECN_RETAINED   : % of current CG_ECNs found in future as HH_ECN or RC_ECN
        PERCENTAGE_CG_RC_ECN_RETAINED: % of current RC_ECNs found in future as HH_ECN or RC_ECN
        NEW_HH_ECN_COUNT             : HH_ECNs in future not present in current as CG_ECN or RC_ECN
        NEW_HH_RC_ECN_COUNT          : RC_ECNs in future not present in current as CG_ECN or RC_ECN

    Args:
        current_portfolio_data : DataFrame — current portfolio state
        new_portfolio_data     : DataFrame — future/recommended portfolio state

    Returns:
        DataFrame with one row per CG_PORTFOLIO_CD
    """
    # Rename RELATED_ECN -> RC_ECN if needed
    if 'RELATED_ECN' in new_portfolio_data.columns:
        new_portfolio_data = new_portfolio_data.rename(columns={'RELATED_ECN': 'RC_ECN'})

    all_port_codes = current_portfolio_data['PORT_CODE'].dropna().unique()
    rows = []

    for port_code in all_port_codes:
        curr = current_portfolio_data[current_portfolio_data['PORT_CODE'] == port_code]
        fut  = new_portfolio_data[new_portfolio_data['PORT_CODE'] == port_code]

        # ---- Current counts ----
        cg_ecns     = curr['CG_ECN'].dropna().unique()
        cg_rc_ecns  = curr['RC_ECN'].dropna().unique()
        cg_count    = len(cg_ecns)
        cg_rc_count = len(cg_rc_ecns)

        # ---- Future counts ----
        hh_ecns     = fut['HH_ECN'].dropna().unique()
        hh_rc_ecns  = fut['RC_ECN'].dropna().unique()
        hh_count    = len(hh_ecns)
        hh_rc_count = len(hh_rc_ecns)

        # ---- Future ECN pool (HH_ECN + RC_ECN combined) for retention check ----
        future_ecn_pool = set(hh_ecns) | set(hh_rc_ecns)

        # ---- Current ECN pool (CG_ECN + RC_ECN combined) for new count check ----
        current_ecn_pool = set(cg_ecns) | set(cg_rc_ecns)

        # ---- PERCENTAGE_CG_ECN_RETAINED ----
        cg_ecns_retained = len(set(cg_ecns) & future_ecn_pool)
        raw_pct_cg       = cg_ecns_retained / cg_count * 100 if cg_count > 0 else 0.0
        pct_cg_retained  = float(pd.Series([raw_pct_cg]).round(2).iloc[0])

        # ---- PERCENTAGE_CG_RC_ECN_RETAINED ----
        cg_rc_ecns_retained = len(set(cg_rc_ecns) & future_ecn_pool)
        raw_pct_cg_rc       = cg_rc_ecns_retained / cg_rc_count * 100 if cg_rc_count > 0 else 0.0
        pct_cg_rc_retained  = float(pd.Series([raw_pct_cg_rc]).round(2).iloc[0])

        # ---- NEW_HH_ECN_COUNT ----
        new_hh_ecn_count = len(set(hh_ecns) - current_ecn_pool)

        # ---- NEW_HH_RC_ECN_COUNT ----
        new_hh_rc_ecn_count = len(set(hh_rc_ecns) - current_ecn_pool)

        rows.append({
            'CG_PORTFOLIO_CD'              : port_code,
            'CG_COUNT'                     : cg_count,
            'CG_RC_COUNT'                  : cg_rc_count,
            'HH_COUNT'                     : hh_count,
            'HH_RC_COUNT'                  : hh_rc_count,
            'PERCENTAGE_CG_ECN_RETAINED'   : pct_cg_retained,
            'PERCENTAGE_CG_RC_ECN_RETAINED': pct_cg_rc_retained,
            'NEW_HH_ECN_COUNT'             : new_hh_ecn_count,
            'NEW_HH_RC_ECN_COUNT'          : new_hh_rc_ecn_count,
        })

    metrics_df = pd.DataFrame(rows)
    print(f"Portfolio metrics generated for {len(metrics_df)} portfolios.")
    return metrics_df


# ==================== USAGE ====================

# metrics_df = generate_portfolio_metrics(
#     current_portfolio_data=CURRENT_PORTFOLIO_DATA,
#     new_portfolio_data=NEW_PORTFOLIO_DATA
# )
#
# metrics_df.head()
