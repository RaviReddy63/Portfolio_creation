import streamlit as st
import pandas as pd

# Import custom modules
from ui_components import (
    setup_page_config, add_logo, create_header, initialize_session_state,
    create_au_filters, create_customer_filters, create_portfolio_button,
    display_summary_statistics, create_portfolio_editor, create_apply_changes_button
)
from data_loader import get_merged_data
from portfolio_creation import process_portfolio_creation, apply_portfolio_changes
from map_visualization import create_combined_map, create_smart_portfolio_map
from portfolio_creation_8 import enhanced_customer_au_assignment_with_two_inmarket_iterations

def main():
    """Main application function"""
    # Setup page
    setup_page_config()
    add_logo()
    
    # Create header and get current page
    page = create_header()
    
    # Initialize session state
    initialize_session_state()
    
    # Load data
    customer_data, banker_data, branch_data, data = get_merged_data()
    
    if page == "Portfolio Assignment":
        portfolio_assignment_page(customer_data, banker_data, branch_data)
    elif page == "Portfolio Mapping":
        portfolio_mapping_page(customer_data, banker_data, branch_data)

def portfolio_assignment_page(customer_data, banker_data, branch_data):
    """Portfolio Assignment page logic"""
    
    # Create AU filters
    selected_aus = create_au_filters(branch_data)
    
    # Create customer filters
    cust_state, role, cust_portcd, max_dist, min_rev, min_deposit = create_customer_filters(customer_data)
    
    # Create portfolio button
    button_clicked = create_portfolio_button()
    
    # Handle button click
    if button_clicked:
        if not selected_aus:
            st.error("Please select at least one AU")
        else:
            st.session_state.should_create_portfolios = True
    
    # Process portfolio creation
    if st.session_state.should_create_portfolios:
        if not selected_aus:
            st.error("Please select at least one AU")
            st.session_state.should_create_portfolios = False
        else:
            portfolios_created, portfolio_summaries = process_portfolio_creation(
                selected_aus, customer_data, banker_data, branch_data,
                role, cust_state, cust_portcd, max_dist, min_rev, min_deposit
            )
            
            if portfolios_created:
                st.session_state.portfolios_created = portfolios_created
                st.session_state.portfolio_summaries = portfolio_summaries
                st.session_state.should_create_portfolios = False
            else:
                st.session_state.should_create_portfolios = False
    
    # Display results
    display_portfolio_results(branch_data)

def display_portfolio_results(branch_data):
    """Display portfolio results if they exist"""
    if 'portfolios_created' in st.session_state and st.session_state.portfolios_created:
        portfolios_created = st.session_state.portfolios_created
        portfolio_summaries = st.session_state.get('portfolio_summaries', {})
        
        # Show Portfolio Summary Tables and Geographic Distribution
        st.markdown("----")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Portfolio Summary Tables")
            display_portfolio_tables(portfolios_created, portfolio_summaries, branch_data)
        
        with col2:
            st.subheader("Geographic Distribution")
            display_geographic_map(portfolios_created, branch_data)
    else:
        # Show message when no portfolios exist
        if st.session_state.get('portfolios_created') is not None:
            st.warning("No customers found for the selected AUs with current filters.")
    
    # Show recommendation reassignment table if it exists
    display_reassignment_recommendations()

def display_portfolio_tables(portfolios_created, portfolio_summaries, branch_data):
    """Display portfolio summary tables"""
    if len(portfolios_created) > 1:
        # Multiple AU case - use tabs
        au_tabs = st.tabs([f"AU {au_id}" for au_id in portfolios_created.keys()])
        
        for tab_idx, (au_id, tab) in enumerate(zip(portfolios_created.keys(), au_tabs)):
            with tab:
                display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, True)
    else:
        # Single AU case
        au_id = list(portfolios_created.keys())[0]
        display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, False)

def display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, is_multi_au):
    """Display table for a single AU"""
    if au_id in portfolio_summaries:
        portfolio_df = pd.DataFrame(portfolio_summaries[au_id])
        portfolio_df = portfolio_df.sort_values('Available for this portfolio', ascending=False).reset_index(drop=True)
        
        # Create editable dataframe
        edited_df = create_portfolio_editor(portfolio_df, au_id, is_multi_au)
        
        # Store the edited data
        st.session_state.portfolio_controls[au_id] = edited_df
        
        # Add Apply Changes button
        if create_apply_changes_button(au_id, not is_multi_au):
            apply_portfolio_changes(au_id, branch_data)
        
        # Display summary statistics
        au_filtered_data = st.session_state.portfolios_created[au_id]
        display_summary_statistics(au_filtered_data)

def display_geographic_map(portfolios_created, branch_data):
    """Display the geographic distribution map"""
    # Create preview portfolios for map display - one portfolio per AU
    preview_portfolios = {}
    
    for au_id, filtered_data in portfolios_created.items():
        if not filtered_data.empty:
            preview_portfolios[f"AU_{au_id}_Portfolio"] = filtered_data
    
    # Display the map with preview portfolios
    if preview_portfolios:
        combined_map = create_combined_map(preview_portfolios, branch_data)
        if combined_map:
            st.plotly_chart(combined_map, use_container_width=True)
    else:
        st.info("No customers selected for map display")

def display_reassignment_recommendations():
    """Display recommendation reassignment table if it exists"""
    if ('recommend_reassignment' in st.session_state and 
        isinstance(st.session_state.recommend_reassignment, pd.DataFrame) and 
        not st.session_state.recommend_reassignment.empty):
        st.markdown("----")
        st.subheader("Recommended Reassignments")
        st.dataframe(st.session_state.recommend_reassignment, use_container_width=True)

def portfolio_mapping_page(customer_data, banker_data, branch_data):
    """Portfolio Mapping page logic with advanced clustering"""
    st.subheader("Smart Portfolio Mapping")
    
    # Create customer filters (reuse from Portfolio Assignment)
    cust_state, role, cust_portcd, max_dist, min_rev, min_deposit = create_customer_filters_for_mapping(customer_data)
    
    # Create Smart Portfolio Generation button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")  # Empty space
    with col2:
        generate_button = st.button("Generate Smart Portfolios", key="generate_smart_portfolios", type="primary")
    
    # Handle button click
    if generate_button:
        st.session_state.should_generate_smart_portfolios = True
    
    # Process smart portfolio generation
    if st.session_state.get('should_generate_smart_portfolios', False):
        generate_smart_portfolios(customer_data, branch_data, cust_state, role, cust_portcd, min_rev, min_deposit)
        st.session_state.should_generate_smart_portfolios = False
    
    # Display results if they exist
    display_smart_portfolio_results(customer_data, branch_data)

def create_customer_filters_for_mapping(customer_data):
    """Create customer selection criteria filters for Portfolio Mapping"""
    col_header2, col_clear2 = st.columns([9, 1])
    with col_header2:
        st.subheader("Customer Selection Criteria")
    with col_clear2:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_mapping_filters", help="Clear customer selection filters", type="secondary"):
            # Clear customer filters for mapping
            st.session_state.mapping_filter_cust_state = []
            st.session_state.mapping_filter_role = []
            st.session_state.mapping_filter_cust_portcd = []
            st.session_state.mapping_filter_min_rev = 5000
            st.session_state.mapping_filter_min_deposit = 100000
            # Clear smart portfolio results
            if 'smart_portfolio_results' in st.session_state:
                del st.session_state.smart_portfolio_results
            st.experimental_rerun()
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col2_or, col3 = st.columns([1, 1, 0.1, 1])
        
        with col1:
            cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
            default_cust_states = st.session_state.get('mapping_filter_cust_state', [])
            default_cust_states = [s for s in default_cust_states if s in cust_state_options]
            cust_state = st.multiselect("Customer State", cust_state_options, default=default_cust_states, key="mapping_cust_state")
            st.session_state.mapping_filter_cust_state = cust_state
            if not cust_state:
                cust_state = None
        
        with col2:
            role_options = list(customer_data['TYPE'].dropna().unique())
            default_roles = st.session_state.get('mapping_filter_role', [])
            default_roles = [r for r in default_roles if r in role_options]
            role = st.multiselect("Role", role_options, default=default_roles, key="mapping_role")
            st.session_state.mapping_filter_role = role
            if not role:
                role = None
        
        with col2_or:
            st.markdown("<div style='text-align: center; padding-top: 8px; font-weight: bold;'>-OR-</div>", unsafe_allow_html=True)
        
        with col3:
            customer_data_temp = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            portfolio_options = list(customer_data_temp['PORT_CODE'].dropna().unique())
            default_portfolios = st.session_state.get('mapping_filter_cust_portcd', [])
            default_portfolios = [p for p in default_portfolios if p in portfolio_options]
            cust_portcd = st.multiselect("Portfolio Code", portfolio_options, default=default_portfolios, key="mapping_cust_portcd")
            st.session_state.mapping_filter_cust_portcd = cust_portcd
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5 = st.columns(2)
        with col4:
            min_rev = st.slider("Minimum Revenue", 0, 20000, value=st.session_state.get('mapping_filter_min_rev', 5000), step=1000, key="mapping_min_revenue")
            st.session_state.mapping_filter_min_rev = min_rev
        with col5:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, value=st.session_state.get('mapping_filter_min_deposit', 100000), step=5000, key="mapping_min_deposit")
            st.session_state.mapping_filter_min_deposit = min_deposit
    
    return cust_state, role, cust_portcd, None, min_rev, min_deposit

def apply_customer_filters_for_mapping(customer_data, cust_state, role, cust_portcd, min_rev, min_deposit):
    """Apply customer filters for Portfolio Mapping"""
    filtered_data = customer_data.copy()
    
    # Ensure CG_ECN is preserved
    if 'CG_ECN' not in filtered_data.columns:
        st.error("CG_ECN column missing from customer data!")
        return pd.DataFrame()
    
    # Apply Customer State filter
    if cust_state is not None:
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
    
    # Apply Role OR Portfolio Code filter (combined with OR logic)
    if role is not None or cust_portcd is not None:
        role_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        portfolio_condition = pd.Series([False] * len(filtered_data), index=filtered_data.index)
        
        # Check role condition
        if role is not None:
            filtered_data['TYPE_CLEAN'] = filtered_data['TYPE'].fillna('').str.strip().str.lower()
            role_clean = [r.strip().lower() for r in role]
            role_condition = filtered_data['TYPE_CLEAN'].isin(role_clean)
            filtered_data = filtered_data.drop('TYPE_CLEAN', axis=1)
        
        # Check portfolio code condition
        if cust_portcd is not None:
            # Use CG_PORTFOLIO_CD instead of renaming to PORT_CODE
            portfolio_condition = filtered_data['CG_PORTFOLIO_CD'].isin(cust_portcd)
        
        # Apply OR logic: keep rows that match either role OR portfolio code
        combined_condition = role_condition | portfolio_condition
        filtered_data = filtered_data[combined_condition]
    
    # Apply other filters
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    return filtered_data

def generate_smart_portfolios(customer_data, branch_data, cust_state, role, cust_portcd, min_rev, min_deposit):
    """Generate smart portfolios using advanced clustering"""
    
    # Apply customer filters
    filtered_customers = apply_customer_filters_for_mapping(
        customer_data, cust_state, role, cust_portcd, min_rev, min_deposit
    )
    
    if len(filtered_customers) == 0:
        st.error("No customers found with the selected filters. Please adjust your criteria.")
        return
    
    st.info(f"Processing {len(filtered_customers)} customers for smart portfolio generation...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        progress_bar.progress(10)
        status_text.text("Initializing clustering algorithm...")
        
        # Run the enhanced clustering algorithm
        progress_bar.progress(30)
        status_text.text("Running advanced clustering analysis...")
        
        # Call the enhanced clustering function
        smart_portfolio_results = enhanced_customer_au_assignment_with_two_inmarket_iterations(
            filtered_customers, branch_data
        )
        
        progress_bar.progress(80)
        status_text.text("Processing results...")
        
        # Store results in session state
        st.session_state.smart_portfolio_results = smart_portfolio_results
        st.session_state.filtered_customers_count = len(filtered_customers)
        
        progress_bar.progress(100)
        status_text.text("Smart portfolios generated successfully!")
        
        # Clear progress indicators after a brief delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Successfully generated smart portfolios for {len(smart_portfolio_results)} customers!")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error generating smart portfolios: {str(e)}")

def display_smart_portfolio_results(customer_data, branch_data):
    """Display smart portfolio results like Portfolio Assignment"""
    
    if 'smart_portfolio_results' not in st.session_state or len(st.session_state.smart_portfolio_results) == 0:
        st.info("Click 'Generate Smart Portfolios' to create optimized customer assignments.")
        return
    
    results_df = st.session_state.smart_portfolio_results
    
    # Convert smart portfolio results to Portfolio Assignment format
    smart_portfolios_created = {}
    
    # Group by AU
    for au in results_df['ASSIGNED_AU'].unique():
        au_data = results_df[results_df['ASSIGNED_AU'] == au].copy()
        
        # Add required columns for Portfolio Assignment format
        au_data['AU'] = au
        
        # Get AU coordinates from branch_data
        au_branch = branch_data[branch_data['AU'] == au]
        if not au_branch.empty:
            au_data['BRANCH_LAT_NUM'] = au_branch.iloc[0]['BRANCH_LAT_NUM']
            au_data['BRANCH_LON_NUM'] = au_branch.iloc[0]['BRANCH_LON_NUM']
        
        # Rename columns to match Portfolio Assignment format
        au_data = au_data.rename(columns={
            'ECN': 'CG_ECN',
            'DISTANCE_TO_AU': 'Distance'
        })
        
        # Merge with original customer_data to get financial information
        customer_data_subset = customer_data[['CG_ECN', 'CG_PORTFOLIO_CD', 'BANK_REVENUE', 'DEPOSIT_BAL', 'TYPE']].copy()
        au_data = au_data.merge(customer_data_subset, on='CG_ECN', how='left', suffixes=('', '_orig'))
        
        # Use original portfolio code if available, otherwise create smart portfolio IDs
        au_data['PORT_CODE'] = au_data['CG_PORTFOLIO_CD'].fillna('SMART_' + au_data['TYPE'])
        
        # Use original financial data
        au_data['BANK_REVENUE'] = au_data['BANK_REVENUE'].fillna(0)
        au_data['DEPOSIT_BAL'] = au_data['DEPOSIT_BAL'].fillna(0)
        
        # Use original TYPE if different from smart assignment
        au_data['TYPE'] = au_data['TYPE_orig'].fillna(au_data['TYPE'])
        
        # Clean up duplicate columns
        au_data = au_data.drop(['CG_PORTFOLIO_CD', 'TYPE_orig'], axis=1, errors='ignore')
        
        smart_portfolios_created[au] = au_data
    
    st.markdown("----")
    
    # Display results in the same format as Portfolio Assignment
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Smart Portfolio Summary")
        display_smart_portfolio_tables(smart_portfolios_created, branch_data)
    
    with col2:
        st.subheader("Geographic Distribution")
        display_smart_geographic_map(smart_portfolios_created, branch_data)

def display_smart_portfolio_tables(smart_portfolios_created, branch_data):
    """Display smart portfolio summary tables like Portfolio Assignment"""
    if len(smart_portfolios_created) > 1:
        # Multiple AU case - use tabs
        au_tabs = st.tabs([f"AU {au_id}" for au_id in smart_portfolios_created.keys()])
        
        for tab_idx, (au_id, tab) in enumerate(zip(smart_portfolios_created.keys(), au_tabs)):
            with tab:
                display_single_smart_au_table(au_id, smart_portfolios_created, branch_data)
    else:
        # Single AU case
        au_id = list(smart_portfolios_created.keys())[0]
        display_single_smart_au_table(au_id, smart_portfolios_created, branch_data)

def display_single_smart_au_table(au_id, smart_portfolios_created, branch_data):
    """Display table for a single smart portfolio AU"""
    if au_id in smart_portfolios_created:
        au_data = smart_portfolios_created[au_id]
        
        # Create portfolio summary similar to Portfolio Assignment
        portfolio_summary = create_smart_portfolio_summary(au_data, au_id)
        
        if portfolio_summary:
            portfolio_df = pd.DataFrame(portfolio_summary)
            
            # Display the portfolio summary table
            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
            
            # Display summary statistics (same as Portfolio Assignment)
            display_summary_statistics(au_data)

def create_smart_portfolio_summary(au_data, au_id):
    """Create portfolio summary for smart portfolios matching Portfolio Assignment format"""
    portfolio_summary = []
    
    # Group by actual portfolio code (like in Portfolio Assignment)
    grouped = au_data[au_data['PORT_CODE'].notna()].groupby("PORT_CODE")
    
    for pid, group in grouped:
        # Get total customers for this portfolio from original data (similar to Portfolio Assignment logic)
        total_customer = len(au_data[au_data['PORT_CODE'] == pid])
        
        # Determine portfolio type
        portfolio_type = "Unknown"
        if not group.empty:
            # Get the most common type for this portfolio
            types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
            if not types.empty:
                portfolio_type = types.index[0]
            else:
                # If no non-unmanaged types, use the first type
                portfolio_type = group['TYPE'].iloc[0] if len(group) > 0 else "Unknown"
        
        portfolio_summary.append({
            'Include': True,
            'Portfolio ID': pid,
            'Portfolio Type': portfolio_type,
            'Total Customers': total_customer,
            'Available for this portfolio': len(group),
            'Select': len(group)
        })
    
    # Add unmanaged customers (like in Portfolio Assignment)
    unmanaged_customers = au_data[
        (au_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
        (au_data['PORT_CODE'].isna())
    ]
    
    if not unmanaged_customers.empty:
        portfolio_summary.append({
            'Include': True,
            'Portfolio ID': 'UNMANAGED',
            'Portfolio Type': 'Unmanaged',
            'Total Customers': len(unmanaged_customers),
            'Available for this portfolio': len(unmanaged_customers),
            'Select': len(unmanaged_customers)
        })
    
    return portfolio_summary

def display_smart_geographic_map(smart_portfolios_created, branch_data):
    """Display the geographic distribution map for smart portfolios"""
    # Convert to format expected by create_combined_map
    preview_portfolios = {}
    
    for au_id, au_data in smart_portfolios_created.items():
        if not au_data.empty:
            preview_portfolios[f"AU_{au_id}_Smart_Portfolio"] = au_data
    
    # Display the map with smart portfolios
    if preview_portfolios:
        combined_map = create_combined_map(preview_portfolios, branch_data)
        if combined_map:
            st.plotly_chart(combined_map, use_container_width=True)
    else:
        st.info("No customers selected for map display")

if __name__ == "__main__":
    main()
