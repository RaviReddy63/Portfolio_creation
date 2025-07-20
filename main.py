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
    """Display smart portfolio results in tabs"""
    
    if 'smart_portfolio_results' not in st.session_state or len(st.session_state.smart_portfolio_results) == 0:
        st.info("Click 'Generate Smart Portfolios' to create optimized customer assignments.")
        return
    
    results_df = st.session_state.smart_portfolio_results
    
    st.markdown("----")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📋 Portfolio Details", "🗺️ Geographic Map"])
    
    with tab1:
        display_smart_portfolio_summary(results_df)
    
    with tab2:
        display_smart_portfolio_details(results_df, branch_data)
    
    with tab3:
        display_smart_portfolio_map(results_df, branch_data)

def display_smart_portfolio_summary(results_df):
    """Display summary statistics for smart portfolios"""
    st.subheader("Smart Portfolio Summary")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers Assigned", len(results_df))
    
    with col2:
        avg_distance = results_df['DISTANCE_TO_AU'].mean()
        st.metric("Average Distance", f"{avg_distance:.1f} miles")
    
    with col3:
        unique_aus = results_df['ASSIGNED_AU'].nunique()
        st.metric("AUs Utilized", unique_aus)
    
    with col4:
        max_distance = results_df['DISTANCE_TO_AU'].max()
        st.metric("Max Distance", f"{max_distance:.1f} miles")
    
    # Portfolio type breakdown
    st.subheader("Portfolio Type Breakdown")
    
    type_summary = results_df.groupby('TYPE').agg({
        'ECN': 'count',
        'DISTANCE_TO_AU': ['mean', 'max']
    }).round(2)
    type_summary.columns = ['Customer Count', 'Avg Distance (miles)', 'Max Distance (miles)']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(type_summary, use_container_width=True)
    
    with col2:
        # Create pie chart for portfolio types
        type_counts = results_df['TYPE'].value_counts()
        st.bar_chart(type_counts)

def display_smart_portfolio_details(results_df, branch_data):
    """Display detailed portfolio information"""
    st.subheader("AU Portfolio Details")
    
    # Group by AU and calculate statistics
    au_summary = results_df.groupby(['ASSIGNED_AU', 'TYPE']).agg({
        'ECN': 'count',
        'DISTANCE_TO_AU': ['mean', 'max', 'min']
    }).round(2)
    
    au_summary.columns = ['Customer Count', 'Avg Distance', 'Max Distance', 'Min Distance']
    au_summary = au_summary.reset_index()
    
    # Add branch information
    branch_info = branch_data[['AU', 'CITY', 'STATECODE']].rename(columns={'AU': 'ASSIGNED_AU'})
    au_summary = au_summary.merge(branch_info, on='ASSIGNED_AU', how='left')
    
    # Reorder columns
    display_columns = ['ASSIGNED_AU', 'CITY', 'STATECODE', 'TYPE', 'Customer Count', 
                      'Avg Distance', 'Max Distance', 'Min Distance']
    au_summary = au_summary[display_columns]
    
    # Sort by AU and Type
    au_summary = au_summary.sort_values(['ASSIGNED_AU', 'TYPE'])
    
    st.dataframe(au_summary, use_container_width=True, hide_index=True)
    
    # Show detailed customer list if requested
    with st.expander("View Detailed Customer Assignments"):
        # Add filters for the detailed view
        selected_au = st.selectbox("Filter by AU", ['All'] + sorted(results_df['ASSIGNED_AU'].unique()))
        selected_type = st.selectbox("Filter by Type", ['All'] + sorted(results_df['TYPE'].unique()))
        
        filtered_details = results_df.copy()
        
        if selected_au != 'All':
            filtered_details = filtered_details[filtered_details['ASSIGNED_AU'] == selected_au]
        
        if selected_type != 'All':
            filtered_details = filtered_details[filtered_details['TYPE'] == selected_type]
        
        # Display filtered results
        display_columns = ['ECN', 'BILLINGCITY', 'BILLINGSTATE', 'ASSIGNED_AU', 'TYPE', 'DISTANCE_TO_AU']
        st.dataframe(filtered_details[display_columns], use_container_width=True, hide_index=True)

def display_smart_portfolio_map(results_df, branch_data):
    """Display geographic map of smart portfolio assignments"""
    st.subheader("Geographic Distribution")
    
    if len(results_df) > 0:
        # Create the smart portfolio map
        smart_map = create_smart_portfolio_map(results_df, branch_data)
        if smart_map:
            st.plotly_chart(smart_map, use_container_width=True)
        else:
            st.error("Unable to create map visualization")
    else:
        st.info("No portfolio data to display on map")

if __name__ == "__main__":
    main()
