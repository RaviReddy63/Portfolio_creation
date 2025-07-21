import streamlit as st
import pandas as pd

# Import custom modules
from ui_components import (
    add_logo, create_header, initialize_session_state,
    create_au_filters, create_customer_filters, create_portfolio_button,
    display_summary_statistics, create_portfolio_editor, create_apply_changes_button
)
from data_loader import get_merged_data
from portfolio_creation import process_portfolio_creation, apply_portfolio_changes
from map_visualization import create_combined_map, create_smart_portfolio_map
from portfolio_creation_8 import enhanced_customer_au_assignment_with_two_inmarket_iterations

def setup_page_config():
    """Configure the Streamlit page"""
    st.set_page_config("Portfolio Creation tool", layout="wide")

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

def display_portfolio_tables(portfolios_created, portfolio_summaries, branch_data):
    """Display portfolio summary tables - Always use tabs"""
    # Always use tabs regardless of number of AUs (removed conditional logic)
    au_tabs = st.tabs([f"AU {au_id}" for au_id in portfolios_created.keys()])
    
    for tab_idx, (au_id, tab) in enumerate(zip(portfolios_created.keys(), au_tabs)):
        with tab:
            display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, True)

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
            portfolio_condition = filtered_data['CG_PORTFOLIO_CD'].isin(cust_portcd)
        
        # Apply OR logic: keep rows that match either role OR portfolio code
        combined_condition = role_condition | portfolio_condition
        filtered_data = filtered_data[combined_condition]
    
    # Simple exclusivity check: if filtering for low priority roles, exclude high priority customers
    if role is not None:
        role_clean = [r.strip().lower() for r in role]
        if any(r in ['unassigned', 'unmanaged'] for r in role_clean):
            # Exclude customers who have INMARKET or CENTRALIZED assignments anywhere in the dataset
            priority_customers = customer_data[
                customer_data['TYPE'].str.lower().str.strip().isin(['inmarket', 'centralized'])
            ]['CG_ECN'].unique()
            
            if len(priority_customers) > 0:
                filtered_data = filtered_data[~filtered_data['CG_ECN'].isin(priority_customers)]
    
    # Apply other filters
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    # Remove duplicates
    filtered_data = filtered_data.drop_duplicates(subset=['CG_ECN'], keep='first')
    
    return filtered_data

def clean_portfolio_data(data):
    """Clean portfolio data by removing duplicates and handling priority conflicts"""
    if data.empty:
        return data
    
    # Step 1: Remove exact duplicates (same ECN with identical records)
    data_cleaned = data.drop_duplicates(subset=['CG_ECN'], keep='first')
    
    # Step 2: Handle portfolio type priority conflicts
    # Priority: INMARKET/CENTRALIZED > Unassigned/Unmanaged
    
    # Group by ECN to find customers with multiple portfolio assignments
    ecn_groups = data.groupby('CG_ECN')
    
    final_records = []
    priority_removed_count = 0
    duplicate_removed_count = len(data) - len(data_cleaned)
    
    for ecn, group in ecn_groups:
        if len(group) == 1:
            # Single record, keep as is
            final_records.append(group.iloc[0])
        else:
            # Multiple records for same ECN - apply priority logic
            group_types = group['TYPE'].str.lower().str.strip()
            
            # Check if we have both high priority (INMARKET/CENTRALIZED) and low priority (Unassigned/Unmanaged)
            high_priority_mask = group_types.isin(['inmarket', 'centralized'])
            low_priority_mask = group_types.isin(['unassigned', 'unmanaged'])
            
            if high_priority_mask.any() and low_priority_mask.any():
                # Conflict detected - keep only high priority records
                high_priority_records = group[high_priority_mask]
                final_records.append(high_priority_records.iloc[0])  # Take first high priority record
                priority_removed_count += len(group) - 1
            else:
                # No conflict, just keep the first record (already handled by drop_duplicates)
                final_records.append(group.iloc[0])
    
    # Convert back to DataFrame
    if final_records:
        result_df = pd.DataFrame(final_records).reset_index(drop=True)
    else:
        result_df = pd.DataFrame()
    
    # Log cleaning results if any changes were made
    if duplicate_removed_count > 0 or priority_removed_count > 0:
        import streamlit as st
        st.info(f"Data cleaned: Removed {duplicate_removed_count} duplicates and {priority_removed_count} lower priority records. Final records: {len(result_df)}")
    
    return result_df

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
        
        # Use original portfolio code if available, otherwise use N/A
        au_data['PORT_CODE'] = au_data['CG_PORTFOLIO_CD'].fillna('N/A')
        
        # Use original financial data
        au_data['BANK_REVENUE'] = au_data['BANK_REVENUE'].fillna(0)
        au_data['DEPOSIT_BAL'] = au_data['DEPOSIT_BAL'].fillna(0)
        
        # Use original TYPE if different from smart assignment
        au_data['TYPE'] = au_data['TYPE_orig'].fillna(au_data['TYPE'])
        
        # Clean up duplicate columns
        au_data = au_data.drop(['CG_PORTFOLIO_CD', 'TYPE_orig'], axis=1, errors='ignore')
        
        smart_portfolios_created[au] = au_data
    
    st.markdown("----")
    
    # Display results in two sections with equal column width
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Smart Portfolio Summary")
        display_smart_portfolio_tables(smart_portfolios_created, branch_data)
    
    with col2:
        st.subheader("Global Portfolio Control")
        display_global_portfolio_control_component(results_df, customer_data, branch_data)
    
    # Geographic Distribution below with full width
    st.markdown("----")
    st.subheader("Geographic Distribution")
    display_smart_geographic_map(smart_portfolios_created, branch_data)[0]['BRANCH_LON_NUM']
        
        # Rename columns to match Portfolio Assignment format
        au_data = au_data.rename(columns={
            'ECN': 'CG_ECN',
            'DISTANCE_TO_AU': 'Distance'
        })
        
        # Merge with original customer_data to get financial information
        customer_data_subset = customer_data[['CG_ECN', 'CG_PORTFOLIO_CD', 'BANK_REVENUE', 'DEPOSIT_BAL', 'TYPE']].copy()
        au_data = au_data.merge(customer_data_subset, on='CG_ECN', how='left', suffixes=('', '_orig'))
        
        # Use original portfolio code if available, otherwise use N/A
        au_data['PORT_CODE'] = au_data['CG_PORTFOLIO_CD'].fillna('N/A')
        
        # Use original financial data
        au_data['BANK_REVENUE'] = au_data['BANK_REVENUE'].fillna(0)
        au_data['DEPOSIT_BAL'] = au_data['DEPOSIT_BAL'].fillna(0)
        
        # Use original TYPE if different from smart assignment
        au_data['TYPE'] = au_data['TYPE_orig'].fillna(au_data['TYPE'])
        
        # Clean up duplicate columns
        au_data = au_data.drop(['CG_PORTFOLIO_CD', 'TYPE_orig'], axis=1, errors='ignore')
        
        smart_portfolios_created[au] = au_data
    
    st.markdown("----")
    
    # Display results in two sections with equal column width
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Smart Portfolio Summary")
        display_smart_portfolio_tables(smart_portfolios_created, branch_data)
    
    with col2:
        st.subheader("Global Portfolio Control")
        display_global_portfolio_control_component(results_df, customer_data, branch_data)
    
    # Geographic Distribution below with full width
    st.markdown("----")
    st.subheader("Geographic Distribution")
    display_smart_geographic_map(smart_portfolios_created, branch_data)

def clean_smart_portfolio_results(results_df):
    """Clean smart portfolio results by removing duplicates and handling conflicts"""
    if results_df.empty:
        return results_df
    
    original_count = len(results_df)
    
    # Step 1: Remove exact duplicates based on ECN
    cleaned_df = results_df.drop_duplicates(subset=['ECN'], keep='first')
    duplicate_removed = original_count - len(cleaned_df)
    
    # Step 2: Handle TYPE priority conflicts for same ECN
    # Priority: INMARKET/CENTRALIZED > any other types
    
    ecn_groups = results_df.groupby('ECN')
    final_records = []
    priority_conflicts_resolved = 0
    
    for ecn, group in ecn_groups:
        if len(group) == 1:
            # Single record, keep as is
            final_records.append(group.iloc[0])
        else:
            # Multiple records for same ECN
            group_types = group['TYPE'].str.upper().str.strip()
            
            # Priority order: INMARKET > CENTRALIZED > others
            if 'INMARKET' in group_types.values:
                # Keep INMARKET record
                inmarket_record = group[group_types == 'INMARKET'].iloc[0]
                final_records.append(inmarket_record)
                priority_conflicts_resolved += len(group) - 1
            elif 'CENTRALIZED' in group_types.values:
                # Keep CENTRALIZED record
                centralized_record = group[group_types == 'CENTRALIZED'].iloc[0]
                final_records.append(centralized_record)
                priority_conflicts_resolved += len(group) - 1
            else:
                # No priority conflicts, keep first record
                final_records.append(group.iloc[0])
    
    # Convert back to DataFrame
    if final_records:
        result_df = pd.DataFrame(final_records).reset_index(drop=True)
    else:
        result_df = pd.DataFrame()
    
    # Log cleaning results
    total_removed = original_count - len(result_df)
    if total_removed > 0:
        st.info(f"Smart portfolio data cleaned: Removed {duplicate_removed} duplicates and resolved {priority_conflicts_resolved} priority conflicts. Final records: {len(result_df)}")
    
    return result_df

def display_smart_portfolio_tables(smart_portfolios_created, branch_data):
    """Display smart portfolio summary tables - Always use tabs"""
    # Always use tabs regardless of number of AUs (removed conditional logic)
    au_tabs = st.tabs([f"AU {au_id}" for au_id in smart_portfolios_created.keys()])
    
    for tab_idx, (au_id, tab) in enumerate(zip(smart_portfolios_created.keys(), au_tabs)):
        with tab:
            display_single_smart_au_table(au_id, smart_portfolios_created, branch_data)

def display_single_smart_au_table(au_id, smart_portfolios_created, branch_data):
    """Display table for a single smart portfolio AU"""
    if au_id in smart_portfolios_created:
        au_data = smart_portfolios_created[au_id]
        
        # Create portfolio summary similar to Portfolio Assignment
        portfolio_summary = create_smart_portfolio_summary(au_data, au_id)
        
        if portfolio_summary:
            portfolio_df = pd.DataFrame(portfolio_summary)
            
            # Create editable dataframe (same as Portfolio Assignment)
            edited_df = create_smart_portfolio_editor(portfolio_df, au_id)
            
            # Store the edited data in session state
            if 'smart_portfolio_controls' not in st.session_state:
                st.session_state.smart_portfolio_controls = {}
            st.session_state.smart_portfolio_controls[au_id] = edited_df
            
            # Add Apply Changes button (same as Portfolio Assignment)
            if create_smart_apply_changes_button(au_id):
                apply_smart_portfolio_changes(au_id, smart_portfolios_created, branch_data)
            
            # Display summary statistics (same as Portfolio Assignment)
            display_summary_statistics(au_data)

def display_global_portfolio_control_component(results_df, customer_data, branch_data):
    """Display unified global portfolio control component with table, button and statistics"""
    
    # Generate global portfolio summary (same as before)
    global_summary = generate_global_portfolio_summary(results_df, customer_data)
    
    if global_summary:
        # Create Global Portfolio tab (to match the AU tabs structure)
        global_tab = st.tabs(["Global Control"])
        
        with global_tab[0]:
            # Display editable table
            edited_summary = create_global_control_editor(global_summary)
            
            # Store in session state
            st.session_state.global_portfolio_controls = edited_summary
            
            # Apply Changes button with same style as AU Apply Changes
            if st.button("Apply Global Changes", key="apply_global_smart_changes", type="primary"):
                apply_global_smart_changes(edited_summary, global_summary, customer_data, branch_data)
            
            # Display global statistics (similar to AU summary statistics)
            display_global_portfolio_statistics(results_df)

def display_global_portfolio_statistics(results_df):
    """Display summary statistics for global portfolios in horizontal format"""
    
    if len(results_df) > 0:
        # Calculate metrics
        total_customers = len(results_df)
        avg_distance = results_df['DISTANCE_TO_AU'].mean()
        
        # Calculate distinct AUs for each portfolio type
        inmarket_aus = results_df[results_df['TYPE'] == 'INMARKET']['ASSIGNED_AU'].nunique()
        centralized_aus = results_df[results_df['TYPE'] == 'CENTRALIZED']['ASSIGNED_AU'].nunique()
        
        # Display metrics in horizontal format (same as AU Summary Statistics)
        st.subheader("Global Summary Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col_b:
            st.metric("Average Distance (Miles)", f"{avg_distance:.1f}")
        
        with col_c:
            st.metric("In-Market Portfolios", inmarket_aus)
        
        with col_d:
            st.metric("Centralized Portfolios", centralized_aus)
        
    else:
        # Show empty state
        st.subheader("Global Summary Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Total Customers", "0")
        
        with col_b:
            st.metric("Average Distance (Miles)", "0.0")
        
        with col_c:
            st.metric("In-Market Portfolios", "0")
        
        with col_d:
            st.metric("Centralized Portfolios", "0")

def generate_global_portfolio_summary(results_df, customer_data):
    """Generate global portfolio summary across all AUs"""
    
    # Group by Portfolio ID across all AUs
    portfolio_aggregates = {}
    
    # Get all unique portfolio IDs from results
    for _, row in results_df.iterrows():
        ecn = row['ECN']
        
        # Find original portfolio code
        original_customer = customer_data[customer_data['CG_ECN'] == ecn]
        if not original_customer.empty:
            portfolio_code = original_customer.iloc[0].get('CG_PORTFOLIO_CD', 'N/A')
            if pd.isna(portfolio_code):
                portfolio_code = 'N/A'
        else:
            portfolio_code = 'N/A'
        
        portfolio_type = row['TYPE']
        
        if portfolio_code not in portfolio_aggregates:
            portfolio_aggregates[portfolio_code] = {
                'Portfolio ID': portfolio_code,
                'Portfolio Type': portfolio_type,
                'customers': [],
                'total_available': 0
            }
        
        portfolio_aggregates[portfolio_code]['customers'].append(row)
        portfolio_aggregates[portfolio_code]['total_available'] += 1
    
    # Convert to summary list
    summary_list = []
    for portfolio_id, data in portfolio_aggregates.items():
        # Get total customers for this portfolio from original data
        if portfolio_id == 'N/A':
            total_customers = len([c for c in data['customers']])
        else:
            total_customers = len(customer_data[customer_data['CG_PORTFOLIO_CD'] == portfolio_id])
        
        summary_list.append({
            'Include': True,
            'Portfolio ID': portfolio_id,
            'Portfolio Type': data['Portfolio Type'],
            'Total Customers': total_customers,
            'Available': data['total_available'],
            'Select': data['total_available']
        })
    
    # Sort by Portfolio ID
    summary_list.sort(key=lambda x: x['Portfolio ID'])
    
    return summary_list

def create_global_control_editor(global_summary):
    """Create editable global control table"""
    
    df = pd.DataFrame(global_summary)
    
    # Create column configuration
    column_config = {
        "Include": st.column_config.CheckboxColumn(
            "Include",
            help="Include portfolio in all AUs (affects all portfolios)"
        ),
        "Portfolio ID": st.column_config.TextColumn(
            "Portfolio ID",
            help="Unique portfolio identifier",
            disabled=True
        ),
        "Portfolio Type": st.column_config.TextColumn(
            "Portfolio Type",
            help="Type of portfolio",
            disabled=True
        ),
        "Total Customers": st.column_config.NumberColumn(
            "Total Customers",
            help="Total customers in this portfolio in original data",
            disabled=True
        ),
        "Available": st.column_config.NumberColumn(
            "Available",
            help="Total customers from this portfolio across all AUs",
            disabled=True
        ),
        "Select": st.column_config.NumberColumn(
            "Select",
            help="Number of customers to select (will keep closest)",
            min_value=0,
            step=1
        )
    }
    
    # Display editable table with increased height to match AU tables
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=350,  # Increased height to match AU tables
        key="global_smart_control_editor"
    )
    
    return edited_df

def apply_global_smart_changes(edited_summary, original_summary, customer_data, branch_data):
    """Apply global changes and regenerate smart portfolios"""
    
    with st.spinner("Applying global changes and regenerating portfolios..."):
        try:
            # Get current filters from session state
            cust_state = st.session_state.get('mapping_filter_cust_state', [])
            role = st.session_state.get('mapping_filter_role', [])
            cust_portcd = st.session_state.get('mapping_filter_cust_portcd', [])
            min_rev = st.session_state.get('mapping_filter_min_rev', 5000)
            min_deposit = st.session_state.get('mapping_filter_min_deposit', 100000)
            
            # Apply customer filters to get base filtered customers
            filtered_customers = apply_customer_filters_for_mapping(
                customer_data, cust_state if cust_state else None, 
                role if role else None, cust_portcd if cust_portcd else None, 
                min_rev, min_deposit
            )
            
            if len(filtered_customers) == 0:
                st.error("No customers found with current filters.")
                return
            
            # Apply portfolio-level filters based on global controls
            final_filtered_customers = apply_portfolio_level_filters(
                filtered_customers, edited_summary, original_summary, branch_data
            )
            
            if len(final_filtered_customers) == 0:
                st.error("No customers remaining after portfolio filters.")
                return
            
            # Regenerate smart portfolios with filtered customers
            st.info(f"Regenerating portfolios with {len(final_filtered_customers)} customers...")
            
            smart_portfolio_results = enhanced_customer_au_assignment_with_two_inmarket_iterations(
                final_filtered_customers, branch_data
            )
            
            # Update session state
            st.session_state.smart_portfolio_results = smart_portfolio_results
            
            st.success(f"Successfully regenerated portfolios with {len(smart_portfolio_results)} customers!")
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error applying global changes: {str(e)}")

def apply_portfolio_level_filters(filtered_customers, edited_summary, original_summary, branch_data):
    """Apply portfolio-level include/select filters"""
    
    from utils import haversine_distance
    
    final_customers = []
    
    # Get the AUs that were used in the current smart portfolio results
    current_results = st.session_state.get('smart_portfolio_results', pd.DataFrame())
    if not current_results.empty:
        identified_aus = current_results['ASSIGNED_AU'].unique().tolist()
    else:
        identified_aus = []
    
    # Process each portfolio
    for _, edited_row in edited_summary.iterrows():
        portfolio_id = edited_row['Portfolio ID']
        include = edited_row['Include']
        select_count = edited_row['Select']
        
        # Skip if not included
        if not include:
            continue
        
        # Get customers for this portfolio
        if portfolio_id == 'N/A':
            # Customers without original portfolio
            portfolio_customers = filtered_customers[
                filtered_customers['CG_PORTFOLIO_CD'].isna()
            ].copy()
        else:
            # Customers with specific portfolio code
            portfolio_customers = filtered_customers[
                filtered_customers['CG_PORTFOLIO_CD'] == portfolio_id
            ].copy()
        
        if len(portfolio_customers) == 0:
            continue
        
        # Apply select count filter by keeping closest customers to any identified AU
        if select_count < len(portfolio_customers):
            portfolio_customers = select_closest_customers_to_any_au(
                portfolio_customers, select_count, identified_aus, branch_data
            )
        elif select_count > len(portfolio_customers):
            # Can't select more than available
            select_count = len(portfolio_customers)
        
        final_customers.append(portfolio_customers)
    
    # Combine all selected customers
    if final_customers:
        return pd.concat(final_customers, ignore_index=True)
    else:
        return pd.DataFrame()

def select_closest_customers_to_any_au(portfolio_customers, select_count, selected_aus, branch_data):
    """Select closest customers to any of the AUs that would be created"""
    
    from utils import haversine_distance
    
    if not selected_aus or len(selected_aus) == 0:
        # If no specific AUs, just return first N customers sorted by a consistent criteria
        return portfolio_customers.head(select_count)
    
    # Calculate minimum distance to any AU for each customer
    customers_with_distance = []
    
    for idx, customer in portfolio_customers.iterrows():
        min_distance = float('inf')
        
        for au_id in selected_aus:
            # Get AU coordinates
            au_row = branch_data[branch_data['AU'] == au_id]
            if au_row.empty:
                continue
                
            au_lat = au_row.iloc[0]['BRANCH_LAT_NUM']
            au_lon = au_row.iloc[0]['BRANCH_LON_NUM']
            
            # Calculate distance
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'], au_lat, au_lon
            )
            
            if distance < min_distance:
                min_distance = distance
        
        customers_with_distance.append({
            'customer': customer,
            'min_distance': min_distance,
            'index': idx
        })
    
    # Sort by distance and select closest
    customers_with_distance.sort(key=lambda x: x['min_distance'])
    selected_indices = [item['index'] for item in customers_with_distance[:select_count]]
    
    return portfolio_customers.loc[selected_indices]

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
        (au_data['TYPE'].str.lower().str.strip() == 'unassigned') |
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

def create_smart_portfolio_editor(portfolio_df, au_id):
    """Create an editable portfolio dataframe for smart portfolios"""
    column_config = {
        "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio in selection"),
        "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
        "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
        "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
        "Available for this portfolio": st.column_config.NumberColumn("Available for this portfolio", disabled=True),
        "Select": st.column_config.NumberColumn(
            "Select",
            help="Number of customers to select from this portfolio",
            min_value=0,
            step=1
        )
    }
    
    return st.data_editor(
        portfolio_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=350,  # Match height with other tables
        key=f"smart_portfolio_editor_{au_id}_{len(portfolio_df)}"
    )

def create_smart_apply_changes_button(au_id):
    """Create Apply Changes button for smart portfolio AU"""
    return st.button(f"Apply Changes for AU {au_id}", key=f"apply_smart_changes_{au_id}")

def apply_smart_portfolio_changes(au_id, smart_portfolios_created, branch_data):
    """Apply portfolio selection changes for smart portfolios"""
    with st.spinner("Applying selection changes..."):
        if (au_id in st.session_state.smart_portfolio_controls and 
            au_id in smart_portfolios_created):
            
            # Get the edited controls
            control_data = st.session_state.smart_portfolio_controls[au_id]
            original_data = smart_portfolios_created[au_id].copy()
            
            # Apply selection changes
            updated_au_data = apply_smart_selection_changes(original_data, control_data)
            
            # Update the smart portfolios in session state
            if 'smart_portfolio_results' in st.session_state:
                # Update the results dataframe
                results_df = st.session_state.smart_portfolio_results.copy()
                
                # Remove old AU data
                results_df = results_df[results_df['ASSIGNED_AU'] != au_id]
                
                # Add updated AU data (convert back to results format)
                if not updated_au_data.empty:
                    updated_results = updated_au_data.rename(columns={
                        'CG_ECN': 'ECN',
                        'Distance': 'DISTANCE_TO_AU'
                    })[['ECN', 'BILLINGCITY', 'BILLINGSTATE', 'LAT_NUM', 'LON_NUM', 'ASSIGNED_AU', 'DISTANCE_TO_AU', 'TYPE']]
                    
                    results_df = pd.concat([results_df, updated_results], ignore_index=True)
                
                # Update session state
                st.session_state.smart_portfolio_results = results_df
                
                # Update the smart portfolios created
                smart_portfolios_created[au_id] = updated_au_data
            
            st.success("Portfolio selection updated!")
            st.experimental_rerun()

def apply_smart_selection_changes(original_data, control_data):
    """Apply the selection changes from portfolio controls to filter customers"""
    
    selected_customers = []
    
    # Process each portfolio selection
    for _, row in control_data.iterrows():
        portfolio_id = row['Portfolio ID']
        select_count = row['Select']
        include = row.get('Include', True)
        
        # Only include portfolios that are checked (include=True) and have select_count > 0
        if not include or select_count <= 0:
            continue
            
        if portfolio_id == 'UNMANAGED':
            # Handle unmanaged customers
            unmanaged_customers = original_data[
                (original_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                (original_data['PORT_CODE'].isna()) |
                (original_data['PORT_CODE'].str.startswith('SMART_'))
            ].copy()
            
            if not unmanaged_customers.empty:
                # Sort by distance (closest first) and take the requested count
                unmanaged_sorted = unmanaged_customers.sort_values('Distance').head(select_count)
                selected_customers.append(unmanaged_sorted)
                
        else:
            # Handle regular portfolios
            portfolio_customers = original_data[original_data['PORT_CODE'] == portfolio_id].copy()
            
            if not portfolio_customers.empty:
                # Sort by distance (closest first) and take the requested count
                portfolio_sorted = portfolio_customers.sort_values('Distance').head(select_count)
                selected_customers.append(portfolio_sorted)
    
    # Combine all selected customers for this AU
    if selected_customers:
        final_customers = pd.concat(selected_customers, ignore_index=True)
        return final_customers
    else:
        # No customers selected for this AU
        return pd.DataFrame()

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
