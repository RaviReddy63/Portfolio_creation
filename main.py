import streamlit as st
import pandas as pd

# Import custom modules
from ui_components import (
    add_logo, create_header, initialize_session_state,
    create_au_filters, create_customer_filters, create_portfolio_button,
    display_summary_statistics, create_portfolio_editor, create_apply_changes_button,
    create_customer_filters_for_mapping, create_save_buttons
)
from data_loader import load_data
from portfolio_creation import process_portfolio_creation, apply_portfolio_changes
from map_visualization import create_combined_map, create_smart_portfolio_map
from portfolio_creation_8 import enhanced_customer_au_assignment_with_two_inmarket_iterations
from utils import merge_dfs

def setup_page_config():
    """Configure the Streamlit page"""
    st.set_page_config("Portfolio Creation tool", layout="wide")

def get_merged_data():
    """Load and merge all data with initial cleanup"""
    customer_data, banker_data, branch_data = load_data()
    
    # Initial data cleanup - remove conflicting portfolio assignments
    customer_data = clean_initial_data(customer_data)
    
    data = merge_dfs(customer_data, banker_data, branch_data)
    return customer_data, banker_data, branch_data, data

def clean_initial_data(customer_data):
    """Clean initial data by removing Unassigned/Unmanaged rows for customers who also have In-Market/Centralized assignments"""
    
    if customer_data.empty:
        return customer_data
    
    original_count = len(customer_data)
    
    # Step 1: Priority Conflict Resolution
    # Find customers (ECNs) who have In-Market/Centralized assignments
    priority_customers = customer_data[
        customer_data['TYPE'].str.lower().str.strip().isin(['in-market', 'inmarket', 'centralized'])
    ]['CG_ECN'].unique()
    
    # Remove Unassigned/Unmanaged rows for these priority customers
    mask_to_remove = (
        customer_data['CG_ECN'].isin(priority_customers) & 
        customer_data['TYPE'].str.lower().str.strip().isin(['unassigned', 'unmanaged'])
    )
    
    cleaned_data = customer_data[~mask_to_remove].copy()
    priority_removed = original_count - len(cleaned_data)
    
    # Step 2: Remove duplicate ECNs (keep only first occurrence)
    final_data = cleaned_data.drop_duplicates(subset=['CG_ECN'], keep='first')
    duplicate_removed = len(cleaned_data) - len(final_data)
    
    # Log cleanup results
    total_removed = original_count - len(final_data)
    if total_removed > 0:
        print(f"Data cleanup: Removed {priority_removed} priority conflicts and {duplicate_removed} duplicate ECNs. Total removed: {total_removed}")
    
    return final_data

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
    
    # Store branch_data in session state for save functions
    st.session_state.branch_data = branch_data
    
    if page == "Portfolio Assignment":
        portfolio_assignment_page(customer_data, banker_data, branch_data)
    elif page == "Portfolio Mapping":
        portfolio_mapping_page(customer_data, banker_data, branch_data)

def portfolio_assignment_page(customer_data, banker_data, branch_data):
    """Portfolio Assignment page logic"""
    
    # Store customer_data in session state for save functions
    st.session_state.customer_data = customer_data
    
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
        
        # Create button row with Apply Changes and Save buttons
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.write("")  # Empty space
        
        with col2:
            apply_clicked = create_apply_changes_button(au_id, not is_multi_au)
        
        with col3:
            save_au_clicked = st.button(f"Save AU {au_id}", key=f"save_au_{au_id}", type="secondary")
        
        with col4:
            save_all_clicked = st.button("Save All", key=f"save_all_from_au_{au_id}", type="secondary")
        
        # Handle button clicks
        if apply_clicked:
            apply_portfolio_changes(au_id, branch_data)
        
        if save_au_clicked:
            save_single_au_portfolio(au_id, portfolios_created, st.session_state.get('customer_data'))
        
        if save_all_clicked:
            save_all_portfolios(portfolios_created, st.session_state.get('customer_data'))
        
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
    
    # Apply other filters
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    return filtered_data

def generate_smart_portfolios(customer_data, branch_data, cust_state, role, cust_portcd, min_rev, min_deposit):
    """Generate smart portfolios using advanced clustering"""
    
    # Clear global data when generating new portfolios
    if 'global_portfolio_df' in st.session_state:
        del st.session_state.global_portfolio_df
    
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
        
        # Clear any existing smart portfolio controls to force refresh
        if 'smart_portfolio_controls' in st.session_state:
            st.session_state.smart_portfolio_controls = {}
        
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
    
    # Store customer_data in session state for save functions
    st.session_state.customer_data = customer_data
    
    if 'smart_portfolio_results' not in st.session_state or len(st.session_state.smart_portfolio_results) == 0:
        st.info("Click 'Generate Smart Portfolios' to create optimized customer assignments.")
        return
    
    # Always use the current results from session state
    results_df = st.session_state.smart_portfolio_results
    
    # Convert smart portfolio results to Portfolio Assignment format - regenerate every time
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
    display_smart_geographic_map(smart_portfolios_created, branch_data)

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
            
            # Create button row with Apply Changes and Save All buttons
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.write("")  # Empty space
            
            with col2:
                apply_clicked = create_smart_apply_changes_button(au_id)
            
            with col3:
                save_all_clicked = st.button("Save All Portfolios", key=f"save_all_smart_{au_id}", type="secondary")
            
            # Handle button clicks
            if apply_clicked:
                apply_smart_portfolio_changes(au_id, smart_portfolios_created, branch_data)
            
            if save_all_clicked:
                save_all_smart_portfolios(smart_portfolios_created, st.session_state.get('customer_data'))
            
            # Display summary statistics (same as Portfolio Assignment)
            display_summary_statistics(au_data)

def display_global_portfolio_control_component(results_df, customer_data, branch_data):
    """Display unified global portfolio control component with table, button and statistics"""
    
    # Create Global Portfolio tab
    global_tab = st.tabs(["Global Control"])
    
    with global_tab[0]:
        # Initialize data only once
        if 'global_portfolio_df' not in st.session_state:
            global_summary = generate_global_portfolio_summary(results_df, customer_data)
            if global_summary:
                st.session_state.global_portfolio_df = pd.DataFrame(global_summary)
        
        # Display editor if data exists
        if 'global_portfolio_df' in st.session_state:
            
            # Simple data editor - do NOT update session state automatically
            edited_df = st.data_editor(
                st.session_state.global_portfolio_df,
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include"),
                    "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
                    "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
                    "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
                    "Available": st.column_config.NumberColumn("Available", disabled=True),
                    "Select": st.column_config.NumberColumn("Select", min_value=0, step=1)
                },
                hide_index=True,
                use_container_width=True,
                height=350,
                key="global_editor"
            )
            
            # Apply button - only regenerates data when clicked
            if st.button("Apply Global Changes", key="apply_global", type="primary"):
                apply_global_changes_final(edited_df, customer_data, branch_data)
            
            # Use updated results from session state, not the original parameter
            current_results = st.session_state.get('smart_portfolio_results', results_df)
            display_global_portfolio_statistics(current_results)

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
        
        # Find original portfolio code and TYPE
        original_customer = customer_data[customer_data['CG_ECN'] == ecn]
        if not original_customer.empty:
            portfolio_code = original_customer.iloc[0].get('CG_PORTFOLIO_CD', 'N/A')
            if pd.isna(portfolio_code):
                portfolio_code = 'N/A'
            # Use original customer TYPE, not algorithm TYPE
            original_type = original_customer.iloc[0].get('TYPE', 'Unknown')
        else:
            portfolio_code = 'N/A'
            original_type = 'Unknown'
        
        if portfolio_code not in portfolio_aggregates:
            portfolio_aggregates[portfolio_code] = {
                'Portfolio ID': portfolio_code,
                'Portfolio Type': original_type,  # Use original TYPE
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
            'Portfolio Type': data['Portfolio Type'],  # This is now original TYPE
            'Total Customers': total_customers,
            'Available': data['total_available'],
            'Select': data['total_available']
        })
    
    # Sort by Portfolio ID
    summary_list.sort(key=lambda x: x['Portfolio ID'])
    
    return summary_list

def apply_global_changes_final(edited_df, customer_data, branch_data):
    """Apply changes using the edited dataframe"""
    
    with st.spinner("Applying changes..."):
        try:
            # Get filters
            cust_state = st.session_state.get('mapping_cust_state')
            role = st.session_state.get('mapping_role')  
            cust_portcd = st.session_state.get('mapping_cust_portcd')
            min_rev = st.session_state.get('mapping_min_revenue', 5000)
            min_deposit = st.session_state.get('mapping_min_deposit', 100000)
            
            # Apply customer filters
            filtered_customers = apply_customer_filters_for_mapping(
                customer_data, cust_state, role, cust_portcd, min_rev, min_deposit
            )
            
            # Apply portfolio selections from edited_df
            final_customers = []
            for _, row in edited_df.iterrows():
                if not row['Include'] or row['Select'] <= 0:
                    continue
                    
                portfolio_id = row['Portfolio ID']
                select_count = int(row['Select'])
                
                if portfolio_id == 'N/A':
                    portfolio_customers = filtered_customers[filtered_customers['CG_PORTFOLIO_CD'].isna()]
                else:
                    portfolio_customers = filtered_customers[filtered_customers['CG_PORTFOLIO_CD'] == portfolio_id]
                
                if len(portfolio_customers) > 0:
                    selected = portfolio_customers.head(select_count)
                    final_customers.append(selected)
            
            if final_customers:
                combined_customers = pd.concat(final_customers, ignore_index=True)
                
                # Regenerate portfolios
                smart_results = enhanced_customer_au_assignment_with_two_inmarket_iterations(
                    combined_customers, branch_data
                )
                
                st.session_state.smart_portfolio_results = smart_results
                
                # Clear global data to regenerate next time
                if 'global_portfolio_df' in st.session_state:
                    del st.session_state.global_portfolio_df
                
                st.success(f"Applied changes with {len(smart_results)} customers!")
            else:
                st.error("No customers selected")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

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
        height=350,
        key=f"smart_portfolio_editor_{au_id}"
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

def prepare_portfolio_for_export(au_data, customer_data, branch_data):
    """Prepare portfolio data in the required export format"""
    from utils import haversine_distance
    
    if au_data.empty or customer_data.empty:
        return pd.DataFrame()
    
    # Merge with customer_data to get all required fields
    export_data = au_data.merge(
        customer_data[['CG_ECN', 'CG_PORTFOLIO_CD', 'TYPE', 'LAT_NUM', 'LON_NUM', 
                      'BILLINGCITY', 'BILLINGSTATE', 'BANK_REVENUE', 'CG_GROSS_SALES', 
                      'DEPOSIT_BAL', 'BANKER_FIRSTNAME', 'BANKER_LASTNAME', 
                      'BILLINGSTREET', 'CG_NAME']],
        on='CG_ECN',
        how='left',
        suffixes=('', '_orig')
    )
    
    # Get AU information from branch_data
    au_id = au_data['AU'].iloc[0]
    au_info = branch_data[branch_data['AU'] == au_id]
    
    if not au_info.empty:
        branch_lat = au_info.iloc[0]['BRANCH_LAT_NUM']
        branch_lon = au_info.iloc[0]['BRANCH_LON_NUM']
    else:
        branch_lat = au_data['BRANCH_LAT_NUM'].iloc[0] if 'BRANCH_LAT_NUM' in au_data.columns else 0
        branch_lon = au_data['BRANCH_LON_NUM'].iloc[0] if 'BRANCH_LON_NUM' in au_data.columns else 0
    
    # Calculate distance from customer to new AU
    export_data['DISTANCE'] = export_data.apply(
        lambda row: haversine_distance(
            row['LAT_NUM'], row['LON_NUM'], 
            branch_lat, branch_lon
        ), axis=1
    )
    
    # Prepare final export format
    final_export = pd.DataFrame({
        'CG_ECN': export_data['CG_ECN'],
        'CG_PORTFOLIO_CD': export_data['CG_PORTFOLIO_CD'],
        'TYPE': export_data['TYPE'],
        'LAT_NUM': export_data['LAT_NUM'],
        'LON_NUM': export_data['LON_NUM'],
        'BILLINGCITY': export_data['BILLINGCITY'],
        'BILLINGSTATE': export_data['BILLINGSTATE'],
        'DISTANCE': export_data['DISTANCE'],
        'AU_NBR': au_id,
        'BRANCH_LAT_NUM': branch_lat,
        'BRANCH_LON_NUM': branch_lon,
        'BANK_REVENUE': export_data['BANK_REVENUE'],
        'CG_GROSS_SALES': export_data['CG_GROSS_SALES'],
        'DEPOSIT_BAL': export_data['DEPOSIT_BAL'],
        'CURRENT_BANKER_FIRSTNAME': export_data['BANKER_FIRSTNAME'],
        'CURRENT_BANKER_LASTNAME': export_data['BANKER_LASTNAME'],
        'NAME': export_data['CG_NAME'],
        'BILLINGSTREET': export_data['BILLINGSTREET'],
        'BILLINGCITY': export_data['BILLINGCITY']
    })
    
    return final_export

def save_single_au_portfolio(au_id, portfolios_created, customer_data):
    """Save a single AU portfolio to CSV"""
    if au_id not in portfolios_created or customer_data is None:
        st.error("No data available to save")
        return
    
    try:
        # Get branch_data from session state or reload
        branch_data = st.session_state.get('branch_data')
        if branch_data is None:
            from data_loader import load_data
            _, _, branch_data = load_data()
        
        # Prepare data for export
        au_data = portfolios_created[au_id]
        export_data = prepare_portfolio_for_export(au_data, customer_data, branch_data)
        
        if export_data.empty:
            st.error("No data to export")
            return
        
        # Convert to CSV
        csv_data = export_data.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label=f"Download AU {au_id} Portfolio CSV",
            data=csv_data,
            file_name=f"portfolio_au_{au_id}.csv",
            mime="text/csv",
            key=f"download_au_{au_id}"
        )
        
        st.success(f"Portfolio for AU {au_id} prepared for download ({len(export_data)} customers)")
        
    except Exception as e:
        st.error(f"Error saving portfolio: {str(e)}")

def save_all_portfolios(portfolios_created, customer_data):
    """Save all portfolios to a single CSV"""
    if not portfolios_created or customer_data is None:
        st.error("No data available to save")
        return
    
    try:
        # Get branch_data from session state or reload
        branch_data = st.session_state.get('branch_data')
        if branch_data is None:
            from data_loader import load_data
            _, _, branch_data = load_data()
        
        all_portfolio_data = []
        
        # Process each AU portfolio
        for au_id, au_data in portfolios_created.items():
            if not au_data.empty:
                export_data = prepare_portfolio_for_export(au_data, customer_data, branch_data)
                if not export_data.empty:
                    all_portfolio_data.append(export_data)
        
        if not all_portfolio_data:
            st.error("No data to export")
            return
        
        # Combine all portfolios
        combined_data = pd.concat(all_portfolio_data, ignore_index=True)
        
        # Convert to CSV
        csv_data = combined_data.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download All Portfolios CSV",
            data=csv_data,
            file_name="all_portfolios.csv",
            mime="text/csv",
            key="download_all_portfolios"
        )
        
        st.success(f"All portfolios prepared for download ({len(combined_data)} customers across {len(portfolios_created)} AUs)")
        
    except Exception as e:
        st.error(f"Error saving all portfolios: {str(e)}")

def save_all_smart_portfolios(smart_portfolios_created, customer_data):
    """Save all smart portfolios to a single CSV"""
    if not smart_portfolios_created or customer_data is None:
        st.error("No data available to save")
        return
    
    try:
        # Get branch_data from session state or reload
        branch_data = st.session_state.get('branch_data')
        if branch_data is None:
            from data_loader import load_data
            _, _, branch_data = load_data()
        
        all_portfolio_data = []
        
        # Process each AU portfolio
        for au_id, au_data in smart_portfolios_created.items():
            if not au_data.empty:
                export_data = prepare_portfolio_for_export(au_data, customer_data, branch_data)
                if not export_data.empty:
                    all_portfolio_data.append(export_data)
        
        if not all_portfolio_data:
            st.error("No data to export")
            return
        
        # Combine all portfolios
        combined_data = pd.concat(all_portfolio_data, ignore_index=True)
        
        # Convert to CSV
        csv_data = combined_data.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download All Smart Portfolios CSV",
            data=csv_data,
            file_name="all_smart_portfolios.csv",
            mime="text/csv",
            key="download_all_smart_portfolios"
        )
        
        st.success(f"All smart portfolios prepared for download ({len(combined_data)} customers across {len(smart_portfolios_created)} AUs)")
        
    except Exception as e:
        st.error(f"Error saving all smart portfolios: {str(e)}")

if __name__ == "__main__":
    main()
