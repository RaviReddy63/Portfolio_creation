import streamlit as st
import pandas as pd

def show_home_page():
    """Display home page content"""
    from data_loader import get_merged_data
    from home_tab import show_home_tab_content
    
    customer_data, banker_data, branch_data, _ = get_merged_data()
    show_home_tab_content(customer_data, banker_data, branch_data)

def show_my_requests_page():
    """Display My Requests page"""
    st.subheader("My Requests")
    st.info("This section is under development. Portfolio assignment requests will appear here.")

def show_portfolio_assignment_page():
    """Display Portfolio Assignment page"""
    from data_loader import get_merged_data
    from main import portfolio_assignment_page
    
    customer_data, banker_data, branch_data, _ = get_merged_data()
    portfolio_assignment_page(customer_data, banker_data, branch_data)

def show_portfolio_mapping_page():
    """Display Portfolio Mapping page"""
    from data_loader import get_merged_data
    from main import portfolio_mapping_page
    
    customer_data, banker_data, branch_data, _ = get_merged_data()
    portfolio_mapping_page(customer_data, banker_data, branch_data)

def show_ask_ai_page():
    """Display Ask AI page"""
    st.subheader("Ask AI")
    st.info("AI-powered insights coming soon!")

def show_q1_2026_move_page():
    """Show Q1 2026 Move page with all logic inline to avoid circular import"""
    from data_loader import load_hh_data
    from portfolio_creation_8 import enhanced_customer_au_assignment_with_two_inmarket_iterations
    from utils import clean_portfolio_data, validate_no_duplicates, prepare_portfolio_for_export_deduplicated
    from map_visualization import create_combined_map
    
    # Load HH customer data (already mapped columns)
    hh_customer_data, branch_data = load_hh_data()
    
    if hh_customer_data.empty:
        st.error("Failed to load HH_DF.csv. Please check the file and try again.")
        return
    
    # Store in session state
    st.session_state.hh_customer_data = hh_customer_data
    st.session_state.branch_data = branch_data
    
    # Main page content - ALL INLINE
    st.subheader("Q1 2026 Move - Smart Portfolio Mapping")
    
    # Create customer filters
    cust_state, cs_new_ns, min_rev, min_deposit, min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius = create_customer_filters_for_q1_2026(hh_customer_data)
    
    # Create Smart Portfolio Generation button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")
    with col2:
        generate_button = st.button("Generate Smart Portfolios", key="generate_q1_2026_portfolios", type="primary")
    
    # Process when button is clicked
    if generate_button:
        # Clear global data
        if 'q1_2026_portfolio_df' in st.session_state:
            del st.session_state.q1_2026_portfolio_df
        
        # Apply filters
        filtered_data = hh_customer_data.copy()
        if 'CG_ECN' not in filtered_data.columns:
            st.error("CG_ECN column missing from customer data!")
            return
        if cust_state is not None:
            filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
        if cs_new_ns is not None:
            if 'CS_NEW_NS' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['CS_NEW_NS'].isin(cs_new_ns)]
        filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
        filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
        
        if len(filtered_data) == 0:
            st.error("No customers found with the selected filters. Please adjust your criteria.")
            return
        
        # Clean filtered customers
        filtered_data = clean_portfolio_data(filtered_data)
        st.info(f"Processing {len(filtered_data):,} customers for Q1 2026 portfolio generation...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            progress_bar.progress(10)
            status_text.text("Initializing clustering algorithm...")
            
            # Calculate derived values
            min_size = min_portfolio_size
            max_size_proximity = max_portfolio_size
            max_size_inmarket = max_portfolio_size - 10
            max_size_centralized = max_portfolio_size - 10
            radius_inmarket_first = inmarket_radius
            radius_inmarket_second = inmarket_radius * 2
            radius_centralized = centralized_radius
            
            progress_bar.progress(30)
            status_text.text("Running advanced clustering analysis...")
            
            q1_2026_results = enhanced_customer_au_assignment_with_two_inmarket_iterations(
                filtered_data, 
                branch_data,
                min_size=min_size,
                max_inmarket_size=max_size_inmarket,
                max_centralized_size=max_size_centralized,
                max_proximity_size=max_size_proximity,
                inmarket_radius_first=radius_inmarket_first,
                inmarket_radius_second=radius_inmarket_second,
                centralized_radius=radius_centralized
            )
            
            progress_bar.progress(80)
            status_text.text("Processing and cleaning results...")
            
            q1_2026_results = clean_portfolio_data(q1_2026_results)
            
            is_clean, duplicate_ids = validate_no_duplicates(q1_2026_results, 'ECN')
            if not is_clean:
                st.warning(f"Removed {len(duplicate_ids)} duplicate customers in final results")
                q1_2026_results = q1_2026_results.drop_duplicates(subset=['ECN'], keep='first')
            
            st.session_state.q1_2026_portfolio_results = q1_2026_results
            st.session_state.q1_2026_filtered_customers_count = len(filtered_data)
            
            progress_bar.progress(100)
            status_text.text("Q1 2026 portfolios generated successfully!")
            
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Successfully generated Q1 2026 portfolios for {len(q1_2026_results):,} customers!")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error generating Q1 2026 portfolios: {str(e)}")
    
    # Display results
    if 'q1_2026_portfolio_results' in st.session_state and len(st.session_state.q1_2026_portfolio_results) > 0:
        results_df = st.session_state.q1_2026_portfolio_results
        results_df = clean_portfolio_data(results_df)
        
        is_clean, duplicate_ids = validate_no_duplicates(results_df, 'ECN')
        if not is_clean:
            st.warning(f"Cleaned {len(duplicate_ids)} duplicate customers from display")
            results_df = results_df.drop_duplicates(subset=['ECN'], keep='first')
            st.session_state.q1_2026_portfolio_results = results_df
        
        # Convert results to portfolio format
        q1_2026_portfolios_created = {}
        
        for au in results_df['ASSIGNED_AU'].unique():
            au_data = results_df[results_df['ASSIGNED_AU'] == au].copy()
            au_data = clean_portfolio_data(au_data)
            au_data['AU'] = au
            
            au_branch = branch_data[branch_data['AU'] == au]
            if not au_branch.empty:
                au_data['BRANCH_LAT_NUM'] = au_branch.iloc[0]['BRANCH_LAT_NUM']
                au_data['BRANCH_LON_NUM'] = au_branch.iloc[0]['BRANCH_LON_NUM']
            
            au_data = au_data.rename(columns={'ECN': 'CG_ECN', 'DISTANCE_TO_AU': 'Distance'})
            
            hh_data_subset = hh_customer_data[['CG_ECN', 'BANK_REVENUE', 'DEPOSIT_BAL']].copy()
            hh_data_subset = clean_portfolio_data(hh_data_subset)
            
            au_data = au_data.merge(hh_data_subset, on='CG_ECN', how='left', suffixes=('', '_orig'))
            au_data = clean_portfolio_data(au_data)
            
            au_data['BANK_REVENUE'] = au_data['BANK_REVENUE'].fillna(0)
            au_data['DEPOSIT_BAL'] = au_data['DEPOSIT_BAL'].fillna(0)
            
            q1_2026_portfolios_created[au] = au_data
        
        st.markdown("----")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Smart Portfolio Summary")
            au_tabs = st.tabs([f"AU {au_id}" for au_id in q1_2026_portfolios_created.keys()])
            
            for tab_idx, (au_id, tab) in enumerate(zip(q1_2026_portfolios_created.keys(), au_tabs)):
                with tab:
                    if au_id in q1_2026_portfolios_created:
                        au_data = q1_2026_portfolios_created[au_id]
                        display_summary_statistics(au_data)
        
        with col2:
            st.subheader("Save Portfolios")
            if st.button("Save All Q1 2026 Portfolios", key="save_all_q1_2026", type="primary"):
                try:
                    all_portfolio_data = []
                    
                    for au_id, au_data in q1_2026_portfolios_created.items():
                        if not au_data.empty:
                            export_data = prepare_portfolio_for_export_deduplicated(au_data, hh_customer_data, branch_data)
                            if not export_data.empty:
                                all_portfolio_data.append(export_data)
                    
                    if all_portfolio_data:
                        combined_data = pd.concat(all_portfolio_data, ignore_index=True)
                        combined_data = clean_portfolio_data(combined_data)
                        
                        is_clean, duplicate_ids = validate_no_duplicates(combined_data, 'CG_ECN')
                        if not is_clean:
                            st.warning(f"Removed {len(duplicate_ids)} duplicate customers from Q1 2026 export")
                            combined_data = combined_data.drop_duplicates(subset=['CG_ECN'], keep='first')
                        
                        csv_data = combined_data.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Q1 2026 Portfolios CSV",
                            data=csv_data,
                            file_name="q1_2026_portfolios.csv",
                            mime="text/csv",
                            key="download_q1_2026_portfolios"
                        )
                        
                        st.success(f"Q1 2026 portfolios prepared for download ({len(combined_data):,} customers across {len(q1_2026_portfolios_created):,} AUs)")
                    else:
                        st.error("No data to export")
                        
                except Exception as e:
                    st.error(f"Error saving Q1 2026 portfolios: {str(e)}")
        
        # Geographic Distribution
        st.markdown("----")
        st.subheader("Geographic Distribution")
        
        preview_portfolios = {}
        for au_id, au_data in q1_2026_portfolios_created.items():
            if not au_data.empty:
                preview_portfolios[f"AU_{au_id}_Q1_2026"] = au_data
        
        if preview_portfolios:
            combined_map = create_combined_map(preview_portfolios, branch_data)
            if combined_map:
                st.plotly_chart(combined_map, use_container_width=True)
        else:
            st.info("No customers selected for map display")
    else:
        st.info("Set your customer filters above, then click 'Generate Smart Portfolios' to create AI-optimized assignments for Q1 2026 move.")

def create_au_filters(branch_data):
    """Create AU selection filters"""
    st.subheader("AU Selection")
    
    # Get unique AUs
    au_options = sorted(branch_data['AU'].unique())
    
    # Create multiselect for AUs
    selected_aus = st.multiselect(
        "Select Administrative Units (AUs)",
        options=au_options,
        default=None,
        help="Select one or more AUs to create portfolios for"
    )
    
    return selected_aus

def create_customer_filters(customer_data):
    """Create customer filter UI for Portfolio Assignment"""
    
    st.subheader("Customer Selection Filters")
    
    # Get unique values for dropdowns
    state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
    role_options = sorted(list(customer_data['TYPE'].dropna().unique()))
    portfolio_options = sorted(list(customer_data['CG_PORTFOLIO_CD'].dropna().unique()))
    cs_new_ns_options = sorted(list(customer_data['CS_NEW_NS'].dropna().unique()))
    
    # Create filter columns
    col1, col2 = st.columns(2)
    
    with col1:
        # State filter
        cust_state = st.multiselect(
            "Customer State",
            options=state_options,
            default=None,
            key='assignment_cust_state'
        )
        
        # Role filter
        role = st.multiselect(
            "Customer Role/Type",
            options=role_options,
            default=None,
            key='assignment_role'
        )
        
        # Max distance filter
        max_dist = st.slider(
            "Maximum Distance (miles)",
            min_value=0,
            max_value=500,
            value=100,
            step=10,
            key='assignment_max_dist'
        )
    
    with col2:
        # Portfolio Code filter
        cust_portcd = st.multiselect(
            "Portfolio Code",
            options=portfolio_options,
            default=None,
            key='assignment_cust_portcd'
        )
        
        # CS_NEW_NS filter
        cs_new_ns = st.multiselect(
            "CS NEW NS",
            options=cs_new_ns_options,
            default=None,
            key='assignment_cs_new_ns'
        )
    
    # Revenue and deposit filters
    col3, col4 = st.columns(2)
    
    with col3:
        min_rev = st.slider(
            "Minimum Revenue ($)",
            min_value=0,
            max_value=int(customer_data['BANK_REVENUE'].max()),
            value=0,
            step=1000,
            key='assignment_min_revenue'
        )
    
    with col4:
        min_deposit = st.slider(
            "Minimum Deposit ($)",
            min_value=0,
            max_value=int(customer_data['DEPOSIT_BAL'].max()),
            value=0,
            step=1000,
            key='assignment_min_deposit'
        )
    
    return cust_state, role, cust_portcd, cs_new_ns, max_dist, min_rev, min_deposit

def create_customer_filters_for_mapping(customer_data):
    """Create customer filter UI for Portfolio Mapping with portfolio size and radius sliders"""
    
    st.subheader("Customer Selection Filters")
    
    # Create two-column layout for header and clear button
    col_header, col_clear = st.columns([9, 1])
    
    with col_header:
        st.write("Filter customers by various criteria:")
    
    with col_clear:
        if st.button("Clear", key="clear_filters_mapping"):
            # Clear all filter-related session state
            filter_keys = [
                'mapping_cust_state', 'mapping_role', 'mapping_cust_portcd', 'mapping_cs_new_ns',
                'mapping_min_revenue', 'mapping_min_deposit', 'mapping_max_dist',
                'min_portfolio_size', 'max_portfolio_size',
                'inmarket_radius', 'centralized_radius'
            ]
            for key in filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Get unique values for dropdowns
    state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
    role_options = sorted(list(customer_data['TYPE'].dropna().unique()))
    portfolio_options = sorted(list(customer_data['CG_PORTFOLIO_CD'].dropna().unique()))
    cs_new_ns_options = sorted(list(customer_data['CS_NEW_NS'].dropna().unique()))
    
    # Create filter columns
    col1, col2 = st.columns(2)
    
    with col1:
        # State filter
        cust_state = st.multiselect(
            "Customer State",
            options=state_options,
            default=None,
            key='mapping_cust_state'
        )
        
        # Role filter
        role = st.multiselect(
            "Customer Role/Type",
            options=role_options,
            default=None,
            key='mapping_role'
        )
        
        # Max distance filter
        max_dist = st.slider(
            "Maximum Distance (miles)",
            min_value=0,
            max_value=500,
            value=100,
            step=10,
            key='mapping_max_dist'
        )
        
        # Minimum revenue filter
        min_rev = st.slider(
            "Minimum Revenue ($)",
            min_value=0,
            max_value=int(customer_data['BANK_REVENUE'].max()),
            value=5000,
            step=1000,
            key='mapping_min_revenue'
        )
    
    with col2:
        # Portfolio Code filter
        cust_portcd = st.multiselect(
            "Portfolio Code",
            options=portfolio_options,
            default=None,
            key='mapping_cust_portcd'
        )
        
        # CS_NEW_NS filter
        cs_new_ns = st.multiselect(
            "CS NEW NS",
            options=cs_new_ns_options,
            default=None,
            key='mapping_cs_new_ns'
        )
        
        # Minimum deposit filter
        min_deposit = st.slider(
            "Minimum Deposit ($)",
            min_value=0,
            max_value=int(customer_data['DEPOSIT_BAL'].max()),
            value=100000,
            step=10000,
            key='mapping_min_deposit'
        )
    
    # Portfolio size configuration section
    st.subheader("Portfolio Size Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        min_portfolio_size = st.slider(
            "Minimum Portfolio Size",
            min_value=50,
            max_value=300,
            value=200,
            step=10,
            help="Minimum number of customers per portfolio",
            key='min_portfolio_size'
        )
    
    with col4:
        max_portfolio_size = st.slider(
            "Maximum Portfolio Size",
            min_value=100,
            max_value=500,
            value=250,
            step=10,
            help="Maximum number of customers per portfolio",
            key='max_portfolio_size'
        )
    
    # Radius configuration section
    st.subheader("Radius Configuration")
    col5, col6 = st.columns(2)
    
    with col5:
        inmarket_radius = st.slider(
            "In-Market Radius (miles)",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Radius for in-market portfolio assignments (first pass uses this, second pass uses 2x this value)",
            key='inmarket_radius'
        )
    
    with col6:
        centralized_radius = st.slider(
            "Centralized Radius (miles)",
            min_value=50,
            max_value=1000,
            value=100,
            step=10,
            help="Radius for centralized portfolio assignments",
            key='centralized_radius'
        )
    
    return cust_state, role, cust_portcd, cs_new_ns, max_dist, min_rev, min_deposit, min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius

def create_customer_filters_for_q1_2026(hh_customer_data):
    """Create customer filter UI for Q1 2026 Move - NO Role and NO Portfolio filters"""
    
    st.subheader("Customer Selection Filters")
    
    # Create two-column layout for header and clear button - FIXED: Added comma
    col_header2, col_clear2 = st.columns([9, 1])
    
    with col_header2:
        st.write("Filter customers by various criteria:")
    
    with col_clear2:
        if st.button("Clear", key="clear_filters_q1_2026"):
            # Clear all filter-related session state
            filter_keys = [
                'q1_2026_cust_state', 'q1_2026_cs_new_ns',
                'q1_2026_min_revenue', 'q1_2026_min_deposit', 
                'q1_2026_min_portfolio_size', 'q1_2026_max_portfolio_size',
                'q1_2026_inmarket_radius', 'q1_2026_centralized_radius'
            ]
            for key in filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Get unique values for dropdowns
    cust_state_options = list(hh_customer_data['BILLINGSTATE'].dropna().unique())
    cs_new_ns_options = sorted(list(hh_customer_data['CS_NEW_NS'].dropna().unique()))
    
    # Create filter columns
    col1, col2 = st.columns(2)
    
    with col1:
        # State filter
        cust_state = st.multiselect(
            "Customer State",
            options=cust_state_options,
            default=None,
            key='q1_2026_cust_state'
        )
        
        # Minimum revenue filter
        min_rev = st.slider(
            "Minimum Revenue ($)",
            min_value=0,
            max_value=int(hh_customer_data['BANK_REVENUE'].max()),
            value=0,
            step=1000,
            key='q1_2026_min_revenue'
        )
    
    with col2:
        # CS_NEW_NS filter
        cs_new_ns = st.multiselect(
            "CS NEW NS",
            options=cs_new_ns_options,
            default=None,
            key='q1_2026_cs_new_ns'
        )
        
        # Minimum deposit filter
        min_deposit = st.slider(
            "Minimum Deposit ($)",
            min_value=0,
            max_value=int(hh_customer_data['DEPOSIT_BAL'].max()),
            value=0,
            step=1000,
            key='q1_2026_min_deposit'
        )
    
    # Portfolio size configuration section
    st.subheader("Portfolio Size Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        min_portfolio_size = st.slider(
            "Minimum Portfolio Size",
            min_value=50,
            max_value=300,
            value=200,
            step=10,
            help="Minimum number of customers per portfolio",
            key='q1_2026_min_portfolio_size'
        )
    
    with col4:
        max_portfolio_size = st.slider(
            "Maximum Portfolio Size",
            min_value=100,
            max_value=500,
            value=250,
            step=10,
            help="Maximum number of customers per portfolio",
            key='q1_2026_max_portfolio_size'
        )
    
    # Radius configuration section
    st.subheader("Radius Configuration")
    col5, col6 = st.columns(2)
    
    with col5:
        inmarket_radius = st.slider(
            "In-Market Radius (miles)",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Radius for in-market portfolio assignments (first pass uses this, second pass uses 2x this value)",
            key='q1_2026_inmarket_radius'
        )
    
    with col6:
        centralized_radius = st.slider(
            "Centralized Radius (miles)",
            min_value=50,
            max_value=1000,
            value=100,
            step=10,
            help="Radius for centralized portfolio assignments",
            key='q1_2026_centralized_radius'
        )
    
    return cust_state, cs_new_ns, min_rev, min_deposit, min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius

def create_portfolio_button():
    """Create portfolio creation button"""
    return st.button("Create Portfolios", type="primary", key="create_portfolios_button")

def display_summary_statistics(portfolio_data):
    """Display summary statistics for a portfolio"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", f"{len(portfolio_data):,}")
        st.metric("Avg Revenue", f"${portfolio_data['BANK_REVENUE'].mean():,.0f}")
    
    with col2:
        st.metric("Total Revenue", f"${portfolio_data['BANK_REVENUE'].sum():,.0f}")
        st.metric("Avg Deposit", f"${portfolio_data['DEPOSIT_BAL'].mean():,.0f}")
    
    with col3:
        st.metric("Total Deposits", f"${portfolio_data['DEPOSIT_BAL'].sum():,.0f}")
        if 'CG_GROSS_SALES' in portfolio_data.columns:
            st.metric("Total Gross Sales", f"${portfolio_data['CG_GROSS_SALES'].sum():,.0f}")

def create_portfolio_editor(portfolio_df, au_id, is_multi_au=True):
    """Create an editable portfolio dataframe"""
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
    
    key_suffix = f"_{au_id}" if is_multi_au else ""
    
    return st.data_editor(
        portfolio_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=350,
        key=f"portfolio_editor{key_suffix}"
    )

def create_apply_changes_button(au_id, is_single_au=False):
    """Create Apply Changes button"""
    key_suffix = "" if is_single_au else f"_{au_id}"
    return st.button(f"Apply Changes", key=f"apply_changes{key_suffix}")
