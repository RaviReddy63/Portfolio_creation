import streamlit as st
import pandas as pd

# Import custom modules
from data_loader import get_merged_data
from portfolio_creation import process_portfolio_creation, apply_portfolio_changes
from map_visualization import create_combined_map, create_smart_portfolio_map
from portfolio_creation_8 import enhanced_customer_au_assignment_with_two_inmarket_iterations
from utils import (
    clean_portfolio_data, remove_customer_duplicates, validate_no_duplicates,
    prepare_portfolio_for_export_deduplicated
)

def setup_page_config():
    """Configure the Streamlit page"""
    st.set_page_config("Portfolio Creation tool", layout="wide")
    
    # Hide Streamlit's default header only
    st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Adjust main content area to account for hidden header */
        .main .block-container {
            padding-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def add_logo():
    """Add custom header with logo and text"""
    import base64
    
    logo_html = ""
    try:
        with open("logo.svg", "rb") as f:
            svg_data = f.read()
            svg_base64 = base64.b64encode(svg_data).decode()
            logo_html = f'<img src="data:image/svg+xml;base64,{svg_base64}" style="height: 40px; width: 250px; margin-right: 15px; object-fit: contain;">'
    except:
        try:
            with open("logo.png", "rb") as f:
                png_data = f.read()
                png_base64 = base64.b64encode(png_data).decode()
                logo_html = f'<img src="data:image/png;base64,{png_base64}" style="height: 40px; width: 250px; margin-right: 15px; object-fit: contain;">'
        except:
            pass
    
    # Create custom header with logo and text that spans full width
    st.markdown(f"""
    <div style="
        background-color: rgb(215, 30, 40);
        color: white;
        padding: 8px 20px;
        margin: -1rem -1rem 2rem -1rem;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        display: flex;
        align-items: center;
        border-bottom: 3px solid rgb(255, 205, 65);
        box-sizing: border-box;
    ">
        {logo_html}
        <span style="
            font-size: 1.2rem;
            font-weight: bold;
            border-left: 2px solid white;
            padding-left: 15px;
            margin-left: 10px;
        ">Banker Placement Tool</span>
    </div>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'all_portfolios': {},
        'portfolio_controls': {},
        'recommend_reassignment': {},
        'should_create_portfolios': False,
        'should_generate_smart_portfolios': False,
        'smart_portfolio_controls': {}
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def create_header():
    """Create the page header with tab navigation and content"""
    from ui_components import (
        show_home_page, show_my_requests_page, show_portfolio_assignment_page,
        show_portfolio_mapping_page, show_ask_ai_page, show_q1_2026_move_page
    )
    
    # Navigation tabs with all 6 pages
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Home", "My Requests", "Portfolio Assignment", "Portfolio Mapping", "Ask AI", "Q1_2026_Move"])
    
    with tab1:
        # Home content
        show_home_page()
        
    with tab2:
        # My Requests content
        show_my_requests_page()
        
    with tab3:
        # Portfolio Assignment content
        show_portfolio_assignment_page()
        
    with tab4:
        # Portfolio Mapping content
        show_portfolio_mapping_page()
        
    with tab5:
        # Ask AI chat interface
        show_ask_ai_page()
    
    with tab6:
        # Q1 2026 Move content
        show_q1_2026_move_page()
    
    return None

def portfolio_assignment_page(customer_data, banker_data, branch_data):
    """Portfolio Assignment page logic"""
    from ui_components import (
        create_au_filters, create_customer_filters, create_portfolio_button,
        display_summary_statistics, create_portfolio_editor, create_apply_changes_button
    )
    
    # Store customer_data in session state for save functions
    st.session_state.customer_data = customer_data
    st.session_state.branch_data = branch_data
    
    # Create AU filters
    selected_aus = create_au_filters(branch_data)
    
    # Create customer filters  
    cust_state, role, cust_portcd, cs_new_ns, max_dist, min_rev, min_deposit = create_customer_filters(customer_data)
    
    # Create portfolio button
    button_clicked = create_portfolio_button()
    
    # ONLY process when button is clicked
    if button_clicked:
        if not selected_aus:
            st.error("Please select at least one AU")
        else:
            # Show loading message
            with st.spinner("Creating portfolios..."):
                portfolios_created, portfolio_summaries = process_portfolio_creation(
                    selected_aus, customer_data, banker_data, branch_data,
                    role, cust_state, cust_portcd, cs_new_ns, max_dist, min_rev, min_deposit
                )
                
                if portfolios_created:
                    st.session_state.portfolios_created = portfolios_created
                    st.session_state.portfolio_summaries = portfolio_summaries
                    st.success("Portfolios created successfully!")
                else:
                    st.warning("No customers found for the selected AUs with current filters.")
    
    # Display results ONLY if they exist in session state
    if 'portfolios_created' in st.session_state and st.session_state.portfolios_created:
        display_portfolio_results(branch_data)
    else:
        # Show helpful message when no portfolios exist yet
        if selected_aus:
            st.info(f"Selected {len(selected_aus)} AU(s). Click 'Create Portfolios' to generate customer assignments.")
        else:
            st.info("Select AUs and set filters, then click 'Create Portfolios' to begin.")

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
    # Always use tabs regardless of number of AUs
    au_tabs = st.tabs([f"AU {au_id}" for au_id in portfolios_created.keys()])
    
    for tab_idx, (au_id, tab) in enumerate(zip(portfolios_created.keys(), au_tabs)):
        with tab:
            display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, True)

def display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, is_multi_au):
    """Display table for a single AU"""
    from ui_components import create_portfolio_editor, create_apply_changes_button, display_summary_statistics
    
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
    """Portfolio Mapping page logic with advanced clustering - MODIFIED to capture portfolio size AND radius parameters"""
    from ui_components import create_customer_filters_for_mapping
    
    st.subheader("Smart Portfolio Mapping")
    
    # Create customer filters - NOW RETURNS 11 VALUES (added min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius)
    cust_state, role, cust_portcd, cs_new_ns, max_dist, min_rev, min_deposit, min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius = create_customer_filters_for_mapping(customer_data)
    
    # Create Smart Portfolio Generation button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")  # Empty space
    with col2:
        generate_button = st.button("Generate Smart Portfolios", key="generate_smart_portfolios", type="primary")
    
    # ONLY process when button is clicked - PASS NEW PARAMETERS INCLUDING RADIUS
    if generate_button:
        # Show loading message and process
        generate_smart_portfolios(customer_data, branch_data, cust_state, role, cust_portcd, cs_new_ns, min_rev, min_deposit, 
                                 min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius)
    
    # Display results ONLY if they exist in session state
    if 'smart_portfolio_results' in st.session_state and len(st.session_state.smart_portfolio_results) > 0:
        display_smart_portfolio_results(customer_data, branch_data)
    else:
        # Show helpful message when no smart portfolios exist yet
        st.info("Set your customer filters above, then click 'Generate Smart Portfolios' to create AI-optimized assignments.")

def apply_customer_filters_for_mapping(customer_data, cust_state, role, cust_portcd, cs_new_ns, min_rev, min_deposit):
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
    
    # Apply CS_NEW_NS filter
    if cs_new_ns is not None:
        if 'CS_NEW_NS' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['CS_NEW_NS'].isin(cs_new_ns)]
    
    # Apply other filters
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    return filtered_data

def generate_smart_portfolios(customer_data, branch_data, cust_state, role, cust_portcd, cs_new_ns, min_rev, min_deposit,
                              min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius):
    """Generate smart portfolios using advanced clustering with deduplication - MODIFIED to accept and apply portfolio size AND radius parameters"""
    
    # Clear global data when generating new portfolios
    if 'global_portfolio_df' in st.session_state:
        del st.session_state.global_portfolio_df
    
    # Apply customer filters
    filtered_customers = apply_customer_filters_for_mapping(
        customer_data, cust_state, role, cust_portcd, cs_new_ns, min_rev, min_deposit
    )
    
    if len(filtered_customers) == 0:
        st.error("No customers found with the selected filters. Please adjust your criteria.")
        return
    
    # Clean filtered customers to remove duplicates
    filtered_customers = clean_portfolio_data(filtered_customers)
    
    st.info(f"Processing {len(filtered_customers):,} customers for smart portfolio generation...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        progress_bar.progress(10)
        status_text.text("Initializing clustering algorithm...")
        
        # CALCULATE DERIVED VALUES BASED ON USER INPUT
        min_size = min_portfolio_size  # Use slider value directly
        max_size_proximity = max_portfolio_size  # Maximum after proximity = slider value
        max_size_inmarket = max_portfolio_size - 10  # Maximum in-market = slider value - 10
        max_size_centralized = max_portfolio_size - 10  # Maximum centralized = slider value - 10
        
        # RADIUS VALUES from sliders
        radius_inmarket_first = inmarket_radius  # First iteration radius (e.g., 20 miles)
        radius_inmarket_second = inmarket_radius * 2  # Second iteration radius (2x first, e.g., 40 miles)
        radius_centralized = centralized_radius  # Centralized radius (e.g., 100 miles)
        
        # Run the enhanced clustering algorithm with deduplication - PASS CALCULATED VALUES INCLUDING RADIUS
        progress_bar.progress(30)
        status_text.text("Running advanced clustering analysis...")
        
        # Use your existing clustering with input/output cleaning and NEW PARAMETERS
        smart_portfolio_results = enhanced_customer_au_assignment_with_two_inmarket_iterations(
            filtered_customers, 
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
        
        # Clean the output from your clustering algorithm
        smart_portfolio_results = clean_portfolio_data(smart_portfolio_results)
        
        # Validate results are clean
        is_clean, duplicate_ids = validate_no_duplicates(smart_portfolio_results, 'ECN')
        if not is_clean:
            st.warning(f"Removed {len(duplicate_ids)} duplicate customers in final results")
            smart_portfolio_results = smart_portfolio_results.drop_duplicates(subset=['ECN'], keep='first')
        
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
        
        st.success(f"Successfully generated smart portfolios for {len(smart_portfolio_results):,} customers!")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error generating smart portfolios: {str(e)}")

def display_smart_portfolio_results(customer_data, branch_data):
    """Display smart portfolio results like Portfolio Assignment with deduplication"""
    
    # Store customer_data in session state for save functions
    st.session_state.customer_data = customer_data
    
    if 'smart_portfolio_results' not in st.session_state or len(st.session_state.smart_portfolio_results) == 0:
        st.info("Click 'Generate Smart Portfolios' to create optimized customer assignments.")
        return
    
    # Always use the current results from session state and clean them
    results_df = st.session_state.smart_portfolio_results
    results_df = clean_portfolio_data(results_df)  # Clean on display
    
    # Validate and show cleaning stats
    is_clean, duplicate_ids = validate_no_duplicates(results_df, 'ECN')
    if not is_clean:
        st.warning(f"Cleaned {len(duplicate_ids)} duplicate customers from display")
        results_df = results_df.drop_duplicates(subset=['ECN'], keep='first')
        st.session_state.smart_portfolio_results = results_df  # Update session state
    
    # Convert smart portfolio results to Portfolio Assignment format - regenerate every time
    smart_portfolios_created = {}
    
    # Group by AU
    for au in results_df['ASSIGNED_AU'].unique():
        au_data = results_df[results_df['ASSIGNED_AU'] == au].copy()
        
        # Clean AU data
        au_data = clean_portfolio_data(au_data)
        
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
        customer_data_subset = clean_portfolio_data(customer_data_subset)  # Clean before merge
        
        au_data = au_data.merge(customer_data_subset, on='CG_ECN', how='left', suffixes=('', '_orig'))
        
        # Clean after merge
        au_data = clean_portfolio_data(au_data)
        
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
    # Always use tabs regardless of number of AUs
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
            from ui_components import display_summary_statistics
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
            st.metric("In-Market Portfolios", f"{inmarket_aus:,}")
        
        with col_d:
            st.metric("Centralized Portfolios", f"{centralized_aus:,}")
        
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
            
            # Clean original data
            original_data = clean_portfolio_data(original_data)
            
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
                
                # Clean and validate final results
                results_df = clean_portfolio_data(results_df)
                
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
                # Clean and sort by distance (closest first) and take the requested count
                unmanaged_customers = clean_portfolio_data(unmanaged_customers)
                unmanaged_sorted = unmanaged_customers.sort_values('Distance').head(select_count)
                selected_customers.append(unmanaged_sorted)
                
        else:
            # Handle regular portfolios
            portfolio_customers = original_data[original_data['PORT_CODE'] == portfolio_id].copy()
            
            if not portfolio_customers.empty:
                # Clean and sort by distance (closest first) and take the requested count
                portfolio_customers = clean_portfolio_data(portfolio_customers)
                portfolio_sorted = portfolio_customers.sort_values('Distance').head(select_count)
                selected_customers.append(portfolio_sorted)
    
    # Combine all selected customers for this AU
    if selected_customers:
        final_customers = pd.concat(selected_customers, ignore_index=True)
        final_customers = clean_portfolio_data(final_customers)
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

def save_all_smart_portfolios(smart_portfolios_created, customer_data):
    """Save all smart portfolios to a single CSV"""
    if not smart_portfolios_created or customer_data is None:
        st.error("No data available to save")
        return
    
    try:
        # Get branch_data from session state
        branch_data = st.session_state.get('branch_data')
        if branch_data is None:
            customer_data, banker_data, branch_data, _ = get_merged_data()
        
        all_portfolio_data = []
        
        # Process each AU portfolio
        for au_id, au_data in smart_portfolios_created.items():
            if not au_data.empty:
                export_data = prepare_portfolio_for_export_deduplicated(au_data, customer_data, branch_data)
                if not export_data.empty:
                    all_portfolio_data.append(export_data)
        
        if not all_portfolio_data:
            st.error("No data to export")
            return
        
        # Combine all portfolios
        combined_data = pd.concat(all_portfolio_data, ignore_index=True)
        
        # Final cleaning and validation
        combined_data = clean_portfolio_data(combined_data)
        is_clean, duplicate_ids = validate_no_duplicates(combined_data, 'CG_ECN')
        if not is_clean:
            st.warning(f"Removed {len(duplicate_ids)} duplicate customers from smart portfolio export")
            combined_data = combined_data.drop_duplicates(subset=['CG_ECN'], keep='first')
        
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
        
        st.success(f"All smart portfolios prepared for download ({len(combined_data):,} customers across {len(smart_portfolios_created):,} AUs)")
        
    except Exception as e:
        st.error(f"Error saving all smart portfolios: {str(e)}")

def apply_global_changes_final(edited_df, customer_data, branch_data):
    """Apply changes using the edited dataframe with deduplication"""
    
    with st.spinner("Applying changes..."):
        try:
            # Get CURRENT filters from UI widgets (not cached session state)
            current_filters = get_current_mapping_filters()
            
            # Apply customer filters to get ALL customers that match current UI selection
            all_filtered_customers = apply_customer_filters_for_mapping(
                customer_data, 
                current_filters['cust_state'], 
                current_filters['role'], 
                current_filters['cust_portcd'],
                current_filters['cs_new_ns'],
                current_filters['min_rev'], 
                current_filters['min_deposit']
            )
            
            # Clean filtered customers
            all_filtered_customers = clean_portfolio_data(all_filtered_customers)
            
            # Get current AU assignments to calculate distances
            current_results = st.session_state.get('smart_portfolio_results', pd.DataFrame())
            identified_aus = current_results['ASSIGNED_AU'].unique().tolist() if not current_results.empty else []
            
            # Start with ALL filtered customers, then apply Global Control selections
            final_customers = []
            
            # Track which customers have been selected through portfolio controls
            selected_customer_ecns = set()
            
            # Apply portfolio selections from edited_df for specific portfolios
            for _, row in edited_df.iterrows():
                if not row['Include'] or row['Select'] <= 0:
                    continue
                    
                portfolio_id = row['Portfolio ID']
                select_count = int(row['Select'])
                
                # Get customers for this portfolio from the FULL filtered customer set
                if portfolio_id == 'N/A':
                    portfolio_customers = all_filtered_customers[
                        all_filtered_customers['CG_PORTFOLIO_CD'].isna()
                    ]
                else:
                    portfolio_customers = all_filtered_customers[
                        all_filtered_customers['CG_PORTFOLIO_CD'] == portfolio_id
                    ]
                
                if len(portfolio_customers) > 0:
                    # Clean portfolio customers before selection
                    portfolio_customers = clean_portfolio_data(portfolio_customers)
                    
                    # Select closest customers to nearest AUs
                    selected = select_closest_customers_to_aus(
                        portfolio_customers, select_count, identified_aus, branch_data
                    )
                    
                    if not selected.empty:
                        final_customers.append(selected)
                        
                        # Track selected customers
                        selected_customer_ecns.update(selected['CG_ECN'].tolist())
            
            # Add any remaining Unassigned/Unmanaged customers that weren't explicitly controlled
            unassigned_unmanaged = all_filtered_customers[
                (all_filtered_customers['TYPE'].str.lower().str.strip().isin(['unassigned', 'unmanaged'])) &
                (~all_filtered_customers['CG_ECN'].isin(selected_customer_ecns))
            ]
            
            if not unassigned_unmanaged.empty:
                unassigned_unmanaged = clean_portfolio_data(unassigned_unmanaged)
                final_customers.append(unassigned_unmanaged)
            
            if final_customers:
                combined_customers = pd.concat(final_customers, ignore_index=True)
                
                # Clean combined customers
                combined_customers = clean_portfolio_data(combined_customers)
                
                # Get portfolio size parameters from session state
                min_portfolio_size = st.session_state.get('min_portfolio_size', 200)
                max_portfolio_size = st.session_state.get('max_portfolio_size', 250)
                
                # Get radius parameters from session state
                inmarket_radius = st.session_state.get('inmarket_radius', 20)
                centralized_radius = st.session_state.get('centralized_radius', 100)
                
                # Calculate derived values
                min_size = min_portfolio_size
                max_size_proximity = max_portfolio_size
                max_size_inmarket = max_portfolio_size - 10
                max_size_centralized = max_portfolio_size - 10
                
                # Calculate radius values
                radius_inmarket_first = inmarket_radius
                radius_inmarket_second = inmarket_radius * 2
                radius_centralized = centralized_radius
                
                # Regenerate portfolios with cleaned data using your existing clustering
                smart_results = enhanced_customer_au_assignment_with_two_inmarket_iterations(
                    combined_customers, 
                    branch_data,
                    min_size=min_size,
                    max_inmarket_size=max_size_inmarket,
                    max_centralized_size=max_size_centralized,
                    max_proximity_size=max_size_proximity,
                    inmarket_radius_first=radius_inmarket_first,
                    inmarket_radius_second=radius_inmarket_second,
                    centralized_radius=radius_centralized
                )
                
                # Clean the output from your clustering algorithm
                smart_results = clean_portfolio_data(smart_results)
                
                st.session_state.smart_portfolio_results = smart_results
                
                # Clear cached data to force updates
                if 'global_portfolio_df' in st.session_state:
                    del st.session_state.global_portfolio_df
                
                # Clear Smart Portfolio Summary controls to force refresh
                if 'smart_portfolio_controls' in st.session_state:
                    st.session_state.smart_portfolio_controls = {}
                
                st.success(f"Applied changes with {len(smart_results):,} customers (from {len(all_filtered_customers):,} total filtered)!")
            else:
                st.error("No customers selected")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def get_current_mapping_filters():
    """Get current filter values from UI widgets"""
    return {
        'cust_state': st.session_state.get('mapping_cust_state'),
        'role': st.session_state.get('mapping_role'),
        'cust_portcd': st.session_state.get('mapping_cust_portcd'),
        'cs_new_ns': st.session_state.get('mapping_cs_new_ns'),
        'min_rev': st.session_state.get('mapping_min_revenue', 5000),
        'min_deposit': st.session_state.get('mapping_min_deposit', 100000)
    }

def select_closest_customers_to_aus(portfolio_customers, select_count, identified_aus, branch_data):
    """Select customers closest to any of the identified AUs"""
    from utils import haversine_distance
    
    if len(portfolio_customers) <= select_count:
        return portfolio_customers
    
    if not identified_aus:
        # If no AUs identified, return first N customers
        return portfolio_customers.head(select_count)
    
    # Calculate minimum distance to any AU for each customer
    customers_with_min_distance = []
    
    for idx, customer in portfolio_customers.iterrows():
        min_distance = float('inf')
        
        # Find distance to closest AU
        for au_id in identified_aus:
            au_row = branch_data[branch_data['AU'] == au_id]
            if au_row.empty:
                continue
                
            au_lat = au_row.iloc[0]['BRANCH_LAT_NUM']
            au_lon = au_row.iloc[0]['BRANCH_LON_NUM']
            
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'], au_lat, au_lon
            )
            
            if distance < min_distance:
                min_distance = distance
        
        customers_with_min_distance.append({
            'index': idx,
            'min_distance': min_distance
        })
    
    # Sort by minimum distance and select closest N customers
    customers_with_min_distance.sort(key=lambda x: x['min_distance'])
    selected_indices = [item['index'] for item in customers_with_min_distance[:select_count]]
    
    return portfolio_customers.loc[selected_indices]

def prepare_portfolio_for_export(au_data, customer_data, branch_data):
    """Prepare portfolio data in the required export format with deduplication"""
    return prepare_portfolio_for_export_deduplicated(au_data, customer_data, branch_data)

def save_single_au_portfolio(au_id, portfolios_created, customer_data):
    """Save a single AU portfolio to CSV"""
    if au_id not in portfolios_created or customer_data is None:
        st.error("No data available to save")
        return
    
    try:
        # Get branch_data from session state
        branch_data = st.session_state.get('branch_data')
        if branch_data is None:
            customer_data, banker_data, branch_data, _ = get_merged_data()
        
        # Prepare data for export
        au_data = portfolios_created[au_id]
        export_data = prepare_portfolio_for_export_deduplicated(au_data, customer_data, branch_data)
        
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
        
        st.success(f"Portfolio for AU {au_id} prepared for download ({len(export_data):,} customers)")
        
    except Exception as e:
        st.error(f"Error saving portfolio: {str(e)}")

def save_all_portfolios(portfolios_created, customer_data):
    """Save all portfolios to a single CSV"""
    if not portfolios_created or customer_data is None:
        st.error("No data available to save")
        return
    
    try:
        # Get branch_data from session state
        branch_data = st.session_state.get('branch_data')
        if branch_data is None:
            customer_data, banker_data, branch_data, _ = get_merged_data()
        
        all_portfolio_data = []
        
        # Process each AU portfolio
        for au_id, au_data in portfolios_created.items():
            if not au_data.empty:
                export_data = prepare_portfolio_for_export_deduplicated(au_data, customer_data, branch_data)
                if not export_data.empty:
                    all_portfolio_data.append(export_data)
        
        if not all_portfolio_data:
            st.error("No data to export")
            return
        
        # Combine all portfolios
        combined_data = pd.concat(all_portfolio_data, ignore_index=True)
        
        # Final cleaning and validation
        combined_data = clean_portfolio_data(combined_data)
        is_clean, duplicate_ids = validate_no_duplicates(combined_data, 'CG_ECN')
        if not is_clean:
            st.warning(f"Removed {len(duplicate_ids)} duplicate customers from combined export")
            combined_data = combined_data.drop_duplicates(subset=['CG_ECN'], keep='first')
        
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
        
        st.success(f"All portfolios prepared for download ({len(combined_data):,} customers across {len(portfolios_created):,} AUs)")
        
    except Exception as e:
        st.error(f"Error saving all portfolios: {str(e)}")

# ============================================================================
# Q1 2026 MOVE TAB FUNCTIONS
# ============================================================================

def q1_2026_move_page(hh_customer_data, branch_data):
    """Q1 2026 Move page logic - Similar to Portfolio Mapping but with HH_DF.csv data"""
    from ui_components import create_customer_filters_for_q1_2026
    
    st.subheader("Q1 2026 Move - Smart Portfolio Mapping")
    
    # Create customer filters - NO ROLE OR PORTFOLIO CODE filters
    cust_state, cs_new_ns, min_rev, min_deposit, min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius = create_customer_filters_for_q1_2026(hh_customer_data)
    
    # Create Smart Portfolio Generation button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")  # Empty space
    with col2:
        generate_button = st.button("Generate Smart Portfolios", key="generate_q1_2026_portfolios", type="primary")
    
    # ONLY process when button is clicked
    if generate_button:
        # Show loading message and process
        generate_q1_2026_portfolios(hh_customer_data, branch_data, cust_state, cs_new_ns, min_rev, min_deposit, 
                                    min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius)
    
    # Display results ONLY if they exist in session state
    if 'q1_2026_portfolio_results' in st.session_state and len(st.session_state.q1_2026_portfolio_results) > 0:
        display_q1_2026_results(hh_customer_data, branch_data)
    else:
        # Show helpful message when no smart portfolios exist yet
        st.info("Set your customer filters above, then click 'Generate Smart Portfolios' to create AI-optimized assignments for Q1 2026 move.")

def apply_customer_filters_for_q1_2026(hh_customer_data, cust_state, cs_new_ns, min_rev, min_deposit):
    """Apply customer filters for Q1 2026 Move - NO ROLE OR PORTFOLIO CODE"""
    filtered_data = hh_customer_data.copy()
    
    # Ensure CG_ECN is preserved (already mapped from HH_ECN)
    if 'CG_ECN' not in filtered_data.columns:
        st.error("CG_ECN column missing from customer data!")
        return pd.DataFrame()
    
    # Apply Customer State filter
    if cust_state is not None:
        filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
    
    # Apply CS_NEW_NS filter (mapped from NEW_SEGMENT)
    if cs_new_ns is not None:
        if 'CS_NEW_NS' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['CS_NEW_NS'].isin(cs_new_ns)]
    
    # Apply other filters
    filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
    filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
    
    return filtered_data

def generate_q1_2026_portfolios(hh_customer_data, branch_data, cust_state, cs_new_ns, min_rev, min_deposit,
                                min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius):
    """Generate Q1 2026 portfolios using advanced clustering - Similar to generate_smart_portfolios"""
    
    # Clear global data when generating new portfolios
    if 'q1_2026_portfolio_df' in st.session_state:
        del st.session_state.q1_2026_portfolio_df
    
    # Apply customer filters
    filtered_customers = apply_customer_filters_for_q1_2026(
        hh_customer_data, cust_state, cs_new_ns, min_rev, min_deposit
    )
    
    if len(filtered_customers) == 0:
        st.error("No customers found with the selected filters. Please adjust your criteria.")
        return
    
    # Clean filtered customers to remove duplicates
    filtered_customers = clean_portfolio_data(filtered_customers)
    
    st.info(f"Processing {len(filtered_customers):,} customers for Q1 2026 portfolio generation...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        progress_bar.progress(10)
        status_text.text("Initializing clustering algorithm...")
        
        # CALCULATE DERIVED VALUES BASED ON USER INPUT
        min_size = min_portfolio_size
        max_size_proximity = max_portfolio_size
        max_size_inmarket = max_portfolio_size - 10
        max_size_centralized = max_portfolio_size - 10
        
        # RADIUS VALUES from sliders
        radius_inmarket_first = inmarket_radius
        radius_inmarket_second = inmarket_radius * 2
        radius_centralized = centralized_radius
        
        # Run the enhanced clustering algorithm
        progress_bar.progress(30)
        status_text.text("Running advanced clustering analysis...")
        
        q1_2026_results = enhanced_customer_au_assignment_with_two_inmarket_iterations(
            filtered_customers, 
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
        
        # Clean the output
        q1_2026_results = clean_portfolio_data(q1_2026_results)
        
        # Validate results are clean
        is_clean, duplicate_ids = validate_no_duplicates(q1_2026_results, 'ECN')
        if not is_clean:
            st.warning(f"Removed {len(duplicate_ids)} duplicate customers in final results")
            q1_2026_results = q1_2026_results.drop_duplicates(subset=['ECN'], keep='first')
        
        # Store results in session state
        st.session_state.q1_2026_portfolio_results = q1_2026_results
        st.session_state.q1_2026_filtered_customers_count = len(filtered_customers)
        
        progress_bar.progress(100)
        status_text.text("Q1 2026 portfolios generated successfully!")
        
        # Clear progress indicators after a brief delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Successfully generated Q1 2026 portfolios for {len(q1_2026_results):,} customers!")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error generating Q1 2026 portfolios: {str(e)}")

def display_q1_2026_results(hh_customer_data, branch_data):
    """Display Q1 2026 portfolio results - SIMPLIFIED VERSION (No Global Control, No Tables)"""
    
    # Store customer_data in session state for save functions
    st.session_state.hh_customer_data = hh_customer_data
    
    if 'q1_2026_portfolio_results' not in st.session_state or len(st.session_state.q1_2026_portfolio_results) == 0:
        st.info("Click 'Generate Smart Portfolios' to create optimized customer assignments for Q1 2026.")
        return
    
    # Always use the current results from session state and clean them
    results_df = st.session_state.q1_2026_portfolio_results
    results_df = clean_portfolio_data(results_df)
    
    # Validate and show cleaning stats
    is_clean, duplicate_ids = validate_no_duplicates(results_df, 'ECN')
    if not is_clean:
        st.warning(f"Cleaned {len(duplicate_ids)} duplicate customers from display")
        results_df = results_df.drop_duplicates(subset=['ECN'], keep='first')
        st.session_state.q1_2026_portfolio_results = results_df
    
    # Convert results to portfolio format
    q1_2026_portfolios_created = {}
    
    # Group by AU
    for au in results_df['ASSIGNED_AU'].unique():
        au_data = results_df[results_df['ASSIGNED_AU'] == au].copy()
        
        # Clean AU data
        au_data = clean_portfolio_data(au_data)
        
        # Add required columns
        au_data['AU'] = au
        
        # Get AU coordinates from branch_data
        au_branch = branch_data[branch_data['AU'] == au]
        if not au_branch.empty:
            au_data['BRANCH_LAT_NUM'] = au_branch.iloc[0]['BRANCH_LAT_NUM']
            au_data['BRANCH_LON_NUM'] = au_branch.iloc[0]['BRANCH_LON_NUM']
        
        # Rename columns to match expected format
        au_data = au_data.rename(columns={
            'ECN': 'CG_ECN',
            'DISTANCE_TO_AU': 'Distance'
        })
        
        # Merge with original HH data to get financial information
        hh_data_subset = hh_customer_data[['CG_ECN', 'BANK_REVENUE', 'DEPOSIT_BAL']].copy()
        hh_data_subset = clean_portfolio_data(hh_data_subset)
        
        au_data = au_data.merge(hh_data_subset, on='CG_ECN', how='left', suffixes=('', '_orig'))
        
        # Clean after merge
        au_data = clean_portfolio_data(au_data)
        
        # Use original financial data
        au_data['BANK_REVENUE'] = au_data['BANK_REVENUE'].fillna(0)
        au_data['DEPOSIT_BAL'] = au_data['DEPOSIT_BAL'].fillna(0)
        
        q1_2026_portfolios_created[au] = au_data
    
    st.markdown("----")
    
    # Display ONLY AU Summary Statistics - NO TABLES, NO GLOBAL CONTROL
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Smart Portfolio Summary")
        display_q1_2026_portfolio_summary(q1_2026_portfolios_created, branch_data)
    
    with col2:
        st.subheader("Save Portfolios")
        # Add save button
        if st.button("Save All Q1 2026 Portfolios", key="save_all_q1_2026", type="primary"):
            save_all_q1_2026_portfolios(q1_2026_portfolios_created, hh_customer_data, branch_data)
    
    # Geographic Distribution below with full width
    st.markdown("----")
    st.subheader("Geographic Distribution")
    display_q1_2026_geographic_map(q1_2026_portfolios_created, branch_data)

def display_q1_2026_portfolio_summary(q1_2026_portfolios_created, branch_data):
    """Display Q1 2026 portfolio summary - ONLY AU SUMMARY STATISTICS (No tables)"""
    # Always use tabs regardless of number of AUs
    au_tabs = st.tabs([f"AU {au_id}" for au_id in q1_2026_portfolios_created.keys()])
    
    for tab_idx, (au_id, tab) in enumerate(zip(q1_2026_portfolios_created.keys(), au_tabs)):
        with tab:
            display_single_q1_2026_au_summary(au_id, q1_2026_portfolios_created)

def display_single_q1_2026_au_summary(au_id, q1_2026_portfolios_created):
    """Display summary statistics for a single Q1 2026 AU - ONLY STATISTICS"""
    if au_id in q1_2026_portfolios_created:
        au_data = q1_2026_portfolios_created[au_id]
        
        # Display summary statistics ONLY (same as Portfolio Assignment)
        from ui_components import display_summary_statistics
        display_summary_statistics(au_data)

def display_q1_2026_geographic_map(q1_2026_portfolios_created, branch_data):
    """Display the geographic distribution map for Q1 2026 portfolios"""
    # Convert to format expected by create_combined_map
    preview_portfolios = {}
    
    for au_id, au_data in q1_2026_portfolios_created.items():
        if not au_data.empty:
            preview_portfolios[f"AU_{au_id}_Q1_2026"] = au_data
    
    # Display the map
    if preview_portfolios:
        combined_map = create_combined_map(preview_portfolios, branch_data)
        if combined_map:
            st.plotly_chart(combined_map, use_container_width=True)
    else:
        st.info("No customers selected for map display")

def save_all_q1_2026_portfolios(q1_2026_portfolios_created, hh_customer_data, branch_data):
    """Save all Q1 2026 portfolios to a single CSV"""
    if not q1_2026_portfolios_created or hh_customer_data is None:
        st.error("No data available to save")
        return
    
    try:
        all_portfolio_data = []
        
        # Process each AU portfolio
        for au_id, au_data in q1_2026_portfolios_created.items():
            if not au_data.empty:
                export_data = prepare_portfolio_for_export_deduplicated(au_data, hh_customer_data, branch_data)
                if not export_data.empty:
                    all_portfolio_data.append(export_data)
        
        if not all_portfolio_data:
            st.error("No data to export")
            return
        
        # Combine all portfolios
        combined_data = pd.concat(all_portfolio_data, ignore_index=True)
        
        # Final cleaning and validation
        combined_data = clean_portfolio_data(combined_data)
        is_clean, duplicate_ids = validate_no_duplicates(combined_data, 'CG_ECN')
        if not is_clean:
            st.warning(f"Removed {len(duplicate_ids)} duplicate customers from Q1 2026 export")
            combined_data = combined_data.drop_duplicates(subset=['CG_ECN'], keep='first')
        
        # Convert to CSV
        csv_data = combined_data.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download Q1 2026 Portfolios CSV",
            data=csv_data,
            file_name="q1_2026_portfolios.csv",
            mime="text/csv",
            key="download_q1_2026_portfolios"
        )
        
        st.success(f"Q1 2026 portfolios prepared for download ({len(combined_data):,} customers across {len(q1_2026_portfolios_created):,} AUs)")
        
    except Exception as e:
        st.error(f"Error saving Q1 2026 portfolios: {str(e)}")

# ============================================================================
# END Q1 2026 MOVE TAB FUNCTIONS
# ============================================================================

def main():
    """Main application function"""
    # Setup page directly
    setup_page_config()
    add_logo()
    
    # Initialize session state
    initialize_session_state()
    
    # Create header
    create_header()

if __name__ == "__main__":
    main()
