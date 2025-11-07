import streamlit as st
import pandas as pd
from data_loader import get_merged_data, load_hh_data
from portfolio_creation_8 import enhanced_customer_au_assignment_with_two_inmarket_iterations
from utils import clean_portfolio_data, validate_no_duplicates, prepare_portfolio_for_export_deduplicated
from map_visualization import create_combined_map

def show_home_page():
    """Display home page content"""
    from home_tab import show_home_tab_content
    show_home_tab_content()

def show_my_requests_page():
    """Display My Requests page"""
    st.subheader("My Requests")
    st.info("This section is under development. Portfolio assignment requests will appear here.")

def show_portfolio_assignment_page():
    """Display Portfolio Assignment page"""
    from main import portfolio_assignment_page
    portfolio_assignment_page()

def show_portfolio_mapping_page():
    """Display Portfolio Mapping page"""
    from main import portfolio_mapping_page
    portfolio_mapping_page()

def show_ask_ai_page():
    """Display Ask AI page"""
    st.subheader("Ask AI")
    st.info("AI-powered insights coming soon!")

def create_customer_filters(customer_data, branch_data):
    """Create customer filter UI with portfolio size and radius sliders"""
    
    st.subheader("Customer Selection Filters")
    
    # Create two-column layout for header and clear button
    col_header, col_clear = st.columns([9, 1])
    
    with col_header:
        st.write("Filter customers by various criteria:")
    
    with col_clear:
        if st.button("Clear", key="clear_filters_mapping"):
            # Clear all filter-related session state
            filter_keys = [
                'cust_role', 'cust_portfolio', 'cust_state', 'cs_new_ns',
                'min_revenue', 'min_deposit', 'min_portfolio_size', 'max_portfolio_size',
                'inmarket_radius', 'centralized_radius'
            ]
            for key in filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Get unique values for dropdowns
    role_options = ['All'] + sorted(list(customer_data['TYPE'].dropna().unique()))
    portfolio_options = ['All'] + sorted(list(customer_data['CG_PORTFOLIO_CD'].dropna().unique()))
    state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
    cs_new_ns_options = sorted(list(customer_data['CS_NEW_NS'].dropna().unique()))
    
    # Create filter columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Role filter
        cust_role = st.selectbox(
            "Customer Role",
            options=role_options,
            key='cust_role'
        )
        
        # State filter
        cust_state = st.multiselect(
            "Customer State",
            options=state_options,
            default=None,
            key='cust_state'
        )
        
        # Minimum revenue filter
        min_rev = st.slider(
            "Minimum Revenue ($)",
            min_value=0,
            max_value=int(customer_data['BANK_REVENUE'].max()),
            value=0,
            step=1000,
            key='min_revenue'
        )
    
    with col2:
        # Portfolio filter
        cust_portfolio = st.selectbox(
            "Portfolio Code",
            options=portfolio_options,
            key='cust_portfolio'
        )
        
        # CS_NEW_NS filter
        cs_new_ns = st.multiselect(
            "CS NEW NS",
            options=cs_new_ns_options,
            default=None,
            key='cs_new_ns'
        )
        
        # Minimum deposit filter
        min_deposit = st.slider(
            "Minimum Deposit ($)",
            min_value=0,
            max_value=int(customer_data['DEPOSIT_BAL'].max()),
            value=0,
            step=1000,
            key='min_deposit'
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
    
    return cust_role, cust_portfolio, cust_state, cs_new_ns, min_rev, min_deposit, min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius

def create_customer_filters_for_q1_2026(hh_customer_data, branch_data):
    """Create customer filter UI for Q1 2026 Move - REMOVED Role and Portfolio filters"""
    
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

def show_q1_2026_move_page():
    """Show complete Q1 2026 Move functionality - All logic here to avoid circular import"""
    
    # Load HH customer data (already mapped columns)
    hh_customer_data, branch_data = load_hh_data()
    
    if hh_customer_data.empty:
        st.error("Failed to load HH_DF.csv. Please check the file and try again.")
        return
    
    st.session_state.hh_customer_data = hh_customer_data
    st.session_state.branch_data = branch_data
    
    # Main page content
    st.subheader("Q1 2026 Move - Smart Portfolio Mapping")
    
    # Create customer filters (NO Role, NO Portfolio Code)
    cust_state, cs_new_ns, min_rev, min_deposit, min_portfolio_size, max_portfolio_size, inmarket_radius, centralized_radius = create_customer_filters_for_q1_2026(hh_customer_data, branch_data)
    
    # Generate button
    col1, col2 = st.columns([5, 1])
    with col2:
        generate_button = st.button("Generate Smart Portfolios", key="generate_q1_2026_portfolios_btn", type="primary")
    
    if generate_button:
        # Clear previous results
        if 'q1_2026_portfolio_df' in st.session_state:
            del st.session_state.q1_2026_portfolio_df
        
        # Apply filters
        filtered_customers = apply_customer_filters_for_q1_2026(
            hh_customer_data, cust_state, cs_new_ns, min_rev, min_deposit
        )
        
        if len(filtered_customers) == 0:
            st.error("No customers found with the selected filters.")
            return
        
        # Generate portfolios
        generate_q1_2026_portfolios(
            filtered_customers, branch_data,
            min_portfolio_size, max_portfolio_size,
            inmarket_radius, centralized_radius
        )
    
    # Display results
    if st.session_state.q1_2026_portfolio_results is not None:
        display_q1_2026_results(st.session_state.q1_2026_portfolio_results, branch_data)

def apply_customer_filters_for_q1_2026(hh_customer_data, state, cs_new_ns, min_rev, min_deposit):
    """Apply customer filters for Q1 2026 Move - NO Role or Portfolio filters"""
    filtered = hh_customer_data.copy()
    
    if state:
        filtered = filtered[filtered['BILLINGSTATE'].isin(state)]
    
    if cs_new_ns:
        filtered = filtered[filtered['CS_NEW_NS'].isin(cs_new_ns)]
    
    if min_rev > 0:
        filtered = filtered[filtered['BANK_REVENUE'] >= min_rev]
    
    if min_deposit > 0:
        filtered = filtered[filtered['DEPOSIT_BAL'] >= min_deposit]
    
    return filtered

def generate_q1_2026_portfolios(customer_data, branch_data, min_size, max_size, inmarket_radius, centralized_radius):
    """Generate Q1 2026 portfolios using clustering algorithm"""
    
    with st.spinner("Generating Q1 2026 portfolios..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Running portfolio assignment algorithm...")
        progress_bar.progress(30)
        
        # Run the clustering algorithm
        results_df = enhanced_customer_au_assignment_with_two_inmarket_iterations(
            customer_data,
            branch_data,
            min_portfolio_size=min_size,
            max_portfolio_size_inmarket=max_size - 10,
            max_portfolio_size_proximity=max_size,
            max_portfolio_size_centralized=max_size - 10,
            inmarket_radius_miles=inmarket_radius,
            centralized_radius_miles=centralized_radius
        )
        
        progress_bar.progress(70)
        status_text.text("Processing results...")
        
        # Store results
        st.session_state.q1_2026_portfolio_results = results_df
        st.session_state.q1_2026_filtered_customers_count = len(customer_data)
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        st.success(f"✅ Successfully created Q1 2026 portfolios for {len(results_df):,} customers")

def display_q1_2026_results(results_df, branch_data):
    """Display Q1 2026 results - SIMPLIFIED version (NO Global Control, NO tables)"""
    st.subheader("Smart Portfolio Summary")
    
    # Get unique AUs
    assigned_aus = sorted(results_df['ASSIGNED_AU'].dropna().unique())
    
    if len(assigned_aus) == 0:
        st.warning("No portfolios were created")
        return
    
    # Display ONLY AU summary statistics (NO tables)
    display_q1_2026_portfolio_summary(results_df, assigned_aus, branch_data)
    
    # Geographic visualization
    st.subheader("Geographic Distribution")
    display_q1_2026_geographic_map(results_df, branch_data)
    
    # Save button
    if st.button("Save All Q1 2026 Portfolios", key="save_all_q1_2026_portfolios_btn"):
        save_all_q1_2026_portfolios(results_df, st.session_state.hh_customer_data, branch_data)

def display_q1_2026_portfolio_summary(results_df, assigned_aus, branch_data):
    """Display Q1 2026 portfolio summary in tabs - SIMPLIFIED"""
    tabs = st.tabs([f"AU {au}" for au in assigned_aus])
    
    for i, au in enumerate(assigned_aus):
        with tabs[i]:
            display_single_q1_2026_au_summary(results_df, au, branch_data)

def display_single_q1_2026_au_summary(results_df, au, branch_data):
    """Display summary for a single AU - ONLY statistics, NO table"""
    au_data = results_df[results_df['ASSIGNED_AU'] == au]
    
    # Get branch info
    branch_info = branch_data[branch_data['AU'] == au].iloc[0] if len(branch_data[branch_data['AU'] == au]) > 0 else None
    
    if branch_info is not None:
        st.write(f"**Branch:** {branch_info['BRANCH_NAME']}")
        st.write(f"**Location:** {branch_info['LAT']:.4f}, {branch_info['LON']:.4f}")
    
    # Display ONLY statistics (NO table)
    display_summary_statistics(au_data)

def display_q1_2026_geographic_map(results_df, branch_data):
    """Display Q1 2026 geographic map"""
    fig = create_combined_map(results_df, branch_data)
    st.plotly_chart(fig, use_container_width=True)

def save_all_q1_2026_portfolios(results_df, hh_customer_data, branch_data):
    """Save all Q1 2026 portfolios to CSV"""
    try:
        assigned_aus = sorted(results_df['ASSIGNED_AU'].dropna().unique())
        all_portfolios = []
        
        for au in assigned_aus:
            au_data = results_df[results_df['ASSIGNED_AU'] == au]
            portfolio_export = prepare_portfolio_for_export_deduplicated(au_data, hh_customer_data, branch_data)
            all_portfolios.append(portfolio_export)
        
        combined_data = pd.concat(all_portfolios, ignore_index=True)
        
        # Check for duplicates
        if combined_data['CG_ECN'].duplicated().any():
            duplicate_ids = combined_data[combined_data['CG_ECN'].duplicated(keep=False)]['CG_ECN'].unique()
            st.warning(f"Found {len(duplicate_ids)} duplicate customers in export. Keeping first occurrence.")
            combined_data = combined_data.drop_duplicates(subset=['CG_ECN'], keep='first')
        
        csv_data = combined_data.to_csv(index=False)
        
        st.download_button(
            label="Download Q1 2026 Portfolios CSV",
            data=csv_data,
            file_name="q1_2026_portfolios.csv",
            mime="text/csv",
            key="download_q1_2026_portfolios_csv"
        )
        
        st.success(f"Q1 2026 portfolios prepared for download ({len(combined_data):,} customers)")
        
    except Exception as e:
        st.error(f"Error saving Q1 2026 portfolios: {str(e)}")

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
