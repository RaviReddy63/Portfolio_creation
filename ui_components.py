import streamlit as st
import pandas as pd

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
    """Show Q1 2026 Move page - Uses local import to avoid circular dependency"""
    from data_loader import load_hh_data
    
    # Load HH customer data (already mapped columns)
    hh_customer_data, branch_data = load_hh_data()
    
    if hh_customer_data.empty:
        st.error("Failed to load HH_DF.csv. Please check the file and try again.")
        return
    
    # Store in session state
    st.session_state.hh_customer_data = hh_customer_data
    st.session_state.branch_data = branch_data
    
    # LOCAL IMPORT to avoid circular dependency
    # This works because main.py is fully loaded by the time this function is called
    from main import q1_2026_move_page
    
    # Call the main Q1 2026 logic from main.py
    q1_2026_move_page(hh_customer_data, branch_data)

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
