import streamlit as st
import pandas as pd

def setup_page_config():
    """Configure the Streamlit page"""
    st.set_page_config("Portfolio Creation tool", layout="wide")
    
    # Custom CSS for red header
    st.markdown("""
    <style>
        [data-testid="stHeader"] {
            background-color: rgb(215, 30, 40) !important;
        }
        /* Style for clear filters buttons to look like header text */
        div[data-testid="column"] button[kind="secondary"] {
            background: none !important;
            border: none !important;
            padding: 0 !important;
            color: #1f77b4 !important;
            text-decoration: underline !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            box-shadow: none !important;
            height: auto !important;
            margin-top: 0.5rem !important;
        }
        div[data-testid="column"] button[kind="secondary"]:hover {
            color: #0d47a1 !important;
            background: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

def add_logo():
    """Add logo to the page"""
    try:
        st.logo("logo.svg")
    except:
        try:
            st.logo("logo.png")
        except:
            st.info("To display a logo, place logo.svg or logo.png in your project directory")

def create_header():
    """Create the page header with navigation"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Portfolio Creation Tool")
    with col2:
        st.markdown("")  # Add some spacing
        page = st.radio("", ["Portfolio Assignment", "Portfolio Mapping"], horizontal=True, key="page_nav")
    return page

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        # Portfolio Assignment variables
        'all_portfolios': {},
        'portfolio_controls': {},
        'recommend_reassignment': {},
        'should_create_portfolios': False,
        'filter_states': [],
        'filter_cities': [],
        'filter_selected_aus': [],
        'filter_cust_state': [],
        'filter_role': [],
        'filter_cust_portcd': [],
        'filter_max_dist': 20,
        'filter_min_rev': 5000,
        'filter_min_deposit': 100000,
        
        # Portfolio Mapping variables
        'should_generate_smart_portfolios': False,
        'mapping_filter_cust_state': [],
        'mapping_filter_role': [],
        'mapping_filter_cust_portcd': [],
        'mapping_filter_min_rev': 5000,
        'mapping_filter_min_deposit': 100000
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def create_au_filters(branch_data):
    """Create AU selection filters"""
    col_header1, col_clear1 = st.columns([9, 1])
    with col_header1:
        st.subheader("Select AUs for Portfolio Creation")
    with col_clear1:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_au_filters", help="Clear AU selection filters", type="secondary"):
            # Clear AU filters
            st.session_state.filter_states = []
            st.session_state.filter_cities = []
            st.session_state.filter_selected_aus = []
            # Clear created portfolios
            if 'portfolios_created' in st.session_state:
                del st.session_state.portfolios_created
            if 'portfolio_summaries' in st.session_state:
                del st.session_state.portfolio_summaries
            st.session_state.portfolio_controls = {}
            st.experimental_rerun()
    
    # Multi-select for AUs with expander
    with st.expander("Select AUs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_states = list(branch_data['STATECODE'].dropna().unique())
            default_states = [s for s in st.session_state.filter_states if s in available_states]
            states = st.multiselect("State", available_states, default=default_states, key="states")
            st.session_state.filter_states = states
        
        # Filter branch data based on selected states
        if states:
            filtered_branch_data = branch_data[branch_data['STATECODE'].isin(states)]
        else:
            filtered_branch_data = branch_data
            
        with col2:
            available_cities = list(filtered_branch_data['CITY'].dropna().unique())
            default_cities = [c for c in st.session_state.filter_cities if c in available_cities]
            cities = st.multiselect("City", available_cities, default=default_cities, key="cities")
            st.session_state.filter_cities = cities
        
        # Filter further based on selected cities
        if cities:
            filtered_branch_data = filtered_branch_data[filtered_branch_data['CITY'].isin(cities)]
        
        with col3:
            available_aus = list(filtered_branch_data['AU'].dropna().unique())
            default_aus = [a for a in st.session_state.filter_selected_aus if a in available_aus]
            selected_aus = st.multiselect("AU", available_aus, default=default_aus, key="selected_aus")
            st.session_state.filter_selected_aus = selected_aus
    
    return selected_aus

def create_customer_filters(customer_data):
    """Create customer selection criteria filters"""
    col_header2, col_clear2 = st.columns([9, 1])
    with col_header2:
        st.subheader("Customer Selection Criteria")
    with col_clear2:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_customer_filters", help="Clear customer selection filters", type="secondary"):
            # Clear customer filters
            st.session_state.filter_cust_state = []
            st.session_state.filter_role = []
            st.session_state.filter_cust_portcd = []
            st.session_state.filter_max_dist = 20
            st.session_state.filter_min_rev = 5000
            st.session_state.filter_min_deposit = 100000
            # Clear created portfolios
            if 'portfolios_created' in st.session_state:
                del st.session_state.portfolios_created
            if 'portfolio_summaries' in st.session_state:
                del st.session_state.portfolio_summaries
            st.session_state.portfolio_controls = {}
            st.experimental_rerun()
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col2_or, col3 = st.columns([1, 1, 0.1, 1])
        
        with col1:
            cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
            default_cust_states = [s for s in st.session_state.filter_cust_state if s in cust_state_options]
            cust_state = st.multiselect("Customer State", cust_state_options, default=default_cust_states, key="cust_state")
            st.session_state.filter_cust_state = cust_state
            if not cust_state:
                cust_state = None
        
        with col2:
            role_options = list(customer_data['TYPE'].dropna().unique())
            default_roles = [r for r in st.session_state.filter_role if r in role_options]
            role = st.multiselect("Role", role_options, default=default_roles, key="role")
            st.session_state.filter_role = role
            if not role:
                role = None
        
        with col2_or:
            st.markdown("<div style='text-align: center; padding-top: 8px; font-weight: bold;'>-OR-</div>", unsafe_allow_html=True)
        
        with col3:
            customer_data_temp = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            portfolio_options = list(customer_data_temp['PORT_CODE'].dropna().unique())
            default_portfolios = [p for p in st.session_state.filter_cust_portcd if p in portfolio_options]
            cust_portcd = st.multiselect("Portfolio Code", portfolio_options, default=default_portfolios, key="cust_portcd")
            st.session_state.filter_cust_portcd = cust_portcd
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5, col6 = st.columns(3)
        with col4:
            max_dist = st.slider("Max Distance (miles)", 1, 100, value=st.session_state.filter_max_dist, key="max_distance")
            st.session_state.filter_max_dist = max_dist
        with col5:
            min_rev = st.slider("Minimum Revenue", 0, 20000, value=st.session_state.filter_min_rev, step=1000, key="min_revenue")
            st.session_state.filter_min_rev = min_rev
        with col6:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, value=st.session_state.filter_min_deposit, step=5000, key="min_deposit")
            st.session_state.filter_min_deposit = min_deposit
    
    return cust_state, role, cust_portcd, max_dist, min_rev, min_deposit

def create_portfolio_button():
    """Create the right-aligned Create Portfolios button"""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")  # Empty space
    with col2:
        return st.button("Create Portfolios", key="create_portfolios", type="primary")

def display_summary_statistics(au_filtered_data):
    """Display summary statistics for an AU"""
    if not au_filtered_data.empty:
        st.subheader("AU Summary Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Customers", len(au_filtered_data))
        with col_b:
            st.metric("Avg Distance (Miles)", f"{au_filtered_data['Distance'].mean():.1f}")
        with col_c:
            avg_revenue_k = au_filtered_data['BANK_REVENUE'].mean() / 1000
            st.metric("Average Revenue", f"{avg_revenue_k:.1f}K")
        with col_d:
            avg_deposit_mm = au_filtered_data['DEPOSIT_BAL'].mean() / 1000000
            st.metric("Average Deposits", f"{avg_deposit_mm:.1f}MM")

def create_portfolio_editor(portfolio_df, au_id, is_multi_au=False):
    """Create an editable portfolio dataframe"""
    if is_multi_au:
        column_config = {
            "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio in selection"),
            "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
            "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
            "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
            "Available for all new portfolios": st.column_config.NumberColumn("Available for all new portfolios", disabled=True),
            "Available for this portfolio": st.column_config.NumberColumn("Available for this portfolio", disabled=True),
            "Select": st.column_config.NumberColumn(
                "Select",
                help="Number of customers to select from this portfolio",
                min_value=0,
                step=1
            )
        }
    else:
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
        key=f"portfolio_editor_{au_id}_{len(portfolio_df)}"
    )

def create_apply_changes_button(au_id, is_single_au=False):
    """Create Apply Changes button for an AU"""
    key_suffix = "_single" if is_single_au else ""
    return st.button(f"Apply Changes for AU {au_id}", key=f"apply_changes_{au_id}{key_suffix}")
