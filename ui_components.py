import streamlit as st
import pandas as pd

def setup_page_config():
    """Configure the Streamlit page"""
    st.set_page_config("Portfolio Creation tool", layout="wide")
    
    # Hide Streamlit's default header and remove default padding
    st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Remove default left and right padding from main container */
        .main .block-container {
            padding-top: 1rem;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        
        /* Style navigation buttons */
        div[data-testid="column"]:first-child .stButton > button {
            color: black !important;
            font-weight: bold !important;
            background-color: transparent !important;
            border: 1px solid #ccc !important;
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
    """Add custom header with logo and text only"""
    # Create a custom header section instead of modifying Streamlit's header
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

def create_header():
    """Create the page header with two-column layout"""
    # Initialize current page if not set (default to Home)
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Create two columns: navigation (15%) and main content (85%)
    col_nav, col_main = st.columns([15, 85])
    
    with col_nav:
        # Add custom styling for selected button
        st.markdown(f"""
        <style>
            /* Style for selected navigation button */
            .nav-button-selected {{
                background-color: rgb(215, 30, 40) !important;
                color: white !important;
                font-weight: bold !important;
                border: 1px solid rgb(215, 30, 40) !important;
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Navigation buttons with conditional styling
        if st.button("Home", key="nav_home", use_container_width=True):
            st.session_state.current_page = "Home"
        
        if st.button("My Requests", key="nav_requests", use_container_width=True):
            st.session_state.current_page = "My Requests"
        
        if st.button("Portfolio Assignment", key="nav_portfolio_assignment", use_container_width=True):
            st.session_state.current_page = "Portfolio Assignment"
        
        if st.button("Portfolio Mapping", key="nav_portfolio_mapping", use_container_width=True):
            st.session_state.current_page = "Portfolio Mapping"
    
    with col_main:
        # Add right padding to second column content
        st.markdown('<div style="padding-right: 1rem;">', unsafe_allow_html=True)
        
        # Show ALL content for selected page in right column
        if st.session_state.current_page == "Home":
            st.markdown("### Welcome to Banker Placement Tool")
            st.info("This is the home page - content coming soon.")
            
        elif st.session_state.current_page == "My Requests":
            st.markdown("### My Requests")
            st.info("My Requests functionality - content coming soon.")
            
        elif st.session_state.current_page == "Portfolio Assignment":
            # Put Portfolio Assignment content directly here
            show_portfolio_assignment_content()
            
        elif st.session_state.current_page == "Portfolio Mapping":
            # Put Portfolio Mapping content directly here
            show_portfolio_mapping_content()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Always return None - no content handled by main.py
    return None

def show_portfolio_assignment_content():
    """Display Portfolio Assignment content"""
    from data_loader import get_merged_data
    
    # Load data
    customer_data, banker_data, branch_data, data = get_merged_data()
    
    # Store branch_data in session state for save functions
    st.session_state.branch_data = branch_data
    st.session_state.customer_data = customer_data
    
    # Create AU filters
    selected_aus = create_au_filters(branch_data)
    
    # Create customer filters
    cust_state, role, cust_portcd, max_dist, min_rev, min_deposit = create_customer_filters(customer_data)
    
    # Create portfolio button
    button_clicked = create_portfolio_button()
    
    # Handle button click and portfolio creation logic
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
            from portfolio_creation import process_portfolio_creation
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
    display_portfolio_assignment_results(branch_data)

def show_portfolio_mapping_content():
    """Display Portfolio Mapping content"""
    from data_loader import get_merged_data
    
    st.subheader("Smart Portfolio Mapping")
    
    # Load data
    customer_data, banker_data, branch_data, data = get_merged_data()
    st.session_state.customer_data = customer_data
    
    # Create customer filters (reuse from Portfolio Assignment)
    cust_state, role, cust_portcd, max_dist, min_rev, min_deposit = create_customer_filters_for_mapping(customer_data)
    
    # Create Smart Portfolio Generation button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")  # Empty space
    with col2:
        generate_button = st.button("Generate Smart Portfolios", key="generate_smart_portfolios", type="primary")
    
    # Handle button click and smart portfolio logic
    if generate_button:
        st.session_state.should_generate_smart_portfolios = True
    
    # Process smart portfolio generation
    if st.session_state.get('should_generate_smart_portfolios', False):
        from main import generate_smart_portfolios, apply_customer_filters_for_mapping
        generate_smart_portfolios(customer_data, branch_data, cust_state, role, cust_portcd, min_rev, min_deposit)
        st.session_state.should_generate_smart_portfolios = False
    
    # Display results if they exist
    display_smart_portfolio_mapping_results(customer_data, branch_data)

def display_portfolio_assignment_results(branch_data):
    """Display portfolio assignment results"""
    if 'portfolios_created' in st.session_state and st.session_state.portfolios_created:
        from main import display_portfolio_tables, display_geographic_map
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

def display_smart_portfolio_mapping_results(customer_data, branch_data):
    """Display smart portfolio mapping results"""
    from main import display_smart_portfolio_results
    if 'smart_portfolio_results' not in st.session_state or len(st.session_state.smart_portfolio_results) == 0:
        st.info("Click 'Generate Smart Portfolios' to create optimized customer assignments.")
        return
    
    display_smart_portfolio_results(customer_data, branch_data)

def initialize_session_state():
    """Initialize all session state variables - avoid conflicting with widget keys"""
    session_vars = {
        # Portfolio Assignment variables
        'all_portfolios': {},
        'portfolio_controls': {},
        'recommend_reassignment': {},
        'should_create_portfolios': False,
        
        # Portfolio Mapping variables
        'should_generate_smart_portfolios': False,
        'smart_portfolio_controls': {}
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
            # Clear AU filters by clearing the widget keys
            for key in ["states", "cities", "selected_aus"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear created portfolios
            if 'portfolios_created' in st.session_state:
                del st.session_state.portfolios_created
            if 'portfolio_summaries' in st.session_state:
                del st.session_state.portfolio_summaries
            st.session_state.portfolio_controls = {}
    
    # Multi-select for AUs with expander
    with st.expander("Select AUs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_states = list(branch_data['STATECODE'].dropna().unique())
            states = st.multiselect("State", available_states, key="states")
        
        # Filter branch data based on selected states
        if states:
            filtered_branch_data = branch_data[branch_data['STATECODE'].isin(states)]
        else:
            filtered_branch_data = branch_data
            
        with col2:
            available_cities = list(filtered_branch_data['CITY'].dropna().unique())
            cities = st.multiselect("City", available_cities, key="cities")
        
        # Filter further based on selected cities
        if cities:
            filtered_branch_data = filtered_branch_data[filtered_branch_data['CITY'].isin(cities)]
        
        with col3:
            available_aus = list(filtered_branch_data['AU'].dropna().unique())
            selected_aus = st.multiselect("AU", available_aus, key="selected_aus")
    
    return selected_aus

def create_customer_filters(customer_data):
    """Create customer selection criteria filters"""
    col_header2, col_clear2 = st.columns([9, 1])
    with col_header2:
        st.subheader("Customer Selection Criteria")
    with col_clear2:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_customer_filters", help="Clear customer selection filters", type="secondary"):
            # Clear customer filters by clearing widget keys
            filter_keys = ["cust_state", "role", "cust_portcd", "max_distance", "min_revenue", "min_deposit"]
            for key in filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear created portfolios
            if 'portfolios_created' in st.session_state:
                del st.session_state.portfolios_created
            if 'portfolio_summaries' in st.session_state:
                del st.session_state.portfolio_summaries
            st.session_state.portfolio_controls = {}
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col2_or, col3 = st.columns([1, 1, 0.1, 1])
        
        with col1:
            cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
            cust_state = st.multiselect("Customer State", cust_state_options, key="cust_state")
            if not cust_state:
                cust_state = None
        
        with col2:
            role_options = list(customer_data['TYPE'].dropna().unique())
            role = st.multiselect("Role", role_options, key="role")
            if not role:
                role = None
        
        with col2_or:
            st.markdown("<div style='text-align: center; padding-top: 8px; font-weight: bold;'>-OR-</div>", unsafe_allow_html=True)
        
        with col3:
            customer_data_temp = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            portfolio_options = list(customer_data_temp['PORT_CODE'].dropna().unique())
            cust_portcd = st.multiselect("Portfolio Code", portfolio_options, key="cust_portcd")
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5, col6 = st.columns(3)
        with col4:
            max_dist = st.slider("Max Distance (miles)", 1, 100, value=20, key="max_distance")
        with col5:
            min_rev = st.slider("Minimum Revenue", 0, 20000, value=5000, step=1000, key="min_revenue")
        with col6:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, value=100000, step=5000, key="min_deposit")
    
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
        key=f"portfolio_editor_{au_id}"
    )

def create_apply_changes_button(au_id, is_single_au=False):
    """Create Apply Changes button for an AU"""
    key_suffix = "_single" if is_single_au else ""
    return st.button(f"Apply Changes for AU {au_id}", key=f"apply_changes_{au_id}{key_suffix}")

def create_save_buttons(au_id, is_single_au=False):
    """Create Save buttons for an AU"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write("")  # Empty space
    
    with col2:
        key_suffix = "_single" if is_single_au else ""
        save_au = st.button(f"Save AU {au_id}", key=f"save_au_{au_id}{key_suffix}", type="secondary")
    
    with col3:
        save_all = st.button("Save All", key=f"save_all_{au_id}{key_suffix}", type="secondary")
    
    return save_au, save_all

def create_customer_filters_for_mapping(customer_data):
    """Create customer selection criteria filters for Portfolio Mapping"""
    col_header2, col_clear2 = st.columns([9, 1])
    with col_header2:
        st.subheader("Customer Selection Criteria")
    with col_clear2:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_mapping_filters", help="Clear customer selection filters", type="secondary"):
            # Clear customer filters for mapping by clearing widget keys
            mapping_filter_keys = ["mapping_cust_state", "mapping_role", "mapping_cust_portcd", "mapping_min_revenue", "mapping_min_deposit"]
            for key in mapping_filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear smart portfolio results
            if 'smart_portfolio_results' in st.session_state:
                del st.session_state.smart_portfolio_results
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2, col2_or, col3 = st.columns([1, 1, 0.1, 1])
        
        with col1:
            cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
            cust_state = st.multiselect("Customer State", cust_state_options, key="mapping_cust_state")
            if not cust_state:
                cust_state = None
        
        with col2:
            role_options = list(customer_data['TYPE'].dropna().unique())
            role = st.multiselect("Role", role_options, key="mapping_role")
            if not role:
                role = None
        
        with col2_or:
            st.markdown("<div style='text-align: center; padding-top: 8px; font-weight: bold;'>-OR-</div>", unsafe_allow_html=True)
        
        with col3:
            customer_data_temp = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
            portfolio_options = list(customer_data_temp['PORT_CODE'].dropna().unique())
            cust_portcd = st.multiselect("Portfolio Code", portfolio_options, key="mapping_cust_portcd")
            if not cust_portcd:
                cust_portcd = None
        
        col4, col5 = st.columns(2)
        with col4:
            min_rev = st.slider("Minimum Revenue", 0, 20000, value=5000, step=1000, key="mapping_min_revenue")
        with col5:
            min_deposit = st.slider("Minimum Deposit", 0, 200000, value=100000, step=5000, key="mapping_min_deposit")
    
    return cust_state, role, cust_portcd, None, min_rev, min_deposit
