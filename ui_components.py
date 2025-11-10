import streamlit as st
import pandas as pd

def show_home_page():
    """Show Home page content with portfolio dashboard"""
    from data_loader import get_merged_data
    
    customer_data, banker_data, branch_data, _ = get_merged_data()
    
    # Import and show home tab content
    try:
        from home_tab import show_home_tab_content
        show_home_tab_content(customer_data, banker_data, branch_data)
    except ImportError:
        # Fallback if home_tab module doesn't exist
        show_basic_home_content(customer_data, banker_data, branch_data)

def show_basic_home_content(customer_data, banker_data, branch_data):
    """Show basic home content if home_tab module is not available"""
    st.markdown("### 🏠 Welcome to Banker Placement Tool")
    st.markdown("*Optimize customer-banker assignments with advanced analytics*")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(customer_data):,}")
    
    with col2:
        st.metric("Active Bankers", f"{len(banker_data):,}")
    
    with col3:
        st.metric("Banking Units", f"{len(branch_data):,}")
    
    with col4:
        avg_revenue = customer_data['BANK_REVENUE'].mean()
        if avg_revenue >= 1000000:
            st.metric("Avg Customer Revenue", f"${avg_revenue/1000000:.1f}M")
        else:
            st.metric("Avg Customer Revenue", f"${avg_revenue/1000:.1f}K")
    
    st.markdown("---")
    st.info("📊 Use the **Portfolio Assignment** tab to create custom portfolios, or try **Portfolio Mapping** for AI-optimized assignments.")

def show_my_requests_page():
    """Show My Requests page content"""
    st.markdown("### 📋 My Requests")
    st.info("My Requests functionality - content coming soon.")

def show_ask_ai_page():
    """Show Ask AI chat interface"""
    # Initialize session state for chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        # Add initial AI message
        ai_intro = """👋 Hello! I'm your AI Assistant for the Banker Placement Tool.

I can help you analyze customer information and portfolios:

🔍 **Customer Analytics**: Get counts of customers in portfolios or specific areas
📊 **Portfolio Insights**: Analyze customer distributions and demographics  
💰 **Revenue Analysis**: Break down customer revenue and deposit patterns
🎯 **Opportunities**: Identify growth opportunities and market gaps
📍 **Geographic Data**: Understand customer locations and coverage areas
📋 **Product Details**: Get information about customer products and services

Just ask me anything about your customers, portfolios, or market opportunities!"""
        
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": ai_intro
        })
    
    st.markdown("### 🤖 Ask AI Assistant")
    st.markdown("*Get insights about your customers, portfolios, and market opportunities*")
    st.markdown("---")
    
    # Custom CSS for chat styling
    st.markdown("""
    <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 10px 0;
            margin-left: 20%;
            text-align: right;
        }
        .ai-message {
            background-color: white;
            color: #333;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 10px 0;
            margin-right: 20%;
            border: 1px solid #dee2e6;
        }
        .message-header {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat container with custom styling
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        if message["role"] == "assistant":
            st.markdown(f"""
            <div class="ai-message">
                <div class="message-header">🤖 AI Assistant</div>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="user-message">
                <div class="message-header">👤 You</div>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input using text_input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input("Ask me about customers, portfolios, or opportunities...", key="chat_input", placeholder="Type your question here...")
    
    with col2:
        send_button = st.button("Send", type="primary")
    
    if send_button and user_input:
        # Add user message to chat
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # AI response
        ai_response = f"""I understand you're asking about: **"{user_input}"**

Currently, I'm in demonstration mode. Here's what I can help you with:

🔍 **Customer Analytics**: "How many customers are in Portfolio ABC?" or "Show me customers in Dallas area"
📊 **Portfolio Insights**: "What's the average revenue in my portfolios?" or "Which areas have the most customers?"
💰 **Revenue Analysis**: "What are the top revenue-generating customer segments?"
🎯 **Opportunities**: "Where should I focus my next banking expansion?"
📍 **Geographic Data**: "Show me customer density by region"
📋 **Product Details**: "What products do customers in Portfolio XYZ use most?"

*Full AI capabilities coming soon! This will connect to your customer database for real-time insights.*"""
        
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": ai_response
        })
        
        # Clear the input and rerun
        st.session_state.chat_input = ""
        st.rerun()

def show_portfolio_assignment_page():
    """Show complete Portfolio Assignment functionality"""
    from data_loader import get_merged_data
    from main import portfolio_assignment_page
    
    customer_data, banker_data, branch_data, data = get_merged_data()
    
    # Store data in session state for save functions
    st.session_state.branch_data = branch_data
    st.session_state.customer_data = customer_data
    
    # Call the main portfolio assignment logic
    portfolio_assignment_page(customer_data, banker_data, branch_data)

def show_portfolio_mapping_page():
    """Show complete Portfolio Mapping functionality"""
    from data_loader import get_merged_data
    from main import portfolio_mapping_page
    
    customer_data, banker_data, branch_data, data = get_merged_data()
    
    st.session_state.customer_data = customer_data
    
    # Call the main portfolio mapping logic
    portfolio_mapping_page(customer_data, banker_data, branch_data)

def show_q1_2024_move_page():
    """Show complete Q1 2024 Move functionality"""
    from data_loader import load_hh_data
    from main import q1_2024_move_page
    
    hh_data, branch_data = load_hh_data()
    
    if hh_data.empty:
        st.error("Unable to load HH_DF.csv data. Please check the file exists.")
        return
    
    st.session_state.q1_hh_data = hh_data
    
    # Call the main Q1 2024 move logic
    q1_2024_move_page(hh_data, branch_data)

def initialize_session_state():
    """Initialize all session state variables - REMOVED, now in main.py"""
    pass

def create_au_filters(branch_data):
    """Create AU selection filters - NO CALCULATIONS, just UI"""
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
            # Clear created portfolios when filters change
            for key in ['portfolios_created', 'portfolio_summaries', 'portfolio_controls']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Multi-select for AUs with expander
    with st.expander("Select AUs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        # Pre-calculate options ONCE and store in session state
        if 'filter_options' not in st.session_state:
            st.session_state.filter_options = {
                'all_states': sorted(branch_data['STATECODE'].dropna().unique().tolist()),
                'all_cities': sorted(branch_data['CITY'].dropna().unique().tolist()),
                'all_aus': branch_data[['AU', 'NAME', 'STATECODE', 'CITY']].dropna().to_dict('records')
            }
        
        with col1:
            # Use pre-calculated states - NO CALCULATION HERE
            states = st.multiselect("State", st.session_state.filter_options['all_states'], key="states")
        
        with col2:
            # Filter cities based on selected states - MINIMAL CALCULATION
            if states:
                available_cities = [au['CITY'] for au in st.session_state.filter_options['all_aus'] 
                                  if au['STATECODE'] in states]
                available_cities = sorted(list(set(available_cities)))
            else:
                available_cities = st.session_state.filter_options['all_cities']
            
            cities = st.multiselect("City", available_cities, key="cities")
        
        with col3:
            # Filter AUs based on selected states and cities - MINIMAL CALCULATION
            filtered_aus = st.session_state.filter_options['all_aus']
            
            if states:
                filtered_aus = [au for au in filtered_aus if au['STATECODE'] in states]
            if cities:
                filtered_aus = [au for au in filtered_aus if au['CITY'] in cities]
            
            # Create AU options - LIGHTWEIGHT OPERATION
            au_options = []
            au_mapping = {}
            
            for au in filtered_aus:
                au_number = au['AU']
                au_name = au['NAME']
                display_text = f"{au_name} - {au_number}"
                au_options.append(display_text)
                au_mapping[display_text] = au_number
            
            au_options = sorted(au_options)
            
            selected_au_displays = st.multiselect("AU", au_options, key="selected_aus")
            
            # Convert back to AU numbers - LIGHTWEIGHT OPERATION
            selected_aus = [au_mapping[display] for display in selected_au_displays if display in au_mapping]
    
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
            filter_keys = ["cust_state", "role", "cust_portcd", "cs_new_ns", "max_distance", "min_revenue", "min_deposit"]
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
        
        col4, col5, col6, col7 = st.columns(4)
        
        with col4:
            # CS_NEW_NS Filter (multiselect)
            if 'CS_NEW_NS' in customer_data.columns:
                cs_new_ns_options = [0, 1, 2, 3, 4]
                cs_new_ns = st.multiselect(
                    "CS NEW NS", 
                    options=cs_new_ns_options,
                    key="cs_new_ns"
                )
                if not cs_new_ns:
                    cs_new_ns = None
            else:
                cs_new_ns = None
                st.info("CS_NEW_NS not in data")
        
        with col5:
            max_dist = st.slider("Max Distance (miles)", 1, 100, value=20, key="max_distance")
        
        with col6:
            min_rev = st.select_slider("Minimum Revenue", 
                                     options=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 
                                            11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000],
                                     value=5000,
                                     format_func=lambda x: f"${x:,}",
                                     key="min_revenue")
        with col7:
            min_deposit = st.select_slider("Minimum Deposit",
                                         options=[0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000],
                                         value=100000,
                                         format_func=lambda x: f"${x:,}",
                                         key="min_deposit")
    
    return cust_state, role, cust_portcd, cs_new_ns, max_dist, min_rev, min_deposit

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
            st.metric("Total Customers", f"{len(au_filtered_data):,}")
        with col_b:
            st.metric("Avg Distance (Miles)", f"{au_filtered_data['Distance'].mean():.1f}")
        with col_c:
            avg_revenue = au_filtered_data['BANK_REVENUE'].mean()
            if avg_revenue >= 1000000:
                st.metric("Average Revenue", f"${avg_revenue/1000000:.1f}M")
            else:
                st.metric("Average Revenue", f"${avg_revenue/1000:.1f}K")
        with col_d:
            avg_deposit = au_filtered_data['DEPOSIT_BAL'].mean()
            if avg_deposit >= 1000000:
                st.metric("Average Deposits", f"${avg_deposit/1000000:.1f}M")
            else:
                st.metric("Average Deposits", f"${avg_deposit/1000:.1f}K")

def create_portfolio_editor(portfolio_df, au_id, is_multi_au=False):
    """Create an editable portfolio dataframe"""
    if is_multi_au:
        column_config = {
            "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio in selection"),
            "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
            "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
            "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True, format="%d"),
            "Available for all new portfolios": st.column_config.NumberColumn("Available for all new portfolios", disabled=True, format="%d"),
            "Available for this portfolio": st.column_config.NumberColumn("Available for this portfolio", disabled=True, format="%d"),
            "Select": st.column_config.NumberColumn(
                "Select",
                help="Number of customers to select from this portfolio",
                min_value=0,
                step=1,
                format="%d"
            )
        }
    else:
        column_config = {
            "Include": st.column_config.CheckboxColumn("Include", help="Check to include this portfolio in selection"),
            "Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
            "Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
            "Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True, format="%d"),
            "Available for this portfolio": st.column_config.NumberColumn("Available for this portfolio", disabled=True, format="%d"),
            "Select": st.column_config.NumberColumn(
                "Select",
                help="Number of customers to select from this portfolio",
                min_value=0,
                step=1,
                format="%d"
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
            mapping_filter_keys = ["mapping_cust_state", "mapping_role", "mapping_cust_portcd", "mapping_cs_new_ns", "mapping_min_revenue", "mapping_min_deposit", "mapping_inmarket_radius_1", "mapping_centralized_radius", "mapping_min_size", "mapping_max_size"]
            for key in mapping_filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear smart portfolio results
            if 'smart_portfolio_results' in st.session_state:
                del st.session_state.smart_portfolio_results
            st.rerun()
    
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
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # CS_NEW_NS Filter (multiselect)
            if 'CS_NEW_NS' in customer_data.columns:
                cs_new_ns_options = [0, 1, 2, 3, 4]
                cs_new_ns = st.multiselect(
                    "CS NEW NS", 
                    options=cs_new_ns_options,
                    key="mapping_cs_new_ns"
                )
                if not cs_new_ns:
                    cs_new_ns = None
            else:
                cs_new_ns = None
                st.info("CS_NEW_NS not in data")
        
        with col5:
            min_rev = st.select_slider("Minimum Revenue", 
                                     options=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 
                                            11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000],
                                     value=5000,
                                     format_func=lambda x: f"${x:,}",
                                     key="mapping_min_revenue")
        with col6:
            min_deposit = st.select_slider("Minimum Deposit",
                                         options=[0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000],
                                         value=100000,
                                         format_func=lambda x: f"${x:,}",
                                         key="mapping_min_deposit")
        
        # Portfolio Configuration (Radius & Size)
        st.markdown("---")
        st.markdown("**Portfolio Configuration (Radius & Size)**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            inmarket_radius_1 = st.slider(
                "INMARKET Radius (miles)", 
                5, 100, 
                value=20, 
                key="mapping_inmarket_radius_1",
                help="Radius for INMARKET clustering"
            )
        
        with col2:
            centralized_radius = st.slider(
                "CENTRALIZED Radius (miles)", 
                50, 1000, 
                value=100, 
                key="mapping_centralized_radius",
                help="Maximum radius for CENTRALIZED portfolios"
            )
        
        with col3:
            min_size = st.slider(
                "Min Portfolio Size", 
                150, 800, 
                value=200, 
                key="mapping_min_size",
                help="Minimum customers per portfolio"
            )
        
        with col4:
            max_size = st.slider(
                "Max Portfolio Size", 
                150, 800, 
                value=225, 
                key="mapping_max_size",
                help="Maximum customers per portfolio"
            )
        
        # Calculate inmarket_radius_2 and max_customers_per_branch
        inmarket_radius_2 = inmarket_radius_1 + 20
        max_customers_per_branch = max_size - 25
        
        # Display calculated values in one line
        st.info(f"ℹ️ 2nd INMARKET radius: **{inmarket_radius_2} miles** | Max per branch (1st iteration): **{max_customers_per_branch}** customers")
    
    return cust_state, role, cust_portcd, cs_new_ns, None, min_rev, min_deposit, inmarket_radius_1, inmarket_radius_2, centralized_radius, min_size, max_size

def create_customer_filters_for_q1_move(hh_data):
    """Create customer selection criteria filters for Q1 2024 Move - NO Role or Portfolio Code"""
    col_header2, col_clear2 = st.columns([9, 1])
    with col_header2:
        st.subheader("Customer Selection Criteria")
    with col_clear2:
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_q1_filters", help="Clear customer selection filters", type="secondary"):
            # Clear customer filters for Q1 by clearing widget keys
            q1_filter_keys = ["q1_cust_state", "q1_cs_new_ns", "q1_min_revenue", "q1_min_deposit", "q1_inmarket_radius_1", "q1_centralized_radius", "q1_min_size", "q1_max_size"]
            for key in q1_filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear Q1 smart portfolio results
            if 'q1_smart_portfolio_results' in st.session_state:
                del st.session_state.q1_smart_portfolio_results
            st.rerun()
    
    with st.expander("Customer Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            cust_state_options = list(hh_data['BILLINGSTATE'].dropna().unique())
            cust_state = st.multiselect("Customer State", cust_state_options, key="q1_cust_state")
            if not cust_state:
                cust_state = None
        
        with col2:
            # CS_NEW_NS Filter (multiselect)
            if 'CS_NEW_NS' in hh_data.columns:
                cs_new_ns_options = [0, 1, 2, 3, 4]
                cs_new_ns = st.multiselect(
                    "CS NEW NS", 
                    options=cs_new_ns_options,
                    key="q1_cs_new_ns"
                )
                if not cs_new_ns:
                    cs_new_ns = None
            else:
                cs_new_ns = None
                st.info("CS_NEW_NS not in data")
        
        col3, col4 = st.columns(2)
        
        with col3:
            min_rev = st.select_slider("Minimum Revenue", 
                                     options=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 
                                            11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000],
                                     value=5000,
                                     format_func=lambda x: f"${x:,}",
                                     key="q1_min_revenue")
        with col4:
            min_deposit = st.select_slider("Minimum Deposit",
                                         options=[0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000],
                                         value=100000,
                                         format_func=lambda x: f"${x:,}",
                                         key="q1_min_deposit")
        
        # Portfolio Configuration (Radius & Size)
        st.markdown("---")
        st.markdown("**Portfolio Configuration (Radius & Size)**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            inmarket_radius_1 = st.slider(
                "INMARKET Radius (miles)", 
                5, 100, 
                value=20, 
                key="q1_inmarket_radius_1",
                help="Radius for INMARKET clustering"
            )
        
        with col2:
            centralized_radius = st.slider(
                "CENTRALIZED Radius (miles)", 
                50, 1000, 
                value=100, 
                key="q1_centralized_radius",
                help="Maximum radius for CENTRALIZED portfolios"
            )
        
        with col3:
            min_size = st.slider(
                "Min Portfolio Size", 
                150, 800, 
                value=200, 
                key="q1_min_size",
                help="Minimum customers per portfolio"
            )
        
        with col4:
            max_size = st.slider(
                "Max Portfolio Size", 
                150, 800, 
                value=225, 
                key="q1_max_size",
                help="Maximum customers per portfolio"
            )
        
        # Calculate inmarket_radius_2 and max_customers_per_branch
        inmarket_radius_2 = inmarket_radius_1 + 20
        max_customers_per_branch = max_size - 25
        
        # Display calculated values in one line
        st.info(f"ℹ️ 2nd INMARKET radius: **{inmarket_radius_2} miles** | Max per branch (1st iteration): **{max_customers_per_branch}** customers")
    
    # Return 10 values (no role, no portfolio code)
    return cust_state, cs_new_ns, None, min_rev, min_deposit, inmarket_radius_1, inmarket_radius_2, centralized_radius, min_size, max_size
