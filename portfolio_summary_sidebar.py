import streamlit as st
import pandas as pd
import numpy as np

def create_portfolio_summary_sidebar():
    """Create right sidebar with portfolio summary for Portfolio Assignment"""
    
    # Check if portfolios exist
    if ('portfolios_created' not in st.session_state or 
        not st.session_state.portfolios_created or
        'portfolio_controls' not in st.session_state):
        return
    
    # Create sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("📋 Portfolio Summary")
        st.markdown("*Global portfolio control across all AUs*")
        
        # Generate aggregated portfolio summary
        aggregated_summary = generate_aggregated_portfolio_summary()
        
        if aggregated_summary:
            # Display the editable summary table
            edited_summary = display_editable_aggregated_portfolio_table(aggregated_summary)
            
            # Add Apply Global Changes button
            if st.button("🌍 Apply Global Changes", key="apply_global_changes", type="primary"):
                apply_global_portfolio_changes(edited_summary, aggregated_summary)
            
            # Add summary metrics
            display_portfolio_summary_metrics(aggregated_summary)
        else:
            st.info("No portfolio data available")

def generate_aggregated_portfolio_summary():
    """Generate aggregated summary across all AUs"""
    
    portfolios_created = st.session_state.portfolios_created
    portfolio_controls = st.session_state.portfolio_controls
    
    # Dictionary to aggregate portfolio data
    portfolio_aggregates = {}
    
    # Process each AU
    for au_id, au_data in portfolios_created.items():
        if au_id not in portfolio_controls:
            continue
            
        control_data = portfolio_controls[au_id]
        
        # Process each portfolio in this AU
        for _, portfolio_row in control_data.iterrows():
            portfolio_id = portfolio_row['Portfolio ID']
            include = portfolio_row.get('Include', True)
            select_count = portfolio_row.get('Select', 0)
            portfolio_type = portfolio_row.get('Portfolio Type', 'Unknown')
            total_customers = portfolio_row.get('Total Customers', 0)
            available_this_au = portfolio_row.get('Available for this portfolio', 0)
            
            # Initialize portfolio if not exists
            if portfolio_id not in portfolio_aggregates:
                portfolio_aggregates[portfolio_id] = {
                    'Portfolio ID': portfolio_id,
                    'Portfolio Type': portfolio_type,
                    'Total Customers': total_customers,
                    'Available Customers': 0,
                    'Total Customers After Transfer': 0,
                    'AUs': [],
                    'Include': True,  # Default to True, will be False if any AU excludes it
                    'Select': 0
                }
            
            # Update aggregated data
            portfolio_aggregates[portfolio_id]['Available Customers'] += available_this_au
            
            if include:
                portfolio_aggregates[portfolio_id]['Total Customers After Transfer'] += select_count
                portfolio_aggregates[portfolio_id]['Select'] += select_count
            else:
                portfolio_aggregates[portfolio_id]['Include'] = False
            
            # Track which AUs this portfolio appears in
            portfolio_aggregates[portfolio_id]['AUs'].append({
                'AU': au_id,
                'Available': available_this_au,
                'Selected': select_count if include else 0,
                'Include': include
            })
    
    # Convert to list and sort
    summary_list = []
    for portfolio_id, data in portfolio_aggregates.items():
        summary_list.append({
            'Include': data['Include'],
            'Portfolio ID': portfolio_id,
            'Portfolio Type': data['Portfolio Type'],
            'Total Customers': data['Total Customers'],
            'Available Customers': data['Available Customers'],
            'Total Customers After Transfer': data['Total Customers After Transfer'],
            'Select': data['Select'],
            'AUs Count': len(data['AUs'])
        })
    
    # Sort by Portfolio ID
    summary_list.sort(key=lambda x: x['Portfolio ID'])
    
    return summary_list

def display_editable_aggregated_portfolio_table(aggregated_summary):
    """Display the editable aggregated portfolio summary table"""
    
    df = pd.DataFrame(aggregated_summary)
    
    # Remove the unwanted columns
    display_columns = ['Include', 'Portfolio ID', 'Total Customers', 'Total Customers After Transfer', 'Select']
    df_display = df[display_columns].copy()
    
    # Add portfolio type for color coding (hidden from display)
    df_display['Portfolio Type'] = df['Portfolio Type']
    
    # Create column configuration
    column_config = {
        "Include": st.column_config.CheckboxColumn(
            "Include",
            help="Include portfolio in final assignment (affects all AUs)"
        ),
        "Portfolio ID": st.column_config.TextColumn(
            "Portfolio ID",
            help="Unique portfolio identifier",
            disabled=True
        ),
        "Total Customers": st.column_config.NumberColumn(
            "Total Customers",
            help="Total customers in this portfolio across all data",
            disabled=True
        ),
        "Total Customers After Transfer": st.column_config.NumberColumn(
            "After Transfer",
            help="Total customers selected across all AUs",
            disabled=True
        ),
        "Select": st.column_config.NumberColumn(
            "Select",
            help="Total customers to select (will keep nearest to selected AUs)",
            min_value=0,
            step=1
        )
    }
    
    # Apply color coding based on portfolio type
    def get_color_for_type(portfolio_type):
        portfolio_type = portfolio_type.lower()
        
        if 'inmarket' in portfolio_type:
            return '#e8f5e8'  # Light green
        elif 'centralized' in portfolio_type:
            return '#e8f0ff'  # Light blue  
        elif 'unmanaged' in portfolio_type:
            return '#fff8e1'  # Light yellow
        elif 'unassigned' in portfolio_type:
            return '#fce4ec'  # Light pink
        else:
            return '#f5f5f5'  # Light gray for others
    
    # Create a styled dataframe
    df_styled = df_display.drop('Portfolio Type', axis=1).copy()
    
    # Display the editable table with custom CSS for row coloring
    st.markdown("""
    <style>
    .portfolio-table {
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the editable table
    edited_df = st.data_editor(
        df_styled,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=400,
        key="global_portfolio_editor"
    )
    
    # Add the Portfolio Type back for processing
    edited_df['Portfolio Type'] = df['Portfolio Type'].values
    edited_df['Available Customers'] = df['Available Customers'].values
    edited_df['AUs Count'] = df['AUs Count'].values
    
    return edited_df

def apply_global_portfolio_changes(edited_summary, original_summary):
    """Apply global changes to all portfolios across all AUs"""
    
    with st.spinner("Applying global portfolio changes..."):
        try:
            # Get current data
            portfolios_created = st.session_state.portfolios_created.copy()
            portfolio_controls = st.session_state.portfolio_controls.copy()
            
            # Get all selected AUs (from branch data where portfolios exist)
            selected_aus = list(portfolios_created.keys())
            
            # Process each portfolio change
            for _, edited_row in edited_summary.iterrows():
                portfolio_id = edited_row['Portfolio ID']
                new_include = edited_row['Include']
                new_select = edited_row['Select']
                
                # Find original values
                original_row = None
                for orig_row in original_summary:
                    if orig_row['Portfolio ID'] == portfolio_id:
                        original_row = orig_row
                        break
                
                if not original_row:
                    continue
                
                original_include = original_row['Include']
                original_select = original_row['Select']
                
                # Apply changes if different
                if new_include != original_include or new_select != original_select:
                    apply_portfolio_change_across_aus(
                        portfolio_id, new_include, new_select, 
                        portfolios_created, portfolio_controls, selected_aus
                    )
            
            # Update the actual customer data in portfolios_created
            update_portfolios_created_data(portfolios_created, portfolio_controls, selected_aus)
            
            # Update session state
            st.session_state.portfolios_created = portfolios_created
            st.session_state.portfolio_controls = portfolio_controls
            
            st.success("Global portfolio changes applied successfully!")
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error applying global changes: {str(e)}")

def apply_portfolio_change_across_aus(portfolio_id, new_include, new_select, portfolios_created, portfolio_controls, selected_aus):
    """Apply portfolio changes across all AUs"""
    
    if not new_include:
        # Exclude portfolio from all AUs
        for au_id in selected_aus:
            if au_id in portfolio_controls:
                control_data = portfolio_controls[au_id]
                for idx, row in control_data.iterrows():
                    if row['Portfolio ID'] == portfolio_id:
                        control_data.at[idx, 'Include'] = False
                        control_data.at[idx, 'Select'] = 0
        return
    
    # Handle select number reduction using distance-based logic
    if new_select > 0:
        apply_distance_based_selection(portfolio_id, new_select, portfolios_created, portfolio_controls, selected_aus)

def update_portfolios_created_data(portfolios_created, portfolio_controls, selected_aus):
    """Update the actual customer data in portfolios_created based on controls"""
    
    # Import the portfolio logic function
    from portfolio_logic import apply_portfolio_selection_changes
    from data_loader import get_merged_data
    
    # Get branch data
    _, _, branch_data, _ = get_merged_data()
    
    # Apply the selection changes to update the actual customer data
    updated_portfolios = apply_portfolio_selection_changes(
        portfolios_created, 
        portfolio_controls, 
        selected_aus, 
        branch_data
    )
    
    # Update portfolios_created with the filtered data
    for au_id in selected_aus:
        if au_id in updated_portfolios:
            portfolios_created[au_id] = updated_portfolios[au_id]
        else:
            # If no customers selected for this AU, create empty dataframe
            portfolios_created[au_id] = portfolios_created[au_id].iloc[0:0]  # Empty with same structure

def apply_distance_based_selection(portfolio_id, target_select, portfolios_created, portfolio_controls, selected_aus):
    """Apply distance-based customer selection across all AUs"""
    
    # Import haversine function from utils
    from utils import haversine_distance
    
    # Step 1: Collect all customers for this portfolio across all AUs
    all_portfolio_customers = []
    
    for au_id in selected_aus:
        if au_id not in portfolios_created:
            continue
            
        au_data = portfolios_created[au_id]
        
        # Get customers for this portfolio
        if portfolio_id == 'UNMANAGED':
            portfolio_customers = au_data[
                (au_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                (au_data['PORT_CODE'].isna())
            ]
        else:
            portfolio_customers = au_data[au_data['PORT_CODE'] == portfolio_id]
        
        for idx, customer in portfolio_customers.iterrows():
            all_portfolio_customers.append({
                'customer_idx': idx,
                'customer_data': customer,
                'current_au': au_id,
                'lat': customer['LAT_NUM'],
                'lon': customer['LON_NUM'],
                'ecn': customer.get('CG_ECN', customer.get('ECN', ''))
            })
    
    if not all_portfolio_customers:
        return
    
    # Step 2: Calculate distance to nearest selected AU for each customer
    from data_loader import get_merged_data
    _, _, branch_data, _ = get_merged_data()
    
    for customer in all_portfolio_customers:
        min_distance = float('inf')
        nearest_au = None
        
        for au_id in selected_aus:
            # Get AU coordinates
            au_row = branch_data[branch_data['AU'] == au_id]
            if au_row.empty:
                continue
                
            au_lat = au_row.iloc[0]['BRANCH_LAT_NUM']
            au_lon = au_row.iloc[0]['BRANCH_LON_NUM']
            
            # Calculate distance
            distance = haversine_distance(customer['lat'], customer['lon'], au_lat, au_lon)
            
            if distance < min_distance:
                min_distance = distance
                nearest_au = au_id
        
        customer['nearest_au'] = nearest_au
        customer['min_distance'] = min_distance
    
    # Step 3: Sort by distance and keep only target number
    all_portfolio_customers.sort(key=lambda x: x['min_distance'])
    selected_customers = all_portfolio_customers[:target_select]
    
    # Step 4: Group selected customers by their nearest AU
    au_customer_groups = {}
    for customer in selected_customers:
        nearest_au = customer['nearest_au']
        if nearest_au not in au_customer_groups:
            au_customer_groups[nearest_au] = []
        au_customer_groups[nearest_au].append(customer)
    
    # Step 5: Update portfolio controls for each AU
    for au_id in selected_aus:
        if au_id not in portfolio_controls:
            continue
            
        control_data = portfolio_controls[au_id]
        customers_for_this_au = au_customer_groups.get(au_id, [])
        
        # Update the select count for this portfolio in this AU
        for idx, row in control_data.iterrows():
            if row['Portfolio ID'] == portfolio_id:
                control_data.at[idx, 'Include'] = True
                control_data.at[idx, 'Select'] = len(customers_for_this_au)
                
                # Update available count to match the customers assigned to this AU
                available_count = len(customers_for_this_au)
                
                # Check if we have the 'Available for this portfolio' column
                if 'Available for this portfolio' in control_data.columns:
                    control_data.at[idx, 'Available for this portfolio'] = available_count
                elif 'Available' in control_data.columns:
                    control_data.at[idx, 'Available'] = available_count
                
                break

def display_portfolio_summary_metrics(aggregated_summary):
    """Display summary metrics"""
    
    df = pd.DataFrame(aggregated_summary)
    
    if not df.empty:
        st.markdown("### 📊 Summary Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_portfolios = len(df)
            included_portfolios = len(df[df['Include'] == True])
            st.metric("Total Portfolios", total_portfolios)
            st.metric("Included Portfolios", included_portfolios)
        
        with col2:
            total_available = df['Available Customers'].sum()
            total_selected = df['Select'].sum()
            st.metric("Available Customers", total_available)
            st.metric("Selected Customers", total_selected)
        
        # Portfolio type breakdown with color indicators
        if 'Portfolio Type' in df.columns:
            st.markdown("### 📈 Portfolio Types")
            type_counts = df['Portfolio Type'].value_counts()
            
            for portfolio_type, count in type_counts.items():
                selected_in_type = df[df['Portfolio Type'] == portfolio_type]['Select'].sum()
                
                # Add color indicator
                color_indicator = ""
                portfolio_type_lower = portfolio_type.lower()
                if 'inmarket' in portfolio_type_lower:
                    color_indicator = "🟢"
                elif 'centralized' in portfolio_type_lower:
                    color_indicator = "🔵"
                elif 'unmanaged' in portfolio_type_lower:
                    color_indicator = "🟡"
                elif 'unassigned' in portfolio_type_lower:
                    color_indicator = "🩷"
                else:
                    color_indicator = "⚪"
                
                st.write(f"{color_indicator} **{portfolio_type}**: {count} portfolios, {selected_in_type:,} customers selected")

# Main function to call from main.py
def render_portfolio_summary_sidebar():
    """Main function to render the portfolio summary sidebar"""
    try:
        create_portfolio_summary_sidebar()
    except Exception as e:
        st.sidebar.error(f"Error loading portfolio summary: {str(e)}")
