import streamlit as st
import pandas as pd

# Import custom modules
from ui_components import (
    setup_page_config, add_logo, create_header, initialize_session_state,
    create_au_filters, create_customer_filters, create_portfolio_button,
    display_summary_statistics, create_portfolio_editor, create_apply_changes_button
)
from data_loader import get_merged_data
from portfolio_creation import process_portfolio_creation, apply_portfolio_changes
from map_visualization import create_combined_map

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
        portfolio_mapping_page(data)

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
    
    # Show recommendation reassignment table if it exists
    display_reassignment_recommendations()

def display_portfolio_tables(portfolios_created, portfolio_summaries, branch_data):
    """Display portfolio summary tables"""
    if len(portfolios_created) > 1:
        # Multiple AU case - use tabs
        au_tabs = st.tabs([f"AU {au_id}" for au_id in portfolios_created.keys()])
        
        for tab_idx, (au_id, tab) in enumerate(zip(portfolios_created.keys(), au_tabs)):
            with tab:
                display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, True)
    else:
        # Single AU case
        au_id = list(portfolios_created.keys())[0]
        display_single_au_table(au_id, portfolio_summaries, portfolios_created, branch_data, False)

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

def display_reassignment_recommendations():
    """Display recommendation reassignment table if it exists"""
    if ('recommend_reassignment' in st.session_state and 
        isinstance(st.session_state.recommend_reassignment, pd.DataFrame) and 
        not st.session_state.recommend_reassignment.empty):
        st.markdown("----")
        st.subheader("Recommended Reassignments")
        st.dataframe(st.session_state.recommend_reassignment, use_container_width=True)

def portfolio_mapping_page(data):
    """Portfolio Mapping page logic"""
    st.subheader("Portfolio Mapping")
    
    # Portfolio mapping functionality
    st.info("Portfolio Mapping functionality coming soon...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution by Type")
        if not data.empty and 'TYPE' in data.columns:
            type_counts = data['TYPE'].value_counts()
            st.bar_chart(type_counts)
    
    with col2:
        st.subheader("Customer Distribution by State")
        if not data.empty and 'BILLINGSTATE' in data.columns:
            state_counts = data['BILLINGSTATE'].value_counts().head(10)
            st.bar_chart(state_counts)
    
    if not data.empty:
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(data))
        with col2:
            if 'BANK_REVENUE' in data.columns:
                st.metric("Total Revenue", f"${data['BANK_REVENUE'].sum():,.0f}")
        with col3:
            if 'DEPOSIT_BAL' in data.columns:
                st.metric("Total Deposits", f"${data['DEPOSIT_BAL'].sum():,.0f}")
        with col4:
            if 'PORT_CODE' in data.columns:
                st.metric("Unique Portfolios", data['PORT_CODE'].nunique())

if __name__ == "__main__":
    main()
