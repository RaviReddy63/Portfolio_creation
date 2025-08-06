import streamlit as st
import pandas as pd
from portfolio_logic import filter_customers_for_au, reassign_to_nearest_au, apply_portfolio_selection_changes
from utils import format_number

def create_portfolio_summary(filtered_data, au_id, customer_data):
    """Create portfolio summary for an AU"""
    portfolio_summary = []
    
    # Group by portfolio for assigned customers
    grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
    
    for pid, group in grouped:
        total_customer = len(customer_data[customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})["PORT_CODE"] == pid])
        
        # Determine portfolio type
        portfolio_type = "Unknown"
        if not group.empty:
            types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
            if not types.empty:
                portfolio_type = types.index[0]
        
        portfolio_summary.append({
            'AU': au_id,
            'Portfolio ID': pid,
            'Portfolio Type': portfolio_type,
            'Total Customers': total_customer,
            'Available': len(group),
            'Select': len(group)
        })
    
    # Add unmanaged customers
    unmanaged_customers = filtered_data[
        (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
        (filtered_data['PORT_CODE'].isna())
    ]
    
    if not unmanaged_customers.empty:
        portfolio_summary.append({
            'AU': au_id,
            'Portfolio ID': 'UNMANAGED',
            'Portfolio Type': 'Unmanaged',
            'Total Customers': len(customer_data[
                (customer_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                (customer_data['PORT_CODE'].isna())
            ]),
            'Available': len(unmanaged_customers),
            'Select': len(unmanaged_customers)
        })
    
    return portfolio_summary

def recalculate_portfolio_summaries(portfolios_created, customer_data):
    """Recalculate portfolio summaries after reassignment"""
    portfolio_summaries = {}
    
    # First, calculate totals across all AUs for each portfolio
    all_portfolio_counts = {}
    for au_id, filtered_data in portfolios_created.items():
        # Count regular portfolios
        grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
        for pid, group in grouped:
            if pid not in all_portfolio_counts:
                all_portfolio_counts[pid] = 0
            all_portfolio_counts[pid] += len(group)
        
        # Count unmanaged customers
        unmanaged_customers = filtered_data[
            (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
            (filtered_data['PORT_CODE'].isna())
        ]
        if not unmanaged_customers.empty:
            if 'UNMANAGED' not in all_portfolio_counts:
                all_portfolio_counts['UNMANAGED'] = 0
            all_portfolio_counts['UNMANAGED'] += len(unmanaged_customers)
    
    for au_id, filtered_data in portfolios_created.items():
        portfolio_summary = []
        
        # Group by portfolio for assigned customers
        grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
        
        for pid, group in grouped:
            total_customer = len(customer_data[customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})["PORT_CODE"] == pid])
            
            # Determine portfolio type
            portfolio_type = "Unknown"
            if not group.empty:
                types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
                if not types.empty:
                    portfolio_type = types.index[0]
            
            summary_item = {
                'Include': True,
                'Portfolio ID': pid,
                'Portfolio Type': portfolio_type,
                'Total Customers': total_customer,
                'Available for this portfolio': len(group),
                'Select': len(group)
            }
            
            # Add "Available for all new portfolios" column only if multiple AUs
            if len(portfolios_created) > 1:
                summary_item['Available for all new portfolios'] = all_portfolio_counts.get(pid, 0)
                # Reorder columns
                summary_item = {
                    'Include': True,
                    'Portfolio ID': summary_item['Portfolio ID'],
                    'Portfolio Type': summary_item['Portfolio Type'],
                    'Total Customers': summary_item['Total Customers'],
                    'Available for all new portfolios': summary_item['Available for all new portfolios'],
                    'Available for this portfolio': summary_item['Available for this portfolio'],
                    'Select': summary_item['Select']
                }
            
            portfolio_summary.append(summary_item)
        
        # Add unmanaged customers
        unmanaged_customers = filtered_data[
            (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
            (filtered_data['PORT_CODE'].isna())
        ]
        
        if not unmanaged_customers.empty:
            summary_item = {
                'Include': True,
                'Portfolio ID': 'UNMANAGED',
                'Portfolio Type': 'Unmanaged',
                'Total Customers': len(customer_data[
                    (customer_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                    (customer_data['PORT_CODE'].isna())
                ]),
                'Available for this portfolio': len(unmanaged_customers),
                'Select': len(unmanaged_customers)
            }
            
            # Add "Available for all new portfolios" column only if multiple AUs
            if len(portfolios_created) > 1:
                summary_item['Available for all new portfolios'] = all_portfolio_counts.get('UNMANAGED', 0)
                # Reorder columns
                summary_item = {
                    'Include': True,
                    'Portfolio ID': summary_item['Portfolio ID'],
                    'Portfolio Type': summary_item['Portfolio Type'],
                    'Total Customers': summary_item['Total Customers'],
                    'Available for all new portfolios': summary_item['Available for all new portfolios'],
                    'Available for this portfolio': summary_item['Available for this portfolio'],
                    'Select': summary_item['Select']
                }
            
            portfolio_summary.append(summary_item)
        
        portfolio_summaries[au_id] = portfolio_summary
    
    return portfolio_summaries

def process_portfolio_creation(selected_aus, customer_data, banker_data, branch_data, role, cust_state, cust_portcd, max_dist, min_rev, min_deposit):
    """Main function to process portfolio creation"""
    portfolios_created = {}
    portfolio_summaries = {}
    
    for au_id in selected_aus:
        # Filter customers for this AU
        filtered_data, au_row = filter_customers_for_au(
            customer_data, banker_data, au_id, branch_data, 
            role, cust_state, cust_portcd, max_dist, min_rev, min_deposit
        )
        
        if not filtered_data.empty:
            portfolio_summary = create_portfolio_summary(filtered_data, au_id, customer_data)
            portfolios_created[au_id] = filtered_data
            portfolio_summaries[au_id] = portfolio_summary
    
    if portfolios_created:
        # Apply nearest AU reassignment
        with st.spinner("Reassigning customers to nearest AUs..."):
            portfolios_created, reassignment_summary = reassign_to_nearest_au(
                portfolios_created, selected_aus, branch_data
            )
        
        total_portfolios = len(portfolios_created)
        st.success(f"Portfolios created for {format_number(total_portfolios)} AUs")
        
        # Recalculate portfolio summaries after reassignment
        portfolio_summaries = recalculate_portfolio_summaries(portfolios_created, customer_data)
        
        return portfolios_created, portfolio_summaries
    else:
        st.warning("No customers found for the selected AUs with current filters.")
        return None, None

def apply_portfolio_changes(au_id, branch_data):
    """Apply portfolio selection changes for a specific AU"""
    with st.spinner("Applying selection changes..."):
        if 'portfolios_created' in st.session_state and au_id in st.session_state.portfolios_created:
            updated_portfolios = apply_portfolio_selection_changes(
                st.session_state.portfolios_created, 
                st.session_state.portfolio_controls, 
                [au_id], 
                branch_data
            )
            
            # Update only this AU's portfolio
            if au_id in updated_portfolios:
                st.session_state.portfolios_created[au_id] = updated_portfolios[au_id]
            
            st.success("Portfolio selection updated!")
