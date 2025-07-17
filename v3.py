import streamlit as st
import pandas as pd

# Custom CSS for the +/- buttons
st.markdown("""
<style>
    .quantity-selector {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0px;
        margin: 0;
    }

    .qty-btn {
        width: 28px;
        height: 28px;
        border: 1px solid #d1d5db;
        background-color: #f9fafb;
        color: #374151;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.2s ease;
        user-select: none;
    }

    .qty-btn:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
    }

    .qty-btn:active {
        background-color: #d1d5db;
        transform: scale(0.95);
    }

    .qty-btn.minus {
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
        border-right: none;
    }

    .qty-btn.plus {
        border-top-left-radius: 0;
        border-bottom-left-radius: 0;
        border-left: none;
    }

    .qty-input {
        width: 50px;
        height: 28px;
        border: 1px solid #d1d5db;
        border-left: none;
        border-right: none;
        text-align: center;
        font-size: 14px;
        background-color: white;
        outline: none;
        border-radius: 0;
    }

    .qty-input:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 0 1px #ff4b4b;
    }

    /* Hide default number input arrows */
    .qty-input::-webkit-outer-spin-button,
    .qty-input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    .qty-input[type=number] {
        -moz-appearance: textfield;
    }
</style>
""", unsafe_allow_html=True)

def create_quantity_selector(portfolio_id, current_value, max_value, form_id):
    """Create a quantity selector with +/- buttons"""
    
    # Create unique keys for each portfolio
    minus_key = f"minus_{portfolio_id}_{form_id}"
    plus_key = f"plus_{portfolio_id}_{form_id}"
    
    # Create columns for the selector
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Minus button
        minus_clicked = st.button(
            "−", 
            key=minus_key,
            help="Decrease quantity",
            use_container_width=True
        )
    
    with col2:
        # Number input (display only)
        st.markdown(
            f'<div style="text-align: center; padding: 4px; border: 1px solid #d1d5db; background: white; border-radius: 4px;">{current_value}</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        # Plus button
        plus_clicked = st.button(
            "+", 
            key=plus_key,
            help="Increase quantity",
            use_container_width=True
        )
    
    # Handle button clicks
    if minus_clicked and current_value > 0:
        return current_value - 1
    elif plus_clicked and current_value < max_value:
        return current_value + 1
    else:
        return current_value

def create_portfolio_summary_table(portfolio_data, form_id):
    """Create the portfolio summary table with +/- selectors"""
    
    # Initialize session state for this form if not exists
    if f'portfolio_selections_{form_id}' not in st.session_state:
        st.session_state[f'portfolio_selections_{form_id}'] = {}
        # Initialize with default values (all available)
        for _, row in portfolio_data.iterrows():
            st.session_state[f'portfolio_selections_{form_id}'][row['Portfolio ID']] = row['Available']
    
    st.subheader("Portfolio Summary & Customer Selection")
    
    # Create the table headers
    col_headers = st.columns([2, 2, 1.5, 1.5, 2, 2, 1.5])
    headers = ['Portfolio ID', 'Portfolio Type', 'Total Customers', 'Available', 'Total Revenue', 'Total Deposits', 'Select']
    
    for i, header in enumerate(headers):
        with col_headers[i]:
            st.markdown(f"**{header}**")
    
    st.divider()
    
    # Create rows for each portfolio
    for _, row in portfolio_data.iterrows():
        portfolio_id = row['Portfolio ID']
        max_available = row['Available']
        
        # Get current selection from session state
        current_selection = st.session_state[f'portfolio_selections_{form_id}'].get(portfolio_id, max_available)
        
        # Create columns for this row
        cols = st.columns([2, 2, 1.5, 1.5, 2, 2, 1.5])
        
        with cols[0]:
            st.write(portfolio_id)
        with cols[1]:
            st.write(row['Portfolio Type'])
        with cols[2]:
            st.write(row['Total Customers'])
        with cols[3]:
            st.write(max_available)
        with cols[4]:
            st.write(row['Total Revenue'])
        with cols[5]:
            st.write(row['Total Deposits'])
        with cols[6]:
            # Create the quantity selector
            new_selection = create_quantity_selector(
                portfolio_id, 
                current_selection, 
                max_available, 
                form_id
            )
            
            # Update session state if value changed
            if new_selection != current_selection:
                st.session_state[f'portfolio_selections_{form_id}'][portfolio_id] = new_selection
                st.rerun()
    
    # Return the current selections
    return st.session_state[f'portfolio_selections_{form_id}']

# Alternative implementation using HTML/JavaScript (more visually appealing)
def create_advanced_portfolio_table(portfolio_data, form_id):
    """Create portfolio table with HTML-based +/- buttons"""
    
    # Initialize session state
    if f'portfolio_selections_{form_id}' not in st.session_state:
        st.session_state[f'portfolio_selections_{form_id}'] = {}
        for _, row in portfolio_data.iterrows():
            st.session_state[f'portfolio_selections_{form_id}'][row['Portfolio ID']] = row['Available']
    
    st.subheader("Portfolio Summary & Customer Selection")
    
    # Create the HTML table
    table_html = """
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <thead>
            <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Portfolio ID</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Portfolio Type</th>
                <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">Total Customers</th>
                <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">Available</th>
                <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Total Revenue</th>
                <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Total Deposits</th>
                <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">Select</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in portfolio_data.iterrows():
        portfolio_id = row['Portfolio ID']
        current_selection = st.session_state[f'portfolio_selections_{form_id}'].get(portfolio_id, row['Available'])
        
        table_html += f"""
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px; border: 1px solid #dee2e6;">{portfolio_id}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6;">{row['Portfolio Type']}</td>
                <td style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">{row['Total Customers']}</td>
                <td style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">{row['Available']}</td>
                <td style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">{row['Total Revenue']}</td>
                <td style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">{row['Total Deposits']}</td>
                <td style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">
                    <div class="quantity-selector">
                        <button class="qty-btn minus" onclick="updateQuantity('{portfolio_id}', {form_id}, -1, {row['Available']})">−</button>
                        <input type="number" class="qty-input" id="{portfolio_id}_{form_id}" value="{current_selection}" min="0" max="{row['Available']}" readonly>
                        <button class="qty-btn plus" onclick="updateQuantity('{portfolio_id}', {form_id}, 1, {row['Available']})">+</button>
                    </div>
                </td>
            </tr>
        """
    
    table_html += """
        </tbody>
    </table>
    
    <script>
        function updateQuantity(portfolioId, formId, change, maxValue) {
            const inputId = portfolioId + '_' + formId;
            const input = document.getElementById(inputId);
            const currentValue = parseInt(input.value);
            const newValue = currentValue + change;
            
            if (newValue >= 0 && newValue <= maxValue) {
                input.value = newValue;
                // You would need to implement a callback to update Streamlit session state
                // This would require additional JavaScript-Python communication
            }
        }
    </script>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Create individual controls below the table for actual functionality
    st.markdown("**Adjust Selections:**")
    
    updated_selections = {}
    cols = st.columns(len(portfolio_data))
    
    for i, (_, row) in enumerate(portfolio_data.iterrows()):
        portfolio_id = row['Portfolio ID']
        current_selection = st.session_state[f'portfolio_selections_{form_id}'].get(portfolio_id, row['Available'])
        
        with cols[i]:
            new_value = st.number_input(
                f"{portfolio_id}",
                min_value=0,
                max_value=row['Available'],
                value=current_selection,
                key=f"input_{portfolio_id}_{form_id}"
            )
            updated_selections[portfolio_id] = new_value
    
    # Update session state
    st.session_state[f'portfolio_selections_{form_id}'] = updated_selections
    
    return updated_selections

# Main function to integrate into your existing code
def data_filteration_with_quantity_selector(customer_data, branch_data, banker_data, form_id):
    """Enhanced data_filteration function with quantity selector"""
    
    st.subheader(f"Form {form_id}")
    
    # ... (your existing AU selection and customer filtering code remains the same) ...
    
    # After filtering, create the portfolio summary
    if not filtered_data.empty:
        # Create portfolio summary data
        portfolio_summary = []
        
        # Group by portfolio for assigned customers
        grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
        
        for pid, group in grouped:
            total_customer = len(customer_data[customer_data["PORT_CODE"] == pid])
            
            # Determine portfolio type
            portfolio_type = "Unknown"
            if not group.empty:
                types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
                if not types.empty:
                    portfolio_type = types.index[0]
            
            portfolio_summary.append({
                'Portfolio ID': pid,
                'Portfolio Type': portfolio_type,
                'Total Customers': total_customer,
                'Available': len(group),
                'Total Revenue': f"${group['BANK_REVENUE'].sum():,.0f}",
                'Total Deposits': f"${group['DEPOSIT_BAL'].sum():,.0f}"
            })
        
        # Add unmanaged customers
        unmanaged_customers = filtered_data[
            (filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
            (filtered_data['PORT_CODE'].isna())
        ]
        
        if not unmanaged_customers.empty:
            portfolio_summary.append({
                'Portfolio ID': 'UNMANAGED',
                'Portfolio Type': 'Unmanaged',
                'Total Customers': len(customer_data[
                    (customer_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                    (customer_data['PORT_CODE'].isna())
                ]),
                'Available': len(unmanaged_customers),
                'Total Revenue': f"${unmanaged_customers['BANK_REVENUE'].sum():,.0f}",
                'Total Deposits': f"${unmanaged_customers['DEPOSIT_BAL'].sum():,.0f}"
            })
        
        # Create the portfolio table with quantity selectors
        portfolio_df = pd.DataFrame(portfolio_summary)
        selections = create_portfolio_summary_table(portfolio_df, form_id)
        
        # Display current selections summary
        st.markdown("**Current Selections Summary:**")
        total_selected = sum(selections.values())
        st.metric("Total Selected Customers", total_selected)
        
        # Show breakdown
        for portfolio_id, count in selections.items():
            if count > 0:
                st.write(f"• {portfolio_id}: {count} customers")
    
    # ... (rest of your existing code remains the same) ...
    
    return filtered_data, selections

# Example usage in your main app
def main():
    st.title("Portfolio Creation Tool")
    
    # Example portfolio data
    portfolio_data = pd.DataFrame([
        {
            'Portfolio ID': 'PORT_001',
            'Portfolio Type': 'Commercial',
            'Total Customers': 45,
            'Available': 23,
            'Total Revenue': '$2,450,000',
            'Total Deposits': '$8,900,000'
        },
        {
            'Portfolio ID': 'PORT_002',
            'Portfolio Type': 'Private Banking',
            'Total Customers': 28,
            'Available': 18,
            'Total Revenue': '$1,850,000',
            'Total Deposits': '$12,300,000'
        },
        {
            'Portfolio ID': 'UNMANAGED',
            'Portfolio Type': 'Unmanaged',
            'Total Customers': 15,
            'Available': 8,
            'Total Revenue': '$680,000',
            'Total Deposits': '$2,100,000'
        }
    ])
    
    # Demo the quantity selector
    form_id = 1
    selections = create_portfolio_summary_table(portfolio_data, form_id)
    
    st.markdown("---")
    st.markdown("**Selected Quantities:**")
    st.json(selections)

if __name__ == "__main__":
    main()
