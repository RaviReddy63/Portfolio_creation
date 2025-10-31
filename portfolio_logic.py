import pandas as pd
import numpy as np
from utils import haversine_distance, clean_portfolio_data, remove_customer_duplicates, validate_no_duplicates

def filter_customers_for_au(customer_data, banker_data, selected_au, branch_data, role, cust_state, cust_portcd, cs_new_ns, max_dist, min_rev, min_deposit):
    """Filter customers for a specific AU based on criteria with deduplication"""
    
    # Clean input data first
    customer_data = clean_portfolio_data(customer_data)
    banker_data = clean_portfolio_data(banker_data)
    
    # Get AU data
    AU_row = branch_data[branch_data['AU'] == int(selected_au)].iloc[0]
    AU_lat = AU_row['BRANCH_LAT_NUM']
    AU_lon = AU_row['BRANCH_LON_NUM']
    
    # Filter customers by distance box (convert miles to degrees)
    box_lat = max_dist/69  # 1 degree lat ≈ 69 miles
    box_lon = max_dist/ (69 * np.cos(np.radians(AU_lat)))
    
    customer_data_boxed = customer_data[(customer_data['LAT_NUM'] >= AU_lat - box_lat) &
                                        (customer_data['LAT_NUM'] <= AU_lat + box_lat) &
                                        (customer_data['LON_NUM'] <= AU_lon + box_lon) &
                                        (customer_data['LON_NUM'] >= AU_lon - box_lon)]
    
    # Calculate distances
    customer_data_boxed['Distance'] = customer_data_boxed.apply(
        lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], AU_lat, AU_lon), axis=1
    )
    
    # Ensure PORT_CODE column exists
    if 'PORT_CODE' not in customer_data_boxed.columns:
        customer_data_boxed = customer_data_boxed.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    
    filtered_data = customer_data_boxed.merge(banker_data, on="PORT_CODE", how='left')
    
    # Clean after merge
    filtered_data = clean_portfolio_data(filtered_data)
    
    # Apply distance filter for all roles except CENTRALIZED
    if role is None or (role is not None and not any(r.lower().strip() == 'centralized' for r in role)):
        filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
    
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
            portfolio_condition = filtered_data['PORT_CODE'].isin(cust_portcd)
        
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
    
    # Add AU information
    filtered_data['AU'] = selected_au
    filtered_data['BRANCH_LAT_NUM'] = AU_lat
    filtered_data['BRANCH_LON_NUM'] = AU_lon
    
    # Final cleaning before return
    filtered_data = clean_portfolio_data(filtered_data)
    
    return filtered_data, AU_row

def apply_portfolio_selection_changes(portfolios_created, portfolio_controls, selected_aus, branch_data):
    """Apply the selection changes from portfolio controls to filter customers with deduplication"""
    
    updated_portfolios = {}
    
    for au_id in selected_aus:
        if au_id not in portfolios_created or au_id not in portfolio_controls:
            continue
            
        original_data = portfolios_created[au_id].copy()
        original_data = clean_portfolio_data(original_data)  # Clean input
        
        control_data = portfolio_controls[au_id]
        
        # Start with empty list for this AU
        selected_customers = []
        
        # Process each portfolio selection
        for _, row in control_data.iterrows():
            portfolio_id = row['Portfolio ID']
            select_count = row['Select']
            include = row.get('Include', True)  # Default to True (included)
            
            # Only include portfolios that are checked (include=True) and have select_count > 0
            if not include or select_count <= 0:
                continue
                
            if portfolio_id == 'UNMANAGED':
                # Handle unmanaged customers
                unmanaged_customers = original_data[
                    (original_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
                    (original_data['PORT_CODE'].isna())
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
            au_final_customers = pd.concat(selected_customers, ignore_index=True)
            # Final cleaning
            au_final_customers = clean_portfolio_data(au_final_customers)
            updated_portfolios[au_id] = au_final_customers
        else:
            # No customers selected for this AU
            updated_portfolios[au_id] = pd.DataFrame()
    
    return updated_portfolios

def reassign_to_nearest_au(portfolios_created, selected_aus, branch_data):
    """Reassign customers to their nearest AU from the selected AUs with deduplication"""
    
    if not portfolios_created or not selected_aus:
        return portfolios_created, {}
    
    # Combine all customers from all portfolios
    all_customers = []
    for au_id, customers_df in portfolios_created.items():
        if not customers_df.empty:
            # Clean before processing
            customers_df = clean_portfolio_data(customers_df)
            customers_copy = customers_df.copy()
            customers_copy['original_au'] = au_id
            all_customers.append(customers_copy)
    
    if not all_customers:
        return portfolios_created, {}
    
    combined_customers = pd.concat(all_customers, ignore_index=True)
    # Clean combined customers
    combined_customers = clean_portfolio_data(combined_customers)
    
    # Get AU coordinates for selected AUs
    au_coordinates = {}
    for au_id in selected_aus:
        au_row = branch_data[branch_data['AU'] == au_id]
        if not au_row.empty:
            au_coordinates[au_id] = (au_row.iloc[0]['BRANCH_LAT_NUM'], au_row.iloc[0]['BRANCH_LON_NUM'])
    
    # Reassign each customer to nearest AU
    reassignment_summary = []
    
    for idx, customer in combined_customers.iterrows():
        if pd.isna(customer['LAT_NUM']) or pd.isna(customer['LON_NUM']):
            continue
            
        nearest_au = None
        min_distance = float('inf')
        
        # Check distance to all selected AUs
        for au_id, (au_lat, au_lon) in au_coordinates.items():
            distance = haversine_distance(customer['LAT_NUM'], customer['LON_NUM'], au_lat, au_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_au = au_id
        
        # Update customer data
        original_au = customer['original_au']
        if nearest_au and nearest_au != original_au:
            # Update AU information
            combined_customers.at[idx, 'AU'] = nearest_au
            combined_customers.at[idx, 'BRANCH_LAT_NUM'] = au_coordinates[nearest_au][0]
            combined_customers.at[idx, 'BRANCH_LON_NUM'] = au_coordinates[nearest_au][1]
            combined_customers.at[idx, 'Distance'] = min_distance
            
            # Track reassignment
            reassignment_summary.append({
                'Customer': customer.get('CG_ECN', 'N/A'),
                'Original AU': original_au,
                'New AU': nearest_au,
                'New Distance': min_distance,
                'Portfolio': customer.get('PORT_CODE', 'N/A'),
                'Revenue': customer.get('BANK_REVENUE', 0)
            })
        elif nearest_au:
            # Update distance even if AU doesn't change
            combined_customers.at[idx, 'Distance'] = min_distance
    
    # Rebuild portfolios_created with reassigned customers
    new_portfolios_created = {}
    for au_id in selected_aus:
        au_customers = combined_customers[combined_customers['AU'] == au_id]
        if not au_customers.empty:
            # Remove the temporary 'original_au' column and clean
            au_customers = au_customers.drop('original_au', axis=1)
            au_customers = clean_portfolio_data(au_customers)
            new_portfolios_created[au_id] = au_customers.reset_index(drop=True)
    
    return new_portfolios_created, reassignment_summary

def recommend_reassignment(all_portfolios):
    """Recommend reassignment of customers to nearest portfolios"""
    if not all_portfolios:
        return pd.DataFrame()
    
    combine_df = pd.concat([df.assign(original_portfolio=portfolio_id) for portfolio_id, df in all_portfolios.items()], ignore_index=True)
    
    # Clean combined data
    combine_df = clean_portfolio_data(combine_df)
    
    au_map = {portfolio_id: (df["BRANCH_LAT_NUM"].iloc[0], df["BRANCH_LON_NUM"].iloc[0])
              for portfolio_id, df in all_portfolios.items()
              if not df.empty}
                
    records = []
    for _, row in combine_df.iterrows():
        best_portfolio = None
        min_dist = float("inf")
        for portfolio_id, (au_lat, au_lon) in au_map.items():
            dist = haversine_distance(row['LAT_NUM'], row['LON_NUM'], au_lat, au_lon)
            if dist < min_dist:
                best_portfolio = portfolio_id
                min_dist = dist
                
        row_data = row.to_dict()
        row_data['recommended_portfolio'] = best_portfolio
        row_data['recommended_dist'] = min_dist
        records.append(row_data)
    
    result_df = pd.DataFrame(records)
    
    # Clean final result
    result_df = clean_portfolio_data(result_df)
    
    return result_df
