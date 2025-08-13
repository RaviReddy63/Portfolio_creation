import math
import pandas as pd
import numpy as np
from math import sin, cos, atan2, radians, sqrt
from io import BytesIO

def haversine_distance(clat, clon, blat, blon):
    """Calculate distance between two points using Haversine formula"""
    if math.isnan(clat) or math.isnan(clon) or math.isnan(blat) or math.isnan(blon):
        return 0
        
    delta_lat = radians(clat - blat)
    delta_lon = radians(clon - blon)
    
    a = sin(delta_lat/2)**2 + cos(radians(clat))*cos(radians(blat))*sin(delta_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    distance = 3959*c  # Earth's radius in miles
    return distance

def merge_dfs(customer_data, banker_data, branch_data):
    """Merge customer, banker, and branch dataframes"""
    customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
    final_table = customer_data.merge(banker_data, on="PORT_CODE", how="left")
    final_table.fillna(0, inplace=True)
    return final_table

def to_excel(all_portfolios):
    """Export portfolios to Excel format"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for portfolio_id, df in all_portfolios.items():
            sheet_name = f"Portfolio_{portfolio_id}"[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

def create_distance_circle(center_lat, center_lon, radius_miles, num_points=100):
    """Create points for a circle around a center point"""
    angles = np.linspace(0, 2*np.pi, num_points)
    circle_lats = []
    circle_lons = []
    
    for angle in angles:
        # Convert miles to degrees (rough approximation)
        lat_offset = radius_miles / 69.0  # 1 degree lat ≈ 69 miles
        lon_offset = radius_miles / (69.0 * math.cos(math.radians(center_lat)))
        
        lat = center_lat + lat_offset * math.cos(angle)
        lon = center_lon + lon_offset * math.sin(angle)
        
        circle_lats.append(lat)
        circle_lons.append(lon)
    
    # Close the circle
    circle_lats.append(circle_lats[0])
    circle_lons.append(circle_lons[0])
    
    return circle_lats, circle_lons

# NEW DEDUPLICATION FUNCTIONS

def remove_customer_duplicates(df, priority_columns=None):
    """
    Remove duplicate customers based on CG_ECN with intelligent priority handling
    
    Args:
        df: DataFrame with customer data
        priority_columns: List of columns to use for prioritizing which duplicate to keep
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty or 'CG_ECN' not in df.columns:
        return df
    
    # Default priority: In-Market > Centralized > Assigned > Unmanaged > Unassigned
    type_priority = {
        'in-market': 1,
        'inmarket': 1, 
        'centralized': 2,
        'assigned': 3,
        'managed': 3,
        'unmanaged': 4,
        'unassigned': 5
    }
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Add priority score for TYPE
    df_clean['_type_priority'] = df_clean['TYPE'].fillna('unassigned').str.lower().str.strip().map(type_priority).fillna(6)
    
    # Add priority score for having valid portfolio code
    df_clean['_portfolio_priority'] = df_clean.get('PORT_CODE', df_clean.get('CG_PORTFOLIO_CD', pd.Series([None]*len(df_clean)))).notna().astype(int)
    
    # Add priority score for distance (lower distance = higher priority)
    if 'Distance' in df_clean.columns:
        df_clean['_distance_priority'] = df_clean['Distance'].fillna(999999)
    elif 'DISTANCE_TO_AU' in df_clean.columns:
        df_clean['_distance_priority'] = df_clean['DISTANCE_TO_AU'].fillna(999999)
    else:
        df_clean['_distance_priority'] = 999999
    
    # Sort by priority (lower numbers = higher priority)
    df_clean = df_clean.sort_values([
        '_type_priority',           # TYPE priority (1=best)
        '_portfolio_priority',      # Has portfolio code (1=yes, 0=no) 
        '_distance_priority',       # Distance (lower=better)
        'BANK_REVENUE'             # Revenue (higher=better)
    ], ascending=[True, False, True, False])
    
    # Keep first occurrence of each CG_ECN (highest priority)
    df_deduplicated = df_clean.drop_duplicates(subset=['CG_ECN'], keep='first')
    
    # Remove temporary priority columns
    priority_cols = ['_type_priority', '_portfolio_priority', '_distance_priority']
    df_deduplicated = df_deduplicated.drop(columns=[col for col in priority_cols if col in df_deduplicated.columns])
    
    return df_deduplicated.reset_index(drop=True)

def clean_portfolio_data(df):
    """
    Comprehensive cleaning of portfolio data to remove all types of duplicates
    """
    if df.empty:
        return df
    
    print(f"Original data: {len(df)} rows")
    
    # Step 1: Remove exact duplicates
    df_clean = df.drop_duplicates()
    print(f"After removing exact duplicates: {len(df_clean)} rows")
    
    # Step 2: Remove customer duplicates with priority
    df_clean = remove_customer_duplicates(df_clean)
    print(f"After removing customer duplicates: {len(df_clean)} rows")
    
    # Step 3: Additional validation - check for any remaining CG_ECN duplicates
    if 'CG_ECN' in df_clean.columns:
        duplicates = df_clean[df_clean.duplicated(subset=['CG_ECN'], keep=False)]
        if not duplicates.empty:
            print(f"Warning: {len(duplicates)} rows still have duplicate CG_ECNs")
            print("Duplicate ECNs:", duplicates['CG_ECN'].unique().tolist())
            # Force removal by keeping first
            df_clean = df_clean.drop_duplicates(subset=['CG_ECN'], keep='first')
            print(f"After force deduplication: {len(df_clean)} rows")
    
    # Check for ECN column (for smart portfolio results)
    if 'ECN' in df_clean.columns:
        duplicates = df_clean[df_clean.duplicated(subset=['ECN'], keep=False)]
        if not duplicates.empty:
            print(f"Warning: {len(duplicates)} rows still have duplicate ECNs")
            print("Duplicate ECNs:", duplicates['ECN'].unique().tolist())
            # Force removal by keeping first
            df_clean = df_clean.drop_duplicates(subset=['ECN'], keep='first')
            print(f"After force deduplication: {len(df_clean)} rows")
    
    return df_clean

def validate_no_duplicates(df, identifier_col='CG_ECN'):
    """
    Validate that there are no duplicates in the final dataset
    """
    if df.empty or identifier_col not in df.columns:
        return True, []
    
    duplicates = df[df.duplicated(subset=[identifier_col], keep=False)]
    
    if duplicates.empty:
        return True, []
    else:
        duplicate_ids = duplicates[identifier_col].unique().tolist()
        return False, duplicate_ids

def enhanced_customer_au_assignment_with_two_inmarket_iterations_deduplicated(customer_df, branch_df):
    """
    Enhanced version with comprehensive deduplication
    """
    # Clean input data first
    print("Cleaning input customer data...")
    customer_df_clean = clean_portfolio_data(customer_df)
    
    # Run the original algorithm
    from portfolio_creation_8 import enhanced_customer_au_assignment_with_two_inmarket_iterations
    result_df = enhanced_customer_au_assignment_with_two_inmarket_iterations(customer_df_clean, branch_df)
    
    # Clean the output
    print("Cleaning output results...")
    result_df_clean = clean_portfolio_data(result_df)
    
    # Final validation
    is_clean, duplicate_ids = validate_no_duplicates(result_df_clean, 'ECN')
    if not is_clean:
        print(f"Warning: Final result still contains duplicates for ECNs: {duplicate_ids}")
        # One more force clean
        result_df_clean = result_df_clean.drop_duplicates(subset=['ECN'], keep='first')
    
    print(f"Final clean result: {len(result_df_clean)} rows")
    return result_df_clean

def prepare_portfolio_for_export_deduplicated(au_data, customer_data, branch_data):
    """
    Enhanced version of prepare_portfolio_for_export with deduplication
    """
    if au_data.empty or customer_data.empty:
        return pd.DataFrame()
    
    print(f"Preparing export for {len(au_data)} customers...")
    
    # Clean input data
    au_data_clean = clean_portfolio_data(au_data)
    customer_data_clean = clean_portfolio_data(customer_data)
    
    # Merge with customer_data to get all required fields
    export_data = au_data_clean.merge(
        customer_data_clean[['CG_ECN', 'CG_PORTFOLIO_CD', 'TYPE', 'LAT_NUM', 'LON_NUM', 
                      'BILLINGCITY', 'BILLINGSTATE', 'BANK_REVENUE', 'CG_GROSS_SALES', 
                      'DEPOSIT_BAL', 'BANKER_FIRSTNAME', 'BANKER_LASTNAME', 
                      'BILLINGSTREET', 'CG_NAME']],
        on='CG_ECN',
        how='left',
        suffixes=('', '_orig')
    )
    
    # Remove duplicates after merge
    export_data = clean_portfolio_data(export_data)
    
    # Get AU information from branch_data
    au_id = au_data_clean['AU'].iloc[0]
    au_info = branch_data[branch_data['AU'] == au_id]
    
    if not au_info.empty:
        branch_lat = au_info.iloc[0]['BRANCH_LAT_NUM']
        branch_lon = au_info.iloc[0]['BRANCH_LON_NUM']
    else:
        branch_lat = au_data_clean['BRANCH_LAT_NUM'].iloc[0] if 'BRANCH_LAT_NUM' in au_data_clean.columns else 0
        branch_lon = au_data_clean['BRANCH_LON_NUM'].iloc[0] if 'BRANCH_LON_NUM' in au_data_clean.columns else 0
    
    # Calculate distance from customer to new AU
    export_data['DISTANCE'] = export_data.apply(
        lambda row: haversine_distance(
            row['LAT_NUM'], row['LON_NUM'], 
            branch_lat, branch_lon
        ), axis=1
    )
    
    # Prepare final export format
    final_export = pd.DataFrame({
        'CG_ECN': export_data['CG_ECN'],
        'CG_PORTFOLIO_CD': export_data['CG_PORTFOLIO_CD'],
        'TYPE': export_data['TYPE'],
        'LAT_NUM': export_data['LAT_NUM'],
        'LON_NUM': export_data['LON_NUM'],
        'BILLINGCITY': export_data['BILLINGCITY'],
        'BILLINGSTATE': export_data['BILLINGSTATE'],
        'DISTANCE': export_data['DISTANCE'],
        'AU_NBR': au_id,
        'BRANCH_LAT_NUM': branch_lat,
        'BRANCH_LON_NUM': branch_lon,
        'BANK_REVENUE': export_data['BANK_REVENUE'],
        'CG_GROSS_SALES': export_data['CG_GROSS_SALES'],
        'DEPOSIT_BAL': export_data['DEPOSIT_BAL'],
        'CURRENT_BANKER_FIRSTNAME': export_data['BANKER_FIRSTNAME'],
        'CURRENT_BANKER_LASTNAME': export_data['BANKER_LASTNAME'],
        'NAME': export_data['CG_NAME'],
        'BILLINGSTREET': export_data['BILLINGSTREET'],
        'BILLINGCITY': export_data['BILLINGCITY']
    })
    
    # Final deduplication
    final_export_clean = clean_portfolio_data(final_export)
    
    print(f"Export prepared: {len(final_export_clean)} customers")
    return final_export_clean
