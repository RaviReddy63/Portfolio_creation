import os
import re
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows

# ==================== CONFIGURATION ====================

OUTPUT_DIR = 'output'

CURRENT_COLS  = ['EMPLOYEE_NAME', 'MANAGER_NAME', 'DIRECTOR_NAME',
                 'PORT_CODE', 'CG_ECN', 'RC_ECN', 'SEGMENT', 'NAME']

FUTURE_COLS   = ['EMPLOYEE_NAME', 'MANAGER_NAME', 'DIRECTOR_NAME',
                 'PORT_CODE', 'HH_ECN', 'RC_ECN', 'SEGMENT', 'NAME']

ADDED_COLS    = ['EMPLOYEE_NAME', 'MANAGER_NAME', 'DIRECTOR_NAME',
                 'PORT_CODE', 'HH_ECN', 'RC_ECN', 'SEGMENT', 'NAME']

REMOVED_COLS  = ['EMPLOYEE_NAME', 'MANAGER_NAME', 'DIRECTOR_NAME',
                 'PORT_CODE', 'CG_ECN', 'RC_ECN', 'SEGMENT', 'NAME', 'DO_NOT_REMOVE']


# ==================== UTILITIES ====================

def sanitize(name):
    """Remove characters not allowed in folder/file names."""
    return re.sub(r'[\\/*?:"<>|]', '_', str(name)).strip()


def write_sheet(wb, sheet_name, df, columns):
    """Write a dataframe to a worksheet with bold headers."""
    ws = wb.create_sheet(title=sheet_name)
    ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    if not df.empty:
        for row in dataframe_to_rows(df[columns], index=False, header=False):
            ws.append(row)


# ==================== DELTA COMPUTATION ====================

def compute_added(current_df, future_df, port_code):
    """RC_ECNs in future portfolio not present in current portfolio for same port code."""
    curr = current_df[current_df['PORT_CODE'] == port_code]['RC_ECN'].dropna().unique()
    fut  = future_df[future_df['PORT_CODE'] == port_code].copy()
    added = fut[~fut['RC_ECN'].isin(curr)].copy()
    return added


def compute_removed(current_df, future_df, port_code):
    """RC_ECNs in current portfolio not present in future portfolio for same port code."""
    fut  = future_df[future_df['PORT_CODE'] == port_code]['RC_ECN'].dropna().unique()
    curr = current_df[current_df['PORT_CODE'] == port_code].copy()
    removed = curr[~curr['RC_ECN'].isin(fut)].copy()
    removed['DO_NOT_REMOVE'] = ''   # Empty column for banker feedback
    return removed


# ==================== MAIN ====================

def generate_portfolio_excels(current_portfolio_data, new_portfolio_data, portfolio_data):
    """
    Generate one Excel file per portfolio code under:
        output / Director / Manager / Banker / PORT_CODE.xlsx

    Args:
        current_portfolio_data : DataFrame — Q1_MOVE_CURRENT_PORTFOLIO
        new_portfolio_data     : DataFrame — Q1_MOVE_NEW_PORTFOLIO
        portfolio_data         : DataFrame — SBRM_PORTFOLIO_DATA
                                 (columns: PORT_CODE, EMPLOYEE_NAME, MANAGER_NAME, DIRECTOR_NAME)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Rename RELATED_ECN → RC_ECN in new portfolio data for consistency
    if 'RELATED_ECN' in new_portfolio_data.columns:
        new_portfolio_data = new_portfolio_data.rename(columns={'RELATED_ECN': 'RC_ECN'})

    all_port_codes = portfolio_data['PORT_CODE'].dropna().unique()
    files_created  = 0

    for port_code in all_port_codes:
        banker_row = portfolio_data[portfolio_data['PORT_CODE'] == port_code]
        if banker_row.empty:
            continue

        banker_row    = banker_row.iloc[0]
        director_name = sanitize(banker_row['DIRECTOR_NAME'])
        manager_name  = sanitize(banker_row['MANAGER_NAME'])
        employee_name = sanitize(banker_row['EMPLOYEE_NAME'])

        # Build folder path
        folder = os.path.join(OUTPUT_DIR, director_name, manager_name, employee_name)
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, f'{sanitize(port_code)}.xlsx')

        # Filter data for this portfolio
        curr_df    = current_portfolio_data[current_portfolio_data['PORT_CODE'] == port_code].copy()
        fut_df     = new_portfolio_data[new_portfolio_data['PORT_CODE'] == port_code].copy()
        added_df   = compute_added(current_portfolio_data, new_portfolio_data, port_code)
        removed_df = compute_removed(current_portfolio_data, new_portfolio_data, port_code)

        # Create workbook
        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet

        write_sheet(wb, 'Current Portfolio',   curr_df,    CURRENT_COLS)
        write_sheet(wb, 'Future Portfolio',    fut_df,     FUTURE_COLS)
        write_sheet(wb, 'New Customers Added', added_df,   ADDED_COLS)
        write_sheet(wb, 'Customers Removed',   removed_df, REMOVED_COLS)

        wb.save(file_path)
        files_created += 1
        print(f'  Created: {file_path}')

    print(f'\nDone. {files_created} files created under ./{OUTPUT_DIR}/')


# ==================== USAGE ====================

# generate_portfolio_excels(
#     current_portfolio_data=CURRENT_PORTFOLIO_DATA,
#     new_portfolio_data=NEW_PORTFOLIO_DATA,
#     portfolio_data=SBRM_PORTFOLIO_DATA
# )
