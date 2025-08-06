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
