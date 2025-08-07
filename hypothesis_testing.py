import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two points on Earth (in miles)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of Earth in miles
    r = 3956
    
    return c * r

def create_distance_buckets(distance):
    """
    Create distance buckets based on the specified ranges
    """
    if distance < 20:
        return '< 20 miles'
    elif distance < 80:
        return '20-80 miles'
    elif distance < 160:
        return '80-160 miles'
    elif distance < 250:
        return '160-250 miles'
    else:
        return '> 250 miles'

# Load your data (replace with your actual file paths)
# client_data = pd.read_csv('CLIENT_GROUP_DF_NEW.csv')
# active_portfolio = pd.read_csv('ACTIVE_PORTFOLIO.csv')
# branch_data = pd.read_csv('BRANCH_DATA.csv')

# For demonstration, I'll show the structure assuming you have the dataframes
# Make sure your dataframes are named: client_data, active_portfolio, branch_data

# Step 1: Merge all data together
# First merge client data with active portfolio
merged_data = client_data.merge(
    active_portfolio, 
    left_on='CG_PORTFOLIO_CD', 
    right_on='PORT_CODE', 
    how='inner'
)

# Then merge with branch data to get branch coordinates
final_data = merged_data.merge(
    branch_data, 
    on='AU', 
    how='inner'
)

print(f"Total records after merging: {len(final_data)}")
print(f"Columns: {final_data.columns.tolist()}")

# Step 2: Calculate haversine distance for each customer
final_data['distance_to_branch'] = final_data.apply(
    lambda row: haversine_distance(
        row['LAT_NUM'], row['LON_NUM'],
        row['BRANCH_LAT_NUM'], row['BRANCH_LON_NUM']
    ), axis=1
)

# Step 3: Create distance buckets for customers
final_data['customer_distance_bucket'] = final_data['distance_to_branch'].apply(create_distance_buckets)

print(f"Distance calculation completed. Sample distances:")
print(final_data[['CG_ECN', 'CG_PORTFOLIO_CD', 'distance_to_branch', 'customer_distance_bucket']].head())

# ===== CUSTOMER-LEVEL ANALYSIS =====
print("\n" + "="*60)
print("CUSTOMER-LEVEL ANALYSIS")
print("="*60)

customer_analysis = final_data.groupby('customer_distance_bucket').agg({
    'CG_ECN': 'count',  # Number of customers
    'BANK_REVENUE': 'mean',
    'DEPOSIT_BAL': 'mean',
    'CG_GROSS_SALES': 'mean'
}).round(2)

# Rename columns for clarity
customer_analysis.columns = [
    'Number_of_Customers', 
    'Avg_Bank_Revenue', 
    'Avg_Deposit_Balance', 
    'Avg_Gross_Sales'
]

# Reorder rows by distance ranges
distance_order = ['< 20 miles', '20-80 miles', '80-160 miles', '160-250 miles', '> 250 miles']
customer_analysis = customer_analysis.reindex([d for d in distance_order if d in customer_analysis.index])

print("\nCustomer Analysis by Distance Buckets:")
print(customer_analysis)

# ===== PORTFOLIO-LEVEL ANALYSIS =====
print("\n" + "="*60)
print("PORTFOLIO-LEVEL ANALYSIS")
print("="*60)

# Step 4: Calculate average distance per portfolio
portfolio_distances = final_data.groupby('CG_PORTFOLIO_CD').agg({
    'distance_to_branch': 'mean',
    'BANK_REVENUE': 'mean',
    'DEPOSIT_BAL': 'mean',
    'CG_GROSS_SALES': 'mean',
    'CG_ECN': 'count'  # Number of customers per portfolio
}).reset_index()

portfolio_distances.columns = [
    'Portfolio_Code', 
    'Avg_Distance_to_Branch', 
    'Avg_Bank_Revenue', 
    'Avg_Deposit_Balance', 
    'Avg_Gross_Sales',
    'Customer_Count'
]

# Create distance buckets for portfolios based on their average distance
portfolio_distances['portfolio_distance_bucket'] = portfolio_distances['Avg_Distance_to_Branch'].apply(create_distance_buckets)

# Portfolio analysis by distance buckets
portfolio_analysis = portfolio_distances.groupby('portfolio_distance_bucket').agg({
    'Portfolio_Code': 'count',  # Number of portfolios
    'Avg_Bank_Revenue': 'mean',
    'Avg_Deposit_Balance': 'mean',
    'Avg_Gross_Sales': 'mean'
}).round(2)

# Rename columns for clarity
portfolio_analysis.columns = [
    'Number_of_Portfolios', 
    'Avg_Bank_Revenue', 
    'Avg_Deposit_Balance', 
    'Avg_Gross_Sales'
]

# Reorder rows by distance ranges
portfolio_analysis = portfolio_analysis.reindex([d for d in distance_order if d in portfolio_analysis.index])

print("\nPortfolio Analysis by Distance Buckets:")
print(portfolio_analysis)

# ===== HYPOTHESIS TESTING =====
print("\n" + "="*60)
print("HYPOTHESIS TESTING")
print("="*60)

from scipy import stats

# HYPOTHESIS 1: Customer-level proximity test
# H0: Average BANK_REVENUE is the same for customers within 20 miles vs. beyond 20 miles
# H1: Average BANK_REVENUE is higher for customers within 20 miles

print("\nHYPOTHESIS TEST 1: Customer-Level Analysis")
print("-" * 50)

customers_within_20 = final_data[final_data['distance_to_branch'] <= 20]['BANK_REVENUE']
customers_beyond_20 = final_data[final_data['distance_to_branch'] > 20]['BANK_REVENUE']

# Remove any NaN values
customers_within_20 = customers_within_20.dropna()
customers_beyond_20 = customers_beyond_20.dropna()

print(f"Customers within 20 miles: {len(customers_within_20)}")
print(f"Customers beyond 20 miles: {len(customers_beyond_20)}")
print(f"Average revenue within 20 miles: ${customers_within_20.mean():,.2f}")
print(f"Average revenue beyond 20 miles: ${customers_beyond_20.mean():,.2f}")

# Perform independent t-test
t_stat_customers, p_value_customers = stats.ttest_ind(customers_within_20, customers_beyond_20)

print(f"\nT-test Results:")
print(f"T-statistic: {t_stat_customers:.4f}")
print(f"P-value: {p_value_customers:.4f}")
print(f"Significance level: 0.05")

if p_value_customers < 0.05:
    print("✓ REJECT null hypothesis - There IS a significant difference in bank revenue")
    if customers_within_20.mean() > customers_beyond_20.mean():
        print("✓ Customers within 20 miles have significantly HIGHER average bank revenue")
    else:
        print("✗ Customers within 20 miles have significantly LOWER average bank revenue")
else:
    print("✗ FAIL to reject null hypothesis - No significant difference found")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(customers_within_20) - 1) * customers_within_20.var() + 
                     (len(customers_beyond_20) - 1) * customers_beyond_20.var()) / 
                     (len(customers_within_20) + len(customers_beyond_20) - 2))
cohens_d_customers = (customers_within_20.mean() - customers_beyond_20.mean()) / pooled_std
print(f"Cohen's d (effect size): {cohens_d_customers:.4f}")

# HYPOTHESIS 2: Portfolio-level proximity test (In-market portfolios only)
# H0: Average BANK_REVENUE is the same for portfolios with avg distance <20 miles vs. >20 miles
# H1: Average BANK_REVENUE is higher for portfolios with avg distance <20 miles

print("\n\nHYPOTHESIS TEST 2: Portfolio-Level Analysis (In-market only)")
print("-" * 60)

# Filter for In-market portfolios only (assuming ROLE_TYPE or TYPE indicates this)
# You may need to adjust this filter based on your data
inmarket_data = final_data[final_data['ROLE_TYPE'].str.contains('In-market', case=False, na=False) | 
                          final_data['TYPE'].str.contains('In-market', case=False, na=False)]

if len(inmarket_data) == 0:
    print("WARNING: No 'In-market' portfolios found. Using all portfolios for analysis.")
    print("Please check your ROLE_TYPE or TYPE columns for correct 'In-market' designation.")
    inmarket_data = final_data.copy()

print(f"In-market records: {len(inmarket_data)}")

# Calculate portfolio-level averages for In-market portfolios
inmarket_portfolio_stats = inmarket_data.groupby('CG_PORTFOLIO_CD').agg({
    'distance_to_branch': 'mean',
    'BANK_REVENUE': 'mean'
}).reset_index()

portfolios_within_20 = inmarket_portfolio_stats[inmarket_portfolio_stats['distance_to_branch'] <= 20]['BANK_REVENUE']
portfolios_beyond_20 = inmarket_portfolio_stats[inmarket_portfolio_stats['distance_to_branch'] > 20]['BANK_REVENUE']

# Remove any NaN values
portfolios_within_20 = portfolios_within_20.dropna()
portfolios_beyond_20 = portfolios_beyond_20.dropna()

print(f"In-market portfolios with avg distance ≤20 miles: {len(portfolios_within_20)}")
print(f"In-market portfolios with avg distance >20 miles: {len(portfolios_beyond_20)}")

if len(portfolios_within_20) > 0 and len(portfolios_beyond_20) > 0:
    print(f"Average revenue for portfolios ≤20 miles: ${portfolios_within_20.mean():,.2f}")
    print(f"Average revenue for portfolios >20 miles: ${portfolios_beyond_20.mean():,.2f}")
    
    # Perform independent t-test
    t_stat_portfolios, p_value_portfolios = stats.ttest_ind(portfolios_within_20, portfolios_beyond_20)
    
    print(f"\nT-test Results:")
    print(f"T-statistic: {t_stat_portfolios:.4f}")
    print(f"P-value: {p_value_portfolios:.4f}")
    print(f"Significance level: 0.05")
    
    if p_value_portfolios < 0.05:
        print("✓ REJECT null hypothesis - There IS a significant difference in portfolio bank revenue")
        if portfolios_within_20.mean() > portfolios_beyond_20.mean():
            print("✓ Portfolios with avg distance ≤20 miles have significantly HIGHER average bank revenue")
        else:
            print("✗ Portfolios with avg distance ≤20 miles have significantly LOWER average bank revenue")
    else:
        print("✗ FAIL to reject null hypothesis - No significant difference found")
    
    # Effect size (Cohen's d)
    pooled_std_portfolios = np.sqrt(((len(portfolios_within_20) - 1) * portfolios_within_20.var() + 
                                   (len(portfolios_beyond_20) - 1) * portfolios_beyond_20.var()) / 
                                   (len(portfolios_within_20) + len(portfolios_beyond_20) - 2))
    cohens_d_portfolios = (portfolios_within_20.mean() - portfolios_beyond_20.mean()) / pooled_std_portfolios
    print(f"Cohen's d (effect size): {cohens_d_portfolios:.4f}")
else:
    print("❌ Cannot perform hypothesis test - insufficient data in one or both groups")

# ADDITIONAL HYPOTHESIS TESTS FOR DISTANCE BUCKETS
print("\n\nADDITIONAL ANALYSIS: ANOVA for Multiple Distance Buckets")
print("-" * 60)

# Customer-level ANOVA across all distance buckets
customer_groups = [group['BANK_REVENUE'].dropna() for name, group in final_data.groupby('customer_distance_bucket')]
customer_group_names = [name for name, group in final_data.groupby('customer_distance_bucket')]

if len(customer_groups) > 1:
    f_stat_customers, p_value_anova_customers = stats.f_oneway(*customer_groups)
    print(f"Customer-level ANOVA across distance buckets:")
    print(f"F-statistic: {f_stat_customers:.4f}")
    print(f"P-value: {p_value_anova_customers:.4f}")
    
    if p_value_anova_customers < 0.05:
        print("✓ Significant differences exist between distance buckets for customer bank revenue")
    else:
        print("✗ No significant differences between distance buckets for customer bank revenue")

# Portfolio-level ANOVA across all distance buckets
portfolio_groups = [group['Avg_Bank_Revenue'].dropna() for name, group in portfolio_distances.groupby('portfolio_distance_bucket')]

if len(portfolio_groups) > 1:
    f_stat_portfolios, p_value_anova_portfolios = stats.f_oneway(*portfolio_groups)
    print(f"\nPortfolio-level ANOVA across distance buckets:")
    print(f"F-statistic: {f_stat_portfolios:.4f}")
    print(f"P-value: {p_value_anova_portfolios:.4f}")
    
    if p_value_anova_portfolios < 0.05:
        print("✓ Significant differences exist between distance buckets for portfolio bank revenue")
    else:
        print("✗ No significant differences between distance buckets for portfolio bank revenue")

# ===== SUMMARY STATISTICS =====
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nOverall Statistics:")
print(f"Total unique customers: {final_data['CG_ECN'].nunique()}")
print(f"Total unique portfolios: {final_data['CG_PORTFOLIO_CD'].nunique()}")
print(f"Average distance across all customers: {final_data['distance_to_branch'].mean():.2f} miles")
print(f"Median distance across all customers: {final_data['distance_to_branch'].median():.2f} miles")
print(f"Max distance: {final_data['distance_to_branch'].max():.2f} miles")
print(f"Min distance: {final_data['distance_to_branch'].min():.2f} miles")

# ===== EXPORT RESULTS =====
print("\n" + "="*60)
print("EXPORTING RESULTS")
print("="*60)

# Save results to CSV files
customer_analysis.to_csv('customer_distance_analysis.csv')
portfolio_analysis.to_csv('portfolio_distance_analysis.csv')
portfolio_distances.to_csv('portfolio_details_with_distances.csv', index=False)

print("Results exported to:")
print("- customer_distance_analysis.csv")
print("- portfolio_distance_analysis.csv") 
print("- portfolio_details_with_distances.csv")

# ===== OPTIONAL: VISUALIZATIONS =====
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Customer count by distance bucket
    customer_analysis['Number_of_Customers'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Number of Customers by Distance Bucket')
    ax1.set_ylabel('Number of Customers')
    ax1.tick_params(axis='x', rotation=45)
    
    # Portfolio count by distance bucket
    portfolio_analysis['Number_of_Portfolios'].plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('Number of Portfolios by Distance Bucket')
    ax2.set_ylabel('Number of Portfolios')
    ax2.tick_params(axis='x', rotation=45)
    
    # Average bank revenue comparison
    ax3.plot(customer_analysis.index, customer_analysis['Avg_Bank_Revenue'], 
             marker='o', label='Customer Level', linewidth=2)
    ax3.plot(portfolio_analysis.index, portfolio_analysis['Avg_Bank_Revenue'], 
             marker='s', label='Portfolio Level', linewidth=2)
    ax3.set_title('Average Bank Revenue by Distance Bucket')
    ax3.set_ylabel('Average Bank Revenue')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # Distance distribution histogram
    final_data['distance_to_branch'].hist(bins=50, ax=ax4, alpha=0.7, color='green')
    ax4.set_title('Distribution of Customer Distances to Branch')
    ax4.set_xlabel('Distance (miles)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(x=20, color='red', linestyle='--', label='20 miles')
    ax4.axvline(x=80, color='orange', linestyle='--', label='80 miles')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('distance_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'distance_analysis_plots.png'")
    
except ImportError:
    print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
