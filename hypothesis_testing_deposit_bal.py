# ===== HYPOTHESIS TESTING FOR DEPOSIT BALANCE =====
print("="*60)
print("HYPOTHESIS TESTING - DEPOSIT BALANCE")
print("="*60)

from scipy import stats

# HYPOTHESIS 1: Customer-level proximity test for DEPOSIT_BAL
print("\nHYPOTHESIS TEST 1: Customer-Level Analysis - DEPOSIT BALANCE")
print("-" * 60)

customers_within_20_dep = final_data[final_data['distance_to_branch'] <= 20]['DEPOSIT_BAL']
customers_beyond_20_dep = final_data[final_data['distance_to_branch'] > 20]['DEPOSIT_BAL']

# Remove any NaN values
customers_within_20_dep = customers_within_20_dep.dropna()
customers_beyond_20_dep = customers_beyond_20_dep.dropna()

print(f"Customers within 20 miles: {len(customers_within_20_dep)}")
print(f"Customers beyond 20 miles: {len(customers_beyond_20_dep)}")
print(f"Average deposit balance within 20 miles: ${customers_within_20_dep.mean():,.2f}")
print(f"Average deposit balance beyond 20 miles: ${customers_beyond_20_dep.mean():,.2f}")

# Perform independent t-test
t_stat_customers_dep, p_value_customers_dep = stats.ttest_ind(customers_within_20_dep, customers_beyond_20_dep)

print(f"\nT-test Results:")
print(f"T-statistic: {t_stat_customers_dep:.4f}")
print(f"P-value: {p_value_customers_dep:.4f}")
print(f"Significance level: 0.05")

if p_value_customers_dep < 0.05:
    print("✓ REJECT null hypothesis - There IS a significant difference in deposit balance")
    if customers_within_20_dep.mean() > customers_beyond_20_dep.mean():
        print("✓ Customers within 20 miles have significantly HIGHER average deposit balance")
    else:
        print("✗ Customers within 20 miles have significantly LOWER average deposit balance")
else:
    print("✗ FAIL to reject null hypothesis - No significant difference found")

# Effect size (Cohen's d)
pooled_std_dep = np.sqrt(((len(customers_within_20_dep) - 1) * customers_within_20_dep.var() + 
                         (len(customers_beyond_20_dep) - 1) * customers_beyond_20_dep.var()) / 
                         (len(customers_within_20_dep) + len(customers_beyond_20_dep) - 2))
cohens_d_customers_dep = (customers_within_20_dep.mean() - customers_beyond_20_dep.mean()) / pooled_std_dep
print(f"Cohen's d (effect size): {cohens_d_customers_dep:.4f}")

# HYPOTHESIS 2: Portfolio-level proximity test for DEPOSIT_BAL (In-market only)
print("\n\nHYPOTHESIS TEST 2: Portfolio-Level Analysis - DEPOSIT BALANCE (In-market only)")
print("-" * 70)

# Filter for In-market portfolios
inmarket_data_dep = final_data[final_data['ROLE_TYPE'].str.contains('In-market', case=False, na=False) | 
                              final_data['TYPE'].str.contains('In-market', case=False, na=False)]

if len(inmarket_data_dep) == 0:
    print("WARNING: No 'In-market' portfolios found. Using all portfolios for analysis.")
    inmarket_data_dep = final_data.copy()

print(f"In-market records: {len(inmarket_data_dep)}")

# Calculate portfolio-level averages for In-market portfolios
inmarket_portfolio_stats_dep = inmarket_data_dep.groupby('CG_PORTFOLIO_CD').agg({
    'distance_to_branch': 'mean',
    'DEPOSIT_BAL': 'mean'
}).reset_index()

portfolios_within_20_dep = inmarket_portfolio_stats_dep[inmarket_portfolio_stats_dep['distance_to_branch'] <= 20]['DEPOSIT_BAL']
portfolios_beyond_20_dep = inmarket_portfolio_stats_dep[inmarket_portfolio_stats_dep['distance_to_branch'] > 20]['DEPOSIT_BAL']

# Remove any NaN values
portfolios_within_20_dep = portfolios_within_20_dep.dropna()
portfolios_beyond_20_dep = portfolios_beyond_20_dep.dropna()

print(f"In-market portfolios with avg distance ≤20 miles: {len(portfolios_within_20_dep)}")
print(f"In-market portfolios with avg distance >20 miles: {len(portfolios_beyond_20_dep)}")

if len(portfolios_within_20_dep) > 0 and len(portfolios_beyond_20_dep) > 0:
    print(f"Average deposit balance for portfolios ≤20 miles: ${portfolios_within_20_dep.mean():,.2f}")
    print(f"Average deposit balance for portfolios >20 miles: ${portfolios_beyond_20_dep.mean():,.2f}")
    
    # Perform independent t-test
    t_stat_portfolios_dep, p_value_portfolios_dep = stats.ttest_ind(portfolios_within_20_dep, portfolios_beyond_20_dep)
    
    print(f"\nT-test Results:")
    print(f"T-statistic: {t_stat_portfolios_dep:.4f}")
    print(f"P-value: {p_value_portfolios_dep:.4f}")
    print(f"Significance level: 0.05")
    
    if p_value_portfolios_dep < 0.05:
        print("✓ REJECT null hypothesis - There IS a significant difference in portfolio deposit balance")
        if portfolios_within_20_dep.mean() > portfolios_beyond_20_dep.mean():
            print("✓ Portfolios with avg distance ≤20 miles have significantly HIGHER average deposit balance")
        else:
            print("✗ Portfolios with avg distance ≤20 miles have significantly LOWER average deposit balance")
    else:
        print("✗ FAIL to reject null hypothesis - No significant difference found")
    
    # Effect size (Cohen's d)
    pooled_std_portfolios_dep = np.sqrt(((len(portfolios_within_20_dep) - 1) * portfolios_within_20_dep.var() + 
                                       (len(portfolios_beyond_20_dep) - 1) * portfolios_beyond_20_dep.var()) / 
                                       (len(portfolios_within_20_dep) + len(portfolios_beyond_20_dep) - 2))
    cohens_d_portfolios_dep = (portfolios_within_20_dep.mean() - portfolios_beyond_20_dep.mean()) / pooled_std_portfolios_dep
    print(f"Cohen's d (effect size): {cohens_d_portfolios_dep:.4f}")
else:
    print("❌ Cannot perform hypothesis test - insufficient data in one or both groups")

# ANOVA TESTS FOR DEPOSIT BALANCE
print("\n\nANOVA ANALYSIS: DEPOSIT BALANCE Across Distance Buckets")
print("-" * 60)

# Customer-level ANOVA across all distance buckets for DEPOSIT_BAL
customer_groups_dep = [group['DEPOSIT_BAL'].dropna() for name, group in final_data.groupby('customer_distance_bucket')]

if len(customer_groups_dep) > 1:
    f_stat_customers_dep, p_value_anova_customers_dep = stats.f_oneway(*customer_groups_dep)
    print(f"Customer-level ANOVA for DEPOSIT_BAL across distance buckets:")
    print(f"F-statistic: {f_stat_customers_dep:.4f}")
    print(f"P-value: {p_value_anova_customers_dep:.4f}")
    
    if p_value_anova_customers_dep < 0.05:
        print("✓ Significant differences exist between distance buckets for customer deposit balance")
    else:
        print("✗ No significant differences between distance buckets for customer deposit balance")

# Portfolio-level ANOVA across all distance buckets for DEPOSIT_BAL
portfolio_groups_dep = [group['Avg_Deposit_Balance'].dropna() for name, group in portfolio_distances.groupby('portfolio_distance_bucket')]

if len(portfolio_groups_dep) > 1:
    f_stat_portfolios_dep, p_value_anova_portfolios_dep = stats.f_oneway(*portfolio_groups_dep)
    print(f"\nPortfolio-level ANOVA for DEPOSIT_BAL across distance buckets:")
    print(f"F-statistic: {f_stat_portfolios_dep:.4f}")
    print(f"P-value: {p_value_anova_portfolios_dep:.4f}")
    
    if p_value_anova_portfolios_dep < 0.05:
        print("✓ Significant differences exist between distance buckets for portfolio deposit balance")
    else:
        print("✗ No significant differences between distance buckets for portfolio deposit balance")

print("\n" + "="*60)
print("DEPOSIT BALANCE HYPOTHESIS TESTING COMPLETE!")
print("="*60)
