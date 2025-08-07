# ===== HYPOTHESIS TESTING FOR INTERACTIONS_COUNT =====
print("="*60)
print("HYPOTHESIS TESTING FOR INTERACTIONS_COUNT")
print("="*60)

from scipy import stats
import numpy as np

# HYPOTHESIS 1: Customer-level proximity test for INTERACTIONS_COUNT
# H0: Average INTERACTIONS_COUNT is the same for customers within 20 miles vs. beyond 20 miles
# H1: Average INTERACTIONS_COUNT is higher for customers within 20 miles

print("\nHYPOTHESIS TEST 1: Customer-Level Analysis - INTERACTIONS_COUNT")
print("-" * 60)

interactions_within_20 = final_data[final_data['distance_to_branch'] <= 20]['INTERACTIONS_COUNT']
interactions_beyond_20 = final_data[final_data['distance_to_branch'] > 20]['INTERACTIONS_COUNT']

# Remove any NaN values
interactions_within_20 = interactions_within_20.dropna()
interactions_beyond_20 = interactions_beyond_20.dropna()

print(f"Customers within 20 miles: {len(interactions_within_20)}")
print(f"Customers beyond 20 miles: {len(interactions_beyond_20)}")
print(f"Average interactions within 20 miles: {interactions_within_20.mean():.2f}")
print(f"Average interactions beyond 20 miles: {interactions_beyond_20.mean():.2f}")

# Perform independent t-test
t_stat_interactions, p_value_interactions = stats.ttest_ind(interactions_within_20, interactions_beyond_20)

print(f"\nT-test Results:")
print(f"T-statistic: {t_stat_interactions:.4f}")
print(f"P-value: {p_value_interactions:.4f}")
print(f"Significance level: 0.05")

if p_value_interactions < 0.05:
    print("✓ REJECT null hypothesis - There IS a significant difference in interactions count")
    if interactions_within_20.mean() > interactions_beyond_20.mean():
        print("✓ Customers within 20 miles have significantly HIGHER average interactions count")
    else:
        print("✗ Customers within 20 miles have significantly LOWER average interactions count")
else:
    print("✗ FAIL to reject null hypothesis - No significant difference found")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(interactions_within_20) - 1) * interactions_within_20.var() + 
                     (len(interactions_beyond_20) - 1) * interactions_beyond_20.var()) / 
                     (len(interactions_within_20) + len(interactions_beyond_20) - 2))
cohens_d_interactions = (interactions_within_20.mean() - interactions_beyond_20.mean()) / pooled_std
print(f"Cohen's d (effect size): {cohens_d_interactions:.4f}")

# HYPOTHESIS 2: Portfolio-level proximity test for INTERACTIONS_COUNT (In-market portfolios only)
print("\n\nHYPOTHESIS TEST 2: Portfolio-Level Analysis - INTERACTIONS_COUNT (In-market only)")
print("-" * 70)

# Filter for In-market portfolios only
inmarket_data = final_data[final_data['ROLE_TYPE'].str.contains('In-market', case=False, na=False) | 
                          final_data['TYPE'].str.contains('In-market', case=False, na=False)]

if len(inmarket_data) == 0:
    print("WARNING: No 'In-market' portfolios found. Using all portfolios for analysis.")
    inmarket_data = final_data.copy()

print(f"In-market records: {len(inmarket_data)}")

# Calculate portfolio-level averages for In-market portfolios
inmarket_portfolio_interactions = inmarket_data.groupby('CG_PORTFOLIO_CD').agg({
    'distance_to_branch': 'mean',
    'INTERACTIONS_COUNT': 'mean'
}).reset_index()

portfolio_interactions_within_20 = inmarket_portfolio_interactions[inmarket_portfolio_interactions['distance_to_branch'] <= 20]['INTERACTIONS_COUNT']
portfolio_interactions_beyond_20 = inmarket_portfolio_interactions[inmarket_portfolio_interactions['distance_to_branch'] > 20]['INTERACTIONS_COUNT']

# Remove any NaN values
portfolio_interactions_within_20 = portfolio_interactions_within_20.dropna()
portfolio_interactions_beyond_20 = portfolio_interactions_beyond_20.dropna()

print(f"In-market portfolios with avg distance ≤20 miles: {len(portfolio_interactions_within_20)}")
print(f"In-market portfolios with avg distance >20 miles: {len(portfolio_interactions_beyond_20)}")

if len(portfolio_interactions_within_20) > 0 and len(portfolio_interactions_beyond_20) > 0:
    print(f"Average interactions for portfolios ≤20 miles: {portfolio_interactions_within_20.mean():.2f}")
    print(f"Average interactions for portfolios >20 miles: {portfolio_interactions_beyond_20.mean():.2f}")
    
    # Perform independent t-test
    t_stat_portfolio_interactions, p_value_portfolio_interactions = stats.ttest_ind(portfolio_interactions_within_20, portfolio_interactions_beyond_20)
    
    print(f"\nT-test Results:")
    print(f"T-statistic: {t_stat_portfolio_interactions:.4f}")
    print(f"P-value: {p_value_portfolio_interactions:.4f}")
    print(f"Significance level: 0.05")
    
    if p_value_portfolio_interactions < 0.05:
        print("✓ REJECT null hypothesis - There IS a significant difference in portfolio interactions count")
        if portfolio_interactions_within_20.mean() > portfolio_interactions_beyond_20.mean():
            print("✓ Portfolios with avg distance ≤20 miles have significantly HIGHER average interactions count")
        else:
            print("✗ Portfolios with avg distance ≤20 miles have significantly LOWER average interactions count")
    else:
        print("✗ FAIL to reject null hypothesis - No significant difference found")
    
    # Effect size (Cohen's d)
    pooled_std_portfolio_interactions = np.sqrt(((len(portfolio_interactions_within_20) - 1) * portfolio_interactions_within_20.var() + 
                                               (len(portfolio_interactions_beyond_20) - 1) * portfolio_interactions_beyond_20.var()) / 
                                               (len(portfolio_interactions_within_20) + len(portfolio_interactions_beyond_20) - 2))
    cohens_d_portfolio_interactions = (portfolio_interactions_within_20.mean() - portfolio_interactions_beyond_20.mean()) / pooled_std_portfolio_interactions
    print(f"Cohen's d (effect size): {cohens_d_portfolio_interactions:.4f}")
else:
    print("❌ Cannot perform hypothesis test - insufficient data in one or both groups")

# ADDITIONAL ANALYSIS: ANOVA for INTERACTIONS_COUNT across Multiple Distance Buckets
print("\n\nADDITIONAL ANALYSIS: ANOVA for INTERACTIONS_COUNT across Distance Buckets")
print("-" * 70)

# Customer-level ANOVA across all distance buckets for INTERACTIONS_COUNT
interaction_groups = [group['INTERACTIONS_COUNT'].dropna() for name, group in final_data.groupby('customer_distance_bucket')]
interaction_group_names = [name for name, group in final_data.groupby('customer_distance_bucket')]

if len(interaction_groups) > 1:
    f_stat_interactions, p_value_anova_interactions = stats.f_oneway(*interaction_groups)
    print(f"Customer-level ANOVA for INTERACTIONS_COUNT across distance buckets:")
    print(f"F-statistic: {f_stat_interactions:.4f}")
    print(f"P-value: {p_value_anova_interactions:.4f}")
    
    if p_value_anova_interactions < 0.05:
        print("✓ Significant differences exist between distance buckets for customer interactions count")
    else:
        print("✗ No significant differences between distance buckets for customer interactions count")
    
    # Show mean interactions count by distance bucket
    print(f"\nMean INTERACTIONS_COUNT by distance bucket:")
    for i, group_name in enumerate(interaction_group_names):
        print(f"  {group_name}: {interaction_groups[i].mean():.2f} (n={len(interaction_groups[i])})")

print("\n" + "="*60)
print("INTERACTIONS_COUNT HYPOTHESIS TESTING COMPLETE!")
print("="*60)
