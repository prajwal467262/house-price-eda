import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted')

# ── Load Data ──────────────────────────────────────────────
# Using Ames Housing Dataset
df = pd.read_csv('data/house_prices.csv')
print(f"Shape: {df.shape}")
print(df.dtypes.value_counts())

# ── Missing Values ────────────────────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing Values:")
print(missing)

fig, ax = plt.subplots(figsize=(10,5))
missing.plot(kind='bar', ax=ax, color='#c8f04a')
ax.set_title('Missing Values per Column')
plt.tight_layout()
plt.savefig('outputs/missing_values.png', dpi=150)

# ── Target Distribution ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12,4))
df['SalePrice'].hist(bins=50, ax=axes[0], color='#7c6af7')
axes[0].set_title('Sale Price Distribution')
np.log1p(df['SalePrice']).hist(bins=50, ax=axes[1], color='#c8f04a')
axes[1].set_title('Log Sale Price Distribution')
plt.tight_layout()
plt.savefig('outputs/price_distribution.png', dpi=150)

# ── Correlation Heatmap ───────────────────────────────────
num_df = df.select_dtypes(include=np.number)
corr   = num_df.corr()
top_corr = corr['SalePrice'].abs().nlargest(15).index

fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(num_df[top_corr].corr(), annot=True, fmt='.2f',
            cmap='RdYlGn', ax=ax, linewidths=0.5)
ax.set_title('Top 15 Features Correlation Matrix')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=150)

# ── Key Relationships ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15,8))
features = ['GrLivArea','TotalBsmtSF','GarageArea','YearBuilt','OverallQual','TotRmsAbvGrd']
for ax, feat in zip(axes.flat, features):
    ax.scatter(df[feat], df['SalePrice'], alpha=0.3, color='#c8f04a', s=5)
    ax.set_xlabel(feat); ax.set_ylabel('SalePrice')
plt.suptitle('Key Feature Relationships with Sale Price')
plt.tight_layout()
plt.savefig('outputs/feature_relationships.png', dpi=150)

print("All plots saved to outputs/")
