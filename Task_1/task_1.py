import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('sales_data.csv')

print("--- First 5 Rows of the Dataset ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

avg_revenue = df['Revenue'].mean()
print(f"\nAverage Revenue: ${avg_revenue:,.2f}")

print("\n--- Summary Statistics ---")
print(df.describe())

category_summary = df.groupby('Product_Category')['Revenue'].sum().reset_index()
print("\n--- Total Revenue by Category ---")
print(category_summary)

plt.figure(figsize=(8, 5))
cat_revenue = df.groupby('Product_Category')['Revenue'].sum()

cat_revenue.plot(kind='bar', color=['darkblue', 'orange', 'green'])
plt.title('Total Revenue by Product Category', fontsize=14)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Advertising_Spend'], df['Revenue'], color='crimson', edgecolors='black', s=100)
plt.title('Advertising Spend vs. Revenue', fontsize=14)
plt.xlabel('Advertising Spend ($)', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
correlation_matrix = df[['Units_Sold', 'Revenue', 'Advertising_Spend', 'Customer_Rating']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap', fontsize=14)
plt.tight_layout()
plt.show()