import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('USA_Housing.csv')
print(df.shape)
print(df.head())

# 2. Check data info
print(df.info())
print(df.describe())

# 3. Check for missing values
print(df.isnull().sum())

# 4. Drop duplicates
df.drop_duplicates(inplace=True)

# 5. Drop non-numeric columns if any (like address)
df.drop(columns=['Address'], inplace=True, errors='ignore')

# 6. Check correlations
print(df.corr())

# 7. Plot correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# 8. Plot price distribution
df['Price'].hist(bins=50)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.savefig('price_dist.png')
plt.show()

# 9. Save cleaned data
df.to_csv('housing_clean.csv', index=False)
print("Saved housing_clean.csv")