import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load cleaned data
df = pd.read_csv('housing_clean.csv')

# 2. Define features and target
X = df.drop(columns=['Price'])
y = df['Price']

print("Features:", list(X.columns))

# 3. Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# 4. Scale the features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 5. Train Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train_sc, y_train)
print("Ridge Regression model trained!")

# 6. Print coefficients
coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nIntercept:", round(model.intercept_, 2))
print("\nCoefficients:")
print(coef_df.to_string(index=False))

# 7. Save model and scaler
joblib.dump(model,  'lr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
X_test.to_csv('X_test.csv',   index=False)
y_test.to_csv('y_test.csv',   index=False)
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
print("\nSaved lr_model.pkl and scaler.pkl")