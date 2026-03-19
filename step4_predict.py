import pandas as pd
import numpy as np
import joblib

# 1. Load model and scaler
model  = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Save a final bundle
bundle = {'model': model, 'scaler': scaler}
joblib.dump(bundle, 'house_price_predictor.pkl')
print("Saved house_price_predictor.pkl")

# 3. Feature columns (must match training order)
feature_cols = [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms',
    'Area Population'
]

# 4. Realistic example houses
examples = pd.DataFrame([
    [79545, 5.68, 7.01, 4.0, 23086],    # average house
    [65000, 4.50, 6.50, 3.0, 20000],    # modest house
    [100000, 7.00, 9.00, 5.0, 40000],   # luxury house
    [68000, 4.00, 6.00, 3.0, 18000],    # small house
    [90000, 6.50, 8.00, 4.0, 35000],    # large suburban house
], columns=feature_cols)

# 5. Predict
examples_sc  = scaler.transform(examples)
predictions  = model.predict(examples_sc)

print("\nExample Predictions:")
print("-" * 35)
labels = ['Average house   ',
          'Modest house    ',
          'Luxury house    ',
          'Small house     ',
          'Large suburban  ']
for label, price in zip(labels, predictions):
    print(f"  {label}: ${price:>12,.0f}")