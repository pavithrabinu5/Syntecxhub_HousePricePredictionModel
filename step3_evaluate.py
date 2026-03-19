import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# 1. Load model, scaler, and test data
model  = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# 2. Make predictions
X_test_sc = scaler.transform(X_test)
y_pred    = model.predict(X_test_sc)

# 3. Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)

print("=" * 40)
print("         MODEL EVALUATION")
print("=" * 40)
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:,.2f}")

# 4. Cross-Validation
print("\n--- Cross Validation (5-Fold) ---")
cv_scores = cross_val_score(model, X_test_sc, y_test, cv=5, scoring='r2')
print("CV R² scores:", [round(s, 4) for s in cv_scores])
print("Mean CV R²  :", round(cv_scores.mean(), 4))
print("Std CV R²   :", round(cv_scores.std(), 4))

# 5. Actual vs Predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted  |  R² = {r2:.4f}')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
print("Saved actual_vs_predicted.png")

# 6. Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.4, color='tomato')
plt.axhline(0, color='black', lw=1.5, linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.tight_layout()
plt.savefig('residuals.png')
plt.show()
print("Saved residuals.png")

# 7. Feature Importance chart
plt.figure(figsize=(8, 5))
coef_df = pd.DataFrame({
    'Feature'    : X_test.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient')
colors = ['tomato' if c < 0 else 'steelblue' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
plt.axvline(0, color='black', lw=0.8, linestyle='--')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Coefficients)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Saved feature_importance.png")