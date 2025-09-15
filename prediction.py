import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Housing dataset
data = {
    "Area": [1000, 1500, 1800, 2400, 3000, 3500, 4000, 4500],
    "Price": [200000, 250000, 280000, 350000, 400000, 450000, 500000, 600000]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Simple linear regression (Area -> Price)
X = df[['Area']]  # predictor
y = df['Price']   # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nRMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Visualizations
plt.figure(figsize=(12,5))

# 1. Data + Regression Line
plt.subplot(1,2,1)
plt.scatter(df['Area'], df['Price'], color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Price vs Area')
plt.legend()

# 2. Actual vs Predicted
plt.subplot(1,2,2)
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted')

plt.tight_layout()
plt.show()