# CHAPTER 4: TRAINING MODELS

# 1. Test the normal equation in Python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # to make this code example reproducible
m = 100  # number of instances
X = 2 * np.random.rand(m, 1)  # column vector
y = 4 + 3 * X + np.random.randn(m, 1)  # column vector
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Data points')
plt.title("Scatter Plot of X vs y")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Next, to calculate the normal equation
from sklearn.preprocessing import add_dummy_feature
X_b = add_dummy_feature(X)  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("With normal equation, the best θ is: ", theta_best)
# Output: With normal equation, the best θ is:  [[4.21509616]
# [2.77011339]]

X_new = np.array([[0], [2]])  # Create a vector
X_new_b = add_dummy_feature(X_new)  # Add a column with all 1s, so that it is allowed to add the bias term
y_predict = X_new_b @ theta_best
print("For the new example, the predicted value is: ", y_predict)
# Output: For the new example, the predicted value is:  [[4.21509616]
#  [9.75532293]]
plt.figure(figsize=(9, 6))
plt.scatter(X, y, color="#1f77b4", alpha=0.7, edgecolor='white', s=70, label="Training data")
plt.plot(X_new, y_predict, "r-", linewidth=2.5, label="Regression line")

# Highlight the predicted points
plt.scatter(X_new, y_predict, color="red", s=80, edgecolor='black', zorder=5)
for i, (x_val, y_val) in enumerate(zip(X_new.flatten(), y_predict.flatten())):
    plt.text(x_val + 0.05, y_val, f"({x_val:.1f}, {y_val:.2f})", fontsize=10, color="darkred")

# Titles and labels
plt.title("Prediction using Normal Equation", fontsize=14, pad=15)
plt.xlabel("X (input feature)", fontsize=12)
plt.ylabel("y (target value)", fontsize=12)

# Add grid, legend, and style tweaks
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()

# Alternatively, we could use LinearRegression to calculate the best θ, and make predictions
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Use LinearRegression directly, we could get the θ: ", lin_reg.intercept_, lin_reg.coef_)
# Output: We could get the θ:  [4.21509616] [[2.77011339]]
# This is the same result as using normal equation
print("Use linearregression to predict, the result is: ", lin_reg.predict(X_new))
# Output: Use linear regression to predict, the result is:  [[4.21509616]
#  [9.75532293]]

# More easily, we could use np.linalg.lstsq() function (lstsq stands for least square)
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print("Use least square function, we could also get the θ:", theta_best_svd)









