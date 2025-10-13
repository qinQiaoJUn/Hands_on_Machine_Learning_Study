import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Original code
# lifesat = pd.read_csv("lifesat.csv", encoding='utf-8')  # The download file location
# X = lifesat[["GDP per capita (USD)"]].values
# y = lifesat[["Life satisfaction"]].values
#
# # Visualize the data
# lifesat.plot(kind='scatter', grid=True,  # Scattered plot （点状图）
#              x="GDP per capita (USD)", y="Life satisfaction")
# plt.axis([23_500, 62_500, 4, 9])
# plt.show()

# Check the structure of this csv file
# with open("D:\PyCharm\Python_projects\lifesat.csv", 'r', encoding='utf-8') as f:
#     for i, line in enumerate(f):
#         print(f"Line {i}: {repr(line)}")
#         if i >= 5:  # 只显示前5行
#             break

lifesat = pd.read_csv("lifesat.csv",
                     encoding='utf-8',
                     sep=',')  # After printing the csv file, we should use "," as the separator

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new)) # output: [[6.30165767]]

