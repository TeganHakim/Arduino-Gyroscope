import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load CSV and columns
data = pd.read_csv("data.csv")

x = data['time']
y = data['roll']

x_data = []
y_data = []

for i in range(len(data)):
    x_data.append([x[i]])
    y_data.append([y[i]])

# Plot outputs
plt.scatter(x_data, y_data, color = 'black')
plt.title('Gyroscope Data')
plt.xlabel('Time')
plt.ylabel('Roll')
plt.xticks(([i for i in range(len(data)) if i % 25 == 0]))
plt.yticks(([i for i in range(-180, 180) if i % 25 == 0]))

# Create linear regression object
regr = LinearRegression()

# Train the model
regr.fit(x_data, y_data)

# Plot outputs
plt.plot(x_data, regr.predict(x_data), color = 'red', linewidth = 3)

# Residuals = Observed Value - Predicted Value
residuals = {"x": [], "y": []}
for i in range(len(data)):
  residuals["x"].append(x_data[i])
  residuals["y"].append(y_data[i])
  residuals["x"].append(x_data[i])
  residuals["y"].append(regr.predict(x_data)[i])

# Generate coorelation coefficient
r = regr.coef_[0][0]
plt.text((len(x_data) / 2 ) - 20, -90, "r = " + str(round(r, 3)), fontsize = 10)

# Plot residual lines
plt.plot(residuals["x"], residuals["y"], color = "orange", linewidth = 1)

plt.show()