import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load and Parse Data
df = pd.read_csv("data.csv")

axis = plt.subplots(nrows=1, ncols=1)

x = df['time']
y = df['roll']

x_data = []
y_data = []
ROLL_RADIUS = 180

for i in range(len(df)):
    x_data.append([x[i]])
    y_data.append([y[i]])


# Plot raw outputs
plt.scatter(x_data, y_data, color = 'black')
plt.title('Gyroscope Data')
plt.xlabel('Time')
plt.ylabel('Roll')
plt.xticks(([i for i in range(len(df)) if i % 25 == 0]))
plt.yticks(([i for i in range(-ROLL_RADIUS, ROLL_RADIUS) if i % 25 == 0]))

# Residuals = Observed Value - Predicted Value
residuals = {"x": [], "y": []}
for i in range(len(df)):
  residuals["x"].append(x_data[i])
  residuals["y"].append(y_data[i])
  residuals["x"].append(x_data[i])
  residuals["y"].append([0])

plt.plot(residuals["x"], residuals["y"], color = "#d3d3d3", linewidth = 1)

# Plot outputs
plt.plot(x_data, y_data, color = 'black', linewidth = 3)
plt.plot([0, ROLL_RADIUS], [0, 0], color = "red", linewidth = 3)

x_axis_y = [0 for i in range(len(df))]
idx = np.argwhere(np.diff(np.sign(np.array(x_axis_y) - np.array(y)))).flatten()
idx = list(idx)
idx.insert(0, 0)
idx = np.array(idx)
plt.plot(np.array(x)[idx], np.array(x_axis_y)[idx], color = "red", marker = "o", markersize = 10)

chunks = []
for i in range(len(idx)):
    if (i != len(idx) - 1):
          chunks.append(df.iloc[idx[i]: idx[i+1]])
    else:
          chunks.append(df.iloc[idx[i]:])

#implement an "ignorechunk" algorithm


for i in range(len(idx)):
    plt.axvline(x = idx[i], color = "red", linestyle = "--")

plt.show()

# Create linear regression object
# regr = LinearRegression()

# Train the model
# regr.fit(x_data, y_data)

# Generate coorelation coefficient
# r = regr.coef_[0][0]
# plt.text((len(x_data) / 2 ) - 20, -90, "r = " + str(round(r, 3)), fontsize = 10)

# Plot residual lines
#Filter main df data -> df1, df2, df3 ...
# df1 = df.iloc[:100]
# df2 = df.iloc[100:]

# print(df1)
# print(df2)