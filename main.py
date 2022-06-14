import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load and Parse Data
df = pd.read_csv("data.csv")

fig, axis = plt.subplots(nrows=2, ncols=1)

x = df['time']
y = df['roll']
yVel = np.gradient(y)


x_data = []
y_data = []
yVel_data = []


for i in range(len(df)):
    x_data.append([x[i]])
    y_data.append([y[i]])
    yVel_data.append([yVel[i]])

# Plot raw outputs
plt.scatter(x_data, y_data, color = 'black')
plt.title('Gyroscope Data')
plt.xlabel('Time')
plt.ylabel('Roll')
plt.xticks(([i for i in range(len(df)) if i % 25 == 0]))
plt.yticks(([i for i in range(-180, 180) if i % 25 == 0]))

# Create linear regression object
regr = LinearRegression()

# Train the model
regr.fit(x_data, y_data)

# Plot outputs
plt.plot(x_data, regr.predict(x_data), color = 'red', linewidth = 3)
plt.plot(x_data, yVel_data, color = 'green', linewidth = 3)

# Residuals = Observed Value - Predicted Value
residuals = {"x": [], "y": []}
for i in range(len(df)):
  residuals["x"].append(x_data[i])
  residuals["y"].append(y_data[i])
  residuals["x"].append(x_data[i])
  residuals["y"].append(regr.predict(x_data)[i])

# Generate coorelation coefficient
r = regr.coef_[0][0]
plt.text((len(x_data) / 2 ) - 20, -90, "r = " + str(round(r, 3)), fontsize = 10)

# Plot residual lines
plt.plot(residuals["x"], residuals["y"], color = "blue", linewidth = 1)

plt.show()




#Filter main df data -> df1, df2, df3 ...
#df1 = df.iloc[:100]
#df2 = df.iloc[100:]

#print(df1)
#print(df2)