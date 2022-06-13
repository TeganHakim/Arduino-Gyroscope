import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load CSV and columns
data = pd.read_csv("data.csv")

x = data['time']
y = data['roll']

# x = x.reshape(len(x),1)
# y = y.reshape(len(y),1)

# Split the data into training/testing sets
x_train = x[:-250]
x_test = x[-250:]

# Split the targets into training/testing sets
y_train = y[:-250]
y_test = y[-250:]

# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Plot outputs
plt.plot(x_test, regr.predict(x_test), color='red',linewidth=3)

plt.show()