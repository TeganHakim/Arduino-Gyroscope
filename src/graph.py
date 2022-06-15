from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import math

# Plot raw outputs
def plot_raw_outputs(df, x_data, y_data, ROLL_RADIUS):
    plt.figure("Data")
    plt.title('Gyroscope Data')
    plt.xlabel('Time')
    plt.ylabel('Roll')
    plt.xticks(([i for i in range(len(df)) if i % 25 == 0]))
    plt.yticks(([i for i in range(-ROLL_RADIUS, ROLL_RADIUS) if i % 25 == 0]))
    plt.scatter(x_data, y_data, color = 'black')
    
# Residuals = Observed Value - Predicted Value
def residuals(df, x_data, y_data):
    residuals = {"x": [], "y": []}
    for i in range(len(df)):
        residuals["x"].append(x_data[i])
        residuals["y"].append(y_data[i])
        residuals["x"].append(x_data[i])
        residuals["y"].append([0])
    plt.plot(residuals["x"], residuals["y"], color = "#d3d3d3", linewidth = 1)

# Graph details
def details(x_data, y_data, ROLL_RADIUS):
     # Plot connection line between points
    plt.plot(x_data, y_data, color = 'black', linewidth = 3)
    # Plot x-axis line
    plt.plot([0, ROLL_RADIUS], [0, 0], color = "red", linewidth = 3)

# Calculate intersection of line and x-axis for chunk detection
def point_intersection(df, x, y):
    x_axis_y = [0 for i in range(len(df))]
    idx = np.argwhere(np.diff(np.sign(np.array(x_axis_y) - np.array(y)))).flatten()
    idx = list(idx)
    idx.insert(0, 0)
    idx = np.array(idx)
    plt.plot(np.array(x)[idx], np.array(x_axis_y)[idx], color = "red", marker = "o", markersize = 10)

    # Draw dashed line depicting boundaries of all chunks
    for i in range(len(idx)):
        plt.axvline(x = idx[i], color = "red", linestyle = "--")

    return idx

#Splits data into chunks based on the values that lie between 2 adjacent x-intercepts
def chunk(df, idx, ERROR_DIST):
    chunks = []
    for i in range(len(idx)):
        if (i != len(idx) - 1):
            chunks.append(df.iloc[idx[i]: idx[i+1]])
        else:
            chunks.append(df.iloc[idx[i]:])

    # "Ignorechunk" algorithm for polluted data
    rem_idx = []
    for i in range(len(chunks)):   
        counter = 0 
        for j in range(len(chunks[i]["roll"])):
            chunks_data = list(chunks[i]["roll"])[j]  
            
            # Detects if the chunk is too small 
            if float(chunks_data) > float(-ERROR_DIST) and float(chunks_data) < float(ERROR_DIST):   
                counter += 1

        if counter == len(chunks[i]["roll"]):
            rem_idx.append(i) 
    
    # Delete the small chunks
    chunks = np.delete(np.array(chunks, dtype = object), rem_idx)
    
    return chunks

    
#Display the filtered chunk graphs
def draw_chunks(chunks):     
    num_plots = math.ceil(len(chunks) / 2.) * 2
    plt.figure("Chunks")
    for i in range(2):
        for j in range(int(num_plots / 2)):
            # Creates multiple plots for each chunk
            ax = plt.subplot2grid((2, int(num_plots / 2)), (i,j))   
            row = num_plots / 2 if i == 1 else 0
            ax.scatter(chunks[int(j + row)]["time"], chunks[int(j + row)]["roll"], color = 'black', linewidth = 3)
            
            # Polynomial Regression
            poly = PolynomialFeatures(degree = 2, include_bias = False)
            poly_features = poly.fit_transform(np.array(chunks[int(j + row)]["time"]).reshape(-1, 1))
            poly_reg_model = LinearRegression()
            poly_reg_model.fit(poly_features, chunks[int(j + row)]["roll"])
            y_predicted = poly_reg_model.predict(poly_features)
            plt.plot(np.array(chunks[int(j + row)]["time"]), y_predicted, color = "red", linewidth = 3)

            # R-squared value (coorelation coefficient) to determine if the model follows an ideal path
            r_squared = poly_reg_model.score(poly_features, chunks[int(j + row)]["roll"])            
            plt.rcParams.update({'font.size': 6})
            plt.title("r² = " + str(round(r_squared, 3)))
            
    plt.show()