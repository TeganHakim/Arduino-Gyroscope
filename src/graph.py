from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import math

# Plot raw outputs
def plot_raw_outputs(df, x_data, y_data, z_data, ROLL_RADIUS):
    #Create the meta-data figure    
    fig, axs = plt.subplots(2, 1, figsize = (15, 15))
    fig.subplots_adjust(hspace = 0.5)
    fig.suptitle("Gyroscope Data", fontsize = 20)
    fig.canvas.set_window_title("Gyroscope Data")

    # Roll vs Time
    axs[0].set_title("Roll", fontsize = 13, fontweight = "bold")
    axs[0].set_xlabel("Time")
    axs[0].set_xticks(([i for i in range(len(df)) if i % 25 == 0])) 
    axs[0].set_ylabel("Roll")
    axs[0].set_yticks(([i for i in range(-ROLL_RADIUS, ROLL_RADIUS) if i % 25 == 0]))
    axs[0].scatter(x_data, y_data, color = 'black')

    # Yaw vs Time
    axs[1].set_title("Yaw", fontsize = 13, fontweight = "bold")
    axs[1].set_xlabel("Time")
    axs[1].set_xticks(([i for i in range(len(df)) if i % 25 == 0])) 
    axs[1].set_ylabel("Yaw")
    axs[1].set_yticks(([i for i in range(-ROLL_RADIUS, ROLL_RADIUS) if i % 25 == 0]))
    axs[1].scatter(x_data, z_data, color = 'black')

    return fig, axs
    
# Residuals = Observed Value - Predicted Value
def residuals(df, axs, x_data, y_data, z_data):
    residuals = {"x": [], "y": [], "z": []}
    for i in range(len(df)):
        residuals["x"].append(x_data[i])
        residuals["y"].append(y_data[i])
        residuals["z"].append(z_data[i])
        residuals["x"].append(x_data[i])
        residuals["y"].append([0])
        residuals["z"].append([0])
    axs[0].plot(residuals["x"], residuals["y"], color = "#d3d3d3", linewidth = 1)
    axs[1].plot(residuals["x"], residuals["z"], color = "#d3d3d3", linewidth = 1)

# Graph details
def details(axs, x_data, y_data, z_data, ROLL_RADIUS):
    # Plot connection line between points
    axs[0].plot(x_data, y_data, color = 'black', linewidth = 3)
    axs[1].plot(x_data, z_data, color = 'black', linewidth = 3)
    # Plot x-axis line
    axs[0].plot([0, ROLL_RADIUS + 10], [0, 0], color = "red", linewidth = 3)
    axs[1].plot([0, ROLL_RADIUS + 10], [0, 0], color = "red", linewidth = 3)

# Calculate intersection of line and x-axis for chunk detection
def point_intersection(df, fig, axs, x, y, z):
    x_axis_y = [0 for i in range(len(df))]
    idx = {"y": [], "z": []}

    # Roll data intersections
    idx["y"] = np.argwhere(np.diff(np.sign(np.array(x_axis_y) - np.array(y)))).flatten()
    idx["y"] = list(idx["y"])
    idx["y"].insert(0, 0)
    idx["y"] = np.array(idx["y"])
    axs[0].plot(np.array(x)[idx["y"]], np.array(x_axis_y)[idx["y"]], color = "red", marker = "o", markersize = 10)

    #Y aw data intersections
    idx["z"] = np.argwhere(np.diff(np.sign(np.array(x_axis_y) - np.array(z)))).flatten()
    idx["z"] = list(idx["z"])
    idx["z"].insert(0, 0)
    idx["z"] = np.array(idx["z"])
    axs[1].plot(np.array(x)[idx["z"]], np.array(x_axis_y)[idx["z"]], color = "red", marker = "o", markersize = 10)

    # Draw dashed line depicting boundaries of all chunks
    for i in range(len(idx["y"])):
        axs[0].axvline(x = idx["y"][i], color = "red", linestyle = "--")
    for i in range(len(idx["z"])):
        axs[1].axvline(x = idx["z"][i], color = "red", linestyle = "--")

    # Save Raw Data     
    fig.savefig("../tests/raw-data.png")

    return idx

# Splits data into chunks based on the values that lie between 2 adjacent x-intercepts
def chunk(df, axs, idx, ERROR_DIST):
    chunks = {"y": [], "z": []}

    # Insert Y-Chunk Data into Chunk array
    for i in range(len(idx["y"])):
        if (i != len(idx["y"]) - 1):
            chunks["y"].append(df.iloc[idx["y"][i]: idx["y"][i+1]])
        else:
            chunks["y"].append(df.iloc[idx["y"][i]:])

    # Insert Z-Chunk Data into Chunk array
    for i in range(len(idx["z"])):
        if (i != len(idx["z"]) - 1):
            chunks["z"].append(df.iloc[idx["z"][i]: idx["z"][i+1]])
        else:
            chunks["z"].append(df.iloc[idx["z"][i]:])

    # "Ignorechunk" algorithm for polluted data for Roll
    rem_idx = {"y": [], "z": []}
    for i in range(len(chunks["y"])):   
        counter_y = 0 
        for j in range(len(chunks["y"][i]["roll"])):
            chunks_data_y = list(chunks["y"][i]["roll"])[j]            
            # Detects if the chunk is too small 
            if float(chunks_data_y) > float(-ERROR_DIST) and float(chunks_data_y) < float(ERROR_DIST):   
                counter_y += 1           
        if counter_y == len(chunks["y"][i]["roll"]):
            rem_idx["y"].append(i) 
    # "Ignorechunk" algorithm for polluted data for Yaw
    for i in range(len(chunks["z"])):   
        counter_z = 0 
        for j in range(len(chunks["z"][i]["yaw"])):
            chunks_data_z = list(chunks["z"][i]["yaw"])[j]            
            # Detects if the chunk is too small 
            if float(chunks_data_z) > float(-ERROR_DIST) and float(chunks_data_z) < float(ERROR_DIST):   
                counter_z += 1           
        if counter_z == len(chunks["z"][i]["yaw"]):
            rem_idx["z"].append(i)
    
    # Delete the small chunks
    chunks["y"] = np.delete(np.array(chunks["y"], dtype = object), rem_idx["y"])
    chunks["z"] = np.delete(np.array(chunks["z"], dtype = object), rem_idx["z"])
    
    return chunks

    
# Display the filtered chunk graphs
def draw_chunks(chunks):     
    num_plots_y = math.ceil(len(chunks["y"]) / 2.) * 2
    num_plots_z = math.ceil(len(chunks["z"]) / 2.) * 2
    roll_y_predicted = 0
    yaw_y_predicted = 0

    # Create the figure of Roll Data Chunks
    plt.figure("Chunks - Roll")
    for i in range(2):
        for j in range(int(num_plots_y / 2)):
            # Creates multiple plots for Roll chunks
            ax = plt.subplot2grid((2, int(num_plots_y / 2)), (i,j))  
            plt.subplots_adjust(hspace = 0.5) 
            row = num_plots_y / 2 if i == 1 else 0
            ax.scatter(chunks["y"][int(j + row)]["time"], chunks["y"][int(j + row)]["roll"], color = 'black', linewidth = 3)
            
            # Polynomial Regression for Roll
            poly = PolynomialFeatures(degree = 2, include_bias = False)
            poly_features = poly.fit_transform(np.array(chunks["y"][int(j + row)]["time"]).reshape(-1, 1))
            poly_reg_model = LinearRegression()
            poly_reg_model.fit(poly_features, chunks["y"][int(j + row)]["roll"])
            roll_y_predicted = poly_reg_model.predict(poly_features)
            plt.plot(np.array(chunks["y"][int(j + row)]["time"]), roll_y_predicted, color = "red", linewidth = 3)

            # R-squared value (coorelation coefficient) to determine if the Roll follows an ideal path
            r_squared = poly_reg_model.score(poly_features, chunks["y"][int(j + row)]["roll"])            
            plt.rcParams.update({'font.size': 6})
            plt.title("r² = " + str(round(r_squared, 3)), fontsize = 13, fontweight = "bold", )

    # Save Roll Chunks     
    plt.savefig("../tests/chunks-roll.png")

    # Create the figure of Yaw Data Chunks
    plt.figure("Chunks - Yaw")
    for i in range(2):
        for j in range(int(num_plots_z / 2)):
            # Creates multiple Yaw chunks
            ax = plt.subplot2grid((2, int(num_plots_z / 2)), (i,j))   
            plt.subplots_adjust(hspace = 0.5)
            row = num_plots_z / 2 if i == 1 else 0
            ax.scatter(chunks["z"][int(j + row)]["time"], chunks["z"][int(j + row)]["yaw"], color = 'black', linewidth = 3)
                
            # Polynomial Regression for Yaw
            poly = PolynomialFeatures(degree = 2, include_bias = False)
            poly_features = poly.fit_transform(np.array(chunks["z"][int(j + row)]["time"]).reshape(-1, 1))
            poly_reg_model = LinearRegression()
            poly_reg_model.fit(poly_features, chunks["z"][int(j + row)]["yaw"])
            yaw_y_predicted = poly_reg_model.predict(poly_features)
            plt.plot(np.array(chunks["z"][int(j + row)]["time"]), yaw_y_predicted, color = "red", linewidth = 3)

            # R-squared value (coorelation coefficient) to determine if the Yaw follows an ideal path
            r_squared = poly_reg_model.score(poly_features, chunks["z"][int(j + row)]["yaw"])            
            plt.rcParams.update({'font.size': 6})
            plt.title("r² = " + str(round(r_squared, 3)), fontsize = 13, fontweight = "bold",)

    predictions = {"roll": roll_y_predicted, "yaw": yaw_y_predicted}

    # Save Yaw Chunks     
    plt.savefig("../tests/chunks-yaw.png")

    return predictions

# Evaluate all chunks
def evaluate_chunks(fig, predictions):
    # Find Steering Ratio and Map values using crazy fucking syntax
    ratio = ((sum(map(lambda x: abs(x), list(predictions["roll"]))) / len(predictions["roll"])) / (sum(map(lambda x: abs(x), list(predictions["yaw"]))) / len(predictions["yaw"])))
    fig.text(0.416, 0.925, "Steering Ratio: " + str(round(ratio, 3)) + ":1", fontsize = 14, fontweight = "bold")

    # Find the error in the predictions
    # avg_roll = ((sum(map(lambda x: abs(x), list(predictions["roll"])))) / len(predictions["roll"]))
    # avg_yaw = ((sum(map(lambda x: abs(x) * ratio, list(predictions["yaw"])))) / len(predictions["yaw"]))
    # if (avg_roll > avg_yaw):
    #     error = avg_roll - avg_yaw
    # else:
    #     error = avg_yaw - avg_roll    
    # print(error)

# Show all plots
def show_plots():
    plt.show()