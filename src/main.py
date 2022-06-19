'''
Title: Gyroscope Steering Visualizer
Description: This program will take in a gyroscope data 
             file and display chunks of data on a graph,
             to analyze it and determine if smooth turning
             is existent.
Author: Tegan Hakim & Tyler Lumpkin
Github: https://github.com/TeganHakim/Arduino-Gyroscope
'''

# Imports
import pandas as pd
import graph

# Constants
ROLL_RADIUS = 180
ERROR_DIST = 25.0

# Load CSV of Gyroscope Data
df = pd.read_csv("../test_data.csv")

# Seperate axis (x, y, z) into distinct data frames
x = df['time']
y = df['roll']
z = df['yaw']

# Initialize datasets & Populate w/ (x, y, z)
x_data = []
y_data = []
z_data = []

for i in range(len(df)):
    x_data.append([x[i]])
    y_data.append([y[i]])
    z_data.append([z[i]])

# Graph raw data
fig, axs = graph.plot_raw_outputs(df, x_data, y_data, z_data, ROLL_RADIUS)

# Graph residuals
graph.residuals(df, axs, x_data, y_data, z_data)

# Graph details
graph.details(axs, x_data, y_data, z_data, ROLL_RADIUS)

# Calculate intersection of line and x-axis for chunk detection
idx = graph.point_intersection(df, fig, axs, x, y, z)

# Chunking algorithm
chunks = graph.chunk(df, axs, idx, ERROR_DIST)

# Graph chunks
predictions = graph.draw_chunks(chunks)

# Evalutate chunks
graph.evaluate_chunks(fig, predictions)

# Show plots
graph.show_plots()