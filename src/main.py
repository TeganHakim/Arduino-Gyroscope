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
df = pd.read_csv("../data.csv")

# Seperate axis (x, y) into distinct data frames
x = df['time']
y = df['roll']

# Initialize datasets & Populate w/ (x,y)
x_data = []
y_data = []

for i in range(len(df)):
    x_data.append([x[i]])
    y_data.append([y[i]])

# Graph raw data
graph.plot_raw_outputs(df, x_data, y_data, ROLL_RADIUS)

# Graph residuals
graph.residuals(df, x_data, y_data)

# Graph details
graph.details(x_data, y_data, ROLL_RADIUS)

# Calculate intersection of line and x-axis for chunk detection
idx = graph.point_intersection(df, x, y)

# Chunking algorithm
chunks = graph.chunk(df, idx, ERROR_DIST)

# Graph chunks
graph.draw_chunks(chunks)