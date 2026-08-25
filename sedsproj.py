# Pandas is used for reading and processing the CSV data
import pandas as pd

# Matplotlib is used for creating and displaying the graph
import matplotlib.pyplot as p

# 1. READ THE DATA
data = pd.read_csv("Depth Data.csv")  # Reads the Depth Data.csv file and stores it in tabular form

# Extract the "Depth (m)" column from the table
# to_numeric() makes sure all the values are treated as numbers
# errors="coerce" converts any invalid/non-numerical values into NaN(Not a Number)
depth = pd.to_numeric(data["Depth (m)"], errors="coerce")

# 2. HANDLE MISSING VALUES
# If there are missing values, interpolate() estimates them using the valid values before and after the missing point.
depth = depth.interpolate()

# 3. FIND SUDDEN SPIKES IN THE SENSOR DATA

# Calculate the rolling median using 5 readings at a time.
# "center=True" means the window is centred around each reading.
# "min_periods=1" allows the calculation to work even at the beginning and the end of the dataset where fewer readings are available.
# The median is useful because it is less affected by extreme values than an average.
rolling_median = depth.rolling(
    5,
    center=True,
    min_periods=1
).median()

# deviation calculates how far each actual reading is from its local median.
# abs() gives the absolute value, so we only care about the size of the difference and not whether it is positive or negative.
# A very large deviation means that the sensor reading is behaving very differently from the readings around it.
deviation = (depth - rolling_median).abs()

# Calculate the local Median Absolute Deviation (MAD).
# This gives us an idea of how much the readings normally fluctuate around each point. Smaller MAD means readings are similar in depth.

local_mad = deviation.rolling(
    5,
    center=True,
    min_periods=1
).median()


# Prevent the MAD value from becoming too small.
# If the MAD were extremely close to zero, dividing by it could produce an extremely large spike score(it's there below).
# We use 5.0 as a minimum allowed value.
local_mad_floor = local_mad.clip(lower=5.0)

# 4. CALCULATE HOW UNUSUAL EACH READING IS

# Compare the deviation of each reading with the normal local deviation.
# A high spike_score means the reading is much further away from its surroundings than normal.
spike_score = deviation / local_mad_floor
# Any reading with a spike score greater than 6 is considered an extreme spike/outlier.
is_spike = spike_score > 6
# Make a copy of the original depth data so that we can modify the copy without losing the original readings.
depth_clean = depth.copy()

# Replace the detected spikes with NaN.
# Instead of allowing an extreme reading to distort the graph, we temporarily remove it and treat it as a missing value.
depth_clean[is_spike] = float("nan")

depth_clean = depth_clean.interpolate() # Estimate the values that replaced the spikes using interpolate().

# 5. SMOOTH THE DATA

# Apply a rolling average using 7 readings at a time.
# Instead of plotting every individual sensor fluctuation, the graph uses the average of nearby readings.
# This reduces small random fluctuations and produces a smoother curve that represents the overall depth trend.
depth_smooth = depth_clean.rolling(
    7,
    center=True,
    min_periods=1
).mean()

# 6. CREATE THE TIME AXIS
time = range(len(depth_smooth))  #the sensor reads time for every second

# 7. SET UP THE ANIMATED GRAPH

# Import Matplotlib's animation functionality
import matplotlib.animation as animation

# Create the figure and the graph's axes
fig, ax = p.subplots()
ax.set_xlim(0, len(depth_smooth)) # The graph starts at 0 seconds and ends at the final reading.
ax.set_ylim(
    depth_smooth.min() - 20,
    depth_smooth.max() + 20
) # Added some buffer value so that the graph doesnt touch the axes.

# Adding labels
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Depth (metres)")

ax.set_title("Ship Depth vs Time") #Tute of graph
ax.grid() # Adding a grid to make it easier to read values from the graph

# Create an empty line. The animation will gradually add points to this line.
line, = ax.plot([])

# 8. UPDATE THE GRAPH

# This function controls what happens during each frame of the animation.
def update(frame):

    # Add all the points from the beginning of the dataset up to the current frame.
    line.set_data(
        range(frame + 1),
        depth_smooth[:frame + 1]
    )
    return line, # Return the updated line to the animation

# 9. RUN THE ANIMATION
 
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(depth_smooth), #creates one frame for every depth reading
    interval=1000,
    repeat=False  #the animation stops after reaching the final reading
)


# Display the animated graph
p.show()