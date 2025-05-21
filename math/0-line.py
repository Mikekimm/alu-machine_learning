#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# Create the x-axis values
x = np.arange(0, 11)
# Create the plot
plt.plot(x, y, 'r-')
# Show the plot
plt.show()
