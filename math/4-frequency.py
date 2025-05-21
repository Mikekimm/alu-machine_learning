#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create the histogram
plt.figure(figsize=(10, 6))

plt.hist(student_grades,
         bins=range(0, 101, 10),  # Bins every 10 units from 0 to 100
         edgecolor='black',  # Black outline for bars
         )

plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

# Set x-axis range to cover all possible grades (0-100)
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.show()
