#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fig, ax = plt.subplots()
ax.bar(['Farrah', 'Fred', 'Felicia'], fruit[0],
       color='red', width=0.5, label='Apples')
ax.bar(['Farrah', 'Fred', 'Felicia'], fruit[1], color='yellow',
       width=0.5, bottom=fruit[0], label='Bananas')
ax.bar(['Farrah', 'Fred', 'Felicia'], fruit[2], color='#ff8000',
       width=0.5, bottom=fruit[0] + fruit[1], label='Oranges')
ax.bar(['Farrah', 'Fred', 'Felicia'], fruit[3], color='#ffe5b4',
       width=0.5, bottom=fruit[0] + fruit[1] + fruit[2], label='Peaches')
ax.set_ylabel('Quantity of Fruit')
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_title('Number of Fruit per Person')
ax.legend()
plt.show()
