import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(600000)
plt.hist(x, bins=200,color='#2a9d8f')
ax = plt.gca()
ax.set_yticklabels([])
plt.show()
