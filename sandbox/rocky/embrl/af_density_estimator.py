# toy task: fitting a uniform distribution using gaussian r.v.s as inputs
import numpy as np
import matplotlib.pyplot as plt


data = np.random.uniform(low=-1, high=1, size=(1000, 2))


plt.plot(data, 'o', color='b')
# plt.xlim(-2, 2)
plt.show()

# Can use MADE?
