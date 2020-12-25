import numpy as np
import matplotlib.pyplot as plt
A = np.random.random((2,2))
B = np.eye(2) - A
x = np.random.random(2)
o = np.random.random(2)
x_ = np.dot(A, x) + np.dot(B, o)
plt.scatter(x[0], x[1], label = 'x')
plt.scatter(o[0], o[1], label = 'o')
plt.scatter(x_[0], x_[1], label = 'x_')
plt.legend()
plt.show()
