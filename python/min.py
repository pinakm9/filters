# minimization test
import numpy as np
import scipy.optimize as opt
import utility as ut
# set parameters
d = 9
A = np.diag([1,2,3,4,5,4,3,2,1])
B = np.diag([1,1,1,1,1,1,1,1,1])
C = np.random.normal(size=(d,d))
D = np.random.normal(size=(d,d))
x0 = np.array([0.0]*d)
y = np.array([0.1]*d)
# function to minimize
def F(x):
    a = x - np.dot(C, x0)
    b = y - np.dot(D, x)
    #print(A.shape, a.shape)
    return 0.5*( np.dot(a.T, np.dot(A, a)) + np.dot(b.T, np.dot(B, b)) )

def F_der(x):
    a = x - np.dot(C, x0)
    b = np.dot(D, x) - y
    return np.dot(a.T, A) + np.dot(b.T, np.dot(B,D))

print(F(np.random.uniform(size=(d,))))

res = opt.minimize(F, x0, method='BFGS', jac=F_der, options={'gtol': 1e-6, 'disp': True})
print(res, F(res.x))
