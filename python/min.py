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

res = opt.minimize(F, x0, method='BFGS', jac=F_der)#, options={'gtol': 1e-6, 'disp': True})
#print(res)

xi = y
rho = np.dot(xi, xi)
eta = xi/np.sqrt(rho)

def F1(z):
    return F(res.x + z*eta) - res.fun - 0.1

def J(z):
    return [z**(d-1)*rho**(1-0.5*d)/np.dot(eta, F_der(res.x + z*eta))]

print("......", J(9))

@ut.timer
def g():
    return opt.fsolve(F1, 0.001, fprime = J)
res1 = g()
print(res1, F1(res1))
