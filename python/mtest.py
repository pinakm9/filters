class C(object):
    def __init__(self, val):
        self.val = val

class D(object):
    def __init__(self, f, **args):
        self.f = lambda x: f(x, **args)

def f(x, obj):
    print(obj.val)

obj = C(50)
F = lambda x: f(x, obj)
"""
# test 1
F(0)
obj.val = 10
F(0)
"""

# test 2
obj_ = D(f, obj = obj)
obj_.f(0)
obj.val = 10
obj_.f(0)

#test 3
def g(x, y):
    print(y)
p = [1,2,3]
g_ = lambda x: g(x,p[1])
g_(7)
p[1]=5
g_(7)
p[1] = 8
g_(7)

#test 4
m = p[1]
def g__(x):
    return g(x, m)

g_(7)
p[1]=5
g_(7)
p[1] = 8
g_(7)

#test 4
s = lambda : p[1]
q = s()
print(q)
p[1] = -273
print(q)
