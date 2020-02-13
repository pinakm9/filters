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
