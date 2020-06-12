def f (a, z,  b, c):
    print("(a,z,  b, c) = {}".format((a,z, b, c)))
    return a+b+c

def g(fn, **algorithm_args):
    return lambda *a: f(*a, **algorithm_args)


f1 = g(f, **{'b': 3, 'c': 4})
f1()
