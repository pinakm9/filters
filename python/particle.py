class C1():
    def __init__(self, func, val):
        self.func = func
        self.val = val


obj1 = C1(func = lambda x: x+1, val = 42)
func = lambda : obj1.val*10
obj2 = C1(func = func, val = 43)
print(obj2.func())
obj1.val += 1
print(obj1.val, obj2.func())
