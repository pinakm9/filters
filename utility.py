# A helper module for various sub-tasks
from time import time
import numpy as np

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print('Time taken by function {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func

def delta(x, x_0):
	"""
	Description:
		Dirac delta function

	Args:
		x: input
		x_0: point where the mass is located

	Returns:
	 	eiter 0.0 or 1.0
	"""
	return 1.0 if np.array_equal(x, x_0) else 0.0

class Picker(object):
    """
    A class defining an object-picker from an array
    """
    def __init__(self, array):
        """
        array = array of objects to pick from
        """
        self.array = array

    def equidistant(self, objs_to_pick, start_pt = 0):
        """
		Description:
        	Picks objs_to_pick equidistant objects starting at the location start_pt
        Returns:
			the picked objects
        """
        increment = int((len(self.array) - start_pt)/objs_to_pick)
        if increment < 1:
            return self.array
        else:
            new_array = [0]*objs_to_pick
            j = start_pt
            for i in range(objs_to_pick):
                new_array[i] = self.array[j]
                j += increment
        return np.array(new_array)
