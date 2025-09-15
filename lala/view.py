import math

class View(tuple):
    def __new__(cls, iterable):
        obj = super().__new__(cls, iterable)
        obj.strides = ()
        return obj
    
    def numel(self):
        return math.prod(self)
