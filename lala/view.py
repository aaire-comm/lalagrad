import math

class View(tuple):
    def __new__(cls, iterable):
        obj = super().__new__(cls, iterable)
        obj.strides = ()
        return obj
    
    def numel(self):
        return math.prod(self)
    
    def matches_buffer(self):
        return self._is_contigous
    
    def __eq__(self, value):
        return math.prod(self) == math.prod(value)


