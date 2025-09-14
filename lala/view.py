class View(tuple):
    def __new__(cls, iterable):
        return super().__new__(cls, iterable)
