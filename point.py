class Point:
    POINT_TYPE = {'END': 0, 'LINE': 1}
    _x = None
    _y = None
    _type = None
    def __init__(self, x, y, type=POINT_TYPE['LINE']):
        self._x = int(x)
        self._y = int(y)
        self._type = type
    

    def get_coords(self):
        return (self._x, self._y)
    

    def get_type(self):
        return self._type

