class Point:
    POINT_TYPE = {'END': 0, 'LINE': 1}
    _x = None
    _y = None
    _type = None
    _color = None
    _size = None
    def __init__(self, x, y, type=POINT_TYPE['LINE'], color=(0, 0, 0), size=2):
        self._x = int(x)
        self._y = int(y)
        self._type = type
        self._color = color
        self._size = size
    

    def get_coords(self):
        return (self._x, self._y)
    

    def get_type(self):
        return self._type


    def get_color(self):
        return self._color

    
    def get_size(self):
        return self._size