import math


class Line:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.__length = self.calculate_length()

    def calculate_length(self):
        length = math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        return length

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, dummy):
        raise AttributeError("You cannot change the length!")

    @property
    def dict_info(self):
        return {
            "key_1": 100,
            "key_2": [200, 201, 202],
            "key_3": {"key_3_1": 3.1, "key_3_2": 3.2},
        }
