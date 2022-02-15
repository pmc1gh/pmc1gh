# Philip Carr
# February 8, 2020
# Vector.py

import math

class Vector(list):
    """A list-based vector class"""

    # def __init__(self, n):
    #     """Constructor for Vector"""
    #     self.components = [0] * n

    def __init__(self, lst):
        """Constructor for Vector"""
        self.components = lst

    def __repr__(self):
        """Printout for Vector"""
        return str(self.components)

    def __getitem__(self, i):
        return self.components[i]

    def __setitem__(self, i, x):
        self.components[i] = x

    def __len__(self):
        return len(self.components)

    def __neg__(self):
        """Negation of the vector"""

        return Vector(list(map(lambda x: -x, self.components)))

    def __add__(self, other):
        """Element-by-element addition, or addition of constant"""

        try:
            return Vector(list(map(lambda x, y: x+y, self.components, other.components)))
        except:
            return Vector(list(map(lambda x: x + other, self.components)))

    def __sub__(self, other):
        """Element-by-element subtraction, or subtraction of constant"""

        try:
            return Vector(list(map(lambda x, y: x-y, self.components, other.components)))
        except:
            return Vector(list(map(lambda x: x - other, self.components)))

    def __mul__(self, other):
        """Element-by-element multiplication, or multiplication of constant"""

        try:
            return Vector(list(map(lambda x, y: x*y, self.components, other.components)))
        except:
            return Vector(list(map(lambda x: x * other, self.components)))

    def __truediv__(self, other):
        """Element-by-element division, or division of constant"""

        try:
            return Vector(list(map(lambda x, y: x/y, self.components, other.components)))
        except:
            return Vector(list(map(lambda x: x / other, self.components)))

    def __radd__(self, other):
        """Element-by-element addition, or addition of constant (from the right)"""

        try:
            return Vector(list(map(lambda x, y: x+y, self.components, other.components)))
        except:
            return Vector(list(map(lambda x: other + x, self.components)))

    def __rsub__(self, other):
        """Element-by-element subtraction, or subtraction of constant (from the right)"""

        try:
            return Vector(list(map(lambda x, y: x-y, self.components, other.components)))
        except:
            return Vector(list(map(lambda x: other - x, self.components)))

    def __rmul__(self, other):
        """Element-by-element multiplication, or multiplication of constant (from the right)"""

        try:
            return Vector(list(map(lambda x, y: x*y, self, other.components)))
        except:
            return Vector(list(map(lambda x: other * x, self.components)))

    def __rtruediv__(self, other):
        """Element-by-element division, or division of constant (from the right)"""

        try:
            return Vector(list(map(lambda x, y: x/y, self, other.components)))
        except:
            return Vector(list(map(lambda x: other / x, self.components)))

    def dot(self, other):
        """Return the dot product of two vectors"""

        return sum(list(map(lambda x, y: x * y, self.components, other.components)))

    def cross(self, other):
        """Return the cross product of two vectors in R^3"""
        assert len(self) == 3 and len(other) == 3, \
               "vectors for cross product must be in R^3"
        i_comp = self[1] * other[2] - self[2] * other[1]
        j_comp = self[2] * other[0] - self[0] * other[2]
        k_comp = self[0] * other[1] - self[1] * other[0]
        return Vector([i_comp, j_comp, k_comp])

    def mag(self):
        """Return the magnitude of a vector"""
        return math.sqrt(self.dot(self))
