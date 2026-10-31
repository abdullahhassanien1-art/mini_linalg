import numpy as np


class Vector:
    def __init__(self, data: list[float]):
        self.data = np.array(data)

    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"

    def __add__(self, other: "Vector") -> "Vector":
        if self.data.shape != other.data.shape:
            raise ValueError("Vectors must have the same dimension for addition.")
        return Vector(self.data + other.data)

    def __sub__(self, other: "Vector") -> "Vector":
        if self.data.shape != other.data.shape:
            raise ValueError("Vectors must have the same dimension for subtraction.")
        return Vector(self.data - other.data)

    def dot(self, other: "Vector") -> float:
        if self.data.shape != other.data.shape:
            raise ValueError("Vectors must have the same dimension for dot product.")
        return float(self.data @ other.data)


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = np.array(data)

    def __repr__(self) -> str:
        return f"Matrix({self.data.tolist()})"

    def __matmul__(self, other):
        """Matrix @ Vector or Matrix @ Matrix, using NumPy instead of loops."""
        if isinstance(other, Vector):
            if self.data.shape[1] != other.data.shape[0]:
                raise ValueError("Matrix column count must match vector dimension.")
            return Vector(self.data @ other.data)

        if isinstance(other, Matrix):
            if self.data.shape[1] != other.data.shape[0]:
                raise ValueError("Matrix inner dimensions must match for multiplication.")
            return Matrix(self.data @ other.data)

        raise TypeError("Unsupported operand type for @")