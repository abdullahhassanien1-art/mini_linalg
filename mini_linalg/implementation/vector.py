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