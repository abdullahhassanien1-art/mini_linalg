from collections.abc import Sequence

import numpy as np

from .vector import Vector


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


def batch_matvec(
    matrices: Sequence[Matrix], vectors: Sequence[Vector]
) -> list[Vector]:
    """Multiply each matrix by its corresponding vector in a batch."""
    if len(matrices) != len(vectors):
        raise ValueError("The number of matrices and vectors must match.")

    return [matrix @ vector for matrix, vector in zip(matrices, vectors)]


def batch_matmul(
    left_matrices: Sequence[Matrix], right_matrices: Sequence[Matrix]
) -> list[Matrix]:
    """Multiply corresponding matrices in two batches."""
    if len(left_matrices) != len(right_matrices):
        raise ValueError("The two matrix batches must have the same size.")

    return [left @ right for left, right in zip(left_matrices, right_matrices)]