class Vector:
    def __init__(self, data: list[float]):
        self.data = data
    
    def __repr__(self) -> str:
        return f"Vector({self.data})"

    def __add__(self, other: 'Vector') -> 'Vector':
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same dimension for addition.")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: 'Vector') -> 'Vector':
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same dimension for subtraction.")
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def dot(self, other: 'Vector') -> float:
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same dimension for dot product.")
        return sum(a * b for a, b in zip(self.data, other.data))


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = data
        
    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def __matmul__(self, other):
        """Supports Matrix-Matrix and Matrix-Vector multiplication using the @ operator or custom methods."""
        if isinstance(other, Vector):
            # Matrix-Vector product: (m x n) @ (n,) -> (m,)
            m_rows = len(self.data)
            n_cols = len(self.data[0])
            if n_cols != len(other.data):
                raise ValueError("Matrix column count must match vector dimension.")
            
            result = [
                sum(self.data[i][j] * other.data[j] for j in range(n_cols))
                for i in range(m_rows)
            ]
            return Vector(result)

        elif isinstance(other, Matrix):
            # Matrix-Matrix product: (m x n) @ (n x p) -> (m x p)
            self_rows = len(self.data)
            self_cols = len(self.data[0])
            other_rows = len(other.data)
            other_cols = len(other.data[0])

            if self_cols != other_rows:
                raise ValueError("Matrix inner dimensions must match for multiplication.")

            result = [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self_cols))
                    for j in range(other_cols)
                ]
                for i in range(self_rows)
            ]
            return Matrix(result)
        
        raise TypeError("Unsupported operand type for @")