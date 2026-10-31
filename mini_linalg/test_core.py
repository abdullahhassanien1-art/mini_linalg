import pytest

from mini_linalg import Matrix, Vector, batch_matvec, batch_matmul


def test_vector_creation():
    v = Vector([1.0, 2.0, 3.0])
    assert list(v.data) == [1.0, 2.0, 3.0]


def test_matrix_creation():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert [list(row) for row in m.data] == [[1.0, 2.0], [3.0, 4.0]]


def test_vector_repr():
    assert repr(Vector([1, 2])) == "Vector([1, 2])"


def test_matrix_shape():
    assert Matrix([[1, 2, 3], [4, 5, 6]]).shape == (2, 3)


def test_matvec_identity():
    eye = Matrix([[1, 0], [0, 1]])
    assert list(eye.matvec(Vector([3, 4])).data) == [3, 4]


def test_matmul_shape():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[1, 0], [0, 1], [1, 1]])
    assert A.matmul(B).shape == (2, 2)


def test_matvec_dimension_mismatch():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        A.matvec(Vector([1, 2]))


def test_matmul_dimension_mismatch():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        A.matmul(B)

def test_batch_matvec():
        matrices = [
            Matrix([[1, 2], [3, 4]]),
            Matrix([[2, 0], [0, 2]]),
        ]
        vectors = [Vector([1, 1]), Vector([3, 4])]

        result = batch_matvec(matrices, vectors)

        assert [vector.data for vector in result] == [[3, 7], [6, 8]]


def test_batch_matmul():
    left_matrices = [
        Matrix([[1, 2], [3, 4]]),
        Matrix([[1, 0], [0, 1]]),
    ]
    right_matrices = [
        Matrix([[2, 0], [0, 2]]),
        Matrix([[5, 6], [7, 8]]),
    ]

    results = batch_matmul(left_matrices, right_matrices)

    assert [matrix.data for matrix in results] == [
        [[2, 4], [6, 8]],
        [[5, 6], [7, 8]],
    ]

def test_batch_matvec_rejects_different_batch_sizes():
    matrices = [Matrix([[1, 0], [0, 1]])]
    vectors = [Vector([1, 2]), Vector([3, 4])]

    with pytest.raises(ValueError):
        batch_matvec(matrices, vectors)

def test_batch_matmul_rejects_different_batch_sizes():
    left_matrices = [Matrix([[1, 0], [0, 1]])]
    right_matrices = [
        Matrix([[1, 0], [0, 1]]),
        Matrix([[2, 0], [0, 2]]),
    ]

    with pytest.raises(ValueError):
        batch_matmul(left_matrices, right_matrices)

