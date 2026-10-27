import pytest

from mini_linalg import Matrix, Vector


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