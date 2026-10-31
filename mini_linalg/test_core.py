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


def test_vector_add():
    assert list((Vector([1, 2, 3]) + Vector([4, 5, 6])).data) == [5, 7, 9]


def test_vector_sub():
    assert list((Vector([4, 5, 6]) - Vector([1, 2, 3])).data) == [3, 3, 3]


def test_vector_dot():
    assert Vector([1, 2, 3]).dot(Vector([4, 5, 6])) == 32


def test_matvec():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    v = Vector([1, 0, 2])
    assert list((A @ v).data) == [7, 16]


def test_matvec_identity():
    eye = Matrix([[1, 0], [0, 1]])
    assert list((eye @ Vector([3, 4])).data) == [3, 4]


def test_matmul():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[1, 0], [0, 1], [1, 1]])
    assert [list(row) for row in (A @ B).data] == [[4, 5], [10, 11]]


def test_matmul_result_size():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[1, 0], [0, 1], [1, 1]])
    result = A @ B
    assert len(result.data) == 2
    assert len(result.data[0]) == 2


def test_vector_add_dimension_mismatch():
    with pytest.raises(ValueError):
        Vector([1, 2]) + Vector([1, 2, 3])


def test_vector_dot_dimension_mismatch():
    with pytest.raises(ValueError):
        Vector([1, 2]).dot(Vector([1, 2, 3]))


def test_matvec_dimension_mismatch():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        A @ Vector([1, 2])


def test_matmul_dimension_mismatch():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        A @ B


def test_matmul_unsupported_type():
    A = Matrix([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        A @ 5