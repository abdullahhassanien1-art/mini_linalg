import pytest
from mini_linalg.core import Vector, Matrix

def test_vector_addition():
    v1 = Vector([1.0, 2.0])
    v2 = Vector([3.0, 4.0])
    v3 = v1 + v2
    assert v3.data == [4.0, 6.0]

def test_vector_dot():
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])
    assert v1.dot(v2) == 32.0

def test_matrix_vector_multiply():
    mat = Matrix([[1.0, 2.0], [3.0, 4.0]])
    vec = Vector([5.0, 6.0])
    res = mat @ vec
    assert isinstance(res, Vector)
    assert res.data == [17.0, 39.0]

def test_matrix_matrix_multiply():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    res = m1 @ m2
    assert isinstance(res, Matrix)
    assert res.data == [[19.0, 22.0], [43.0, 50.0]]

def test_dimension_mismatch_raises():
    v1 = Vector([1.0, 2.0])
    v2 = Vector([3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        _ = v1 + v2