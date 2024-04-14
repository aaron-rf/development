from hello import *


def test_add():
    result = add(1, 2)
    assert result == 1 + 2


def test_multiply():
    result = multiply(2, 4)
    assert result == 2 * 4
