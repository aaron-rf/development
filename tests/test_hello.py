from development.hello import *

import pytest

def test_add():
    result = add(1, 2)
    assert result == 1 + 2


@pytest.mark.parametrize("a, b, result_expected", [(2, 4, 8), (3, 0, 0), (-1, 2, -2)])
def test_multiply(a, b, result_expected):
    result = multiply(a, b)
    assert result == result_expected


@pytest.fixture
def get_random_prime():
    from random import choice
    list_primes = [1, 3, 5, 7, 11, 13]
    current_prime = choice(list_primes)
    return current_prime


def test_divisible_by_2(get_random_prime):
    assert get_random_prime % 2 != 0


class TestClass:
    property_1 = None

    @pytest.fixture(autouse=True)
    def setup(self):
        print("Setup called!")
        self.property_1 = 2

    def test_one(self, setup):
        assert self.property_1 * 5 == 10

    def test_two(self, setup):
        self.property_1 = self.property_1 * 2
        assert self.property_1 == 4
