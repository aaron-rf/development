

import pytest 

from development.class_management import * 

@pytest.fixture
def create_line():

    x1 = 4 
    x2 = 8
    y1 = 10
    y2 = 20

    line = Line(x1, x2, y1, y2)
    return line 

def test_length_property(create_line):
    expected_length = (
        math.sqrt(
            (create_line.x2 - create_line.x1)**2 
            + (create_line.y2 - create_line.y1)**2
            )
    )
    assert create_line.length == expected_length

def test_changing_length_property(create_line):
    with pytest.raises(AttributeError):
        create_line.length = -100

def test_dict_info_property(create_line):
    expected_dict = {
            "key_1": 100,
            "key_2": [200, 201, 202],
            "key_3": {"key_3_1": 3.1, "key_3_2": 3.2}
        }
    assert create_line.dict_info == expected_dict
