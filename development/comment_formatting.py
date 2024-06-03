

### One-Line Docstring ###
def sum(a, b):
    """Return the sum of the two parameters"""
    return a + b

# print(sum.__doc__)
# print(help(sum))



### Multi-Line Docstring ###
def string_reverse(str_1):
    """
    Return the reversed string

    Parameters:
        str_1 (str): The string which is to be reversed

    Returns:
        reverse (str): The string which gets reversed
    
    """
    reverse_str_1 = ""
    i = len(str_1)
    while i > 0:
        reverse_str_1 += str_1[i - 1]
        i = i - 1
    
    return reverse_str_1

#print(string_reverse.__doc__)


### Class Docstring ###
class MyClass:
    """This is the documentation for MyClass."""

    def __init__(self):
        """This is the documentation fo rthe __init__ method"""
        super.__init__()
        
# print(MyClass.__doc__)


### Google's Docstring Style ###
class Vehicles(object):
    """
    The Vehicle object contains a lot of vehicle

    Args:
        arg (str): The arg is used for...
        *args: The variable arguments are used for...
        **kwargs: The keyword arguments are used for...

    Attributes:
        arg (str): This is the attribute for the class
    """
    def __init__(self, arg, *args, **kwargs):
        self.arg = arg

    def calculate_consumption(self, distance, destination):
        """Function to calculate consumption of the car

        Args:
            distance (int): The amount of distance travelled
            destination (bool): If the destination is special

        Raises:
            RuntimeError: if the distance is too much for the curreent level of fuel

        Returns:
            consumption: the car's total consumption
        """
        consumption = distance * 4
        if destination:
            consumption *= 2
        
        if consumption > 100:
            raise RuntimeError("Too much distance")

        return consumption

# print(Vehicles.__doc__)
# vehicles = Vehicles(arg=2)
# print(vehicles.calculate_consumption.__doc__)