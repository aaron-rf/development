

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

# Another example
class Animal:
    """
    A class used to represent an animal

    ...
    
    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    says_tr = "A {name} says {sound}"

    def __init__(self, name, sound, num_legs=4):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal has (default is 4)
        """

        self.name = name
        self.sound = sound
        self.num_legs = num_legs

    def says(self, sound=None):
        """Prints what the animals name is and what sound it makes.
        
        If the argument `sound` isn't passed in, the default Animal
        sound is used.
        
        Parameters
        ----------
        sound : str, optional
            The sound the animal makes (default is None)

        Raises
        ------
        NotImplementedError
            If no sound is set for the animal or passed in as a 
            parameter.
        """
        if self.sound is None and sound is None:
            raise NotImplementedError("Silent Animals are not supported!")
        
        out_sound = self.sound if sound is None else sound
        print(self.says_str.format(name=self.name, sound=out_sound))







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

print(Vehicles.__doc__)
# vehicles = Vehicles(arg=2)
# print(vehicles.calculate_consumption.__doc__)


### reStructuredText Format for Docstrings ###
def foo(var1, var2, *args, **kwargs):
    """ Summary o fthe function i one line

    Several sentences providing an extended description. Refer to 
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1: array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.
    var 2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. `(N, ) ndarray``or ``array_like``.
    *args : iterable
        Other arguments
    **kwargs : iterable
        Other arguments

    Returns
    -------
    out : type
        Explanation of `out`

    Raises
    ------
    Exception
        Explanation of the exception
    
    """
    out = var1 + var2
    return out





### Docstring in Scripts ###
"""Spreadsheet Column Printer

This script allows the user to print to the console all columns in the
spreadsheet. It is assumed that the first row of the spreadsheet is the 
location of the columns.

This tool accepts comma separated value files (.csv) as well as excel
(.xls, .xlsx) files.

This script requires that `pandas` is installed within the Python
environment you are running this script in.

This file can also be importaed as a module and contains the following
functions:

    * get_spreadsheet_cols - returns the column headers of the file
    * main - the main function of the script

"""
import argparse

import pandas as pd

def get_spreadsheet_cols(file_loc, print_cols=False):
    """Gets and prints the spreadsheet's header columns
    
    Parameters
    ----------
    file_loc : str
        The file location of the spreadsheet
    print_cols : bool, optional
        A flag used to print the columns to the console (default is
        False)
    
    Returns
    -------
    list
        a list of strings used that are the header columns
    """

    file_data = pd.read_excel(file_loc)
    col_headers = list(file_data.columns.values)

    if print_cols:
        print("\n".join(col_headers))
    
    return col_headers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_file',
        type=str,
        help="The spreadhseet file to print the columns on"
    )
    args = parser.parse_args()
    get_spreadsheet_cols(args.input_file, print_cols=True)


if __name__ == "__main__":
    main()





