
### Object Creation ###
class Example:
    def __new__(cls):
        # Called before __init__, it returns a new instance of the class
        print("Creating Instance")
        return super(Example, cls).__new__(cls)

    def __init__(self):
        # Called after the instance has been created, 
        # it initializes the object
        print("Initializing Instance")

#ex = Example()

### Object Representation ###
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        # Returns a user-friendly string representation of the object
        return f"{self.name}, aged {self.age}"
    
    def __repr__(self):
        # Returns an unambiguous string representation of the object;
        # ideallly one that could be used to recreate the object
        return f"Person('{self.name}', {self.age})"
    
#p = Person("Alice", 30)
#print(str(p))
#print(repr(p))


### Comparison Magic Methods ###
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __eq__(self, other):
        return self.title == other.title and self.author == other.author
    
#book1 = Book("2002", "Arthur")
#book2 = Book("2002", "Arthur")
#print(book1 == book2)


### Arithmetic and Bitwise Magic Methods ###
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

#v1 = Vector(2, 4)
#v2 = Vector(1, 3)
#v3 = v1 + v2 
#print(v3.x, v3.y)


### Attribute Access and Descriptor Methods ###
class ProtectedAttributes:
    def __init__(self):
        self._protected = "This is protected"
    
    def __getattr__(self, name):
        if name == "secret":
            raise AttributeError("Access Denied")
        return self.__dict__.get(name, f"{name} not found")
    
    def __setattr__(self, name, value):
        if name == "secret":
            raise AttributeError("Cannot modify secret")
        self.__dict__[name] = ValueError

    def __delattr__(self, name):
        if name == "secret":
            raise AttributeError("Cannot delete secret")
        del self.__dict__[name]

        
#obj = ProtectedAttributes()
#print(obj._protected)  # Access allowed
#print(obj.missing)  # Outputs "missing not found"
#obj.secret  # Raises "AttributeError: Access Denied"
#obj.secret = "New"  # Raises "AttributeError: Cannot modify secret"



### Container Magic Methods ###
class Library:
    def __init__(self, books):
        self.books = books

    def __len__(self):
        return len(self.books)
    
    def __getitem__(self, index):
        return self.books[index]
    
#library = Library(["Book1", "Book2", "Book3"])
#print(len(library))
#print(library[1])




### Context Managers ###
class AsyncFileHandler:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    async def __aenter__(self):
        # Called at the beginning of a "with" block, and optionally returns
        # an object that is bound to the variable after the "as" keyword
        # in the "with" statement
        self.file = await aiofiles.open(self.filename, self.mode)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Called after the "with" block. The three arguments are used
        # to manage exceptions
        await self.file.close()

#import asyncio
#import aiofiles
#async def main():
#    async with AsyncFileHandler('example.txt', 'w') as f:
#        await f.write('Hello, aync world!')



### Callable Objects ###
import asyncio
class AsyncAdder:
    def __init__(self, value):
        self.value = value
    
    async def __call__(self, x):
        # Allows an instance of a class to be called as a function
        await asyncio.sleep(1)
        return self.value + x
    
async def main():
    add_ten = AsyncAdder(10)
    result = await add_ten(20)
    print(result)

#asyncio.run(main())




### Using __slots__
class Player:
    # With __slots__ we declare that instances of the class will have 
    # a fixed set of attributes.
    __slots__ = ['name', 'score']

    def __init__(self, name, score):
        self.name = name
        self.score = score

#player1 = Player("Alice", 100)
#print(player1.name)
#print(player1.score)
#player1.age = 25  # <= It raises an error



### Classmethod and Staticmethod Decorator ###

# Classmethods are bound to the class (and not to to the instance), 
# so that they can access the class attributes (but not the instance attributes)
# They are commonly used for factory methods
class Person:
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_full_name(cls, full_name):
        name = full_name.split()[0]
        return cls(name)
    
#john = Person.from_full_name("John Doe")
#print(john.name)

# Staticmethods don't access instance or class data. They do not receive
# an implicif first argument (neither self nor cls). They are useful
# for utility functions that perform a task in isolation.
class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(celsius):
        return (celsius * 9/5) + 32


#fahrenheit = TemperatureConverter.celsius_to_fahrenheit(0)
#print(fahrenheit)




### Property Decorators ###

# They allow for the management of class attributes. We can use them
# to implement getter, setter, and deleter methods

# A getter method is used to access the value of an attribute,
# without directly exposing the attribute itself
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        # A getter method is used to access the value of an attribute,
        # without directly exposing the attribute itself
        return self._celsius
    
#temp = Temperature(100)
#print(temp.celsius)


class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        # A setter method is used to set the value of an attribute. It provides 
        # a controlled way of setting attributes, allowing for validation or computation.
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value

#temp = Temperature(0)
#temp.celsius = -300  # <= Raises ValueError



class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
        
    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        
    @celsius.deleter
    def celsius(self):
        # A deleter method is used to delte an attribute. It's useful when deleting
        # an attribute requires more than just removing it from the object's __dict__
        print("Deleting celsius!")
        del self._celsius

#temp = Temperature(50)
#del temp.celsius


