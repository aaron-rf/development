
### Classes are Objects ###
class Person:
    pass

#print(id(Person))

class Child(Person):
    pass

#print(id(Child))

child = Child()

#print(id(child))

#print(type(Child))
#print(type(child))


class Parent:
    def __init__(self, name, age):
        self.name = name
        self.age = age

parent = Parent("John", 35)


class Parent:

    def __new__(cls, name, age):
        print("new is called")
        return super().__new__(cls)
    

    def __init__(self, name, age):
        print("init is called")
        self.name = name
        self.age = age

#parent = Parent('John', 35)
#print(isinstance(object, object))

### Callables ###

class Parent:
    def __new__(cls, name, age):
        print("new is called")
        return super().__new__(cls)
    
    def __init__(self, name, age):
        print("init is called")
        self.name = name
        self.age = age

    def __call__(self):
        print("Parent here!")

#parent = Parent("John", 35)
#parent()



### Metaclasses ###
class MyMeta(type):
    def __call__(self, *args, **kwargs):
        print(f'{self.__name__} is called'
              f' with args={args}, kwargs={kwargs}' )
        
class Parent(metaclass=MyMeta):
    def __new__(cls, name, age):
        print("new is called")
        return super().__new__(cls)
    
    def __init__(self, name, age):
        print("init is called")
        self.name = name
        self.age = age

#parent = Parent("John", 35)
#print(type(parent))


class MyMeta(type):
    def __call__(cls, *args, **kwargs):
        print(f'{cls.__name__} is called'
              f'with args={args}, kwargs={kwargs}')
        print("metaclass calls __new__")
        obj = cls.__new__(cls, *args, **kwargs)

        if isinstance(obj, cls):
            print("metaclass calls __init__")
            cls.__init__(obj, *args, **kwargs)

        return obj

class Parent(metaclass=MyMeta):
    def __new__(cls, name, age):
        print("new is called")
        return super().__new__(cls)
    
    def __init__(self, name, age):
        print("init is called")
        self.name = name
        self.age = age
        
#parent = Parent("John", 35)
#print(type(parent))
#print(str(parent))
