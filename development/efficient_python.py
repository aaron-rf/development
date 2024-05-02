
### Caching ###
# pip install cachetools
from cachetools import cached, TTLCache, LRUCache, LFUCache, RRCache
import time
@cached(cache=TTLCache(maxsize=1024, ttl=86400))
def get_data():
    """Get some data from a slow API"""
    time.sleep(1)
    return {"data": "some data"}


@cached(cache = {})
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)

@cached(cache=LRUCache(maxsize=3))
def myfun(n):
    s = time.time()
    time.sleep(n)
    print("\nTime taken: ", time.time() - s)
    return (f"I am executed: {n} seconds")

@cached(cache=LFUCache(maxsize=33))
def some_fun():
    pass

@cached(cache=RRCache(maxsize=33))
def some_fun():
    pass

### End Caching ###


### Efficient Operations ###
import math 
def function():
    sqrt = math.sqrt
    for i in range(100):
        sqrt(i)

# String concatenation
my_strings = ['Hello', 'world!']
' '.join(my_strings)

# Profiling
import cProfile
def slow_function():
    total = 0
    for i in range(10_000_000):
        total += i
    return total

cProfile.run('slow_function()')


# Using build-in functions
wordlist = ['...']
newlist = map(str.upper, wordlist)

# Generators
def countdown(n):
    while n > 0:
        yield n 
        n -= 1

for num in countdown(5):
    print(num)


### End Efficient Operations ###




### Least Recently Used (LRU) Cache Strategy ###
from functools import lru_cache 

@lru_cache
def steps_to(stair):
    if stair == 1:
        return 1
    elif stair == 2:
        return 2
    elif stair == 3:
        return 4
    else:
        return (
            steps_to(stair -3)
            + steps_to(stair - 2)
            + steps_to(stair - 1)
        )
    
# Information about the hits & misses of the cache:
# print(steps_to.cache_info())

### END Least Recently Used (LRU) Cache Strategy ###


### Caching with a Dictionary ###
import requests 

cache = dict()

def get_article_from_server(url):
    response = requests.get(url)
    return response.text 

def get_article(url):
    if url not in cache:
        cache[url] = get_article_from_server(url)
    return cache[url]
### END Caching with a Dictionary ###

