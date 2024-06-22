

### Basics of Typing ###
def headline(text: str, centered: bool = False) -> str:
    if not centered:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f" {text.title()}".center(50, "o")

# pip install mypy

#print(headline("python type checking"))
#print(headline("use mypy", centered=True))



### Annotations ###
import math

def circumference(radius: float) -> float:
    return 2 * math.pi * radius

#print(circumference)
#print(circumference.__annotations__)

# mypy -> reveal_type(math.pi)
# radius = 1
# circumference = 2 * math.pi *radius
# reveal_locals()


pi: float = 3.142

def circumference(radius: float) -> float:
    return 2 * pi * radius

#print(__annotations__)


nothing: str
#print(__annotations__)

name: str = "Guido"
pi: float = 3.142
centered: bool = False



### Composite Type Hints ###

from typing import List, Tuple, Dict, Sequence

import random
SUITS = "P C D T".split()
RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()

def create_deck(shuffle: bool = False) -> List[Tuple[str, str]]:
    deck = [(s, r) for r in RANKS for s in SUITS]
    if shuffle:
        random.shuffle(deck)
    return deck

def deal_hands(deck):
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])

def play():
    deck = create_deck(shuffle=True)
    names = "P1 P2 P3 P4".split()
    hands = {n: h for n, h in zip(names, deal_hands(deck))}
    for name, cards in hands.items():
        card_str = " ".join(f"{s}{r}" for (s, r) in cards)
        print(f"{name}: {card_str}")

# play()


names: List[str] = ["Guido", "Jukka", "Ivan"]
version: Tuple[int, int, int] = (3, 7, 1)
options: Dict[str, bool] = {"centered": False, "capitalize": True}

def square(elems: Sequence[float]) -> List[float]:
    return [x**2 for x in elems]



### Type Aliases ###
Card = Tuple[str, str]
Deck = List[Card]

def deal_hands(deck: Deck) -> Tuple[Deck, Deck, Deck, Deck]:
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])

#print(Deck)


### Functions Without Return Values ###
def play(player_name: str) -> None:
    print(f"{player_name} plays")

ret_val = play("Jacob")

from typing import NoReturn
def black_hole() -> NoReturn:
    raise Exception("There is no going back...")



### The Any Type ###

from typing import Any

Card = Tuple[str, str]
Deck = list[Card]

def create_deck(shuffle: bool = False) -> Deck:
    deck = [(s, r) for r in RANKS for s in SUITS]
    if shuffle:
        random.shuffle(deck)
    return deck

def deal_hands(deck: Deck) -> Tuple[Deck, Deck, Deck, Deck]:
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])



from typing import TypeVar

Choosable = TypeVar("Choosable", str, Card)
def choose(items: Sequence[Choosable]) -> Choosable:
    return random.choice(items)


def player_order(names, start=None):
    if start is None:
        start = choose(names)
    start_idx = names.index(start)
    return names[start_idx:] + names[:start_idx]

def play() -> None:
    deck = create_deck(shuffle=True)
    names = "P1 P2 P3 P4".split()
    hands = {n: h for n, h in zip(names, deal_hands(deck))}
    start_player = choose(names)
    turn_over = player_order(names, start=start_player)

    while hands[start_player]:
        for name in turn_over:
            card = choose(hands[name])
            hands[name].remove(card)
            print(f"{name}: {card[0] + card[1]:<3}  ", end="")
        print()

# play()



### Duck Types and Protocols ###
from typing import Sized

def len(obj: Sized) -> int:
    return obj.__len__()


### The Optional Type ###
from typing import Sequence, Optional

def player_order(
        names: Sequence[str], start: Optional[str] = None
) -> Sequence[str]:
    pass



### Annotating Classes ###
import random
import sys

class Card:
    SUITS = "S1 S2 S3 S4".split()
    RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()

    def __init__(self, suit: str, rank: str) -> None:
        self.suit = suit
        self.rank = rank

    def __repr__(self) -> str:
        return f"{self.suit}{self.rank}"
    


class Deck:
    def __init__(self, cards: List[Card]) -> None:
        self.cards = cards
        
    @classmethod
    def create(cls, shuffle: bool = False) -> "Deck":
        # Alternative annotation
        # from __future__ import annotations
        # def create(cls, shuffle: bool = False) -> Deck
        cards = [Card(s, r) for r in Card.RANKS for s in Card.SUITS]
        if shuffle:
            random.shuffle(cards)
        return cls(cards)
    
    def deal(self, num_hands):
        cls = self.__class__
        return tuple(
            cls(self.cards[i::num_hands]) 
            for i in range(num_hands)
        )
    
class Player:
    def __init__(self, name: str, hand: Deck) -> None:
        self.name = name
        self.hand = hand

    def play_card(self):
        card = random.choice(self.hand.cards)
        self.hand.cards.remove(card)
        print(f"{self.name}: {card!r:<3}  ", end="")
        return card


class Game:
    def __init__(self, *names: str):
        deck = Deck.create(shuffle=True)
        self.names = (list(names) + "P1 P2 P3 P4".split())[:4]
        self.hands = {
            n: Player(n, h)
            for n, h in zip(self.names, deck.deal(4))
        }

    def play(self):
        start_player = random.choice(self.names)
        turn_over = self.player_order(start=start_player)

        while self.hands[start_player].hand.cards:
            for name in turn_over:
                self.hands[name].play_card()
            print()

    def player_order(self, start=None):
        if start is None:
            start = random.choice(self.names)
        start_idx = self.names.index(start)
        return self.names[start_idx:] + self.names[:start_idx]
    
# player_names = sys.argv[1:]
# game = Game(*player_games)
# game.play()



### Returning SELF or CLS ###
from datetime import date
from typing import Type, TypeVar
TAnimal = TypeVar("TAnimal", bound="Animal")

class Animal:
    def __init__(self, name: str, birthday: date) -> None:
        self.name = name
        self.birthday = birthday

    @classmethod
    def newborn(cls: Type[TAnimal], name: str) -> TAnimal:
        return cls(name, date.today())
    
    def twin(self: TAnimal, name: str) -> TAnimal:
        cls = self.__class__
        return cls(name, self.birthday)
    

class Dog(Animal):
    def bark(self) -> None:
        print(f"{self.name} says woof!")

#fido = Dog.newborn("Fido")
#pluto = fido.twin("Pluto")
#fido.bark()
#pluto.bark()


### Callables ###
from typing import Callable

def do_twice(func: Callable[[str], str], argument: str) -> None:
    print(func(argument))
    print(func(argument))

def create_greeting(name: str) -> str:
    return f"Hello {name}"

# do_twice(create_greeting, "Jekyll")


### Additional Information ###

# Third-Party Packages
import numpy as np  # type: ignore <= To ignore Mypy warnings about third-party packages
def print_cosine(x: np.ndarray) -> None:
    with np.printoptions(precision=3, suppress=True):
        print(np.cos(x))

x = np.linspace(0, 2 * np.pi, 9)
print_cosine(x)


# Mypy Configuration File
"""mypy.ini
[mypy]

[mypy-numpy]
ignore_missing_imports = True
"""

