# Beginner Python

2 + 4

"Hello World"

1.5 + 2.5

(2^2)-(1-2)

x = 5
x *21

x + 5

magic_number = 70
c = 5
magic_number * x * c
1750
type(1)

type('x')

type(1.0)

type(x)

type("x")

type(" ")

"Hello" + "World"

"Hello" + " " + "World"

name = "Bobby"
"Hello" + " " + name

type("1")

1 + 1

"1" + "1"



str(1)

"Hello" + str(1)

"Hello " + str(1)

"Hello " + str(1.0)

len("Hello")

len("Hello ")

"A" * 5

h = "Happy"
b = "Birthday"
(h + b) * 10

h = "Happy "
b = "Birthday "

print("Hello")

##################################################################################################################

# Booleans

True

type(True)

False

type(False)

0 == 0

0 == 1

0 != 1

# A single = sign is Assignment and a == is Comparison (the same goes for !=)

1 > 0

1 < 0

2 >= 3

-1 <=0

"H" in "Hello"

"H" not in "Hello"

# Only capitalized True and False are booleans, Python is Case Sensitive

if 6 > 5:
    print("SDJK")

# The above results in the boolean True, if the expression is
# true it follows through with the code indented

if 0 > 2:
    print("0 > 2")

# The expression above is false so there is no output

if "banana" in "bananarama":
    print("the 80's suck...")

broth = 15
sister = 12

if broth < sister:
    print("A")
else:
    print("B")

# the work under the if is not true and the execution moves to the else statement

# Compound conditionals

x = 1
x > 0 and x < 2

1 < 2 and "X" in "abc"

# and's can be strung together infinetly, but if one of the booleans is false the entire line is false

"a" in "Hello" or "e" in "Hello"

# or's can also be strung together infinetly and the result will be true even if only one statement is true

1 <= 0 or "a" not in "abc"

# Here the output of the boolean is false becasue both statements are incorrect and there is no other data provided

temp = 32

if temp > 60 and temp < 75:
    print("good")
else:
    print("not good")

hour = 11
if hour < 7 or hour > 23:
    print("Go away")
    print("NO way")
else:
    print("lets go")
    
broth = 12
sister = 15

if broth > sister:
    print("A")
elif sister == broth:
    print("AB")
else:
    print("B")

# else serves as the catch all, elif should serve as an intermediary for any
# other missing conditions you may want to add


##############################################################################################################


# Lists
# Lists store an ordered collection of items

your_list = ["a", "b", "c"]

type(your_list)

len(your_list)

"a" in your_list

"z" in your_list

"z" not in your_list

your_list[0]

# Python starts indexing at 0

your_list.append('d')

# adds d or APPENDS d to the end of the list
# append is a function

len(your_list)

her_list = []
len(her_list)

names = ["Alice", "Amy"]
names.append("Adam")
len(names)

names[0]

names[0] = "Jimmy"
names

names[2] = "Rachel"

names.append('Tim')

# gets the last name from the list
names[-1]

# tells you the last index number of the list
len(names) - 1

# gives you the last name from the list
names[len(names) - 1]

# an easier way to get the last name from a list
names[-1]

names=[]

my_name = 'Bobby'

names = ["Alice", "Bob", "Cassie", "Diane", "Ellen"]
for name in names:
    print(name)

for x in names:
    print(x)

for word in names:
    print("Hello " + word)

name = "Alice"
name[0]

name[0] in ["A", "E", "I", "O", "U"]
name[0] in "AEIOU"

for name in names:
    if name[0] in "AEIOU":
        print("Name starts with vowel: " + name)

vowel_names = []
for name in names:
    if name[0] in "AEIOU":
        vowel_names.append(name)

print(vowel_names)

#the following updates the number total for each price in prices

prices = [1.5, 2.35, 5.99, 16.49]
total = 0
for price in prices:
    total = total + price

print(total)

# this is the built in python function that can sum the prices instead of using the for loop
print(sum(prices))


#Dictionaries are lists of Key/Value pairs, Python does not care about the order of the list, and two
#identical items
#cannot coexist aka Bob and another person named Bob
flavors = {"Alice": "chocolate", "Bob": "strawberry", "Cara": "mint chocolate chip"}
flavors["Alice"]

flavors["Eve"] = "rum raisin"
flavors

"Eve" in flavors

flavors["Bob"] = "vanilla"
flavors

flavors["Cara"] = "vanilla"
flavors

phone_numbers = {}
type(phone_numbers)

#############################################################################################################

# Modules

import random

print(random.randint(1, 6))

cards = ["Jack ", "King ", "Queen ", "Ace"]
print(random.choice(cards))

############################################################################################################

# While Lopps
counter = 0
while counter < 5:
    print("Hello" + str(counter))
    counter = counter + 1


counter = 0
while True:
    print("Hello" + str(counter))
    counter = counter + 1

    if counter >= 5:
        break
    
print("welcome")

while True:
    user_input = input("> ")
    if user_input == "quit":
        print("Goodbye")
        break
    else:
        print(user_input)

##############################################################################################################

f = open("Countries.txt", "r")

for line in f:
    print(line)

f.close()


################################################################################################################

# Classes

class Greeter(object):
    def __init__(self, name):
        self.name = name

    def Hello(self):
        print("Hello " + self.name)

    def Goodbye(self):
        print("Goodbye " + self.name)

g = Greeter("Bob")
g.Hello()
g.Goodbye()

g2 = Greeter("Adam")
g2.Hello()
g2.Goodbye()

import random

class Die(object):
    def __init__(self, sides):
        self.sides = sides
    def roll(self):
        return random.randint(1, self.sides)

d = Die(81)
print(d.roll())
print(d.roll())
print(d.roll())

class Deck(object):
    def shuffle(self):
        suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
        self.cards = []
        for suit in suits:
            for rank in ranks:
                self.cards.append(rank + " of " + suit)

        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()

d = Deck()
d.shuffle()
print(d.deal())
print(d.deal())
print(d.deal())










