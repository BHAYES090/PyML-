# Variables and Printing

age = 23
print(age)

#Use underscore to separate words (snake Case)
friend_age = 30

#Numbers
PI = 3.1415926 #Float
age = 25 #integer

math_operation = 1 * 3 +4 / 2-3
print(math_operation)

float_division = 13 / 3
print(float_division)

integer_division = 13 // 3 # double divisors remove everything after decimal place Does not round numbers
print(integer_division)

# Calcuate Remainder after divison
integer_division = 13 // 5
print(integer_division)

remainder = 13 % 5
print(remainder)

x = 37
remainder = x % 2
print(remainder)

#Strings
my_string = "Hello World"
print(my_string)

#String properties
string_with_quotes = "Hello, its's me."
another_with_quotes = "He Said 'You are amazing!' yesterday"
another_with_quotes = 'He Said "You are amazing!" yesterday'
another_with_quotes = "He Said \"You are amazing!\" yesterday"
another_with_quotes = "He Said 'You are amazing!'" + "yesterday"

multiline = """Hello World!

Welcome to my program
"""
print(multiline)

#convert age (number) to string and print
age = 34
age_to_string = str(age)
print("You are: " + age_to_string)

#Use f-string and .format() to complete the same action
age = 34
print(f"You are: {age}")

name = "Jose"
greeting = f"How are you {name}?"
print(greeting)

name = "Bob"
final_greeting = "How are you {}?"
bob_greeting = final_greeting.format(name)
print(bob_greeting)

#Get User input
my_name = "Bobby"
your_name = input("Enter your name: ")
print(f"Hello {your_name}. My name is {my_name}")

age = input("Enter your age: ")
print(f"You have lived for {age * 12} months.") #Because age is a string in the format, it repeats it 12 times

age = input("Enter your age: ")
age_num = int(age) #changes the age variable (str) to an integer
print(f"You have lived for {age_num * 12} months.")

age = int(input("Enter your age: ")) # creates a single line user input and integer switch 
print(f"You have lived for {age * 12} months.")

#Booleans
truthy = True
falsey = False

age = 20
is_over_age = age >= 18 # Result is True
is_under_age = age <= 18 # Result is False
is_twenty = age == 20 #Result is True

my_number = 5
user_number = int(input("Enter a number: "))

matches = my_number == user_number #boolean to compare user_number and my Number

print(f"You got it right: {matches}.")

# and, &, or
age = int(input("Enter your age: "))
can_learn_py = age > 0 and age < 150 # adding <and> makes is so both rules have to be true for the boolean to be True

print(f"You can learn Python: {can_learn_py}.")

age = int(input("Enter your age: "))
usually_working = age >= 18 or age <= 65 #adding <or> makes it so both rules can be true for he Boolean to be True

print(f"At {age}, you are usually working: {usually_working}")

print(bool(35)) #takes a value and converts it to a boolean
print(bool("String")) #works with a string too

print(bool(0)) #result is false
print(bool("")) #result is false

True and False # here and looks at the first value and if the first value is True it returns the second value

x = True and False
print(x) # looks at the first value and becuase it is True, it gives you the second value

# <and> gives you the first value if it is False otherwise it gives you the second value

x = 35 and 0 #first value is True so 0 is what is returned
print(x)

x = 0 and 35
print(x)

# or will give you the first value if it is True, otherwise it will give you the second value
x = 35 or 0 # here it will return 35 because 35 is True
print(x)

#Here it is turned around
x = 0 or 35
print(x)















