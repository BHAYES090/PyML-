#process items in an iterable, but for whatever reason,
#you can’t or don’t want to use a for loop.

##To manually consume an iterable, use the next()
##function and write your code to catch the StopIteration
##exception. For example, this example manually reads lines from a file:

with open('/etc/passwd') as f:
    try:
        while True:
            line = next(f)
            print(line, end='')
    except StopIteration:
        pass

##Normally, StopIteration is used to signal the end of iteration. However,
##if you’re using next() manually (as shown), you can also instruct it to
##return a terminating value, such as None, instead. For example:

with open('/etc/passwd') as f:
     while True:
         line = next(f, None)
         if line is None:
             break
         print(line, end='')

##In most cases, the for statement is used to consume an iterable.
##However, every now and then, a problem calls for more precise
##control over the underlying iteration mechanism. Thus, it is
##useful to know what actually happens.

##The following interactive example illustrates the basic
##mechanics of what happens during iteration:

items = [1, 2, 3]
# Get the iterator
it = iter(items)     # Invokes items.__iter__()
# Run the iterator
next(it)             # Invokes it.__next__()
next(it)
next(it)
next(it)  # Generates an error

##Subsequent recipes in this chapter expand on
##iteration techniques, and knowledge of the basic
##iterator protocol is assumed. Be sure
##to tuck this first recipe away in your memory.


#################################################################################################################


#implement a custom iteration pattern that’s different than the usual built-in
#functions (e.g., range(), reversed(), etc.).

##If you want to implement a new kind of iteration pattern,
##define it using a generator function. Here’s a
##generator that produces a range of floating-point numbers:

def frange(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment

##To use such a function, you iterate over it using a
##for loop or use it with some other function that consumes an iterable
##(e.g., sum(), list(), etc.). For example:

for n in frange(0, 4, 0.5):
    print(n)

list(frange(0, 1, 0.125))

##The mere presence of the yield statement in a function
##turns it into a generator. Unlike a normal function, a
##generator only runs in response to iteration. Here’s an
##experiment you can try to see the
##underlying mechanics of how such a function works:

def countdown(n):
    print('Starting to count from', n)
    while n > 0:
        yield n
        n -= 1
    print('Done!')


# Create the generator, notice no output appears
c = countdown(3)
c


# Run to first yield and emit a value
next(c)


# Run to the next yield
next(c)


# Run to next yield
next(c)


# Run to next yield (iteration stops)
next(c)

##The key feature is that a generator function only runs
##in response to "next" operations carried out in iteration.
##Once a generator function returns, iteration stops. However,
##the for statement that’s usually used to iterate takes care of these details,
##so you don’t normally need to worry about them.


#################################################################################################################


# iterate in reverse over a sequence.

##Use the built-in reversed() function. For example:

a = [1, 2, 3, 4]
for x in reversed(a):
    print(x)

##Reversed iteration only works if the object in
##question has a size that can be determined or if the
##object implements a _reversed_() special method. If
##neither of these can be satisfied, you’ll have to
##convert the object into a list first. For example:

# Print a file backwards
f = open('somefile')
for line in reversed(list(f)):
    print(line, end='')

##Be aware that turning an iterable into a list as shown
##could consume a lot of memory if it’s large.

class Countdown(object):
    def __init__(self, start):
        self.start = start

    # Forward iterator
    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

    # Reverse iterator
    def __reversed__(self):
        n = 1
        while n <= self.start:
            yield n
            n += 1

##Defining a reversed iterator makes the code much more
##efficient, as it’s no longer necessary to pull the data
##into a list and iterate in reverse on the list.


#################################################################################################################


# iterate over all of the possible combinations or
# permutations of a collection of items.

##The itertools module provides three functions for this task.
##The first of these--itertools.permutations()—takes a collection
##of items and produces a sequence of tuples that rearranges all of
##the items into all possible permutations
##(i.e., it shuffles them into all possible configurations). For example:

items = ['a', 'b', 'c']
from itertools import permutations
for p in permutations(items):
    print(p)

##If you want all permutations of a smaller length, you can
##give an optional length argument. For example:

for p in permutations(items, 2):
    print(p)

##Use itertools.combinations() to produce a sequence of
##combinations of items taken from the input. For example:

from itertools import combinations
for c in combinations(items, 3):
    print(c)

for c in combinations(items, 2):
    print(c)

for c in combinations(items, 1):
    print(c)

##For combinations(), the actual order of the
##elements is not considered. That is, the combination
##('a', 'b') is considered to be the same as ('b', 'a')
##(which is not produced).

##When producing combinations, chosen items are
##removed from the collection of possible candidates
##(i.e., if 'a' has already been chosen, then it is removed from consideration).
##The itertools.combinations_with_replacement() function relaxes this, and allows the same item
##to be chosen more than once. For example:

from itertools import combinations_with_replacement
for c in combinations_with_replacement(items, 3):
    print(c)

##This recipe demonstrates only some of the power
##found in the itertools module. Although you could
##certainly write code to produce permutations and
##combinations yourself, doing so would probably require
##more than a fair bit of thought. When faced with seemingly
##complicated iteration problems, it always pays to look at
##itertools first. If the problem is common,
##chances are a solution is already available.


#################################################################################################################


 #iterate over a sequence, but would like to keep track
 #of which element of the sequence is currently being processed.

##The built-in enumerate() function handles this quite nicely

my_list = ['a', 'b', 'c']
for idx, val in enumerate(my_list):
    print(idx, val)

##For printing output with canonical line numbers
##(where you typically start the numbering at 1 instead of 0),
##you can pass in a start argument:

my_list = ['a', 'b', 'c']
for idx, val in enumerate(my_list, 1):
    print(idx, val)

##This case is especially useful for tracking
##line numbers in files should you want
##to use a line number in an error message:

def parse_data(filename):
    with open(filename, 'rt') as f:
         for lineno, line in enumerate(f, 1):
             fields = line.split()
             try:
                 count = int(fields[1])
                 ...
             except ValueError as e:
                 print('Line {}: Parse error: {}'.format(lineno, e))

##enumerate() can be handy for keeping track of the
##offset into a list for occurrences of certain values,
##for example. So, if you want to map words in a file
##to the lines in which they occur, it can easily be
##accomplished using enumerate() to map each word to the
##line offset in the file where it was found:

word_summary = defaultdict(list)

with open('myfile.txt', 'r') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    # Create a list of words in current line
    words = [w.strip().lower() for w in line.split()]
    for word in words:
        word_summary[word].append(idx)

##If you print word_summary after processing the file,
##it’ll be a dictionary (a defaultdict to be precise), and
##it’ll have a key for each word. The value for each word-key
##will be a list of line numbers that word occurred on. If the
##word occurred twice on a single line, that line number will be
##listed twice, making it possible to identify various simple metrics about the text.

##enumerate() is a nice shortcut for situations where you might be inclined to
##keep your own counter variable. You could write code like this:

lineno = 1
for line in f:
    # Process line
    ...
    lineno += 1

##But it’s usually much more elegant
##(and less error prone) to use enumerate() instead:

for lineno, line in enumerate(f):
    # Process line
    ...
##The value returned by enumerate() is an instance
##of an enumerate object, which is an iterator that returns
##successive tuples consisting of a counter and the value
##returned by calling next() on the sequence you’ve passed in.

##Although a minor point, it’s worth mentioning that sometimes
##it is easy to get tripped up when applying enumerate() to a
##sequence of tuples that are also being unpacked. To do it, you
##have to write code like this:

##``` data = [ (1, 2), (3, 4), (5, 6), (7, 8) ]

##Correct!
##for n, (x, y) in enumerate(data): ...

##Error!
##for n, x, y in enumerate(data): ... ```


################################################################################################################


# iterate over the items contained in more than one sequence at a time.

##To iterate over more than one sequence simultaneously,
##use the zip() function. For example:

xpts = [1, 5, 4, 2, 10, 7]
ypts = [101, 78, 37, 15, 62, 99]
for x, y in zip(xpts, ypts):
    print(x,y)

##zip(a, b) works by creating an iterator that produces
##tuples (x, y) where x is taken from a and y is taken from b.
##Iteration stops whenever one of the input sequences is exhausted.
##Thus, the length of the iteration is the same as the length of the
##shortest input. For example:

a = [1, 2, 3]
b = ['w', 'x', 'y', 'z']
for i in zip(a,b):
    print(i)

##If this behavior is not desired, use itertools.zip_longest() instead. For example:

from itertools import zip_longest
for i in zip_longest(a,b):
    print(i)

for i in zip_longest(a, b, fillvalue=0):
    print(i)

##zip() is commonly used whenever you need to pair data together.
##For example, suppose you have a list of column headers and column values like this:

headers = ['name', 'shares', 'price']
values = ['ACME', 100, 490.1]

##Using zip(), you can pair the values together to make a dictionary like this:

s = dict(zip(headers,values))

##Alternatively, if you are trying to produce output, you can write code like this:

for name, val in zip(headers, values):
    print(name, '=', val)

##It’s less common, but zip() can be passed more than two sequences as input.
##For this case, the resulting tuples have the same number of items in
##them as the number of input sequences. For example:

a = [1, 2, 3]
b = [10, 11, 12]
c = ['x','y','z']
for i in zip(a, b, c):
    print(i)

##Last, but not least, it’s important to emphasize that
##zip() creates an iterator as a result. If you need the
##paired values stored in a list, use the list() function. For example:

zip(a, b)

list(zip(a, b))


#################################################################################################################


##code that uses a while loop to iteratively process
##data because it involves a function or some kind of
##unusual test condition that doesn’t fall into the usual
##iteration pattern.

##A somewhat common scenario in programs involving I/O is to write code like this:

CHUNKSIZE = 8192

def reader(s):
    while True:
        data = s.recv(CHUNKSIZE)
        if data == b'':
            break
        process_data(data)

##Such code can often be replaced using iter(), as follows:

def reader(s):
    for chunk in iter(lambda: s.recv(CHUNKSIZE), b''):
        process_data(chunk)

##If you’re a bit skeptical that it might work,
##you can try a similar example involving files. For example:

import sys
f = open('/etc/passwd')
for chunk in iter(lambda: f.read(10), ''):
    n = sys.stdout.write(chunk)

s

##A little-known feature of the built-in iter()
##function is that it optionally accepts a zero-argument
##callable and sentinel (terminating) value as inputs.
##When used in this way, it creates an iterator that
##repeatedly calls the supplied callable over and over
##again until it returns the value given as a sentinel.

##This particular approach works well with certain
##kinds of repeatedly called functions, such as those
##involving I/O. For example, if you want to read data
##in chunks from sockets or files, you usually have to
##repeatedly execute read() or recv() calls followed by
##an end-of-file test. This recipe simply takes these
##two features and combines them together into a single
##iter() call. The use of lambda in the solution is needed
##to create a callable that takes no arguments, yet still
##supplies the desired size argument to recv() or read().
