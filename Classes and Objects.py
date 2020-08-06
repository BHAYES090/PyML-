# Change the output produced by printing or viewing
# instances to something more sensible.

##To change the string representation of an instance,
##define the __str__() and __repr__() methods. For example:

class Pair(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return 'Pair({0.x!r}, {0.y!r})'.format(self)
    def __str__(self):
        return '({0.x!s}, {0.y!s})'.format(self)

##The __repr__() method returns the code representation
##of an instance, and is usually the text you would type
##to re-create the instance. The built-in repr() function
##returns this text, as does the interactive interpreter
##when inspecting values.

##The __str__() method converts
##the instance to a string, and is the output produced
##by the str() and print() functions. For example:

p = Pair(3, 4)
p
print(p)

##The implementation of this recipe also shows how different
##string representations may be used during formatting.
##Specifically, the special !r formatting code indicates
##that the output of __repr__() should be used instead of
##__str__(), the default. You can try this experiment
##with the preceding class to see this:

p = Pair(3, 4)
print('p is {0!r}'.format(p))
print('p is {0}'.format(p))

##Defining __repr__() and __str__() is often good practice,
##as it can simplify debugging and instance output. For example,
##by merely printing or logging an instance, a programmer will be shown more
##useful information about the instance contents.

##It is standard practice for the output of __repr__() to produce
##text such that eval(repr(x)) == x. If this is not possible or desired,
##then it is common to create a useful textual representation enclosed
##in < and > instead. For example:

##>>> f = open('file.dat')
##>>>f
##<_io.TextIOWrapper name='file.dat' mode='r' encoding='UTF-8'>

##If no __str__() is defined, the output of __repr__() is used as a fallback.

##The use of format() in the solution might look a little funny,
##but the format code {0.x} specifies the x-attribute of argument 0. So, in
##the following function, the 0 is actually the instance self:

def __repr__(self):
      return 'Pair({0.x!r}, {0.y!r})'.format(self)

##As an alternative to this implementation, you could also use the % operator and the following code:

def __repr__(self):
     return 'Pair(%r, %r)' % (self.x, self.y)


##############################################################################


#object to support customized formatting through the format() function and string method.

##To customize string formatting, define the __format__() method on a class. For example:

_formats = {
    'ymd' : '{d.year}-{d.month}-{d.day}',
    'mdy' : '{d.month}/{d.day}/{d.year}',
    'dmy' : '{d.day}/{d.month}/{d.year}'
    }

class Date(object):
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    def __format__(self, code):
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)

##Instances of the Date class now support formatting operations such as the following:

d = Date(2012, 12, 21)
format(d)
format(d, 'mdy')
'The date is {:ymd}'.format(d)
'The date is {:mdy}'.format(d)

##The __format__() method provides a hook into Python’s string
##formatting functionality. It’s important to emphasize that
##the interpretation of format codes is entirely up to the class
##itself. Thus, the codes can be almost anything at all. For example,
##consider the following from the datetime module:

from datetime import date
d = date(2012, 12, 21)
format(d)
format(d,'%A, %B %d, %Y')
'The end is {:%d %b %Y}. Goodbye'.format(d)
