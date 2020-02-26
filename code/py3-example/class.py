class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_stuff(self):
        print('%s %s' %  (self.name, self.score))

b = Student('bart', 99)
c = Student('lisa', 88)
print_stuff(b)
print_stuff(c)
