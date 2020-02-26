#!/usr/bin/env python3
#-*- coding=utf-8 -*-

class Test(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'Test object (name: %s)' % self.name
    __repr__ = __str__

print(Test('Test'))
