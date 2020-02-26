#!/usr/bin/env python
# coding=utf-8

def higher(a, b, f):
    return f(a, b)

def f(x, y):
    return x ** y

print(higher(2, 3, f))
