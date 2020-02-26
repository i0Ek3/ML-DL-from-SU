#!/usr/bin/env python
# coding=utf-8

def odd(x):
    return x & 0x1

print(list(filter(odd, [1,2,3,4,5,6,7])))
print(sorted(['aaa', 'ddd', 'bbb', 'ccc'], key=str.upper))

L = [('bob', 99), ('Adam', 100), ('Lisa', 98)]
print(sorted(L, key=lambda L:L[1], reverse=True))
