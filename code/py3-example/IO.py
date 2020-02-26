#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import os
from io import StringIO
from io import BytesIO

def SIO():
    f = StringIO('hello\nstringio!')
    while True:
        s = f.readline()
        if s == '':
            break
        print(s.strip())

def BIO():
    f = BytesIO()
    f.write('或者'.encode('utf-8'))
    print(f.getvalue())

def basic_io():
    fpath = './io.txt'

    #try:
    #    f = open('./io.txt', 'r')
    #    print(f.read())
    #finally:
    #    if f:
    #        f.close()

    with open(fpath, 'r') as f:
        s = f.read()
        #f.write('This is test.')
        print(s)

    #f = open('./xx.txt', 'r', encoding='gbk', errors='ignore')


def __main__():
    SIO()
    BIO()
    basic_io()
