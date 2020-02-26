#!/usr/bin/env python3

from contextlib import closing, contextmanager
from urllib.request import urlopen

with closing(urlopen('https://www.google.com')) as page:
    for line in page:
        print(line)

@contextmanager
def closing(thing):
    try:
        yield thing
    finally:
        thing.close()

