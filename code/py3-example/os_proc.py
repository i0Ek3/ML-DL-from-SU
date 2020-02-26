#!/usr/bin/env python3

import os

print('Proc (%s) start....' % os.getpid())
pid = os.fork()
if pid == 0:
    print('I am child, my pid = (%s)' % os.getpid())
else:
    print('I am father, my pid = (%s)' % os.getpid())

