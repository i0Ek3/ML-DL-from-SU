#!/usr/bin/env python3

import threading
import time

balance = 0
lock = threading.Lock()

def changeit(n):
    global balance
    balance += n
    balance -= n

def run_thread(n):
    for i in range(1000000):
        changeit(n)

def run_thread_with_lock(n):
    for i in range(1000000):
        lock.acquire()
        try:
            changeit(n)
        finally:
            lock.release()


t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
