# ML-DL-from-SU

> Stanford University Machine Learning(CS229) and Deep Learning(CS230) Course by Andrew Ng.

:tada: :tada: :tada: Bravo, notes modified finished and pdf files save to the docs/, welcome to read. Thanks for PhD huang with his amazing work.

Here are something I learned from Andrew Ng, that's really amazing. And the notes just modified from PhD Huang and I fixed some typos and make some changes. 

The files under the code/tf-example/ just from TensorFlow docs, I make some changes and they all worked well, I just checked it.

Of course, before you execute these files please be sure you have required dependences. 

Have fun, always updating!

## Noted

The ML(CS229)/DL(CS230) vedio courses noticed in this repo producted in 2014, so you should know some libraries have changed in TensorFlow and Keras, please ensure what exactly you do before you run it! 

## Issues

### When I run `python3 xxx.py`, the terminal show me below message:

```
➜ python3 basic_image_classify.py
[1, 2, 3, 4]
Traceback (most recent call last):
  File "basic_image_classify.py", line 6, in <module>
    import tensorflow as tf
  File "/usr/local/lib/python3.7/site-packages/tensorflow/__init__.py", line 101, in <module>
    from tensorflow_core import *
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/__init__.py", line 40, in <module>
    from tensorflow.python.tools import module_util as _module_util
ModuleNotFoundError: No module named 'tensorflow.python.tools'; 'tensorflow.python' is not a package
```
Just remove some non-relative .py files and folders or rename some files, then the command will work again. I don't know why but it really worked!

### `griddata` removed from matlablib, use `scipy.interpolate.griddata`.

