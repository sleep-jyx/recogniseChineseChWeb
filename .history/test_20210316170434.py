import os
import re
import pickle

#f = pickle.load('char_dict')
f = open('char_dict')
print(type(f))
d = dict(f)
print(type(d))
# print(f[1025])
