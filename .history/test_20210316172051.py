import os
import re
import pickle
import json
#f = pickle.load('char_dict')
'''
f = open('char_dict')
print(type(f))
d = f.read()
print(type(d))
'''
# print(f[1025])
with open('char_dict', 'rb') as f:
    data = pickle.load(f)

# 根据值找键
keyIndex = list(data.values()).index(20)
key = list(data.keys())[keyIndex]
print(key)
