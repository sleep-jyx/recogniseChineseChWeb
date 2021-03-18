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
with open(char_dict, 'r') as f:
    data = pickle.load(f)


print("Data type before reconstruction : ", type(data))
