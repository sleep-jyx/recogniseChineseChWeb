import os
import re
import pickle

f = pickle('char_dict', mode='rb')

print(type(f))
print(f[1025])
