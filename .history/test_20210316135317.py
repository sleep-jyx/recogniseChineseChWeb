import os
import re
str = '../data/train\03754\98253.png'
print(type(str))
out = str.split('\\')
#out2 = re.split('[\\]', r'../data/train\03754\98253.png')
print(out)
print(str.split.__doc__)
# print(out2)
