import os
import re
str = '../data/train\03754\98253.png'
out = str.split('\\')
out2 = re.split('[\\|/]', str)
print(out)
print(out2)