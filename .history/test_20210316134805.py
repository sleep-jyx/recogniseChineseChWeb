import os
import re
str = '../data/train\03754\98253.png'
out = str.split('\\')
out2 = re.split('[\\|/]', '../data/train\03754\98253.png')
print(out)
print(out2)
