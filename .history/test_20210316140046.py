import os
import re
str = '../data/train\03754\98253.png'

text = '你好！吃早饭了吗？再见。'
print(type(str))
out = str.split('\\')

out2 = re.split('。|/|\\\\|？', str)
print(out)

print(out2)
